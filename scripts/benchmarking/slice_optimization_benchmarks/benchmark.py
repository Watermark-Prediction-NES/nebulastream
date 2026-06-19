#!/usr/bin/env python3

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure the effect of preemptive slice creation and slice-recycle pool on memory-source
benchmark queries.

Sweep: cartesian product of (query × preemptive_create_horizon_ms × recycle_pool_size × threads).
Each cell runs `systest -b` once and appends one row to results/<run-id>/results.csv.

Knobs (both sentinel-disabled, 0 = off; see SlicePreallocationConfiguration.hpp):
  preemptive_create_horizon_ms  build-side lookahead, creates upcoming slices in one critical section
  recycle_pool_size             LIFO of slices retired by probe-side GC, reused by build-side

The four corners (off/off, on/off, off/on, on/on) isolate each effect and their interaction.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BENCH_DIR.parents[2]
BUILD_DIR = Path(os.environ.get("BENCH_BUILD_DIR_OVERRIDE") or (SOURCE_DIR / "cmake-build-bench"))
SYSTEST_BIN = BUILD_DIR / "nes-systests" / "systest" / "systest"
TESTDATA_DIR = BUILD_DIR / "nes-systests" / "testdata"
RESULTS_ROOT = BENCH_DIR / "results"

### Subset of nes-systests/benchmark/memory-source/, chosen for nontrivial windowing
### (every entry has at least one tumbling/sliding aggregation that exercises the slice store).
QUERIES = {
    "CM1": "nes-systests/benchmark/memory-source/ClusterMonitoring_memory.test:01",
    "CM2": "nes-systests/benchmark/memory-source/ClusterMonitoring_memory.test:02",
    "LRB1": "nes-systests/benchmark/memory-source/LinearRoadBenchmark_memory.test:01",
    "LRB2": "nes-systests/benchmark/memory-source/LinearRoadBenchmark_memory.test:02",
    "MA": "nes-systests/benchmark/memory-source/Manufacturing_memory.test:01",
    "YSB1": "nes-systests/benchmark/memory-source/YahooStreamingBenchmark_with_varsized_memory.test:01",
    "YSB2": "nes-systests/benchmark/memory-source/YahooStreamingBenchmark_with_varsized_memory.test:02",
    "NM5": "nes-systests/benchmark/memory-source/Nexmark_with_varsized_memory.test:04",
    "NM8": "nes-systests/benchmark/memory-source/Nexmark_with_varsized_memory.test:05",
    "NM8_Variant": "nes-systests/benchmark/memory-source/Nexmark_with_varsized_memory.test:06",
}

### 0 = off. Same value used for both knobs so the matrix stays a clean 2x2 by default
### (off/off baseline, preemptive-only, recycle-only, both-on).
PREEMPTIVE_HORIZON_MS = [0, 10]
RECYCLE_POOL_SIZE = [0, 32]
WORKER_THREADS = [1, 16]

### Default worker buffer config — large (1MB) buffers keep cells fast.
BUFFER_SIZE_BYTES = 1_048_576
BUFFERS_IN_GLOBAL_POOL = 40_000
PAGE_SIZE = 8192

### Per-query buffer overrides: (operator_buffer_size, number_of_buffers). The heaviest varsized
### windowed Nexmark queries need finer-grained allocation and more slots; the default 1MB pool makes
### them crawl/stall, so give them 128KiB buffers × a larger pool. Other queries use the fast default.
BUFFER_OVERRIDES = {
    "NM8_Variant": (131_072, 320_000),
    "NM8": (131_072, 320_000),
}

### Per-cell wall-clock cap. The heavy varsized queries can run for many minutes; without a bound a
### single slow/stuck cell blocks the whole sweep. On timeout the cell is recorded as failed.
CELL_TIMEOUT_S = 1800

CMAKE_FLAGS = {
    "CMAKE_BUILD_TYPE": "Benchmark",
    "CMAKE_TOOLCHAIN_FILE": "/vcpkg/scripts/buildsystems/vcpkg.cmake",
    "USE_LIBCXX_IF_AVAILABLE": "OFF",
    "ENABLE_LARGE_TESTS": "1",
    "NES_BUILD_NATIVE:BOOL": "ON",
}
CMAKE_TARGETS = ["systest"]

RESULT_COLS = [
    "run_id", "query", "preemptive_horizon_ms", "recycle_pool_size", "threads",
    "time_s", "tuples_per_second", "bytes_per_second", "failure_reason",
]

def sh(cmd: str, *, env: dict | None = None) -> None:
    print(f"[bench] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


def cmake_configure_and_build() -> None:
    flags = " ".join(f"-D{k}={v}" for k, v in CMAKE_FLAGS.items())
    sh(f"/usr/bin/cmake -G Ninja -S {SOURCE_DIR} -B {BUILD_DIR} {flags}")
    jobs = max(1, (os.cpu_count() or 4) - 2)
    env = dict(os.environ, MOLD_JOBS="1", NINJA_STATUS="[%f/%t %p %es] ")
    sh(f"/usr/bin/cmake --build {BUILD_DIR} --target {' '.join(CMAKE_TARGETS)} -- -j {jobs}", env=env)


def run_cell(qname: str, query_test: str, horizon: int, pool: int, threads: int, cell_dir: Path) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    buf_size, num_buffers = BUFFER_OVERRIDES.get(qname, (BUFFER_SIZE_BYTES, BUFFERS_IN_GLOBAL_POOL))
    worker_cfg = " ".join([
        f"--worker.query_engine.number_of_worker_threads={threads}",
        f"--worker.default_query_execution.execution_mode=COMPILER",
        f"--worker.number_of_buffers_in_global_buffer_manager={num_buffers}",
        f"--worker.default_query_execution.operator_buffer_size={buf_size}",
        f"--worker.default_query_execution.page_size={PAGE_SIZE}",
        f"--worker.default_query_execution.slice_preallocation.preemptive_create_horizon_ms={horizon}",
        f"--worker.default_query_execution.slice_preallocation.recycle_pool_size={pool}",
    ])
    test_path = (SOURCE_DIR / query_test).resolve()
    cmd = (
        f"{SYSTEST_BIN} -b -t {test_path} "
        f"--data {TESTDATA_DIR} --workingDir={cell_dir} -- {worker_cfg}"
    )
    log = cell_dir / "systest.log"
    try:
        with log.open("w") as f:
            subprocess.run(cmd, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=CELL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"failure_reason": f"systest timeout ({CELL_TIMEOUT_S}s)"}
    except subprocess.CalledProcessError as exc:
        return {"failure_reason": f"systest exit {exc.returncode}"}

    results_path = cell_dir / "BenchmarkResults.json"
    if not results_path.exists():
        return {"failure_reason": "no BenchmarkResults.json"}
    rows = json.loads(results_path.read_text())
    if not rows:
        return {"failure_reason": "empty BenchmarkResults.json"}
    ### One systest call → one (or more) query rows. We sum across rows in the same file index
    ### (typically just 1) so the cell yields one summary line.
    return {
        "time_s": sum(float(r["time"]) for r in rows),
        "tuples_per_second": sum(float(r["tuplesPerSecond"]) for r in rows),
        "bytes_per_second": sum(float(r["bytesPerSecond"]) for r in rows),
        "failure_reason": "",
    }


def _fmt_dur(seconds: float) -> str:
    s = int(seconds + 0.5)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def parse_argv(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-build", action="store_true", help="skip cmake configure/build (reuse existing build)")
    p.add_argument("--clean-build", action="store_true", help=f"rm -rf {BUILD_DIR} before configuring")
    p.add_argument("-q", "--queries", nargs="+", help=f"subset of {sorted(QUERIES)}")
    p.add_argument("--horizon", nargs="+", type=int, help=f"override PREEMPTIVE_HORIZON_MS (default {PREEMPTIVE_HORIZON_MS})")
    p.add_argument("--pool", nargs="+", type=int, help=f"override RECYCLE_POOL_SIZE (default {RECYCLE_POOL_SIZE})")
    p.add_argument("--threads", nargs="+", type=int, help=f"override WORKER_THREADS (default {WORKER_THREADS})")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_argv(argv)

    if not args.skip_build:
        if args.clean_build and BUILD_DIR.exists():
            print(f"[bench] removing {BUILD_DIR}", flush=True)
            shutil.rmtree(BUILD_DIR)
        cmake_configure_and_build()

    if not SYSTEST_BIN.exists():
        print(f"[bench] missing {SYSTEST_BIN} — drop --skip-build or fix build", file=sys.stderr)
        return 2

    queries = {k: QUERIES[k] for k in (args.queries or list(QUERIES)) if k in QUERIES}
    if not queries:
        print(f"[bench] no matching queries in {args.queries}", file=sys.stderr)
        return 2
    horizons = args.horizon or PREEMPTIVE_HORIZON_MS
    pools = args.pool or RECYCLE_POOL_SIZE
    threads_list = args.threads or WORKER_THREADS

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"

    cells = list(itertools.product(queries.items(), horizons, pools, threads_list))
    print(f"[bench] {len(cells)} cells → {csv_path}", flush=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        f.flush()

        tty = sys.stdout.isatty()
        start_t = time.monotonic()
        for i, ((qname, qpath), horizon, pool, threads) in enumerate(cells, start=1):
            slug = f"{qname}_h{horizon}_p{pool}_t{threads}"
            cell_dir = run_dir / slug
            prefix = f"[bench] cell {i}/{len(cells)}: {slug}"
            ### On a TTY, show a live "running" line; \r overwrites it in place with the result so each
            ### cell ends up on a single line. Off a TTY (file/non-interactive), skip it to keep logs clean.
            if tty:
                print(f"{prefix} ...", end="", flush=True)
            base = {
                "run_id": run_id, "query": qname,
                "preemptive_horizon_ms": horizon, "recycle_pool_size": pool, "threads": threads,
                "time_s": "", "tuples_per_second": "", "bytes_per_second": "", "failure_reason": "",
            }
            base.update(run_cell(qname, qpath, horizon, pool, threads, cell_dir))
            writer.writerow(base)
            f.flush()
            if base["failure_reason"]:
                status = f"runtime=FAILED  throughput=FAILED  ({base['failure_reason']})"
            else:
                status = f"runtime={base['time_s']:.3f}s  throughput={base['tuples_per_second']:.3f} tuples/s"
            ### ETA from mean time per completed cell × cells remaining.
            elapsed = time.monotonic() - start_t
            eta = elapsed / i * (len(cells) - i)
            print(f"{chr(13) if tty else ''}{prefix} -> {status}  [elapsed {_fmt_dur(elapsed)}, eta {_fmt_dur(eta)}]", flush=True)

    print(f"[bench] done. results: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
