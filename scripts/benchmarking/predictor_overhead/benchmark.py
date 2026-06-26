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

"""Predictor overhead benchmark.

For each (query, predictor) cell: patch a SET (...) clause onto the query, run it under
`systest -b`, and write the BenchmarkResults.json row into results.csv.

Predictor variants:
    off     -- no SET clause: query runs against DefaultTimeBasedSliceStore (no spill wrapper).
    ewma    -- spill enabled, predictive policy, EWMA predictor, in-memory backend.
    kalman  -- same as above but Kalman predictor.
    robustkalman -- same as above but robust/adaptive Kalman predictor.

Both predictor cells use a generous buffer pool (`NUMBER_OF_BUFFERS`) so memory pressure stays
below the policy's highBound (default 0.85): observe() is called every GC tick (paying the
predictor cost) but decide() returns Keep before consulting the predictor, so nothing is
spilled. The throughput delta vs the `off` baseline is the prediction overhead.

CLI:
    --skip-build   reuse existing binaries.
    --clean-build  rm -rf BUILD_DIR before configure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
SOURCE_DIR = _BENCH_DIR.parents[2]
BUILD_DIR = Path(os.environ.get("BENCH_BUILD_DIR_OVERRIDE", SOURCE_DIR / "cmake-build-bench"))
SYSTEST_BIN = BUILD_DIR / "nes-systests" / "systest" / "systest"
TEST_DATA_DIR = BUILD_DIR / "nes-systests" / "testdata"
RESULTS_ROOT = _BENCH_DIR / "results"
SYSTESTS_ROOT = SOURCE_DIR / "nes-systests"

### Same flags as spilling_e2e_benchmarks/config.py — Benchmark build type zeroes logging.
CMAKE_FLAGS = {
    "CMAKE_BUILD_TYPE": "Benchmark",
    "CMAKE_TOOLCHAIN_FILE": "/vcpkg/scripts/buildsystems/vcpkg.cmake",
    "USE_LIBCXX_IF_AVAILABLE": "OFF",
    "ENABLE_LARGE_TESTS": "1",
}
CMAKE_TARGETS = ["systest", "nes-single-node-worker"]

### Subset of `engine-statemanagement-paper/scripts/benchmarking/e2e/run_nes_benchmarks.py` that
### exists in this fork's nes-systests/benchmark/memory-source/. Tuple is (label, test_file, query_idx).
QUERIES = [
    ("CM1", "benchmark/memory-source/ClusterMonitoring_memory.test", 1),
    ("CM2", "benchmark/memory-source/ClusterMonitoring_memory.test", 2),
    ("LRB1", "benchmark/memory-source/LinearRoadBenchmark_memory.test", 1),
    ("LRB2", "benchmark/memory-source/LinearRoadBenchmark_memory.test", 2),
    ("MA", "benchmark/memory-source/Manufacturing_memory.test", 1),
    ("NM4", "benchmark/memory-source/Nexmark_memory.test", 4),
    ("NM5", "benchmark/memory-source/Nexmark_memory.test", 5),
    ("YSB1", "benchmark/memory-source/YahooStreamingBenchmark_memory.test", 1),
    ("YSB2", "benchmark/memory-source/YahooStreamingBenchmark_memory.test", 2),
]

### off → no SET (no spill wrapper); ewma/kalman → predictive policy, in-memory backend.
PREDICTORS = ["off", "ewma", "kalman", "robustkalman"]

### Worker knobs. Buffer pool sized so memoryPressure < 0.85 → predictor never triggers a spill.
### 40000 (40GB at 1MB/buffer) is the smallest pool at which the heaviest query (NM4, a self-join of
### two windowed aggregations) completes at threads=8 without exhausting the global pool; 20000 timed
### out. Stays well under the box's 61GB.
NUMBER_OF_WORKER_THREADS = [4, 8]
BUFFER_SIZE_IN_BYTES = 1048576
NUMBER_OF_BUFFERS = 40000
PAGE_SIZE = 8192
NUM_RUNS_PER_CELL = 3


def spill_set_clause(predictor: str) -> str:
    if predictor == "off":
        return ""
    return (
        f"SET (TRUE AS SPILL.ENABLED, 'predictive' AS SPILL.POLICY, "
        f"'in-memory' AS SPILL.BACKEND, '{predictor}' AS SPILL.PREDICTOR)"
    )


_INTO_RE = re.compile(r"(INTO\s+\w+)\s*;")


def patch_test_file(src: Path, dst: Path, set_clause: str) -> None:
    """Append `set_clause` before the `;` of every `INTO <sink>;` query in `src`."""
    content = src.read_text()
    if set_clause:
        content = _INTO_RE.sub(rf"\1 {set_clause};", content)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content)


def sh(cmd: str, *, env: dict | None = None) -> None:
    print(f"[bench] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


def cmake_configure() -> None:
    flags = " ".join(f"-D{k}={v}" for k, v in CMAKE_FLAGS.items())
    sh(f"/usr/bin/cmake -G Ninja -S {SOURCE_DIR} -B {BUILD_DIR} {flags}")


def cmake_build(targets: list[str]) -> None:
    env = dict(os.environ, MOLD_JOBS="1", NINJA_STATUS="[%f/%t %p %es] ")
    jobs = os.cpu_count() or 4
    sh(f"/usr/bin/cmake --build {BUILD_DIR} --target {' '.join(targets)} -- -j {jobs}", env=env)


@dataclass(frozen=True)
class Cell:
    threads: int
    query_label: str
    test_file: str
    query_idx: int
    predictor: str

    @property
    def slug(self) -> str:
        return f"t{self.threads}_{self.query_label}_{self.predictor}"


def worker_args(threads: int) -> list[str]:
    return [
        f"--worker.query_engine.number_of_worker_threads={threads}",
        f"--worker.default_query_execution.operator_buffer_size={BUFFER_SIZE_IN_BYTES}",
        f"--worker.number_of_buffers_in_global_buffer_manager={NUMBER_OF_BUFFERS}",
        f"--worker.default_query_execution.page_size={PAGE_SIZE}",
    ]


def run_cell(cell: Cell, run_dir: Path) -> dict:
    cell_dir = run_dir / cell.slug
    cell_dir.mkdir(parents=True, exist_ok=True)

    ### Patch the test file into the cell dir; systest accepts any path for `-t`.
    src_test = SYSTESTS_ROOT / cell.test_file
    dst_test = cell_dir / Path(cell.test_file).name
    patch_test_file(src_test, dst_test, spill_set_clause(cell.predictor))

    target = f"{dst_test}:{cell.query_idx:02d}"
    samples_tps: list[float] = []
    samples_bps: list[float] = []
    samples_t: list[float] = []
    failure = ""

    for run in range(NUM_RUNS_PER_CELL):
        working_dir = cell_dir / f"run_{run}"
        if working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True)
        cmd = [
            str(SYSTEST_BIN),
            "-b",
            "-t", target,
            "--data", str(TEST_DATA_DIR),
            f"--workingDir={working_dir}",
            "--",
            *worker_args(cell.threads),
        ]
        log_path = cell_dir / f"run_{run}.log"
        try:
            with log_path.open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            results = json.loads((working_dir / "BenchmarkResults.json").read_text())
            if not results:
                failure = "empty BenchmarkResults.json"
                break
            samples_t.append(float(results[0]["time"]))
            samples_tps.append(float(results[0]["tuplesPerSecond"]))
            samples_bps.append(float(results[0]["bytesPerSecond"]))
        except subprocess.CalledProcessError as e:
            failure = f"systest exit {e.returncode} (see {log_path})"
            break
        except Exception as e:  # noqa: BLE001
            failure = f"{type(e).__name__}: {e}"
            break

    def avg(xs: list[float]) -> str:
        return f"{sum(xs) / len(xs):.3f}" if xs else ""

    return {
        "query": cell.query_label,
        "test_file": cell.test_file,
        "query_idx": cell.query_idx,
        "threads": cell.threads,
        "predictor": cell.predictor,
        "runs": len(samples_t),
        "avg_time_s": avg(samples_t),
        "avg_tuples_per_s": avg(samples_tps),
        "avg_bytes_per_s": avg(samples_bps),
        "failure": failure,
    }


def matrix() -> list[Cell]:
    cells = []
    for threads in NUMBER_OF_WORKER_THREADS:
        for label, tf, qi in QUERIES:
            for p in PREDICTORS:
                cells.append(Cell(threads=threads, query_label=label, test_file=tf, query_idx=qi, predictor=p))
    return cells


def _fmt_dur(seconds: float) -> str:
    s = int(seconds + 0.5)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--clean-build", action="store_true")
    args = p.parse_args(argv)

    if not args.skip_build:
        if args.clean_build and BUILD_DIR.exists():
            print(f"[bench] removing {BUILD_DIR}", flush=True)
            shutil.rmtree(BUILD_DIR)
        cmake_configure()
        cmake_build(CMAKE_TARGETS)

    if not SYSTEST_BIN.exists():
        print(f"error: systest binary not found at {SYSTEST_BIN}", file=sys.stderr)
        return 2

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "results.csv"
    cols = ["query", "test_file", "query_idx", "threads", "predictor",
            "runs", "avg_time_s", "avg_tuples_per_s", "avg_bytes_per_s", "failure"]
    cells = matrix()
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        tty = sys.stdout.isatty()
        start_t = time.monotonic()
        for i, cell in enumerate(cells, start=1):
            prefix = f"[bench] cell {i}/{len(cells)}: {cell.slug}"
            ### On a TTY, show a live "running" line; \r overwrites it in place with the result so each
            ### cell ends up on a single line. Off a TTY (file/non-interactive), skip it to keep logs clean.
            if tty:
                print(f"{prefix} ...", end="", flush=True)
            row = run_cell(cell, run_dir)
            w.writerow(row)
            f.flush()
            if row["failure"]:
                status = f"runtime=FAILED  throughput=FAILED  ({row['failure']})"
            else:
                status = f"runtime={row['avg_time_s']}s  throughput={row['avg_tuples_per_s']} tuples/s"
            ### ETA from mean time per completed cell × cells remaining.
            elapsed = time.monotonic() - start_t
            eta = elapsed / i * (len(cells) - i)
            print(f"{chr(13) if tty else ''}{prefix} -> {status}  [elapsed {_fmt_dur(elapsed)}, eta {_fmt_dur(eta)}]", flush=True)

    print(f"[bench] done. results: {csv_path}", flush=True)
    return 0


def _selfcheck() -> None:
    src = "INTO sinkA;\nINTO sinkB ;\nCREATE LOGICAL SOURCE foo(a INT64);\nSELECT * INTO sinkC;"
    out = _INTO_RE.sub(r"\1 SET_CLAUSE;", src)
    assert "INTO sinkA SET_CLAUSE;" in out
    assert "INTO sinkB SET_CLAUSE;" in out
    assert "INTO sinkC SET_CLAUSE;" in out
    assert "CREATE LOGICAL SOURCE foo(a INT64);" in out, "DDL must be untouched"
    assert spill_set_clause("off") == ""
    assert "'predictive' AS SPILL.POLICY" in spill_set_clause("ewma")
    assert "'kalman' AS SPILL.PREDICTOR" in spill_set_clause("kalman")
    assert "'robustkalman' AS SPILL.PREDICTOR" in spill_set_clause("robustkalman")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        print("ok")
        sys.exit(0)
    sys.exit(main(sys.argv[1:]))
