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

"""Measure the effect of slice recycling and slice group creation on the benchmark_memory queries.

Sweep: query x enable_slice_recycling x enable_slice_group_creation x max_slice_group_size x threads,
pruned so max_slice_group_size only varies while group creation is on. Each cell runs
`systest -b` once (or --runs times, averaged) and appends one row to results/<run-id>/results.csv with

* e2e_*: end-to-end numbers from systest's BenchmarkResults.json (input size / running->stop wall time)
* tl_*: aggregates over the ThroughputListener interval samples parsed from the systest log
        (true parsed-source-tuple rate; requires a build with NES_LOG_LEVEL=INFO)
* slices_*: the per-query "Slice creation statistics" log line (created / wasted races / creation ms)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import config as cfg

RESULT_COLS = [
    "run_id", "query", "enable_slice_recycling", "enable_slice_group_creation", "max_slice_group_size",
    "slice_pool_capacity", "threads", "runs",
    "e2e_time_s", "e2e_tuples_per_second", "e2e_bytes_per_second",
    "tl_mean_tps", "tl_median_tps", "tl_p95_tps", "tl_stddev_tps", "tl_samples",
    "slices_created", "slices_wasted", "slice_creation_ms",
    "failure_reason",
]

### The query id renders as "QueryId(local=<uuid>, distributed=<name>:<n>)" and contains colons,
### so match greedily up to the last ": <rate> tuples/s".
THROUGHPUT_RE = re.compile(r"Throughput for query .*: ([\d.]+) tuples/s")
SLICE_STATS_RE = re.compile(r"Slice creation statistics: (\d+) slices created, (\d+) wasted[^,]*, ([\d.]+) ms")


def sh(cmd: str, *, env: dict | None = None) -> None:
    print(f"[bench] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


### PATH lookup, not /usr/bin/cmake: the runtime-registry glue needs CMake >= 3.30, and on the
### bench hosts the new-enough cmake lives in /usr/local/bin while /usr/bin holds a 3.28 whose
### binaries abort at startup with "duplicate registration". BENCH_CMAKE pins an explicit binary.
CMAKE = os.environ.get("BENCH_CMAKE", "cmake")


def cmake_configure_and_build() -> None:
    flags = " ".join(f"-D{k}={v}" for k, v in cfg.CMAKE_FLAGS.items())
    sh(f"{CMAKE} -G Ninja -S {cfg.SOURCE_DIR} -B {cfg.BUILD_DIR} {flags}")
    jobs = max(1, (os.cpu_count() or 4) - 2)
    env = dict(os.environ, MOLD_JOBS=str(cfg.MOLD_JOBS), NINJA_STATUS="[%f/%t %p %es] ")
    sh(f"{CMAKE} --build {cfg.BUILD_DIR} --target {' '.join(cfg.CMAKE_TARGETS)} -- -j {jobs}", env=env)


def assert_info_logging() -> None:
    """The Benchmark build type caches NES_LOG_LEVEL=LEVEL_NONE, which compiles out every log line
    this harness parses. Refuse to run against such a build."""
    cache = cfg.BUILD_DIR / "CMakeCache.txt"
    if not cache.exists():
        return
    for line in cache.read_text().splitlines():
        if line.startswith("NES_LOG_LEVEL"):
            level = line.split("=", 1)[1]
            assert level in ("INFO", "DEBUG", "TRACE"), (
                f"build at {cfg.BUILD_DIR} has NES_LOG_LEVEL={level}; throughput/slice log lines are "
                f"compiled out. Reconfigure with -DNES_LOG_LEVEL=INFO (drop --skip-build)."
            )
            return


def parse_log_metrics(log_path: Path) -> dict:
    """Pull ThroughputListener samples and the slice-creation statistics out of the systest log."""
    out: dict = {}
    if not log_path.exists():
        return out
    text = log_path.read_text(errors="replace")

    samples = [float(m) for m in THROUGHPUT_RE.findall(text)]
    ### Drop the first and last interval (ramp-up / drain partials) when there are enough samples.
    trimmed = samples[1:-1] if len(samples) > 3 else samples
    if trimmed:
        out["tl_mean_tps"] = statistics.fmean(trimmed)
        out["tl_median_tps"] = statistics.median(trimmed)
        out["tl_p95_tps"] = statistics.quantiles(trimmed, n=20)[-1] if len(trimmed) > 1 else trimmed[0]
        out["tl_stddev_tps"] = statistics.stdev(trimmed) if len(trimmed) > 1 else 0.0
    out["tl_samples"] = len(samples)

    ### One line per windowed operator handler at query stop; sum them for the query.
    slice_stats = SLICE_STATS_RE.findall(text)
    if slice_stats:
        out["slices_created"] = sum(int(s[0]) for s in slice_stats)
        out["slices_wasted"] = sum(int(s[1]) for s in slice_stats)
        out["slice_creation_ms"] = sum(float(s[2]) for s in slice_stats)
    return out


def run_cell(qname: str, query_test: str, recycling: bool, group_creation: bool, group_size: int,
             pool_capacity: int, threads: int, runs: int, cell_dir: Path) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    buf_size, total_memory = cfg.BUFFER_OVERRIDES.get(qname, (cfg.BUFFER_SIZE_BYTES, cfg.TOTAL_MEMORY))
    worker_cfg = " ".join([
        f"--throughput_log_interval_in_ms={cfg.THROUGHPUT_LOG_INTERVAL_MS}",
        f"--worker.query_engine.number_of_worker_threads={threads}",
        "--worker.default_query_execution.execution_mode=COMPILER",
        f"--worker.total_memory_in_bytes={total_memory}",
        f"--worker.unpooled_memory_fraction={cfg.UNPOOLED_FRACTION}",
        f"--worker.default_query_execution.operator_buffer_size={buf_size}",
        f"--worker.default_query_execution.page_size={cfg.PAGE_SIZE}",
        f"--worker.default_query_execution.slice_store.enable_slice_recycling={str(recycling).lower()}",
        f"--worker.default_query_execution.slice_store.slice_pool_capacity={pool_capacity}",
        f"--worker.default_query_execution.slice_store.enable_slice_group_creation={str(group_creation).lower()}",
        f"--worker.default_query_execution.slice_store.max_slice_group_size={group_size}",
    ])
    rel_path, _, idx = query_test.partition(":")
    test_path = f"{(cfg.SOURCE_DIR / rel_path).resolve()}:{idx}" if idx else str((cfg.SOURCE_DIR / rel_path).resolve())

    times: list[float] = []
    tps: list[float] = []
    bps: list[float] = []
    log_metrics: dict = {}
    for run in range(runs):
        ### Fresh working dir per run so a stale BenchmarkResults.json can't be misread as this run's.
        working_dir = cell_dir / f"run_{run}"
        if working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True)
        ### Logs live NEXT TO the working dir, not inside it: systest remove_all()s the working dir
        ### at startup, which would silently delete them.
        log_path = cell_dir / f"run_{run}.nes.log"
        cmd = (
            f"{cfg.SYSTEST_BIN} -b -t {test_path} "
            f"--data {cfg.testdata_dir()} --workingDir={working_dir} --log-path {log_path} -- {worker_cfg}"
        )
        print(f"[bench] $ {cmd}", flush=True)
        stdout_log = cell_dir / f"run_{run}.systest.out"
        try:
            with stdout_log.open("w") as f:
                subprocess.run(cmd, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=cfg.CELL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            ### The timeout kills the shell, not necessarily systest itself; a deadlocked systest
            ### would otherwise linger forever. The working dir is unique per cell, so this is precise.
            subprocess.run(f"pkill -9 -f -- '--workingDir={working_dir}'", shell=True, check=False)
            return {"runs": len(times), "failure_reason": f"systest timeout ({cfg.CELL_TIMEOUT_S}s)"}
        except subprocess.CalledProcessError as exc:
            return {"runs": len(times), "failure_reason": f"systest exit {exc.returncode}"}

        results_path = working_dir / "BenchmarkResults.json"
        if not results_path.exists():
            return {"runs": len(times), "failure_reason": "no BenchmarkResults.json"}
        import json
        rows = json.loads(results_path.read_text())
        if not rows:
            return {"runs": len(times), "failure_reason": "empty BenchmarkResults.json"}
        ### One systest call -> one (or more) query rows; sum rows of the run, then average over runs.
        times.append(sum(float(r["time"]) for r in rows))
        tps.append(sum(float(r["tuplesPerSecond"]) for r in rows))
        bps.append(sum(float(r["bytesPerSecond"]) for r in rows))
        ### TL/slice metrics from the last run only; averaging log-derived aggregates across runs
        ### would mix ramp-up trims - one run's interval series is representative.
        log_metrics = parse_log_metrics(log_path)

    return {
        "runs": len(times),
        "e2e_time_s": sum(times) / len(times),
        "e2e_tuples_per_second": sum(tps) / len(tps),
        "e2e_bytes_per_second": sum(bps) / len(bps),
        "failure_reason": "",
        **log_metrics,
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


def build_cells(queries: dict, recyclings: list, group_creations: list, group_sizes: list,
                pool_capacities: list, threads_list: list):
    """Corners at the defaults, plus each knob swept only where it has an effect: max_slice_group_size
    while group creation is on, slice_pool_capacity while recycling is on (the slice store reads it
    nowhere else)."""
    cells = []
    for (qname, qpath) in queries.items():
        query_threads = [t for t in threads_list if t in cfg.THREAD_OVERRIDES.get(qname, threads_list)]
        for threads in query_threads:
            for recycling in recyclings:
                pools = pool_capacities if recycling else [cfg.DEFAULT_SLICE_POOL_CAPACITY]
                for group_creation in group_creations:
                    sizes = group_sizes if group_creation else [cfg.DEFAULT_MAX_SLICE_GROUP_SIZE]
                    for size in sizes:
                        for pool in pools:
                            cells.append((qname, qpath, recycling, group_creation, size, pool, threads))
    return cells


def parse_argv(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-build", action="store_true", help="skip cmake configure/build (reuse existing build)")
    p.add_argument("--clean-build", action="store_true", help=f"rm -rf {cfg.BUILD_DIR} before configuring")
    p.add_argument("-q", "--queries", nargs="+", help=f"subset of {sorted(cfg.QUERIES)}")
    p.add_argument("--recycling", nargs="+", type=int, choices=[0, 1],
                   help=f"override ENABLE_SLICE_RECYCLING (default {cfg.ENABLE_SLICE_RECYCLING})")
    p.add_argument("--group-creation", nargs="+", type=int, choices=[0, 1],
                   help=f"override ENABLE_SLICE_GROUP_CREATION (default {cfg.ENABLE_SLICE_GROUP_CREATION})")
    p.add_argument("--group-size", nargs="+", type=int,
                   help=f"override MAX_SLICE_GROUP_SIZE (default {cfg.MAX_SLICE_GROUP_SIZE})")
    p.add_argument("--pool-capacity", nargs="+", type=int,
                   help=f"override SLICE_POOL_CAPACITY, 0 = auto (default {cfg.SLICE_POOL_CAPACITY})")
    p.add_argument("--threads", nargs="+", type=int, help=f"override WORKER_THREADS (default {cfg.WORKER_THREADS})")
    p.add_argument("--runs", type=int, help=f"runs per cell, averaged (default {cfg.RUNS_PER_CELL})")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_argv(argv)

    if not args.skip_build:
        if args.clean_build and cfg.BUILD_DIR.exists():
            print(f"[bench] removing {cfg.BUILD_DIR}", flush=True)
            shutil.rmtree(cfg.BUILD_DIR)
        cmake_configure_and_build()

    if not cfg.SYSTEST_BIN.exists():
        print(f"[bench] missing {cfg.SYSTEST_BIN} - drop --skip-build or fix build", file=sys.stderr)
        return 2
    assert_info_logging()

    queries = {k: cfg.QUERIES[k] for k in (args.queries or list(cfg.QUERIES)) if k in cfg.QUERIES}
    if not queries:
        print(f"[bench] no matching queries in {args.queries}", file=sys.stderr)
        return 2
    recyclings = [bool(v) for v in args.recycling] if args.recycling else cfg.ENABLE_SLICE_RECYCLING
    group_creations = [bool(v) for v in args.group_creation] if args.group_creation else cfg.ENABLE_SLICE_GROUP_CREATION
    group_sizes = args.group_size or cfg.MAX_SLICE_GROUP_SIZE
    pool_capacities = args.pool_capacity or cfg.SLICE_POOL_CAPACITY
    threads_list = args.threads or cfg.WORKER_THREADS
    runs = args.runs or cfg.RUNS_PER_CELL

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"

    cells = build_cells(queries, recyclings, group_creations, group_sizes, pool_capacities, threads_list)
    print(f"[bench] {len(cells)} cells -> {csv_path}", flush=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        f.flush()

        tty = sys.stdout.isatty()
        start_t = time.monotonic()
        for i, (qname, qpath, recycling, group_creation, group_size, pool_capacity, threads) in enumerate(cells, start=1):
            slug = f"{qname}_r{int(recycling)}_g{int(group_creation)}_s{group_size}_p{pool_capacity}_t{threads}"
            cell_dir = run_dir / slug
            prefix = f"[bench] cell {i}/{len(cells)}: {slug}"
            if tty:
                print(f"{prefix} ...", end="", flush=True)
            base = {c: "" for c in RESULT_COLS}
            base.update({
                "run_id": run_id, "query": qname,
                "enable_slice_recycling": recycling, "enable_slice_group_creation": group_creation,
                "max_slice_group_size": group_size, "slice_pool_capacity": pool_capacity,
                "threads": threads,
            })
            base.update(run_cell(qname, qpath, recycling, group_creation, group_size, pool_capacity, threads, runs, cell_dir))
            writer.writerow(base)
            f.flush()
            if base["failure_reason"]:
                status = f"FAILED ({base['failure_reason']})"
            else:
                tl = f"{base['tl_mean_tps']:.0f}" if base.get("tl_mean_tps") else "n/a"
                status = f"e2e={base['e2e_tuples_per_second']:.0f} tup/s  tl={tl} tup/s  time={base['e2e_time_s']:.2f}s"
            elapsed = time.monotonic() - start_t
            eta = elapsed / i * (len(cells) - i)
            print(f"{chr(13) if tty else ''}{prefix} -> {status}  [elapsed {_fmt_dur(elapsed)}, eta {_fmt_dur(eta)}]", flush=True)

    print(f"[bench] done. results: {csv_path}", flush=True)
    return 0


def _demo() -> None:
    qid = "QueryId(local=3f03b889-ef8b-49f1-9095-087e1498d35c, distributed=SmartGrid:1)"
    sample_log = (
        f"[12:00:01.000] [I] [w] [SingleNodeWorker.cpp:88] [operator()] Throughput for query {qid}: 100.000 tuples/s (100 tuples in 1000 ms)\n"
        f"[12:00:02.000] [I] [w] [SingleNodeWorker.cpp:88] [operator()] Throughput for query {qid}: 200.500 tuples/s (200 tuples in 1000 ms)\n"
        f"[12:00:03.000] [I] [w] [SingleNodeWorker.cpp:88] [operator()] Throughput for query {qid}: 300.000 tuples/s (300 tuples in 1000 ms)\n"
        f"[12:00:04.000] [I] [w] [SingleNodeWorker.cpp:88] [operator()] Throughput for query {qid}: 400.000 tuples/s (400 tuples in 1000 ms)\n"
        "[12:00:05.000] [I] [1] [q] [w] [x.cpp:84] [stop] Slice creation statistics: 128 slices created, 3 wasted (lost creation race), 1.500 ms total creation time\n"
        "[12:00:05.100] [I] [1] [q] [w] [x.cpp:84] [stop] Slice creation statistics: 64 slices created, 1 wasted (lost creation race), 0.250 ms total creation time\n"
    )
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as tmp:
        tmp.write(sample_log)
        path = Path(tmp.name)
    m = parse_log_metrics(path)
    path.unlink()
    assert m["tl_samples"] == 4, m
    assert m["tl_mean_tps"] == statistics.fmean([200.5, 300.0]), m  ### first/last trimmed
    assert m["slices_created"] == 192 and m["slices_wasted"] == 4, m
    assert abs(m["slice_creation_ms"] - 1.75) < 1e-9, m

    cells = build_cells({"Q": "p:01"}, [False, True], [False, True], [16, 64, 256], [0, 128], [1])
    ### recycling off (pool pinned to default): 1 + 3 sizes = 4; recycling on: (1 + 3) x 2 pools = 8
    assert len(cells) == 12, len(cells)
    assert {c[5] for c in cells if not c[2]} == {cfg.DEFAULT_SLICE_POOL_CAPACITY}, "pool must not vary with recycling off"
    assert {c[4] for c in cells if not c[3]} == {cfg.DEFAULT_MAX_SLICE_GROUP_SIZE}, "size must not vary with grouping off"
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _demo()
        sys.exit(0)
    sys.exit(main(sys.argv[1:]))
