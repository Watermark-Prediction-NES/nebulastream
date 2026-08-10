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

"""Spilling end-to-end benchmark orchestrator.

Reads parameters from `config.py`, enumerates the cartesian product of every knob, and runs
each cell as a fresh worker invocation. One row per cell lands in `results/<run-id>/results.csv`.

The loop never references specific values from `config.py` — appending entries to lists in
that module is sufficient to widen the sweep.

CLI:
    --skip-build      Skip cmake configure/build + cargo build (use existing binaries).
    --dry-matrix      Render every (cell × query) topology + SQL and exit. No processes.
    --only=K=V,K=V    Filter the matrix to a single cell. Keys: threads, buffer_size,
                      no_buffers, spill, var_size, predictor, rate, high_bound, horizon.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config as cfg

### Cell-level spill telemetry, derived from spill_stats.csv by summarize_spill_stats(). Kept as its
### own list so the dry-matrix path can blank exactly these without re-listing them.
SPILL_RESULT_COLS = [
    "total_spills",
    "total_restores",
    "total_spill_failures",
    "barrier_misses",
    "spilled_bytes",
    "restored_bytes",
    "kept_by_prediction",
    "spilled_predictor_cold",
    "spilled_beyond_horizon",
    "avg_spill_latency_ms",
    "avg_restore_latency_ms",
    "max_restore_latency_ms",
    "peak_spilled_slices",
    "spill_stats_samples",
]

RESULT_COLS = [
    "run_id",
    "cell_slug",
    "threads",
    "operator_buffer_size",
    "no_buffers",
    "spill_variant",
    "predictor",
    "high_bound",
    "horizon_ms",
    "var_sized_bytes",
    "ingestion_rate",
    "wall_s",
    "sinks_nonempty",
    "avg_throughput_tps",
    "throughput_samples",
    "peak_rss_bytes",
    "peak_pooled_used",
    "peak_unpooled_used",
    *SPILL_RESULT_COLS,
    "failure_reason",
]

### Worker emits to stdout (captured in worker.log):
###   Throughput for queryId QueryId(local=<uuid>[, distributed=<name>]) in window <ms>-<ms> is X.XXX <unit>Tup/s
### unit ∈ {"", "k", "M", "G", "T"} (formatThroughput in SingleNodeWorker.cpp).
_THROUGHPUT_RE = re.compile(
    r"Throughput for queryId QueryId\(local=([^,)]+)(?:, distributed=([^)]+))?\) "
    r"in window (\d+)-(\d+) is (\d+\.\d+) (\w*)Tup/s"
)
_UNIT_MUL = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def parse_throughput_log(log_path: Path) -> tuple[float, int, list[tuple[int, str, float]]]:
    """Return (avg_tps, sample_count, rows) from a worker.log.

    Mirrors `parse_average_throughput_from_throughput_listener` in nebulastream-espat:
    drop the trailing window (likely partial) when more than one sample exists.
    """
    samples: list[tuple[int, str, float]] = []
    if not log_path.exists():
        return -1.0, 0, samples
    with log_path.open() as f:
        for line in f:
            m = _THROUGHPUT_RE.search(line)
            if not m:
                continue
            qid = m.group(2) or m.group(1)
            window_start = int(m.group(3))
            value = float(m.group(5)) * _UNIT_MUL[m.group(6)]
            samples.append((window_start, qid, value))
    if not samples:
        return -1.0, 0, samples
    values = [v for _, _, v in samples]
    if len(values) > 1:
        values = values[:-1]
    return sum(values) / len(values), len(values), samples


def write_throughput_csv(samples: list[tuple[int, str, float]], csv_path: Path) -> None:
    if not samples:
        return
    base = min(ts for ts, _, _ in samples)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["normalized_timestamp_ms", "query_id", "throughput_tps"])
        for ts, qid, val in samples:
            w.writerow([ts - base, qid, val])


### Memory monitoring: Python samples worker RSS from /proc; C++ samples pooled+unpooled into buffer_usage.csv.
### Both files are merged into memory.csv at cell teardown (union of timestamps, forward-fill on missing).


def _read_vm_rss_bytes(pid: int) -> int | None:
    """Return current VmRSS of pid in bytes from /proc/<pid>/status, or None if unavailable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  ### kB → bytes
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


class RssSampler:
    """Background thread that polls a process's RSS at a fixed cadence and writes rss.csv."""

    def __init__(self, pid: int, csv_path: Path, interval_ms: int) -> None:
        self.pid = pid
        self.csv_path = csv_path
        self.interval_s = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(join_timeout_s)

    def _run(self) -> None:
        with self.csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ms", "rss_bytes"])
            while not self._stop.is_set():
                rss = _read_vm_rss_bytes(self.pid)
                if rss is None:
                    return  ### process gone
                now_ms = int(time.time() * 1000)
                w.writerow([now_ms, rss])
                f.flush()
                if self._stop.wait(self.interval_s):
                    return


def _read_csv_rows(path: Path) -> list[dict]:
    """Read a monitor CSV, dropping any row that is not fully formed.

    The worker and the RSS sampler are killed while they may be mid-write, so the last line of a
    monitor CSV is routinely a partial record. DictReader happily yields it with None values for the
    missing fields, and int(None) then took down the entire sweep at teardown of one cell.
    """
    if not path.exists():
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if None in row or None in row.values() or any(v == "" for v in row.values()):
                continue  ### short (truncated) or over-long line
            rows.append(row)
        return rows


def merge_memory_csv(rss_path: Path, buffer_path: Path, out_path: Path) -> tuple[int, int, int]:
    """Union timestamps from rss.csv and buffer_usage.csv into memory.csv (forward-fill on misses).

    Returns (peak_rss_bytes, peak_pooled_used, peak_unpooled_used). Zero for missing series.
    """
    rss = [(int(r["timestamp_ms"]), int(r["rss_bytes"])) for r in _read_csv_rows(rss_path)]
    buf = [
        (int(r["timestamp_ms"]), int(r["pooled_used"]), int(r["unpooled_used"]))
        for r in _read_csv_rows(buffer_path)
    ]
    if not rss and not buf:
        return 0, 0, 0

    timestamps = sorted({t for t, _ in rss} | {t for t, _, _ in buf})
    base = timestamps[0]
    ### Forward-fill walks each series with an index that advances when the next sample's ts <= cur.
    ri = bi = 0
    cur_rss = 0
    cur_pool = 0
    cur_unp = 0
    peak_rss = peak_pool = peak_unp = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["normalized_timestamp_ms", "rss_bytes", "pooled_used", "unpooled_used"])
        for ts in timestamps:
            while ri < len(rss) and rss[ri][0] <= ts:
                cur_rss = rss[ri][1]
                ri += 1
            while bi < len(buf) and buf[bi][0] <= ts:
                _, cur_pool, cur_unp = buf[bi]
                bi += 1
            w.writerow([ts - base, cur_rss, cur_pool, cur_unp])
            peak_rss = max(peak_rss, cur_rss)
            peak_pool = max(peak_pool, cur_pool)
            peak_unp = max(peak_unp, cur_unp)
    return peak_rss, peak_pool, peak_unp


### Spill statistics: the C++ SpillStatsRegistry samples one row per slice store per interval into
### spill_stats.csv. Counters are cumulative per store, so a cell-level total is the sum over stores
### of each store's LAST sample -- never a sum over rows, which would count every sample again.
SPILL_STATS_COUNTERS = (
    "spills",
    "restores",
    "spill_failures",
    "barrier_misses",
    "spilled_bytes",
    "restored_bytes",
    "kept_by_prediction",
    "spilled_predictor_cold",
    "spilled_beyond_horizon",
    "spill_ns_total",
    "restore_ns_total",
)


def summarize_spill_stats(stats_path: Path) -> dict:
    """Collapse spill_stats.csv to cell-level aggregates and normalize its timestamps in place.

    Returns zeros for every field when the file is missing or empty, so a cell that never spilled and
    a cell that was never instrumented are distinguishable only via `spill_stats_samples`.
    """
    empty = {
        "total_spills": 0,
        "total_restores": 0,
        "total_spill_failures": 0,
        "barrier_misses": 0,
        "spilled_bytes": 0,
        "restored_bytes": 0,
        "kept_by_prediction": 0,
        "spilled_predictor_cold": 0,
        "spilled_beyond_horizon": 0,
        "avg_spill_latency_ms": "",
        "avg_restore_latency_ms": "",
        "max_restore_latency_ms": "",
        "peak_spilled_slices": 0,
        "spill_stats_samples": 0,
    }
    rows = _read_csv_rows(stats_path)
    if not rows:
        return empty

    ### Last sample per store holds the final cumulative value; the file is written in time order.
    last_per_store: dict[str, dict] = {}
    peak_spilled = 0
    max_restore_ns = 0
    for r in rows:
        last_per_store[r["store_id"]] = r
        peak_spilled = max(peak_spilled, int(r["spilled_slices"]))
        max_restore_ns = max(max_restore_ns, int(r["restore_ns_max"]))

    ### `or 0` tolerates a results dir written by an older worker that did not yet emit one of these
    ### columns; a missing counter reads as zero rather than taking down the whole sweep.
    totals = {c: sum(int(r.get(c) or 0) for r in last_per_store.values()) for c in SPILL_STATS_COUNTERS}

    def mean_ms(total_ns: int, count: int) -> str:
        return f"{total_ns / count / 1e6:.3f}" if count else ""

    ### Rewrite with normalized timestamps so the notebook can overlay it on memory.csv/throughput.csv.
    base = min(int(r["timestamp_ms"]) for r in rows)
    with stats_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["normalized_timestamp_ms", *[k for k in rows[0] if k != "timestamp_ms"]])
        w.writeheader()
        for r in rows:
            out = {k: v for k, v in r.items() if k != "timestamp_ms"}
            out["normalized_timestamp_ms"] = int(r["timestamp_ms"]) - base
            w.writerow(out)

    return {
        "total_spills": totals["spills"],
        "total_restores": totals["restores"],
        "total_spill_failures": totals["spill_failures"],
        "barrier_misses": totals["barrier_misses"],
        "spilled_bytes": totals["spilled_bytes"],
        "restored_bytes": totals["restored_bytes"],
        "kept_by_prediction": totals["kept_by_prediction"],
        "spilled_predictor_cold": totals["spilled_predictor_cold"],
        "spilled_beyond_horizon": totals["spilled_beyond_horizon"],
        "avg_spill_latency_ms": mean_ms(totals["spill_ns_total"], totals["spills"]),
        "avg_restore_latency_ms": mean_ms(totals["restore_ns_total"], totals["restores"]),
        "max_restore_latency_ms": f"{max_restore_ns / 1e6:.3f}",
        "peak_spilled_slices": peak_spilled,
        "spill_stats_samples": len(rows),
    }


def parse_argv(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-build", action="store_true", help="skip cmake + cargo build")
    p.add_argument("--clean-build", action="store_true", help=f"rm -rf {{BUILD_DIR}} before configure (forces a fresh build)")
    p.add_argument("--dry-matrix", action="store_true", help="render templates only; no processes")
    p.add_argument("--only", default=None, help="single-cell filter, e.g. threads=1,buffer_size=4096,...")
    return p.parse_args(argv)


def expand_spill_variants() -> Iterable[dict]:
    """Each entry × every PREDICTOR_VARIANT when policy == 'predictive'. Otherwise pass through.

    Adds 'predictor' key (always present) and 'tag' (cell-slug-safe short name).
    """
    for v in cfg.SPILL_VARIANTS:
        if v.get("policy") == "predictive":
            for p in cfg.PREDICTOR_VARIANTS:
                merged = dict(v)
                merged["predictor"] = p
                merged["tag"] = f"{v['name']}-{p}"
                yield merged
        else:
            merged = dict(v)
            merged["predictor"] = ""
            merged["tag"] = v["name"]
            yield merged


@dataclass(frozen=True)
class Cell:
    threads: int
    buf_size: int
    no_buffers: int
    spill: tuple  ### Hashable view of the spill dict for matrix-product purposes
    var_size: int
    rate: int  ### Ingestion rate in tuples/sec per source; 0 = unbounded.
    high_bound: float  ### Memory-pressure ratio above which the policy spills.
    horizon_ms: int  ### Predictive policy only: keep-resident look-ahead.

    @property
    def spill_dict(self) -> dict:
        return dict(self.spill)

    @property
    def slug(self) -> str:
        tag = self.spill_dict["tag"]
        rate = "inf" if self.rate == 0 else str(self.rate)
        base = f"t{self.threads}_bs{self.buf_size}_nb{self.no_buffers}_{tag}_v{self.var_size}_r{rate}"
        ### Only append the knobs that vary, so slugs stay readable when a sweep pins them.
        if len(cfg.SPILL_HIGH_BOUND) > 1:
            base += f"_hb{self.high_bound}"
        if len(cfg.PREDICTION_HORIZON_MS) > 1 and self.spill_dict.get("policy") == "predictive":
            base += f"_hz{self.horizon_ms}"
        return base


def matrix() -> Iterator[Cell]:
    for threads, buf_size, no_buffers, spill, var_size, rate, high_bound, horizon in itertools.product(
        cfg.NUMBER_OF_WORKER_THREADS,
        cfg.OPERATOR_BUFFER_SIZE,
        cfg.NUMBER_OF_BUFFERS,
        list(expand_spill_variants()),
        cfg.VAR_SIZED_BYTES,
        cfg.INGESTION_RATES,
        cfg.SPILL_HIGH_BOUND,
        cfg.PREDICTION_HORIZON_MS,
    ):
        ### The horizon only exists for the predictive policy; without this the sweep would run
        ### len(PREDICTION_HORIZON_MS) identical copies of every reactive and every off cell.
        if spill.get("policy") != "predictive" and horizon != cfg.PREDICTION_HORIZON_MS[0]:
            continue
        yield Cell(
            threads=threads,
            buf_size=buf_size,
            no_buffers=no_buffers,
            spill=tuple(sorted(spill.items())),
            var_size=var_size,
            rate=rate,
            high_bound=high_bound,
            horizon_ms=horizon,
        )


def apply_only_filter(cells: Iterable[Cell], only: str | None) -> Iterator[Cell]:
    if only is None:
        yield from cells
        return
    wanted = dict(kv.split("=", 1) for kv in only.split(","))

    def matches(c: Cell) -> bool:
        sd = c.spill_dict
        if "threads" in wanted and str(c.threads) != wanted["threads"]:
            return False
        if "buffer_size" in wanted and str(c.buf_size) != wanted["buffer_size"]:
            return False
        if "no_buffers" in wanted and str(c.no_buffers) != wanted["no_buffers"]:
            return False
        if "spill" in wanted and sd["name"] != wanted["spill"]:
            return False
        if "predictor" in wanted and sd.get("predictor", "") != wanted["predictor"]:
            return False
        if "var_size" in wanted and str(c.var_size) != wanted["var_size"]:
            return False
        if "rate" in wanted and str(c.rate) != wanted["rate"]:
            return False
        if "high_bound" in wanted and str(c.high_bound) != wanted["high_bound"]:
            return False
        if "horizon" in wanted and str(c.horizon_ms) != wanted["horizon"]:
            return False
        return True

    for c in cells:
        if matches(c):
            yield c


def sh(cmd: str, *, env: dict | None = None) -> None:
    """Run a shell command, streaming stdout/stderr to this process's tty."""
    print(f"[bench] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


def cmake_configure() -> None:
    flags = " ".join(f"-D{k}={v}" for k, v in cfg.CMAKE_FLAGS.items())
    sh(f"/usr/bin/cmake -G Ninja -S {cfg.SOURCE_DIR} -B {cfg.BUILD_DIR} {flags}")


def cmake_build(targets: list[str]) -> None:
    target_arg = " ".join(targets)
    env = dict(os.environ, MOLD_JOBS=str(cfg.MOLD_JOBS), NINJA_STATUS="[%f/%t %p %es] ")
    jobs = os.cpu_count() or 4
    sh(
        f"/usr/bin/cmake --build {cfg.BUILD_DIR} --target {target_arg} -- -j {jobs}",
        env=env,
    )


def cargo_build_release() -> None:
    sh(f"cargo build --release --manifest-path {cfg.GENERATOR_DIR}/Cargo.toml")


@dataclass(frozen=True)
class Query:
    q_index: int
    sql: str
    topology_path: Path
    sink_path: Path
    port_a: int
    port_b: int


def render_query(
    env: Environment,
    q_index: int,
    window: tuple[int, int],
    cell_dir: Path,
) -> Query:
    size_ms, slide_ms = window
    port_a = cfg.GENERATOR_PORT_BASE + 2 * q_index
    port_b = port_a + 1
    sink_path = cell_dir / f"q_{q_index}.out"

    sql = env.get_template("query.sql.jinja").render(
        q_index=q_index,
        size_ms=size_ms,
        slide_ms=slide_ms,
    )
    sql_path = cell_dir / f"q_{q_index}.sql"
    sql_path.write_text(sql)

    topology = env.get_template("topology.yaml.jinja").render(
        q_index=q_index,
        grpc_port=cfg.GRPC_PORT,
        data_address=cfg.DATA_ADDRESS,
        port_a=port_a,
        port_b=port_b,
        sink_path=str(sink_path),
    )
    topology_path = cell_dir / f"topology_q{q_index}.yaml"
    topology_path.write_text(topology)

    return Query(
        q_index=q_index,
        sql=sql,
        topology_path=topology_path,
        sink_path=sink_path,
        port_a=port_a,
        port_b=port_b,
    )


def spawn_generator(port: int, var_size: int, rate: int, cell_dir: Path) -> subprocess.Popen:
    log_path = cell_dir / f"gen_{port}.log"
    log = open(log_path, "w")
    return subprocess.Popen(
        [
            str(cfg.GENERATOR_BIN),
            f"--port={port}",
            f"--var-size={var_size}",
            f"--rate={rate}",
        ],
        stdout=subprocess.PIPE,
        stderr=log,
        start_new_session=True,
        bufsize=1,
        text=True,
    )


def wait_for_ready(proc: subprocess.Popen, timeout_s: float) -> None:
    """Block until the generator prints `READY\\n` on stdout, or raise on timeout."""
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"generator (pid={proc.pid}) did not print READY within {timeout_s}s")
        ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
        if ready:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"generator (pid={proc.pid}) closed stdout before READY")
            if line.strip() == "READY":
                return


def spawn_worker(cell: Cell, cell_dir: Path) -> subprocess.Popen:
    """Start a worker configured for one cell.

    The spill configuration is passed here, at worker scope, rather than as a per-query
    `SET (SPILL.* AS ...)` clause: that clause is bound correctly by the parser but dropped at the
    gRPC boundary (grpc/SerializableQueryPlan.proto carries no spill field), so it never reaches the
    worker when the query is submitted through nes-cli. One worker per cell means worker scope is the
    cell's granularity anyway. See templates/query.sql.jinja.
    """
    log_path = cell_dir / "worker.log"
    log = open(log_path, "w")
    sd = cell.spill_dict
    enabled = bool(sd.get("enabled", False))
    args = [
        str(cfg.WORKER_BIN),
        f"--grpc=0.0.0.0:{cfg.GRPC_PORT}",
        f"--data_address={cfg.DATA_ADDRESS}",
        f"--worker.query_engine.number_of_worker_threads={cell.threads}",
        f"--worker.default_query_execution.operator_buffer_size={cell.buf_size}",
        f"--worker.number_of_buffers_in_global_buffer_manager={cell.no_buffers}",
        f"--worker.buffer_usage_log_path={cell_dir / 'buffer_usage.csv'}",
        f"--worker.buffer_usage_monitor_interval_in_ms={cfg.MEMORY_SAMPLE_INTERVAL_MS}",
        ### Per-cell spill directory: all cells otherwise share /tmp/nes-spill and accumulate there.
        f"--worker.default_query_execution.spill.directory={cell_dir / 'spill'}",
        f"--worker.default_query_execution.spill.stats_log_path={cell_dir / 'spill_stats.csv'}",
        f"--worker.default_query_execution.spill.stats_interval_in_ms={cfg.MEMORY_SAMPLE_INTERVAL_MS}",
        f"--worker.default_query_execution.spill.enabled={'true' if enabled else 'false'}",
    ]
    if enabled:
        args += [
            f"--worker.default_query_execution.spill.policy={sd['policy']}",
            f"--worker.default_query_execution.spill.backend={sd['backend']}",
            f"--worker.default_query_execution.spill.high_memory_bound={cell.high_bound}",
        ]
        if sd.get("policy") == "predictive":
            args += [
                f"--worker.default_query_execution.spill.predictor={sd['predictor']}",
                f"--worker.default_query_execution.spill.prediction_horizon_ms={cell.horizon_ms}",
            ]
    return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_for_port_free(port: int, timeout_s: float) -> None:
    """Block until nothing is accepting connections on `port`.

    A cell's worker is SIGTERMed at teardown but keeps its listening socket for a moment after. The
    next cell's wait_for_grpc would then connect straight to the *dying* worker, declare it ready,
    and query registration would come back `UNAVAILABLE` — which showed up as roughly every other
    cell failing with 'nes-cli start exit 1'.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=0.5):
                pass
        except OSError:
            return  ### nothing listening — the previous worker is gone
        time.sleep(0.25)
    raise TimeoutError(f"gRPC port {port} still occupied after {timeout_s}s; previous worker did not exit")


def wait_for_grpc(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except (ConnectionRefusedError, socket.timeout, OSError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"gRPC port {port} unreachable after {timeout_s}s")
            time.sleep(0.25)


def nes_cli_start(topology: Path, sql: str) -> str:
    """Submit one query via nes-cli; return the PersistedQueryId from stdout."""
    result = subprocess.run(
        [str(cfg.NES_CLI_BIN), "-t", str(topology), "start", sql],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nes-cli start exit {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result.stdout.strip()


def nes_cli_stop(topology: Path, ids: list[str]) -> None:
    subprocess.run(
        [str(cfg.NES_CLI_BIN), "-t", str(topology), "stop", *ids],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def terminate(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the process group; SIGKILL it if it doesn't exit within grace_s. Never raises."""
    if proc.poll() is not None:
        return
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(grace_s)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(5)


def render_cell_artifacts(cell: Cell, cell_dir: Path, env: Environment) -> list[Query]:
    ### The rendered SQL no longer carries the spill configuration — spawn_worker owns it. The cell is
    ### still passed so the template can be re-parameterised without touching this signature.
    return [render_query(env, i, w, cell_dir) for i, w in enumerate(cfg.WINDOWS)]


def run_cell(cell: Cell, cell_dir: Path, run_id: str, env: Environment) -> dict:
    queries = render_cell_artifacts(cell, cell_dir, env)

    generators: list[subprocess.Popen] = []
    worker: subprocess.Popen | None = None
    rss_sampler: RssSampler | None = None
    failure_reason = ""
    wall_s = ""
    sinks_nonempty = ""

    try:
        for q in queries:
            for port in (q.port_a, q.port_b):
                generators.append(spawn_generator(port, cell.var_size, cell.rate, cell_dir))
        for g in generators:
            wait_for_ready(g, cfg.GENERATOR_READY_TIMEOUT_S)

        ### Must precede the spawn: the port has to be free before this worker binds it, otherwise
        ### wait_for_grpc below cannot tell this worker from the previous cell's.
        wait_for_port_free(cfg.GRPC_PORT, cfg.WORKER_READY_TIMEOUT_S)
        worker = spawn_worker(cell, cell_dir)
        rss_sampler = RssSampler(worker.pid, cell_dir / "rss.csv", cfg.MEMORY_SAMPLE_INTERVAL_MS)
        rss_sampler.start()
        wait_for_grpc(cfg.GRPC_PORT, cfg.WORKER_READY_TIMEOUT_S)

        persisted_ids: list[str] = []
        t0 = time.monotonic()
        for q in queries:
            persisted_ids.append(nes_cli_start(q.topology_path, q.sql))
            time.sleep(cfg.QUERY_SUBMIT_INTERVAL_S)

        time.sleep(cfg.POST_SUBMIT_DRAIN_S)
        nes_cli_stop(queries[0].topology_path, persisted_ids)
        wall_s = f"{time.monotonic() - t0:.3f}"

        sinks_nonempty = "1" if all(q.sink_path.exists() and q.sink_path.stat().st_size > 0 for q in queries) else "0"
    except Exception as exc:  # noqa: BLE001 — bench-loop must not crash on one bad cell
        failure_reason = f"{type(exc).__name__}: {exc}"
        if not sinks_nonempty:
            sinks_nonempty = "0"
    finally:
        ### ORDER: rss sampler → worker → generators. Sampler first so we stop reading /proc before
        ### the worker pid is gone; worker before generators so its log isn't drowned in reset noise.
        if rss_sampler is not None:
            rss_sampler.stop()
        if worker is not None:
            terminate(worker, cfg.WORKER_SIGTERM_GRACE_S)
        for g in generators:
            terminate(g, cfg.GENERATOR_SIGTERM_GRACE_S)

    avg_tps, n_samples, samples = parse_throughput_log(cell_dir / "worker.log")
    write_throughput_csv(samples, cell_dir / "throughput.csv")
    peak_rss, peak_pool, peak_unp = merge_memory_csv(
        cell_dir / "rss.csv", cell_dir / "buffer_usage.csv", cell_dir / "memory.csv"
    )
    spill_stats = summarize_spill_stats(cell_dir / "spill_stats.csv")

    ### `sinks_nonempty` above is the only thing the join output is read for, and it has been decided
    ### by now. Keeping the files costs tens of GB per cell at unbounded ingestion.
    if not getattr(cfg, "KEEP_SINK_OUTPUT", False):
        for q in queries:
            with suppress(OSError):
                q.sink_path.unlink(missing_ok=True)

    sd = cell.spill_dict
    return {
        "run_id": run_id,
        "cell_slug": cell.slug,
        "threads": cell.threads,
        "operator_buffer_size": cell.buf_size,
        "no_buffers": cell.no_buffers,
        "spill_variant": sd["name"],
        "predictor": sd.get("predictor", ""),
        "high_bound": cell.high_bound,
        "horizon_ms": cell.horizon_ms,
        "var_sized_bytes": cell.var_size,
        "ingestion_rate": cell.rate,
        "wall_s": wall_s,
        "sinks_nonempty": sinks_nonempty,
        "avg_throughput_tps": f"{avg_tps:.3f}" if avg_tps >= 0 else "",
        "throughput_samples": n_samples,
        "peak_rss_bytes": peak_rss,
        "peak_pooled_used": peak_pool,
        "peak_unpooled_used": peak_unp,
        **spill_stats,
        "failure_reason": failure_reason,
    }


def make_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(cfg.TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


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
    args = parse_argv(argv)
    env = make_jinja_env()

    if not args.skip_build and not args.dry_matrix:
        if args.clean_build:
            for d in (cfg.BUILD_DIR, cfg.GENERATOR_DIR / "target"):
                if d.exists():
                    print(f"[bench] removing {d}", flush=True)
                    shutil.rmtree(d)
        cmake_configure()
        cmake_build(cfg.CMAKE_TARGETS)
        cargo_build_release()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cells = list(apply_only_filter(matrix(), args.only))
    if not cells:
        print(f"[bench] no cells match --only={args.only!r}", file=sys.stderr)
        return 2

    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        f.flush()

        start_t = time.monotonic()
        for i, cell in enumerate(cells, start=1):
            cell_dir = run_dir / cell.slug
            cell_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[bench] cell {i}/{len(cells)}: {cell.slug}", flush=True)

            if args.dry_matrix:
                render_cell_artifacts(cell, cell_dir, env)
                row = {
                    "run_id": run_id,
                    "cell_slug": cell.slug,
                    "threads": cell.threads,
                    "operator_buffer_size": cell.buf_size,
                    "no_buffers": cell.no_buffers,
                    "spill_variant": cell.spill_dict["name"],
                    "predictor": cell.spill_dict.get("predictor", ""),
                    "high_bound": cell.high_bound,
                    "horizon_ms": cell.horizon_ms,
                    "var_sized_bytes": cell.var_size,
                    "ingestion_rate": cell.rate,
                    "wall_s": "",
                    "sinks_nonempty": "",
                    "avg_throughput_tps": "",
                    "throughput_samples": "",
                    "peak_rss_bytes": "",
                    "peak_pooled_used": "",
                    "peak_unpooled_used": "",
                    **{k: "" for k in SPILL_RESULT_COLS},
                    "failure_reason": "dry-matrix",
                }
            else:
                row = run_cell(cell, cell_dir, run_id, env)

            writer.writerow(row)
            f.flush()
            ### ETA from mean time per completed cell × cells remaining.
            elapsed = time.monotonic() - start_t
            eta = elapsed / i * (len(cells) - i)
            print(f"[bench] cell {i}/{len(cells)} done [elapsed {_fmt_dur(elapsed)}, eta {_fmt_dur(eta)}]", flush=True)

    print(f"\n[bench] done. results: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
