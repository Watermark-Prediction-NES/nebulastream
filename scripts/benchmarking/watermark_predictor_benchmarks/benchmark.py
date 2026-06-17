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

"""Wrapper around the C++ WatermarkPredictorBenchmark target.

Builds the binary (unless --skip-build), runs it, captures stdout to output.txt, and
parses the fixed-width table the binary prints into results.csv. No queries run — this
measures predictor accuracy (MAE/RMSE/MAPE/MaxErr) and predictWallClock() timing
(Reps/MeanMs/MinMs/ns-per-predict).

CLI:
    --skip-build      Skip cmake configure/build (use existing binary).
    --clean-build     rm -rf {BUILD_DIR} before configure (forces a fresh build).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config as cfg

RESULT_COLS = ["trace", "predictor", "samples", "mae", "rmse", "mape_pct", "max_err", "reps", "mean_ms", "min_ms", "ns_per_predict"]

### Matches one data row from WatermarkPredictorBenchmark stdout, e.g.
###   "ConstantRate(2.0) clean              EWMA(alpha=0.3)     100   12.34 ...   5   998.20   995.10   9.98"
### Trace name may contain spaces; predictor names are bounded by their fixed pattern.
_ROW_RE = re.compile(
    r"^(?P<trace>\S.*?)\s{2,}"
    r"(?P<predictor>(?:EWMA|Kalman)\([^)]*\))\s+"
    r"(?P<samples>\d+)\s+"
    r"(?P<mae>[\d.]+)\s+"
    r"(?P<rmse>[\d.]+)\s+"
    r"(?P<mape>[\d.]+)\s+"
    r"(?P<max_err>[\d.]+)\s+"
    r"(?P<reps>\d+)\s+"
    r"(?P<mean_ms>[\d.]+)\s+"
    r"(?P<min_ms>[\d.]+)\s+"
    r"(?P<ns_pred>[\d.]+)\s*$"
)


def sh(cmd: str, *, env: dict | None = None) -> None:
    print(f"[bench] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


def cmake_configure() -> None:
    flags = " ".join(f"-D{k}={v}" for k, v in cfg.CMAKE_FLAGS.items())
    sh(f"/usr/bin/cmake -G Ninja -S {cfg.SOURCE_DIR} -B {cfg.BUILD_DIR} {flags}")


def cmake_build(targets: list[str]) -> None:
    target_arg = " ".join(targets)
    env = dict(os.environ, MOLD_JOBS=str(cfg.MOLD_JOBS), NINJA_STATUS="[%f/%t %p %es] ")
    jobs = os.cpu_count() or 4
    sh(f"/usr/bin/cmake --build {cfg.BUILD_DIR} --target {target_arg} -- -j {jobs}", env=env)


def parse_rows(output: str) -> list[dict]:
    rows = []
    for line in output.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        rows.append({
            "trace": m["trace"].strip(),
            "predictor": m["predictor"],
            "samples": int(m["samples"]),
            "mae": float(m["mae"]),
            "rmse": float(m["rmse"]),
            "mape_pct": float(m["mape"]),
            "max_err": float(m["max_err"]),
            "reps": int(m["reps"]),
            "mean_ms": float(m["mean_ms"]),
            "min_ms": float(m["min_ms"]),
            "ns_per_predict": float(m["ns_pred"]),
        })
    return rows


def parse_argv(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-build", action="store_true", help="skip cmake configure/build")
    p.add_argument("--clean-build", action="store_true", help=f"rm -rf {{BUILD_DIR}} before configure")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_argv(argv)

    if not args.skip_build:
        if args.clean_build and cfg.BUILD_DIR.exists():
            print(f"[bench] removing {cfg.BUILD_DIR}", flush=True)
            shutil.rmtree(cfg.BUILD_DIR)
        cmake_configure()
        cmake_build(cfg.CMAKE_TARGETS)

    if not cfg.BENCH_BIN.exists():
        print(f"[bench] binary missing: {cfg.BENCH_BIN}", file=sys.stderr)
        return 2

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench] running {cfg.BENCH_BIN}", flush=True)
    ### Capture stdout (the table we parse); let the binary's stderr inherit ours so its per-cell
    ### progress/ETA streams live instead of being buffered until the run finishes.
    completed = subprocess.run([str(cfg.BENCH_BIN)], check=True, stdout=subprocess.PIPE, text=True)
    (run_dir / "output.txt").write_text(completed.stdout)

    rows = parse_rows(completed.stdout)
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"[bench] done. {len(rows)} rows → {csv_path}", flush=True)
    assert rows, "parser produced no rows — table format probably changed; check output.txt"
    return 0


def _demo() -> None:
    sample = (
        "Trace                                          Predictor              Samples           MAE          RMSE     MAPE(%)        MaxErr    Reps        MeanMs         MinMs       ns/pred\n"
        + "-" * 180 + "\n"
        "ConstantRate(2.0) clean                       EWMA(alpha=0.3)              100         12.34         15.67        3.45         42.10       5        998.20        995.10          9.98\n"
        "ConstantRate(2.0) clean                       Kalman(default)              100          8.20         11.05        2.10         30.00       5       1203.40       1198.70         12.03\n"
        "\n"
    )
    rows = parse_rows(sample)
    assert len(rows) == 2, rows
    assert rows[0]["trace"] == "ConstantRate(2.0) clean"
    assert rows[0]["predictor"] == "EWMA(alpha=0.3)"
    assert rows[0]["samples"] == 100
    assert rows[1]["predictor"] == "Kalman(default)"
    assert rows[1]["mae"] == 8.20
    assert rows[0]["reps"] == 5
    assert rows[0]["ns_per_predict"] == 9.98
    assert rows[1]["min_ms"] == 1198.70
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _demo()
        sys.exit(0)
    sys.exit(main(sys.argv[1:]))
