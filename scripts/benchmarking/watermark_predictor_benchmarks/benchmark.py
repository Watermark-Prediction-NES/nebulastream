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

Builds the binary (unless --skip-build), runs it, captures stdout to output.txt, and splits
the CSV stream it prints into results.csv (one row per scored prediction from the rolling
prequential evaluation) and traces.csv (the clean scenario shapes). No queries run — this
measures predictor accuracy per (eval_offset, horizon) and predictWallClock() timing.

CLI:
    --skip-build      Skip cmake configure/build (use existing binary).
    --clean-build     rm -rf {BUILD_DIR} before configure (forces a fresh build).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config as cfg

### One row per scored prediction. ns_per_predict is denormalised (cell-level timing repeated on
### every row) so the notebook reads a single results.csv with no join.
RESULT_COLS = ["trace", "predictor", "eval_offset", "horizon", "abs_err", "signed_err", "true_wall", "ns_per_predict"]


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


def parse_trace_rows(output: str) -> list[tuple]:
    """Pull the `TRACE,scenario,event,wall` lines the binary emits for the scenario-shape plot."""
    rows = []
    for line in output.splitlines():
        if line.startswith("TRACE,"):
            _, scenario, event, wall = line.split(",", 3)
            rows.append((scenario, int(event), int(wall)))
    return rows


def parse_rows(output: str) -> list[dict]:
    """Split the `ROW,trace,predictor,eval_offset,horizon,abs_err,signed_err,true_wall,ns` lines.
    The last 6 fields are always numeric; split from the right so predictor names containing
    commas (e.g. `MLP(win=16,h=16)`) are captured whole in the left portion."""
    rows = []
    for line in output.splitlines():
        if not line.startswith("ROW,"):
            continue
        left, off, hor, abserr, signerr, truew, nspp = line.rsplit(",", 6)
        _, trace, predictor = left.split(",", 2)
        rows.append({
            "trace": trace,
            "predictor": predictor,
            "eval_offset": int(off),
            "horizon": int(hor),
            "abs_err": float(abserr),
            "signed_err": float(signerr),
            "true_wall": float(truew),
            "ns_per_predict": float(nspp),
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

    ### Per-run archive + the notebook's read location (results/results.csv).
    rows = parse_rows(completed.stdout)
    for csv_path in (run_dir / "results.csv", cfg.RESULTS_ROOT / "results.csv"):
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_COLS)
            w.writeheader()
            w.writerows(rows)

    ### Scenario shapes -> traces.csv next to results.csv, plus the notebook's read location.
    trace_rows = parse_trace_rows(completed.stdout)
    for traces_path in (run_dir / "traces.csv", cfg.RESULTS_ROOT / "traces.csv"):
        with traces_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "event_time", "wall_clock"])
            w.writerows(trace_rows)

    print(f"[bench] done. {len(rows)} rows, {len(trace_rows)} trace samples → {run_dir / 'results.csv'}", flush=True)
    assert rows, "parser produced no rows — table format probably changed; check output.txt"
    return 0


def _demo() -> None:
    trace_sample = "TRACE,Stall(2.0->0),1500,2000\nTRACE,CatchUp(2.0->8.0),3000,2500\n"
    tr = parse_trace_rows(trace_sample)
    assert tr == [("Stall(2.0->0)", 1500, 2000), ("CatchUp(2.0->8.0)", 3000, 2500)], tr

    sample = (
        "ROW,ConstantRate(2.0) clean,EWMA(alpha=0.3),0,500,12.340,12.340,1100.000,9.980\n"
        "ROW,ConstantRate(2.0) clean,RobustAdaptiveKalman,3,5000,-7.100,-7.100,21000.000,13.000\n"
        "TRACE,ConstantRate(2.0),1000,1000\n"  # interleaved trace line must be ignored here
    )
    rows = parse_rows(sample)
    assert len(rows) == 2, rows
    assert rows[0]["trace"] == "ConstantRate(2.0) clean"
    assert rows[0]["predictor"] == "EWMA(alpha=0.3)"
    assert rows[0]["eval_offset"] == 0
    assert rows[0]["horizon"] == 500
    assert rows[0]["abs_err"] == 12.34
    assert rows[0]["ns_per_predict"] == 9.98
    assert rows[1]["predictor"] == "RobustAdaptiveKalman"
    assert rows[1]["horizon"] == 5000
    assert rows[1]["signed_err"] == -7.10
    assert rows[1]["true_wall"] == 21000.0
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _demo()
        sys.exit(0)
    sys.exit(main(sys.argv[1:]))
