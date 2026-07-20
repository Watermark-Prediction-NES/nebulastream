#!/usr/bin/env python3
"""Plots for the out-of-order + fluctuating-rate watermark predictor experiments.

Reads results/results.csv and results/traces.csv (written by benchmark.py or directly
by the C++ binary via the Python parse step). Saves one PNG per figure under results/.

Usage:
    python3 plot_ooo_results.py [--results-dir path]

The six figures produced:
  fig_ooo_00_traces.png         -- raw scenario shapes (wall-clock vs event-time)
  fig_ooo_01_heatmap.png        -- median |err| + sat-rate heatmap, trace × predictor
  fig_ooo_02_rolling.png        -- rolling prequential error on Fluctuating traces
  fig_ooo_03_ooo_sensitivity.png -- median error vs OOO severity (const-rate control)
  fig_ooo_04_fluct_sat.png      -- stall blowup rate + median err on Fluctuating variants
  fig_ooo_05_timing.png         -- predict / observe throughput + accuracy tradeoff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"

PRED_ORDER = [
    "EWMA(alpha=0.3)", "EWMA(alpha=0.5)", "EWMA(alpha=1.0)",
    "Kalman(stable)", "Kalman(reactive)", "RobustAdaptiveKalman",
    "MLP(win=16,h=16)", "NeuralKalman(h=8)",
]
PRED_SHORT = {
    "EWMA(alpha=0.3)":      "EWMA(0.3)",
    "EWMA(alpha=0.5)":      "EWMA(0.5)",
    "EWMA(alpha=1.0)":      "EWMA(1.0)",
    "Kalman(stable)":       "Kalman(S)",
    "Kalman(reactive)":     "Kalman(R)",
    "RobustAdaptiveKalman": "RAKalman",
    "MLP(win=16,h=16)":     "MLP",
    "NeuralKalman(h=8)":    "NeuralKalman",
}

TRACE_ORDER = [
    "ConstantRate(2.0) + mild out-of-order (p=0.20 maxDelay=3)",
    "ConstantRate(2.0) + heavy out-of-order (p=0.50 maxDelay=8)",
    "Fluctuating clean",
    "Fluctuating + mild out-of-order (p=0.20 maxDelay=3)",
    "Fluctuating + heavy out-of-order (p=0.50 maxDelay=8)",
    "Fluctuating + jitter(sd=10) + out-of-order(p=0.20 maxDelay=3)",
]
TRACE_SHORT = {
    "ConstantRate(2.0) + mild out-of-order (p=0.20 maxDelay=3)":        "Const\n+mild OOO",
    "ConstantRate(2.0) + heavy out-of-order (p=0.50 maxDelay=8)":       "Const\n+heavy OOO",
    "Fluctuating clean":                                                  "Fluct\nclean",
    "Fluctuating + mild out-of-order (p=0.20 maxDelay=3)":              "Fluct\n+mild OOO",
    "Fluctuating + heavy out-of-order (p=0.50 maxDelay=8)":             "Fluct\n+heavy OOO",
    "Fluctuating + jitter(sd=10) + out-of-order(p=0.20 maxDelay=3)":   "Fluct\n+jitter+OOO",
}
# saturation threshold: abs_err at/above this is a prediction that hit the MIN_RATE clamp
# (rate→0 during stall → predicted = wall + horizon/1e-12 ≈ huge); not a real miss
SAT = 1e9

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(context="paper", style="whitegrid", font="serif")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
})
PALETTE = dict(zip(PRED_ORDER, sns.color_palette("tab10", len(PRED_ORDER))))
MARKERS = dict(zip(PRED_ORDER, ["o","s","^","D","v","P","X","*"]))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(results_dir / "results.csv")
    df["saturated"] = df["abs_err"] >= SAT
    traces = pd.read_csv(results_dir / "traces.csv")
    return df, traces


def agg_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (trace, predictor) with robust accuracy metrics."""
    TIMING = ["predict_ns_per_op","predict_ops_per_sec","observe_ns_per_op","observe_ops_per_sec"]

    def _agg(g):
        ae, se, tw, sat = g["abs_err"], g["signed_err"], g["true_wall"], g["saturated"]
        unsat = ~sat
        real = unsat & (tw > 0)
        s = {
            "mae":       ae.mean(),
            "mdae":      ae.median(),
            "rmse":      np.sqrt((se**2).mean()),
            "sat_rate":  sat.mean(),
            "max_err":   ae[unsat].max() if unsat.any() else np.nan,
            "mdape_pct": (ae[real] / tw[real] * 100).median() if real.any() else np.nan,
            "samples":   len(g),
        }
        for col in TIMING:
            s[col] = g[col].iloc[0]
        return pd.Series(s)

    cell = (df.groupby(["trace","predictor"], sort=False)
              [["abs_err","signed_err","true_wall","saturated"] + TIMING]
              .apply(_agg).reset_index())
    return cell


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, path: Path, name: str) -> Path:
    out = path / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out


# ---------------------------------------------------------------------------
# Fig 0: scenario shapes
# ---------------------------------------------------------------------------
def fig_traces(traces: pd.DataFrame, out: Path) -> None:
    scenarios = list(traces["scenario"].unique())
    ncol = 2
    nrow = -(-len(scenarios) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.8 * nrow))
    for ax, sc in zip(axes.flat, scenarios):
        g = traces[traces["scenario"] == sc]
        ax.plot(g["event_time"], g["wall_clock"], lw=1.6, color="C0")
        ax.set_title(sc, fontsize=9)
        ax.set_xlabel("watermark (event-time)")
        ax.set_ylabel("wall-clock")
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    fig.suptitle("Base scenario shapes: wall-clock vs watermark event-time", y=1.01, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_ooo_00_traces.png")


# ---------------------------------------------------------------------------
# Fig 1: heatmap (median |err| + sat-rate)
# ---------------------------------------------------------------------------
def fig_heatmap(cell: pd.DataFrame, out: Path) -> None:
    short_preds = [PRED_SHORT[p] for p in PRED_ORDER]
    short_traces = [TRACE_SHORT[t] for t in TRACE_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (metric, label, fmt) in zip(
        axes,
        [("mdae", "Median |err| (wall-clock units, lower = better)", ".0f"),
         ("sat_rate", "Stall-blowup rate (fraction, lower = better)", ".2f")]
    ):
        pivot = (cell[cell["trace"].isin(TRACE_ORDER) & cell["predictor"].isin(PRED_ORDER)]
                 .pivot_table(index="trace", columns="predictor", values=metric)[PRED_ORDER]
                 .reindex(TRACE_ORDER))
        pivot.index = short_traces
        pivot.columns = short_preds

        span = (pivot.max(axis=1) - pivot.min(axis=1)).replace(0, 1)
        norm = pivot.sub(pivot.min(axis=1), axis=0).div(span, axis=0)

        sns.heatmap(norm, annot=pivot, fmt=fmt, cmap="rocket_r", linewidths=0.4,
                    annot_kws={"fontsize": 7.5}, cbar=False, vmin=0, vmax=1, ax=ax)
        # cyan outline for per-row best
        for row_i, vals in enumerate(pivot.values):
            best = np.nanmin(vals)
            for col_i, v in enumerate(vals):
                if v == best:
                    ax.add_patch(mpatches.Rectangle(
                        (col_i, row_i), 1, 1, fill=False, edgecolor="#00d0ff", lw=2))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(label, pad=6)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Accuracy heatmap: out-of-order experiments  (cyan = best in row)", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_ooo_01_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 2: rolling prequential error on Fluctuating traces
# ---------------------------------------------------------------------------
def fig_rolling(df: pd.DataFrame, out: Path) -> None:
    fluct_traces = [t for t in TRACE_ORDER if t.startswith("Fluctuating")]
    ncol = 2
    nrow = -(-len(fluct_traces) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.0 * nrow), sharey=True)

    for ax, trace in zip(axes.flat, fluct_traces):
        sub = df[df["trace"] == trace]
        # aggregate over horizons: mean abs_err per (eval_offset, predictor)
        pivot = (sub[~sub["saturated"]]  # exclude stall-blowup points for the rolling view
                 .groupby(["eval_offset","predictor"])["abs_err"]
                 .mean().reset_index())
        for pred in PRED_ORDER:
            p = pivot[pivot["predictor"] == pred]
            if p.empty:
                continue
            ax.plot(p["eval_offset"], p["abs_err"],
                    color=PALETTE[pred], marker=MARKERS[pred],
                    markevery=max(1, len(p)//15), markersize=4,
                    lw=1.2, label=PRED_SHORT[pred])
        ax.set_title(TRACE_SHORT[trace].replace("\n"," "), fontsize=9)
        ax.set_xlabel("eval offset (samples since warmup)")
        ax.set_ylabel("mean |err| (non-saturated)")
        ax.set_yscale("log")

    for ax in axes.flat[len(fluct_traces):]:
        ax.set_visible(False)

    handles = [mpatches.Patch(color=PALETTE[p], label=PRED_SHORT[p]) for p in PRED_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Rolling prediction error on Fluctuating traces (saturated points excluded)",
                 y=1.06, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_ooo_02_rolling.png")


# ---------------------------------------------------------------------------
# Fig 3: OOO severity sweep on constant-rate control
# ---------------------------------------------------------------------------
def fig_ooo_sensitivity(cell: pd.DataFrame, out: Path) -> None:
    const_traces = [t for t in TRACE_ORDER if t.startswith("ConstantRate")]
    sub = cell[cell["trace"].isin(const_traces)].copy()
    sub["trace_short"] = sub["trace"].map(TRACE_SHORT).str.replace("\n", " ")
    sub["pred_short"] = sub["predictor"].map(PRED_SHORT)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (metric, label, log) in zip(
        axes,
        [("mdae", "Median |err| (wall-clock, lower = better)", True),
         ("sat_rate", "Stall-blowup rate (lower = better)", False)]
    ):
        trace_order_short = [TRACE_SHORT[t].replace("\n"," ") for t in const_traces]
        pred_order_short = [PRED_SHORT[p] for p in PRED_ORDER]
        sns.barplot(sub, x="trace_short", y=metric,
                    hue="pred_short", hue_order=pred_order_short,
                    palette={PRED_SHORT[p]: PALETTE[p] for p in PRED_ORDER},
                    order=trace_order_short, ax=ax)
        ax.set_xlabel("scenario")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=10)
        if log:
            ax.set_yscale("log")
        ax.get_legend().remove()

    handles = [mpatches.Patch(color=PALETTE[p], label=PRED_SHORT[p]) for p in PRED_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Out-of-order severity on ConstantRate stream: median error and blowup rate",
                 y=1.06, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_ooo_03_ooo_sensitivity.png")


# ---------------------------------------------------------------------------
# Fig 4: Fluctuating variants — sat-rate + median error side by side
# ---------------------------------------------------------------------------
def fig_fluct_sat(cell: pd.DataFrame, out: Path) -> None:
    fluct_traces = [t for t in TRACE_ORDER if t.startswith("Fluctuating")]
    sub = cell[cell["trace"].isin(fluct_traces)].copy()
    sub["trace_short"] = sub["trace"].map(TRACE_SHORT).str.replace("\n", " ")
    sub["pred_short"] = sub["predictor"].map(PRED_SHORT)
    trace_order_short = [TRACE_SHORT[t].replace("\n"," ") for t in fluct_traces]
    pred_order_short = [PRED_SHORT[p] for p in PRED_ORDER]
    pal_short = {PRED_SHORT[p]: PALETTE[p] for p in PRED_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (metric, label, log) in zip(
        axes,
        [("sat_rate", "Stall-blowup rate (fraction of predictions)", False),
         ("mdae", "Median |err| across all predictions incl. blowups", True)]
    ):
        sns.barplot(sub, x="trace_short", y=metric,
                    hue="pred_short", hue_order=pred_order_short, palette=pal_short,
                    order=trace_order_short, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        if log:
            ax.set_yscale("log")
        ax.get_legend().remove()

    handles = [mpatches.Patch(color=PALETTE[p], label=PRED_SHORT[p]) for p in PRED_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Fluctuating-rate traces: stall blowup rate and median error per predictor",
                 y=1.06, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_ooo_04_fluct_sat.png")


# ---------------------------------------------------------------------------
# Fig 5: Accuracy vs. observe() throughput
# ---------------------------------------------------------------------------
def fig_timing(cell: pd.DataFrame, out: Path) -> None:
    # use constant-rate rows for accuracy (no stall blowup, fair comparison)
    const = cell[cell["trace"].isin([t for t in TRACE_ORDER if t.startswith("ConstantRate")])].copy()
    timing = cell.drop_duplicates("predictor")[["predictor","predict_ops_per_sec","observe_ops_per_sec"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # scatter: accuracy (MdAE) vs observe throughput
    ax = axes[0]
    for _, row in const.drop_duplicates("predictor").iterrows():
        p = row["predictor"]
        t_row = timing[timing["predictor"] == p].iloc[0]
        ax.scatter(t_row["observe_ops_per_sec"], row["mdae"],
                   color=PALETTE[p], marker=MARKERS[p], s=80, zorder=3,
                   label=PRED_SHORT[p])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("observe() throughput (calls/sec)")
    ax.set_ylabel("Median |err| on ConstantRate+OOO")
    ax.set_title("Accuracy vs. ingestion throughput")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # bar: predict throughput
    ax = axes[1]
    timing_sorted = timing.set_index("predictor").reindex(PRED_ORDER).reset_index()
    ax.bar([PRED_SHORT[p] for p in PRED_ORDER],
           [timing_sorted[timing_sorted["predictor"]==p]["predict_ops_per_sec"].iloc[0]
            if not timing_sorted[timing_sorted["predictor"]==p].empty else 0
            for p in PRED_ORDER],
           color=[PALETTE[p] for p in PRED_ORDER])
    ax.set_yscale("log")
    ax.set_ylabel("predictWallClock() calls/sec")
    ax.set_title("Predict throughput")
    ax.tick_params(axis="x", rotation=35, labelsize=7)

    # bar: observe throughput
    ax = axes[2]
    ax.bar([PRED_SHORT[p] for p in PRED_ORDER],
           [timing_sorted[timing_sorted["predictor"]==p]["observe_ops_per_sec"].iloc[0]
            if not timing_sorted[timing_sorted["predictor"]==p].empty else 0
            for p in PRED_ORDER],
           color=[PALETTE[p] for p in PRED_ORDER])
    ax.set_yscale("log")
    ax.set_ylabel("observe() calls/sec")
    ax.set_title("Observe throughput")
    ax.tick_params(axis="x", rotation=35, labelsize=7)

    fig.suptitle("Latency/throughput: predict is equally fast for all; observe() cost separates ML",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_ooo_05_timing.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args()
    results_dir = Path(args.results_dir)

    print(f"[plot] reading {results_dir}")
    df, traces = load(results_dir)
    cell = agg_cells(df)
    out = results_dir

    print("[plot] generating figures …")
    fig_traces(traces, out)
    fig_heatmap(cell, out)
    fig_rolling(df, out)
    fig_ooo_sensitivity(cell, out)
    fig_fluct_sat(cell, out)
    fig_timing(cell, out)
    print("[plot] done.")


if __name__ == "__main__":
    main()
