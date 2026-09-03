"""
Results export (JSON) and plots (PNG).

Plots follow a quiet, consistent style: thin marks, hairline gridlines, text
in ink tones (never in the series color), a legend only when there are two or
more series, and one selective direct label rather than a number on every
point. Colors come from a validated categorical palette (adjacent slots are
colorblind-distinguishable). matplotlib is an optional dependency; the JSON
export works without it.

    plot_equity({"strategy": (dates, equity)}, "equity.png", "Momentum on BTC")
    plot_permutation_hist(perm_objs, real_obj, p, "hist.png", "In-sample MCPT")
    plot_leverage_sweep(table, "sweep.png")
    save_results(results, "results.json")
"""

import json
import math

import numpy as np
import pandas as pd

# Chart chrome and ink (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots, fixed order (never cycled past the list).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.reset_index().to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def save_results(results, path):
    """Write any nested dict of results (numpy/pandas inside is fine) as JSON."""
    with open(path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plt():
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError("matplotlib is required for plots: "
                          "pip install matplotlib") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _figure(plt, width=9, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120,
                           facecolor=SURFACE)
    _style(ax)
    return fig, ax


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", color=INK, fontsize=12,
                 fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK_2,
                fontsize=9, va="bottom")


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    _plt().close(fig)
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity(series, path, title, ylabel="Equity (x initial)", log=False):
    """
    Line chart of one or more equity curves.

    series : dict name -> (x, y)  (x may be a DatetimeIndex or bar numbers)
             or a single (x, y) tuple.
    A single series gets no legend (the title names it); its final value is
    direct-labeled at the line end. Multiple series get a legend and end
    labels (up to 4).
    """
    plt = _plt()
    if isinstance(series, tuple):
        series = {title: series}
    if len(series) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} series; fold the rest into "
                         f"'Other' or use several charts.")

    fig, ax = _figure(plt)
    _title(ax, title)
    xmax = None
    for i, (name, (x, y)) in enumerate(series.items()):
        color = SERIES[i]
        x = pd.Index(x)
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, color=color, linewidth=2, solid_joinstyle="round",
                solid_capstyle="round", label=name)
        ax.plot([x[-1]], [y[-1]], marker="o", markersize=7, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2)
        if len(series) <= 4:
            ax.annotate(f"{y[-1]:,.2f}", (x[-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", color=INK_2, fontsize=9,
                        va="center")
        xmax = x[-1] if xmax is None else max(xmax, x[-1])
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    if len(series) >= 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    return _finish(fig, path)


def plot_permutation_hist(perm_objectives, real_objective, p_value, path,
                          title, xlabel="Profit factor", bins=40):
    """
    Histogram of the objective across permutations with the real result
    marked -- the picture behind a permutation p-value.
    """
    plt = _plt()
    perm = np.asarray(perm_objectives, dtype=float)
    perm = perm[np.isfinite(perm)]

    fig, ax = _figure(plt)
    verdict = "pass" if p_value < 0.01 else "marginal" if p_value < 0.05 else "fail"
    _title(ax, title,
           f"{len(perm)} permutations · real = {real_objective:.4f} · "
           f"p = {p_value:.4f} ({verdict})")

    ax.hist(perm, bins=bins, color=SERIES[0], rwidth=0.9,
            edgecolor=SURFACE, linewidth=1)
    ax.axvline(real_objective, color=SERIES[1], linewidth=2)
    ymax = ax.get_ylim()[1]
    ax.annotate("real", (real_objective, ymax * 0.96), xytext=(6, 0),
                textcoords="offset points", color=INK_2, fontsize=9,
                va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Permutations")
    return _finish(fig, path)


def plot_leverage_sweep(table, path, title="Prop-firm pass probability vs leverage"):
    """
    Two small multiples from propfirm.sweep_leverage(): P(pass) and
    P(fail by daily loss), each with the historical and bootstrap estimates.
    One measure per axis -- never a dual axis.
    """
    plt = _plt()
    lev = table["leverage"].values
    panels = [
        ("P(pass)", "hist_p_pass", "boot_p_pass"),
        ("P(fail: daily-loss limit)", "hist_p_fail_daily", "boot_p_fail_daily"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120, facecolor=SURFACE)
    for ax, (label, h_col, b_col) in zip(axes, panels):
        _style(ax)
        _title(ax, label)
        for i, (name, col) in enumerate((("historical", h_col),
                                          ("bootstrap", b_col))):
            y = table[col].values.astype(float)
            ax.plot(lev, y, color=SERIES[i], linewidth=2, marker="o",
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=name)
            ax.annotate(f"{y[-1]:.0%}", (lev[-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", color=INK_2, fontsize=9,
                        va="center")
        ax.set_xlabel("Leverage (x equity)")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def export_validation(results, out_dir, prefix="validation"):
    """
    Save a validate.run() result: JSON plus the two permutation histograms.
    Returns the list of files written.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    files = [save_results(results, os.path.join(out_dir, f"{prefix}.json"))]
    is_res = results.get("insample")
    if is_res:
        files.append(plot_permutation_hist(
            is_res["perm_objectives"], is_res["real_objective"],
            is_res["p_value"], os.path.join(out_dir, f"{prefix}_insample.png"),
            "In-sample permutation test"))
    wf_res = results.get("walkforward")
    if wf_res:
        files.append(plot_permutation_hist(
            wf_res["perm_objectives"], wf_res["real_objective"],
            wf_res["p_value"], os.path.join(out_dir, f"{prefix}_walkforward.png"),
            "Walk-forward permutation test"))
    return files


if __name__ == "__main__":
    # Smoke test on synthetic numbers.
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    eq = np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
    print(plot_equity((dates, eq), os.path.join(out, "demo_equity.png"),
                      "Demo equity"))
    print(plot_permutation_hist(rng.normal(1.0, 0.05, 500), 1.15, 0.004,
                                os.path.join(out, "demo_hist.png"),
                                "Demo permutation test"))
    table = pd.DataFrame({"leverage": [0.25, 0.5, 1, 2],
                          "hist_p_pass": [0.32, 0.11, 0.0, 0.0],
                          "boot_p_pass": [0.40, 0.09, 0.0, 0.0],
                          "hist_p_fail_daily": [0.13, 0.66, 0.9, 0.97],
                          "boot_p_fail_daily": [0.19, 0.71, 0.93, 0.97]})
    print(plot_leverage_sweep(table, os.path.join(out, "demo_sweep.png")))
    print(save_results({"a": np.float64(1.5), "b": np.array([1, 2]),
                        "c": float("nan")}, os.path.join(out, "demo.json")))
