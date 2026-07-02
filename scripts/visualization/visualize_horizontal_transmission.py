#!/usr/bin/env python3
"""Forest plot of horizontal (conformist) transmission: per-trait convergence on
the prevailing league level (kappa) with 95% HDI, against the pooled population
estimate mu_kappa. Filled markers = HDI excludes zero (credible convergence);
open markers = not credible. Traits ordered by kappa.

Reads outputs/analysis/horizontal_transmission_results.json. Writes
outputs/visualizations/performance/horizontal_transmission_forest.png. ASCII only.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_config import configure_plots

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs/analysis"

KAPPA_COLOR = "#2E86AB"
POOL_COLOR = "#6f6f6f"


def main():
    d = json.load(open(ANALYSIS / "horizontal_transmission_results.json"))
    rows = list(d["per_phenotype"].values())
    rows.sort(key=lambda r: r["kappa_median"])  # ascending -> highest at top

    configure_plots()
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ys = list(range(len(rows)))

    for y, r in zip(ys, rows):
        k, lo, hi = r["kappa_median"], r["hdi_low"], r["hdi_high"]
        credible = (lo > 0) or (hi < 0)
        is_comp = r.get("is_composite", False)
        ax.plot([lo, hi], [y, y], color=KAPPA_COLOR, lw=2.6, alpha=0.85,
                zorder=3, solid_capstyle="round")
        ax.plot(k, y, "o", ms=13 if is_comp else 9,
                mfc=KAPPA_COLOR if credible else "white",
                mec="black" if is_comp else KAPPA_COLOR,
                mew=1.1 if is_comp else 1.0, zorder=4)

    # pooled mu_kappa (population conformity across the ten sub-traits)
    mp = d["pooled"]
    ax.axvspan(mp["hdi_low"], mp["hdi_high"], color=POOL_COLOR, alpha=0.14, zorder=1)
    ax.axvline(mp["mu_kappa_median"], color=POOL_COLOR, lw=1.6, ls="--", alpha=0.8,
               zorder=2)
    ax.axvline(0, color="gray", lw=1.2, ls="-", alpha=0.6, zorder=1)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("bold" if r.get("is_composite") else "normal")
        tick.set_fontsize(12 if r.get("is_composite") else 10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel(r"Convergence on the prevailing league level  $\kappa$  (per season)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Horizontal transmission: convergence on the prevailing league level",
                 fontsize=14.5, fontweight="bold", pad=14)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=KAPPA_COLOR,
               markeredgecolor="black", markersize=11,
               label=r"$\kappa$ (median, 95% HDI); filled = HDI excludes 0"),
        Line2D([0], [0], ls="--", color=POOL_COLOR,
               label=r"pooled $\mu_\kappa$ = %.2f [%.2f, %.2f]" % (
                   mp["mu_kappa_median"], mp["hdi_low"], mp["hdi_high"])),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.25, ls=":", zorder=0)

    plt.tight_layout()
    out = REPO / "outputs/visualizations/performance/horizontal_transmission_forest.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", out)
    plt.close()


if __name__ == "__main__":
    main()
