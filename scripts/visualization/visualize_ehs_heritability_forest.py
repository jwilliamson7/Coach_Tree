#!/usr/bin/env python3
"""Forest plot of transmissibility (h^2) against its repeatability ceiling.

For each trait, the posterior median h^2 with its 95\% HDI (points + bars) is shown
against the repeatability (open marker), the quantitative-genetics ceiling a
heritability cannot exceed. Traits are ordered by h^2 and colored by family. The
plot makes the confirmatory picture visible at once: schematic-identity traits and
defensive box/rush transmit near their ceiling, whereas offensive aggression is
repeatable (ceiling well above zero) but has h^2 indistinguishable from zero, the
preregistered repeatable-but-not-heritable pattern (H2).

Reads outputs/analysis/ehs_heritability_results.json. Writes
outputs/visualizations/performance/ehs_heritability_forest.png. ASCII only.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_config import configure_plots

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "outputs/analysis"

# entity key (composite_* or subtrait) -> (label, family, is_composite)
# Families: approach = offensive risk/attitude; identity = schematic system;
# defensive = mixed (box/rush are attitude-like, man coverage is a system choice).
ROWS = [
    ("composite_aggression", "Offensive aggression", "approach", True),
    ("fourth_down", "  fourth-down", "approach", False),
    ("pass_heavy", "  pass-heavy", "approach", False),
    ("deep_pass", "  deep-pass", "approach", False),
    ("two_point", "  two-point", "approach", False),
    ("composite_defensive", "Defensive aggression", "defensive", True),
    ("box_stacking", "  box-stacking", "defensive", False),
    ("pass_rush", "  pass-rush", "defensive", False),
    ("man_coverage", "  man-coverage", "defensive", False),
    ("composite_tempo", "Tempo", "identity", True),
    ("no_huddle", "  no-huddle", "identity", False),
    ("pace", "  pace", "identity", False),
    ("shotgun", "Shotgun", "identity", True),
]
# Single colour for all traits; the approach / identity / mixed nuance is made in
# the text, not by segmenting the figure.
H2_COLOR = "#2E86AB"


def main():
    d = json.load(open(ANALYSIS / "ehs_heritability_results.json"))

    def get(key):
        grp = "composites" if key.startswith("composite_") else "subtraits"
        k = key.replace("composite_", "")
        e = d[grp][k]
        return e["c1"], e["c2"]

    configure_plots()
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ys = list(range(len(ROWS)))[::-1]  # top-to-bottom in ROWS order

    for y, (key, label, fam, is_comp) in zip(ys, ROWS):
        c1, c2 = get(key)
        h, hl, hh = c1["h2_median"], c1["hdi_low"], c1["hdi_high"]
        rep, rl, rh = c2["repeatability_median"], c2["hdi_low"], c2["hdi_high"]
        col = H2_COLOR
        # repeatability ceiling: 95% HDI band (behind) + median diamond. Shown with
        # its uncertainty so that h2 medians which cross the ceiling are visibly
        # inside the repeatability's credible interval (h2 ~ repeatability, not > it).
        ax.plot([rl, rh], [y, y], color="gray", lw=6, alpha=0.16, zorder=2,
                solid_capstyle="round")
        ax.plot(rep, y, "D", ms=8, mfc="white", mec="gray", mew=1.4, zorder=4)
        # h2 point + HDI
        ax.plot([hl, hh], [y, y], color=col, lw=2.4, alpha=0.85, zorder=3,
                solid_capstyle="round")
        ax.plot(h, y, "o", ms=12 if is_comp else 8, color=col,
                markeredgecolor="black", markeredgewidth=1.0 if is_comp else 0.6, zorder=4)

    ax.axvline(0, color="gray", lw=1.2, ls="-", alpha=0.6, zorder=1)
    # y-axis labels placed outside the data area (no overlap with the bars)
    ax.set_yticks(ys)
    ax.set_yticklabels([lbl.strip() for _, lbl, _, _ in ROWS])
    for tick, (_, _, _, is_comp) in zip(ax.get_yticklabels(), ROWS):
        tick.set_fontweight("bold" if is_comp else "normal")
        tick.set_fontsize(12 if is_comp else 10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, len(ROWS) - 0.2)
    ax.set_xlim(-0.62, 1.02)
    ax.set_xlabel("Transmissibility  $h^{2}$  (filled) and repeatability ceiling (open diamond)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Heritability against its repeatability ceiling, by trait",
                 fontsize=15, fontweight="bold", pad=14)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=H2_COLOR,
               markeredgecolor="black", markersize=11, label="$h^2$ (median, 95% HDI)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=9,
               label="Repeatability ceiling (median, 95% HDI)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=2, fontsize=10.5, framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.25, ls=":", zorder=0)

    plt.tight_layout()
    out = REPO / "outputs/visualizations/performance/ehs_heritability_forest.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", out)
    plt.close()


if __name__ == "__main__":
    main()
