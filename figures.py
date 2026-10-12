"""Figures for the paper. Reads results/ only.

Figure 1: (a) resting heart rate, per-subject error of the network against the per-subject median under the
within-subject temporal split; (b) subject-identification accuracy of the same network under the two within-subject
protocols. Figure 2: subject-disjoint error of the network relative to the population median, per scenario and
pooled, heart rate and systolic pressure.

    python figures.py
Writes figures/fig1.png, fig2.png (600 dpi) and .pdf.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SLOT = ["#2a78d6", "#eb6834"]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def j(n):
    return json.loads((R / n).read_text(encoding="utf-8"))


def fig1():
    ps, probe = j("per_subject_temporal.json"), j("gate2_vitals.json")["subject_id_probe"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    c = np.array([v["cnn_mae"] for v in ps.values()]); l = np.array([v["subj_median_mae"] for v in ps.values()])
    lim = max(c.max(), l.max()) * 1.08
    a.plot([0, lim], [0, lim], color=GRID, lw=1, zorder=1)
    a.scatter(l, c, s=18, color=SLOT[0], edgecolor=INK, lw=0.4, zorder=2)
    a.set_xlabel("Per-subject median, MAE (beats per minute)"); a.set_ylabel("Network, MAE (beats per minute)")
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.text(0.97, 0.04, f"lookup better on {(l < c).sum()} of {len(c)} subjects", transform=a.transAxes, va="bottom", ha="right", color=INK2)
    a.set_title("(a) Within-subject temporal split", loc="left", color=INK)
    names = ["Random windows", "Temporal 8:2"]
    vals = [100 * probe["random_windows_acc"], 100 * probe["temporal_8_2_acc"]]
    seeds = [100 * np.array(probe["seeds"]["random"]), 100 * np.array(probe["seeds"]["temporal"])]
    b.bar(names, vals, color=SLOT, width=0.55, edgecolor=INK, lw=0.4)
    for i, s in enumerate(seeds):
        b.plot([i] * len(s), s, "o", color=INK, ms=2.5)
    b.axhline(100 / 30, color=INK2, lw=0.8, ls="--"); b.text(1.3, 100 / 30 + 1.5, "chance", color=INK2, fontsize=7)
    b.set_ylim(0, 105); b.set_ylabel("Subject identification accuracy (%)")
    b.set_title("(b) Subject identification", loc="left", color=INK)
    for ax in (a, b):
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1.png", dpi=600); fig.savefig(FIG / "fig1.pdf"); plt.close(fig)


def fig2():
    ev = j("eval_scenarios.json")
    order = [("Resting", "Resting"), ("Valsalva", "Valsalva"), ("Apnea", "Apnea"), ("TiltUp", "Tilt-up"), ("TiltDown", "Tilt-down"), ("all", "All pooled")]
    hr = [ev[k]["hr"]["subject_disjoint_5fold"]["cnn"] / ev[k]["hr"]["subject_disjoint_5fold"]["pop_median"] for k, _ in order]
    bpv = [ev[k]["bp"]["subject_disjoint_5fold"]["cnn_sbp"] / ev[k]["bp"]["subject_disjoint_5fold"]["pop_sbp"] for k, _ in order]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=False)
    x = np.arange(len(order)); labels = [n for _, n in order]
    for ax, vals, title in ((a, hr, "(a) Heart rate"), (b, bpv, "(b) Systolic pressure")):
        colors = [SLOT[0] if v < 1 else SLOT[1] for v in vals]
        ax.bar(x, vals, color=colors, edgecolor=INK, lw=0.4, width=0.6)
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Network MAE / population-median MAE"); ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_ylim(0, max(1.4, max(vals) * 1.1))
    fig.tight_layout()
    fig.savefig(FIG / "fig2.png", dpi=600); fig.savefig(FIG / "fig2.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
    print("wrote figures/fig1.png, fig2.png")
