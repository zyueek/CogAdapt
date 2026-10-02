#!/usr/bin/env python3
"""Render saved aggregate paper results, without preprocessing or model execution."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cogadapt-review-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cogadapt.alignment import rank_correlation


def rows(relative):
    with (ROOT / relative).open(newline="") as handle:
        return list(csv.DictReader(handle))


def array(data, column):
    return np.asarray([float(r[column]) if r[column] else np.nan for r in data])


def save(fig, output, name):
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


def accuracy(output):
    data = rows("results/rq3/table2_accuracy.csv")
    methods = ["Regular FT", "Random-K6", "CKA-K6", "CogAdapt"]
    colors = ["#7a7a7a", "#95876a", "#6f8391", "#225e68"]
    markers = ["s", "D", "^", "o"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, model in zip(axes, ["qwen", "glm"], strict=True):
        for j, (method, color, marker) in enumerate(zip(methods, colors, markers, strict=True)):
            for i, benchmark in enumerate(["lcb", "bcb"]):
                row = next(r for r in data if (r["model"], r["benchmark"], r["condition"]) ==
                           (model, benchmark, method))
                value = float(row["pass_at_1_percent"])
                sd = float(row["sample_sd_pp"]) if row["sample_sd_pp"] else None
                y = i + (j - 1.5) * .17
                ax.errorbar(value, y, xerr=sd, marker=marker, color=color, markersize=7,
                            capsize=3, linestyle="none", label=method if i == 0 else None)
                label = f"{value:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")
                ax.annotate(label, (value + (sd or 0), y), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=9)
        ax.set(title=model.upper() if model == "glm" else "Qwen", xlabel="Execution pass@1 (%)",
               xlim=(12, 47), ylim=(1.5, -.5), yticks=[0, 1],
               yticklabels=["LiveCodeBench", "BigCodeBench"])
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .94))
    fig.suptitle("Execution accuracy on the paper benchmarks", y=1.02, fontsize=15)
    fig.text(.5, -.01, "Random-K6 bars show sample SD across the paper's three masks; not confidence intervals.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .02, 1, .83))
    save(fig, output, "table2_accuracy")


def region_alignment(output):
    from reproduce_four_category_figure import verify, redraw
    percentages, denominators = verify(ROOT / "results/rq1")
    redraw(percentages, denominators, output / "region_retention")


def theta_heatmap(output):
    data = rows("results/rq2/theta_depth_time_cells.csv")
    models = list(dict.fromkeys(r["model"] for r in data))
    families = list(dict.fromkeys(r["family"] for r in data))
    fig, axes = plt.subplots(5, 2, figsize=(12, 12))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d6d6d6")
    for j, model in enumerate(models):
        n_blocks = 48 if model == "Qwen" else 47
        for i, family in enumerate(families):
            matrix = np.full((12, n_blocks), np.nan)
            for row in data:
                if row["model"] == model and row["family"] == family:
                    value = row["length_controlled_spearman_rho"]
                    matrix[int(row["reading_bin"]), int(row["block"])] = float(value) if value else np.nan
            ax = axes[i, j]
            im = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest",
                           cmap=cmap, vmin=-1, vmax=1, extent=(-.5, n_blocks-.5, 0, 1))
            ax.set_title(f"{model}\n{family}", fontsize=11)
            ax.set_yticks([1/6, .5, 5/6], ["Early", "Middle", "Late"], fontsize=9)
            for value in [1/3, 2/3]:
                ax.axhline(value, color="white", linewidth=.8)
            ax.set_xticks([0, 10, 20, 30, 40, n_blocks-1])
            if i == 4:
                ax.set_xlabel("Transformer block (zero-based)")
    fig.suptitle("Length-controlled theta–MoE alignment across depth and reading time", fontsize=15)
    fig.tight_layout(rect=(0, .05, .92, .96), h_pad=1.7)
    cax = fig.add_axes((.94, .20, .016, .60))
    fig.colorbar(im, cax=cax, label="Spearman correlation of size-residualized values")
    fig.text(.48, .014, "32 programs; 12 reading bins. Gray = not applicable; no sign flipping or positive-only filtering.",
             ha="center", fontsize=9)
    save(fig, output, "figure5_theta_depth_time")


def program_triangle(output):
    data = rows("results/rq2/program_values.csv")
    hidden = array(data, "qwen_hidden_cosine_distance_b41")
    rt = array(data, "median_reading_time_s")
    theta = array(data, "global_theta_mean_z")
    panels = [
        (hidden, rt, "Qwen block 41 hidden cosine distance", "Reading time (s; log scale)", False, True),
        (rt, theta, "Reading time (s; log scale)", "Global EEG theta (mean z)", True, False),
        (hidden, theta, "Qwen block 41 hidden cosine distance", "Global EEG theta (mean z)", False, False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, (x, y, xlabel, ylabel, logx, logy) in zip(axes, panels, strict=True):
        ax.scatter(x, y, s=38, facecolor="#28666e", edgecolor="white", linewidth=.5, alpha=.9)
        fx, fy = np.log(x) if logx else x, np.log(y) if logy else y
        grid = np.linspace(fx.min(), fx.max(), 100)
        fitted = np.polyval(np.polyfit(fx, fy, 1), grid)
        ax.plot(np.exp(grid) if logx else grid, np.exp(fitted) if logy else fitted,
                color="#a64b50", linewidth=1.7)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Spearman ρ = {rank_correlation(x, y):.3f}")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Program-level correspondence: Qwen block 41", fontsize=15, y=1.02)
    fig.text(.5, -.02, "Each point is one of 32 programs. Block 41 was selected exploratorily; lines are descriptive fits.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .03, 1, .96), w_pad=2)
    save(fig, output, "figure6_program_correspondence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--figure", choices=["all", "4", "5", "6", "table2"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
        "axes.titlesize": 12, "pdf.fonttype": 42, "figure.facecolor": "white",
    })
    plots = {"table2": accuracy, "4": region_alignment, "5": theta_heatmap, "6": program_triangle}
    for key, plot in plots.items():
        if args.figure in ("all", key):
            plot(args.output_dir)


if __name__ == "__main__":
    main()
