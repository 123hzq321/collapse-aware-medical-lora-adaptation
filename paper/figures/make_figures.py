from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
BLUE = "#2F6B8F"
TEAL = "#2A9D8F"
RED = "#B84A4A"
GOLD = "#D99A2B"
GRAY = "#4A5568"
LIGHT = "#F4F7FA"


def add_box(ax, xy, width, height, text, fc=LIGHT, ec=BLUE, fontsize=10):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1F2937",
        linespacing=1.25,
    )


def add_arrow(ax, start, end, color=GRAY):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def figure1():
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "A  Experimental workflow", fontsize=12, weight="bold", color="#111827")

    boxes = [
        ((0.04, 0.58), "Public medical\nQA datasets"),
        ((0.23, 0.58), "full45k replay\n44,978 examples"),
        ((0.42, 0.58), "LoRA adaptation\nseeds 42/43/44"),
        ((0.61, 0.58), "Same held-out\nmedical QA tasks"),
        ((0.80, 0.58), "Collapse-aware\nbenchmarker"),
    ]
    for xy, text in boxes:
        add_box(ax, xy, 0.15, 0.22, text)

    for i in range(len(boxes) - 1):
        add_arrow(ax, (boxes[i][0][0] + 0.15, 0.69), (boxes[i + 1][0][0], 0.69))

    add_box(ax, (0.42, 0.20), 0.15, 0.18, "Frozen\nbaseline", fc="#FFF8EA", ec=GOLD)
    add_arrow(ax, (0.495, 0.58), (0.495, 0.39), color=GOLD)
    add_arrow(ax, (0.57, 0.29), (0.80, 0.58), color=GOLD)

    add_box(ax, (0.23, 0.12), 0.18, 0.16, "Accuracy, macro-F1,\ndominance, entropy", fc="#EFF8F6", ec=TEAL, fontsize=9)
    add_box(ax, (0.61, 0.12), 0.18, 0.16, "Per-class tables,\nprediction counts", fc="#EFF8F6", ec=TEAL, fontsize=9)
    add_box(ax, (0.80, 0.12), 0.15, 0.16, "Paired tests\nand p-values", fc="#EFF8F6", ec=TEAL, fontsize=9)
    add_arrow(ax, (0.875, 0.58), (0.875, 0.29), color=TEAL)
    add_arrow(ax, (0.80, 0.20), (0.79, 0.20), color=TEAL)
    add_arrow(ax, (0.61, 0.20), (0.41, 0.20), color=TEAL)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2():
    rows = [
        ("PubMedQA 0.5B", 0.210, 0.650, "1/1 -> 0/3"),
        ("PubMedQA 1.5B", 0.620, 0.690, "1/1 -> 0/3"),
        ("PubMedQA 3B", 0.230, 0.643, "1/1 -> 0/3"),
        ("MedMCQA 0.5B", 0.317, 0.414, "1/1 -> 0/3"),
        ("MedMCQA 1.5B", 0.436, 0.539, "0/1 -> 0/3"),
        ("MedMCQA 3B", 0.490, 0.583, "0/1 -> 0/3"),
        ("MedQA 0.5B", 0.288, 0.389, "1/1 -> 0/3"),
        ("MedQA 1.5B", 0.470, 0.535, "0/1 -> 0/3"),
        ("MedQA 3B", 0.496, 0.566, "0/1 -> 0/3"),
    ]
    labels = [row[0] for row in rows]
    frozen = [row[1] for row in rows]
    adapted = [row[2] for row in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh([v + 0.18 for v in y], frozen, height=0.30, color="#D4DAE2", label="Frozen")
    ax.barh([v - 0.18 for v in y], adapted, height=0.30, color=TEAL, label="Full45k LoRA adaptation")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.15, 0.82)
    ax.set_xlabel("Accuracy")
    ax.set_title("Frozen models vs. full45k LoRA adaptation", loc="left", fontsize=12, weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for idx, (_, before, after, collapse) in enumerate(rows):
        ax.text(0.835, idx, f"+{after - before:.3f}", va="center", fontsize=8, color=BLUE)
        ax.text(0.925, idx, collapse, va="center", fontsize=8, color=RED if collapse.startswith("1/1") else GRAY)
    ax.text(0.835, -0.65, "Delta", fontsize=8, weight="bold", color=BLUE)
    ax.text(0.925, -0.65, "Collapse", fontsize=8, weight="bold", color=GRAY)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.22, right=0.78, top=0.92, bottom=0.10)
    fig.savefig(OUT / "figure2_results.pdf", bbox_inches="tight")
    plt.close(fig)


def distribution(ax, x, y, title, labels, values, color):
    width = 0.12
    max_v = max(values)
    ax.text(x, y + 0.23, title, fontsize=9, weight="bold", color="#111827")
    for i, (label, value) in enumerate(zip(labels, values)):
        height = 0.18 * value / max_v
        ax.add_patch(plt.Rectangle((x + i * 0.14, y), width, height, color=color, alpha=0.85))
        ax.text(x + i * 0.14 + width / 2, y - 0.035, label, ha="center", va="top", fontsize=8)


def figure3():
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "A  Mechanism hypothesis", fontsize=12, weight="bold", color="#111827")

    add_box(ax, (0.05, 0.62), 0.16, 0.18, "Frozen small\nmedical LM", fc="#FFF3F0", ec=RED)
    add_box(ax, (0.30, 0.62), 0.18, 0.18, "Label-prior or\nformat bias", fc="#FFF8EA", ec=GOLD)
    add_box(ax, (0.58, 0.62), 0.17, 0.18, "Train LoRA\nadapter only", fc="#EFF8F6", ec=TEAL)
    add_box(ax, (0.80, 0.62), 0.16, 0.18, "Answer-space\nrecalibration", fc="#EFF8F6", ec=TEAL)
    add_arrow(ax, (0.21, 0.71), (0.30, 0.71))
    add_arrow(ax, (0.48, 0.71), (0.58, 0.71))
    add_arrow(ax, (0.75, 0.71), (0.80, 0.71))

    distribution(ax, 0.11, 0.25, "Collapsed prediction distribution", ["yes", "no", "maybe"], [0.17, 0.0, 0.83], RED)
    ax.text(0.09, 0.14, "Dominant-label answers\nlow entropy, low macro-F1", fontsize=9, color=GRAY)

    distribution(ax, 0.70, 0.25, "Recalibrated prediction distribution", ["yes", "no", "maybe"], [0.54, 0.29, 0.17], TEAL)
    ax.text(0.69, 0.14, "Higher coverage\nno collapse", fontsize=9, color=GRAY)

    add_arrow(ax, (0.35, 0.37), (0.70, 0.37), color=TEAL)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "figure3_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    figure1()
    figure2()
    figure3()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
