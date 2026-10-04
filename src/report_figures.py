from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#193047"
BLUE = "#275D92"
TEAL = "#14827E"
MUTED = "#64748B"
PALE_BLUE = "#EAF2FA"
PALE_TEAL = "#E6F5F2"
PALE_GREY = "#F0F3F6"
LINE = "#8292A3"


def save_both(fig, stem):
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )


def add_box(ax, cx, cy, width, height, text, fill, border, *, fontsize=10.5):
    patch = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=border,
        facecolor=fill,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.35,
        transform=ax.transAxes,
        zorder=3,
    )


def add_arrow(ax, start, end, *, dashed=False):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.35,
        linestyle="--" if dashed else "-",
        color=LINE,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(arrow)


def make_workflow():
    fig, ax = plt.subplots(figsize=(12.4, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.978,
        "Chronological preprocessing and validation workflow",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=17,
        fontweight="semibold",
        color=INK,
    )

    add_box(
        ax, 0.5, 0.885, 0.72, 0.075,
        "UCI Bike Sharing hour.csv  |  17,379 hourly records (2011–2012)",
        PALE_BLUE, BLUE, fontsize=11,
    )
    add_box(
        ax, 0.5, 0.755, 0.72, 0.115,
        "Define cnt as the target; exclude casual + registered and record ID instant\n"
        "Document 165 absent timestamps; do not invent zero-rental hours",
        PALE_BLUE, BLUE, fontsize=10.5,
    )
    add_box(
        ax, 0.5, 0.605, 0.72, 0.085,
        "Sort by timestamp  →  chronological 70% / 15% / 15% split",
        PALE_BLUE, BLUE, fontsize=11,
    )

    add_arrow(ax, (0.5, 0.847), (0.5, 0.812))
    add_arrow(ax, (0.5, 0.695), (0.5, 0.650))

    add_box(
        ax, 0.18, 0.415, 0.270, 0.180,
        "TRAIN  |  12,165 rows\nFit one-hot encoding and scaling\nFit Ridge baseline",
        PALE_TEAL, TEAL, fontsize=10.2,
    )
    add_box(
        ax, 0.50, 0.415, 0.270, 0.180,
        "VALIDATION  |  2,607 rows\nUse train-fitted transformations\nCompare calendar and weather inputs",
        PALE_TEAL, TEAL, fontsize=10.2,
    )
    add_box(
        ax, 0.82, 0.415, 0.270, 0.180,
        "TEST  |  2,607 rows\nHeld out during model choices\nNo test error reported",
        PALE_GREY, MUTED, fontsize=10.2,
    )

    for branch_x in (0.18, 0.50, 0.82):
        add_arrow(ax, (0.5, 0.562), (branch_x, 0.515), dashed=branch_x == 0.82)

    add_box(
        ax, 0.34, 0.195, 0.59, 0.135,
        "VALIDATION RESULT  |  same Ridge model and time period\n"
        "Calendar only: MAE 116.52, RMSE 144.75\n"
        "+ Observed weather: MAE 111.10, RMSE 142.34",
        PALE_TEAL, TEAL, fontsize=10.2,
    )
    add_arrow(ax, (0.18, 0.325), (0.19, 0.265))
    add_arrow(ax, (0.50, 0.325), (0.49, 0.265))

    ax.text(
        0.82,
        0.205,
        "Use only after\nfinal selection",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=MUTED,
    )
    add_arrow(ax, (0.82, 0.325), (0.82, 0.250), dashed=True)

    ax.text(
        0.5,
        0.055,
        "Observed weather is an exploratory input scenario, not an advance weather forecast.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    save_both(fig, "bike_preprocessing_workflow")
    plt.close(fig)


def make_validation_chart():
    metrics = ["MAE", "RMSE"]
    calendar = np.array([116.52, 144.75])
    with_weather = np.array([111.10, 142.34])
    positions = np.arange(len(metrics))
    bar_width = 0.29

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_calendar = ax.bar(
        positions - bar_width / 2,
        calendar,
        width=bar_width,
        color=BLUE,
        label="Calendar only",
    )
    bars_weather = ax.bar(
        positions + bar_width / 2,
        with_weather,
        width=bar_width,
        color=TEAL,
        label="Calendar + observed weather",
    )

    for bar in list(bars_calendar) + list(bars_weather):
        ax.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )

    improvements = (calendar - with_weather) / calendar * 100
    for index, improvement in enumerate(improvements):
        ax.text(
            positions[index],
            161,
            f"{improvement:.1f}% lower",
            ha="center",
            fontsize=10.5,
            color=TEAL,
            fontweight="semibold",
        )

    ax.set_ylim(0, 175)
    ax.set_xticks(positions, metrics)
    ax.set_ylabel("Validation error (rentals per hour)", color=INK)
    ax.set_title("Ridge validation error by input set", color=INK, fontsize=15, pad=20)
    ax.grid(axis="y", color="#DCE4EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A8B5C2")
    ax.tick_params(colors=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.015,
        "Same chronological validation period; lower error is better. Weather values were observed, not forecast.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_both(fig, "bike_validation_error_comparison")
    plt.close(fig)


make_workflow()
make_validation_chart()
for path in sorted(OUTPUT_DIR.glob("bike_*")):
    if path.name.startswith(("bike_preprocessing_workflow", "bike_validation_error_comparison")):
        print(path)
