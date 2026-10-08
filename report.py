"""Builds the charts and tables in docs/ from the bundled data.

Usage:  python report.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd

from crossings import quality
from crossings.clean import classify, clean, group, usable
from crossings.districts import assign_districts, load_districts
from crossings.load import TAGS, load_city

DOCS = Path(__file__).resolve().parent / "docs"

KIND_ORDER = ["signals", "zebra", "unmarked", "unknown"]
KIND_LABEL = {"signals": "Traffic lights", "zebra": "Zebra / marked", "unmarked": "Unmarked", "unknown": "Not mapped"}
KIND_COLOR = {"signals": "#2a78d6", "zebra": "#1baf7a", "unmarked": "#eb6834", "unknown": "#a3a29b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def kind_mix_chart(by_district: pd.DataFrame, path: Path) -> None:
    """Horizontal stacked bars: what kind of crossing, per district, sorted by share with lights."""
    table = by_district.sort_values("signals_pct")
    labels = [f"{d}. {name}" for d, name in table.index]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    left = pd.Series(0.0, index=table.index)
    for kind in KIND_ORDER:
        column = "unknown_kind_pct" if kind == "unknown" else f"{kind}_pct"
        ax.barh(labels, table[column], left=left, color=KIND_COLOR[kind], label=KIND_LABEL[kind],
                height=0.72, edgecolor="white", linewidth=1)
        left += table[column]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of crossings")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False)
    ax.set_title("Vienna: what kind of crossing, by district", loc="left", color=INK, pad=28, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def missing_data_chart(by_district: pd.DataFrame, path: Path) -> None:
    """Two small panels sharing the district axis: missing sound info, missing kerb info."""
    table = by_district.sort_values("sound_missing_pct_of_signals")
    labels = [f"{d}. {name}" for d, name in table.index]
    fig, axes = plt.subplots(1, 2, figsize=(9, 7.5), sharey=True)
    panels = [
        ("sound_missing_pct_of_signals", "Traffic lights with no acoustic-signal info"),
        ("kerb_missing_pct", "Crossings with no kerb info"),
    ]
    for ax, (column, title) in zip(axes, panels):
        ax.barh(labels, table[column], color="#2a78d6", height=0.72)
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("% missing in OpenStreetMap")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Vienna: where the data has gaps", x=0.02, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def map_chart(crossings: pd.DataFrame, districts: pd.DataFrame, path: Path) -> None:
    """Every crossing as a dot, coloured by kind, over the district outlines."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for geometry in districts["geometry"]:
        for polygon in getattr(geometry, "geoms", [geometry]):
            x, y = polygon.exterior.xy
            ax.plot(x, y, color="#c3c2b7", linewidth=0.7, zorder=1)
    inside = crossings.dropna(subset=["district"])
    for kind in KIND_ORDER:
        part = inside[inside["kind"] == kind]
        ax.scatter(part["lon"], part["lat"], s=2.5, color=KIND_COLOR[kind], label=f"{KIND_LABEL[kind]} ({len(part):,})",
                   zorder=3 if kind == "unmarked" else 2, linewidths=0)
    ax.set_aspect(1 / 0.665)  # 1° of longitude is ~0.665 of 1° of latitude at 48° N
    ax.axis("off")
    ax.legend(loc="lower left", markerscale=5, frameon=False)
    ax.set_title(f"Vienna: {len(inside):,} street crossings inside the city boundary", loc="left", color=INK,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    districts = load_districts()

    vienna_raw = load_city("vienna")
    budapest_raw = load_city("budapest")
    vienna = assign_districts(clean(vienna_raw), districts)
    table = quality.by_district(vienna)

    kind_mix_chart(table, DOCS / "kind-by-district.png")
    missing_data_chart(table, DOCS / "missing-data-by-district.png")
    map_chart(vienna, districts, DOCS / "map.png")

    cities = pd.DataFrame({
        "Vienna": quality.summary(vienna),
        "Budapest": quality.summary(clean(budapest_raw)),
    })
    nodes = classify(usable(vienna_raw))
    methods = pd.DataFrame({
        method: {
            "crossings": len(g := group(nodes, method)),
            "with_lights": int((g["kind"] == "signals").sum()),
            "largest_group_nodes": int(g["nodes"].max()),
        }
        for method in ["greedy", "dbscan"]
    })

    with open(DOCS / "tables.md", "w", encoding="utf-8") as f:
        f.write("## Vienna vs Budapest\n\n" + cities.to_markdown() + "\n\n")
        f.write("## Grouping method (Vienna)\n\n" + methods.to_markdown() + "\n\n")
        f.write("## Tag completeness, raw Vienna nodes (%)\n\n"
                + quality.tag_completeness(vienna_raw, TAGS).to_frame("% of nodes").to_markdown() + "\n\n")
        f.write("## By district (Vienna)\n\n" + table.to_markdown() + "\n")
    table.to_csv(DOCS / "vienna-districts.csv")
    print(cities, methods, sep="\n\n")
    print(f"\nwrote charts and tables to {DOCS}")


if __name__ == "__main__":
    main()
