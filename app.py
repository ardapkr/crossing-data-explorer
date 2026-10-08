"""Interactive dashboard: pick a Vienna district and see its crossings and data gaps.

Run:  solara run app.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solara

from crossings import quality
from crossings.clean import clean
from crossings.districts import assign_districts, load_districts
from crossings.load import load_city

# Load and clean once when the app starts (~5 s), not on every click.
DISTRICTS = load_districts()
CROSSINGS = assign_districts(clean(load_city("vienna")), DISTRICTS)
TABLE = quality.by_district(CROSSINGS)
CHOICES = ["All of Vienna"] + [f"{d}. {name}" for d, name in TABLE.index]

KIND_COLOR = {"signals": "#2a78d6", "zebra": "#1baf7a", "unmarked": "#eb6834", "unknown": "#a3a29b"}
KIND_LABEL = {"signals": "Traffic lights", "zebra": "Zebra / marked", "unmarked": "Unmarked", "unknown": "Not mapped"}

selected = solara.reactive(CHOICES[0])


def crossings_for(choice: str):
    if choice == CHOICES[0]:
        return CROSSINGS.dropna(subset=["district"])
    number = int(choice.split(".")[0])
    return CROSSINGS[CROSSINGS["district"] == number]


def district_map(crossings):
    fig, ax = plt.subplots(figsize=(6, 5))
    for geometry in DISTRICTS["geometry"]:
        for polygon in getattr(geometry, "geoms", [geometry]):
            ax.plot(*polygon.exterior.xy, color="#c3c2b7", linewidth=0.6)
    for kind, color in KIND_COLOR.items():
        part = crossings[crossings["kind"] == kind]
        ax.scatter(part["lon"], part["lat"], s=4, color=color, label=KIND_LABEL[kind], linewidths=0)
    if len(crossings):  # zoom to the selection
        pad = 0.005
        ax.set_xlim(crossings["lon"].min() - pad, crossings["lon"].max() + pad)
        ax.set_ylim(crossings["lat"].min() - pad, crossings["lat"].max() + pad)
    ax.set_aspect(1 / 0.665)
    ax.axis("off")
    ax.legend(loc="lower left", markerscale=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


@solara.component
def Metric(label: str, value: str):
    with solara.Card(style={"width": "200px"}):
        with solara.Column(gap="2px"):
            solara.Text(label, style={"color": "#52514e", "font-size": "0.85rem"})
            solara.Text(value, style={"font-size": "1.6rem", "font-weight": "600"})


@solara.component
def Page():
    crossings = crossings_for(selected.value)
    numbers = quality.summary(crossings)

    solara.Title("Vienna crossing explorer")
    with solara.Column(style={"padding": "16px", "max-width": "1100px"}):
        solara.Markdown("# Vienna crossing explorer")
        solara.Markdown("Street crossings from OpenStreetMap: how safe they are, and how much is not mapped yet.")
        solara.Select(label="District", value=selected, values=CHOICES)

        with solara.Row(style={"flex-wrap": "wrap"}):
            Metric("Crossings (in city)", f"{int(numbers['crossings']):,}")
            Metric("With traffic lights", f"{numbers['signals_pct']}%")
            Metric("Unmarked", f"{numbers['unmarked_pct']}%")
            Metric("Lights with acoustic signal", f"{numbers['sound_yes_pct_of_signals']}%")
            Metric("Lights, sound not mapped", f"{numbers['sound_missing_pct_of_signals']}%")
            Metric("Kerb not mapped", f"{numbers['kerb_missing_pct']}%")

        solara.FigureMatplotlib(district_map(crossings))

        solara.Markdown("## All districts")
        solara.DataFrame(TABLE.reset_index(), items_per_page=23)
