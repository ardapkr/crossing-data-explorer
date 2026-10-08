"""Safety and data-quality numbers per area.

Two kinds of question:

* **Safety:** how many crossings have traffic lights, and how many of those have an acoustic signal for blind
  people? How many are unmarked?
* **Data quality:** how often is the information simply missing in OpenStreetMap? A missing tag is not "no",
  it is "unknown", and a navigation app has to treat it differently.
"""

import pandas as pd


def _share(mask: pd.Series) -> float:
    """Percentage of True values (0–100), NaN for an empty group."""
    return round(100 * mask.mean(), 1) if len(mask) else float("nan")


def summary(crossings: pd.DataFrame) -> pd.Series:
    """The key numbers for one set of crossings (a whole city or one district)."""
    signals = crossings["kind"] == "signals"
    on_signals = crossings[signals]
    return pd.Series(
        {
            "crossings": len(crossings),
            "signals_pct": _share(signals),
            "zebra_pct": _share(crossings["kind"] == "zebra"),
            "unmarked_pct": _share(crossings["kind"] == "unmarked"),
            "unknown_kind_pct": _share(crossings["kind"] == "unknown"),
            # of the crossings with lights: acoustic signal yes / sound not mapped
            "sound_yes_pct_of_signals": _share(on_signals["sound"] == "yes"),
            "sound_missing_pct_of_signals": _share(on_signals["sound"].isna()),
            "kerb_missing_pct": _share(crossings["kerb"].isna()),
            "tactile_missing_pct": _share(crossings["tactile"].isna()),
        }
    )


def by_district(crossings: pd.DataFrame) -> pd.DataFrame:
    """One row per district (needs the ``district`` column from ``assign_districts``)."""
    inside = crossings.dropna(subset=["district"])
    rows = {
        (int(district), name): summary(group)
        for (district, name), group in inside.groupby(["district", "district_name"])
    }
    table = pd.DataFrame(rows).T
    table.index.names = ["district", "name"]
    return table.astype({"crossings": int})


def tag_completeness(raw_nodes: pd.DataFrame, tags: list[str]) -> pd.Series:
    """For raw OSM nodes: what percentage has each tag at all? Sorted from best to worst mapped."""
    return (raw_nodes[tags].notna().mean() * 100).round(1).sort_values(ascending=False)
