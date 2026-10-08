"""Load raw OpenStreetMap data (an Overpass API download) into a pandas DataFrame."""

import gzip
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The OSM tags we use. Every tag becomes a column; a missing tag becomes NaN.
TAGS = [
    "highway",
    "crossing",
    "crossing:signals",
    "crossing:markings",
    "crossing_ref",
    "crossing:island",
    "traffic_signals",
    "traffic_signals:sound",
    "traffic_signals:vibration",
    "tactile_paving",
    "kerb",
    "level",
    "access",
]


def read_overpass(path: Path) -> dict:
    """Read an Overpass JSON file, plain or gzipped."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def nodes_frame(overpass: dict) -> pd.DataFrame:
    """One row per OSM node: id, lat, lon and one column per tag in TAGS."""
    rows = [
        {"id": e["id"], "lat": e["lat"], "lon": e["lon"], **{t: e.get("tags", {}).get(t) for t in TAGS}}
        for e in overpass["elements"]
        if e["type"] == "node"
    ]
    return pd.DataFrame(rows, columns=["id", "lat", "lon", *TAGS])


def load_city(city: str) -> pd.DataFrame:
    """Raw crossing nodes of 'vienna' or 'budapest' from the bundled snapshot."""
    return nodes_frame(read_overpass(DATA_DIR / f"overpass-{city}.json.gz"))
