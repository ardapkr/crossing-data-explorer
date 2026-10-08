"""Which Vienna district (Bezirk 1–23) is each crossing in?"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from crossings.load import DATA_DIR


def load_districts(path: Path = DATA_DIR / "vienna-districts.geojson") -> pd.DataFrame:
    """District number, name and polygon (from the City of Vienna open data, simplified)."""
    with open(path, encoding="utf-8") as f:
        features = json.load(f)["features"]
    return pd.DataFrame(
        {
            "district": [f["properties"]["district"] for f in features],
            "name": [f["properties"]["name"] for f in features],
            "geometry": [shapely.geometry.shape(f["geometry"]) for f in features],
        }
    )


def assign_districts(crossings: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    """Add ``district`` and ``district_name`` columns. Crossings outside every district get NaN.

    A spatial index (STRtree) first finds the polygons whose bounding box contains the point,
    then only those few polygons are checked exactly, so 9,000 points take milliseconds.
    """
    points = shapely.points(crossings["lon"].to_numpy(), crossings["lat"].to_numpy())
    tree = shapely.STRtree(districts["geometry"].to_numpy())
    point_idx, district_idx = tree.query(points, predicate="within")

    number = np.full(len(crossings), np.nan)
    number[point_idx] = districts["district"].to_numpy()[district_idx]

    out = crossings.copy()
    out["district"] = pd.array(number, dtype="Int64")
    out["district_name"] = out["district"].map(dict(zip(districts["district"], districts["name"])))
    return out
