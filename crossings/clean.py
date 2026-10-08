"""Turn raw OSM nodes into one row per real crossing.

Three steps, each a small function that is easy to test on its own:

1. ``usable``    drop nodes that are not pedestrian crossings (underground, private, "crossing=no").
2. ``classify``  read the messy OSM tags into a few clean columns (kind, sound, kerb...).
3. ``group``     merge nodes that belong to the same crossing (one intersection can have 15 nodes).

Two ways to decide which nodes belong together are implemented and compared in the README:
``greedy_cluster_ids`` (the method the Crosswise app uses) and ``dbscan_cluster_ids`` (scikit-learn).
"""

import math

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6_371_000
CLUSTER_RADIUS_M = 20

# Traffic signals that are not for pedestrians.
NON_PEDESTRIAN_SIGNALS = {"emergency", "cyclist_crossing", "tram_priority", "blinker", "ramp_meter"}

# "Best known" value per attribute when nodes are merged: the first value in the list wins.
# Kerbs are the exception on purpose: if ANY kerb is raised, a wheelchair user must be warned.
BEST = {
    "kind": ["signals", "zebra", "unmarked", "unknown"],
    "sound": ["yes", "no", None],
    "vibration": ["yes", "no", None],
    "kerb": ["raised", "lowered", None],
    "tactile": ["yes", "no", None],
    "island": ["yes", "no", None],
}


def usable(nodes: pd.DataFrame) -> pd.DataFrame:
    """Only nodes that are pedestrian crossings at street level."""
    is_crossing = nodes["highway"].isin(["crossing", "traffic_signals"])
    street_level = nodes["level"].isna() | (nodes["level"] == "0")
    not_private = nodes["access"] != "private"
    allowed = nodes["crossing"] != "no"  # crossing=no means "you may not cross here"
    pedestrian_signal = ~(
        (nodes["highway"] == "traffic_signals") & nodes["traffic_signals"].isin(NON_PEDESTRIAN_SIGNALS)
    )
    return nodes[is_crossing & street_level & not_private & allowed & pedestrian_signal].copy()


def _yes_no(column: pd.Series) -> pd.Series:
    """'yes' / 'no' stay, anything else (missing, 'locate', 'yes;no'...) becomes None = unknown."""
    return column.where(column.isin(["yes", "no"]), None)


def classify(nodes: pd.DataFrame) -> pd.DataFrame:
    """Add clean columns: kind (signals/zebra/unmarked/unknown), sound, vibration, kerb, tactile, island."""
    out = nodes.copy()
    crossing = out["crossing"]
    markings = out["crossing:markings"].fillna("")

    signals = (crossing == "traffic_signals") | (out["crossing:signals"] == "yes") | (
        out["highway"] == "traffic_signals"
    )
    unmarked = (crossing == "unmarked") | (markings == "no")
    zebra = (
        crossing.isin(["zebra", "marked", "uncontrolled"])
        | (out["crossing_ref"] == "zebra")
        | markings.str.contains("zebra")
        | (markings == "yes")
    )
    # Order matters: a node with lights is "signals" even if it also has zebra markings.
    out["kind"] = np.select([signals, unmarked, zebra], ["signals", "unmarked", "zebra"], default="unknown")

    kerb = out["kerb"]
    out["kerb_clean"] = np.select(
        [kerb.isin(["lowered", "flush", "no"]), kerb.isin(["raised", "yes", "regular"])],
        ["lowered", "raised"],
        default=None,
    )
    out["sound"] = _yes_no(out["traffic_signals:sound"])
    out["vibration"] = _yes_no(out["traffic_signals:vibration"])
    out["tactile"] = _yes_no(out["tactile_paving"])
    out["island"] = _yes_no(out["crossing:island"])
    return out.drop(columns="kerb").rename(columns={"kerb_clean": "kerb"})


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (haversine) distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def greedy_cluster_ids(nodes: pd.DataFrame, radius_m: float = CLUSTER_RADIUS_M) -> np.ndarray:
    """Each node joins the nearest existing group whose centre is within ``radius_m``, else starts a new one.

    Same rule as the Crosswise app. A grid of ``radius_m``-sized cells means each node only looks at
    groups in its own and the 8 neighbouring cells, instead of all ~9,000 groups.
    """
    cell = radius_m / 111_320  # metres → degrees of latitude
    grid: dict[tuple[int, int], list[int]] = {}
    centres: list[list[float]] = []  # per group: [lat, lon, node count]
    labels = np.empty(len(nodes), dtype=int)

    order = np.lexsort((nodes["id"].to_numpy(), nodes["lon"].to_numpy(), nodes["lat"].to_numpy()))
    lats, lons = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    for i in order:
        lat, lon = lats[i], lons[i]
        gi = math.floor(lat / cell)
        gj = math.floor(lon * math.cos(math.radians(lat)) / cell)

        best, best_d = None, math.inf
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for g in grid.get((gi + di, gj + dj), []):
                    d = distance_m(lat, lon, centres[g][0], centres[g][1])
                    if d <= radius_m and d < best_d:
                        best, best_d = g, d

        if best is None:
            best = len(centres)
            centres.append([lat, lon, 0])
            grid.setdefault((gi, gj), []).append(best)
        c = centres[best]  # move the centre to the mean of its nodes
        c[0], c[1], c[2] = (c[0] * c[2] + lat) / (c[2] + 1), (c[1] * c[2] + lon) / (c[2] + 1), c[2] + 1
        labels[i] = best
    return labels


def dbscan_cluster_ids(nodes: pd.DataFrame, radius_m: float = CLUSTER_RADIUS_M) -> np.ndarray:
    """Nodes closer than ``radius_m``, directly or through a chain of nodes, share a cluster.

    DBSCAN with min_samples=1 never marks points as noise, so every node ends up in a cluster.
    The haversine metric works on (lat, lon) in radians and returns distances on the unit sphere.
    """
    coords = np.radians(nodes[["lat", "lon"]].to_numpy())
    model = DBSCAN(eps=radius_m / EARTH_RADIUS_M, min_samples=1, metric="haversine", algorithm="ball_tree")
    return model.fit_predict(coords)


METHODS = {"greedy": greedy_cluster_ids, "dbscan": dbscan_cluster_ids}


def _best(values: pd.Series, order: list) -> object:
    ranks = [order.index(v if isinstance(v, str) else None) for v in values]
    return order[min(ranks)]


def group(nodes: pd.DataFrame, method: str = "greedy", radius_m: float = CLUSTER_RADIUS_M) -> pd.DataFrame:
    """One row per crossing: mean position, number of nodes, and the best-known value per attribute."""
    if nodes.empty:
        return pd.DataFrame(columns=["lat", "lon", "nodes", *BEST])
    clustered = nodes.assign(cluster=METHODS[method](nodes, radius_m))
    aggregations = {"lat": ("lat", "mean"), "lon": ("lon", "mean"), "nodes": ("id", "count")}
    for attr, order in BEST.items():
        aggregations[attr] = (attr, lambda s, order=order: _best(s, order))
    return clustered.groupby("cluster").agg(**aggregations).reset_index(drop=True)


def clean(raw_nodes: pd.DataFrame, method: str = "greedy") -> pd.DataFrame:
    """The whole pipeline: raw OSM nodes → one row per crossing."""
    return group(classify(usable(raw_nodes)), method)
