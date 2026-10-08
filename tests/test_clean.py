import math

import pandas as pd
import pytest

from crossings.clean import classify, clean, dbscan_cluster_ids, distance_m, greedy_cluster_ids, group, usable
from crossings.load import TAGS, nodes_frame

# Two points 10 m apart in Vienna: 10 m north is about 0.00009 degrees of latitude.
LAT, LON = 48.2, 16.37
NORTH_10M = 10 / 111_320


def frame(*nodes: dict) -> pd.DataFrame:
    """Build a raw-nodes DataFrame like load.nodes_frame does, from short dicts."""
    elements = [
        {"type": "node", "id": i, "lat": n.pop("lat", LAT), "lon": n.pop("lon", LON), "tags": n}
        for i, n in enumerate(nodes, start=1)
    ]
    return nodes_frame({"elements": elements})


def test_nodes_frame_has_one_column_per_tag():
    df = frame({"highway": "crossing", "kerb": "lowered"})
    assert list(df.columns) == ["id", "lat", "lon", *TAGS]
    assert df.loc[0, "kerb"] == "lowered"
    assert pd.isna(df.loc[0, "crossing"])


class TestUsable:
    def test_keeps_crossings_and_signals(self):
        df = frame({"highway": "crossing"}, {"highway": "traffic_signals"}, {"highway": "bus_stop"})
        assert list(usable(df)["id"]) == [1, 2]

    @pytest.mark.parametrize(
        "tags",
        [
            {"highway": "crossing", "level": "-1"},  # underground passage
            {"highway": "crossing", "access": "private"},
            {"highway": "crossing", "crossing": "no"},  # crossing not allowed
            {"highway": "traffic_signals", "traffic_signals": "tram_priority"},
        ],
    )
    def test_drops_non_pedestrian_nodes(self, tags):
        assert usable(frame(tags)).empty

    def test_level_zero_is_street_level(self):
        assert len(usable(frame({"highway": "crossing", "level": "0"}))) == 1


class TestClassify:
    @pytest.mark.parametrize(
        ("tags", "kind"),
        [
            ({"highway": "traffic_signals"}, "signals"),
            ({"highway": "crossing", "crossing": "traffic_signals"}, "signals"),
            ({"highway": "crossing", "crossing:signals": "yes"}, "signals"),
            ({"highway": "crossing", "crossing": "zebra"}, "zebra"),
            ({"highway": "crossing", "crossing": "uncontrolled"}, "zebra"),
            ({"highway": "crossing", "crossing:markings": "zebra;dots"}, "zebra"),
            ({"highway": "crossing", "crossing": "unmarked"}, "unmarked"),
            ({"highway": "crossing", "crossing:markings": "no"}, "unmarked"),
            ({"highway": "crossing"}, "unknown"),
            # lights win over markings
            ({"highway": "crossing", "crossing": "traffic_signals", "crossing:markings": "zebra"}, "signals"),
        ],
    )
    def test_kind(self, tags, kind):
        assert classify(frame(tags)).loc[0, "kind"] == kind

    @pytest.mark.parametrize(("raw", "clean_value"), [("lowered", "lowered"), ("flush", "lowered"), ("raised", "raised")])
    def test_kerb(self, raw, clean_value):
        assert classify(frame({"highway": "crossing", "kerb": raw})).loc[0, "kerb"] == clean_value

    def test_unclear_values_become_unknown_not_guessed(self):
        row = classify(frame({"highway": "traffic_signals", "traffic_signals:sound": "yes;no"})).loc[0]
        assert row["sound"] is None or pd.isna(row["sound"])
        assert pd.isna(row["kerb"])


def test_distance_m_matches_known_value():
    # one degree of latitude is about 111.2 km
    assert distance_m(48.0, 16.0, 49.0, 16.0) == pytest.approx(111_195, rel=1e-3)
    assert distance_m(LAT, LON, LAT, LON) == 0


class TestClustering:
    def test_nearby_nodes_share_a_cluster_and_far_ones_do_not(self):
        df = frame(
            {"highway": "crossing"},
            {"highway": "crossing", "lat": LAT + NORTH_10M},
            {"highway": "crossing", "lat": LAT + 100 * NORTH_10M},  # 1 km away
        )
        for method in (greedy_cluster_ids, dbscan_cluster_ids):
            labels = method(df)
            assert labels[0] == labels[1]
            assert labels[0] != labels[2]

    def test_dbscan_chains_but_greedy_does_not(self):
        # Five nodes in a line, 15 m apart (60 m end to end). Each one is within 20 m of the next.
        df = frame(*[{"highway": "crossing", "lat": LAT + 1.5 * k * NORTH_10M} for k in range(5)])
        assert len(set(dbscan_cluster_ids(df))) == 1  # one 60 m long "crossing"
        assert len(set(greedy_cluster_ids(df))) > 1  # centres stay within 20 m


class TestGroup:
    def test_best_known_value_wins(self):
        df = classify(
            frame(
                {"highway": "traffic_signals"},
                {"highway": "crossing", "lat": LAT + NORTH_10M, "traffic_signals:sound": "yes"},
            )
        )
        crossing = group(df).iloc[0]
        assert crossing["nodes"] == 2
        assert crossing["sound"] == "yes"  # known beats unknown
        assert crossing["kind"] == "signals"

    def test_raised_kerb_wins_over_lowered(self):
        df = classify(
            frame(
                {"highway": "crossing", "kerb": "lowered"},
                {"highway": "crossing", "kerb": "raised", "lat": LAT + NORTH_10M},
            )
        )
        assert group(df).iloc[0]["kerb"] == "raised"

    def test_centre_is_mean_position(self):
        df = classify(frame({"highway": "crossing"}, {"highway": "crossing", "lat": LAT + NORTH_10M}))
        assert group(df).iloc[0]["lat"] == pytest.approx(LAT + NORTH_10M / 2)

    def test_empty_input(self):
        assert group(classify(frame())).empty


def test_clean_runs_the_whole_pipeline():
    df = frame(
        {"highway": "crossing", "crossing": "zebra"},
        {"highway": "crossing", "crossing": "zebra", "lat": LAT + NORTH_10M},
        {"highway": "crossing", "level": "-1"},
    )
    result = clean(df)
    assert len(result) == 1
    assert result.iloc[0]["kind"] == "zebra"
    assert not math.isnan(result.iloc[0]["lat"])
