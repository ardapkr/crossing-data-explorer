"""Checks on the bundled real OpenStreetMap snapshot (Vienna, 26 Sep 2026).

These pin the numbers quoted in the README, so a change to the cleaning rules that moves them shows up here.
"""

import pytest

from crossings.clean import classify, clean, group, usable
from crossings.districts import assign_districts, load_districts
from crossings.load import load_city


@pytest.fixture(scope="module")
def vienna_raw():
    return load_city("vienna")


def test_raw_and_usable_node_counts(vienna_raw):
    assert len(vienna_raw) == 21_468
    assert len(usable(vienna_raw)) == 21_169


def test_greedy_grouping_matches_the_crosswise_app(vienna_raw):
    # The app's JavaScript build script produced 8,987 crossings from the same download.
    assert len(clean(vienna_raw, "greedy")) == 8_987


def test_dbscan_merges_more(vienna_raw):
    nodes = classify(usable(vienna_raw))
    assert len(group(nodes, "dbscan")) < len(group(nodes, "greedy"))


def test_all_23_districts_get_crossings(vienna_raw):
    located = assign_districts(clean(vienna_raw), load_districts())
    assert located["district"].dropna().nunique() == 23
