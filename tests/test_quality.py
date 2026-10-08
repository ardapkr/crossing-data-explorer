import pandas as pd
import pytest
import shapely

from crossings.districts import assign_districts
from crossings.quality import by_district, summary, tag_completeness


def crossings(*rows: tuple) -> pd.DataFrame:
    """rows of (kind, sound, kerb, tactile)"""
    return pd.DataFrame(rows, columns=["kind", "sound", "kerb", "tactile"])


def test_summary_percentages():
    df = crossings(
        ("signals", "yes", "lowered", "yes"),
        ("signals", None, None, "no"),
        ("zebra", None, None, None),
        ("unmarked", None, "raised", None),
    )
    s = summary(df)
    assert s["crossings"] == 4
    assert s["signals_pct"] == 50.0
    assert s["unmarked_pct"] == 25.0
    assert s["sound_yes_pct_of_signals"] == 50.0  # 1 of the 2 signals
    assert s["sound_missing_pct_of_signals"] == 50.0
    assert s["kerb_missing_pct"] == 50.0


def test_summary_without_signals_has_no_sound_percentage():
    s = summary(crossings(("zebra", None, None, None)))
    assert pd.isna(s["sound_yes_pct_of_signals"])


def test_assign_and_group_by_district():
    # Two square "districts" side by side, and three crossings
    districts = pd.DataFrame(
        {
            "district": [1, 2],
            "name": ["West", "East"],
            "geometry": [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
        }
    )
    df = crossings(("signals", "yes", None, None), ("zebra", None, None, None), ("unmarked", None, None, None))
    df["lat"] = [0.5, 0.5, 5.0]
    df["lon"] = [0.5, 1.5, 5.0]  # the last one is outside both

    located = assign_districts(df, districts)
    assert list(located["district_name"].fillna("-")) == ["West", "East", "-"]

    table = by_district(located)
    assert list(table.index.get_level_values("name")) == ["West", "East"]
    assert table.loc[(1, "West"), "signals_pct"] == 100.0


def test_tag_completeness_is_sorted():
    raw = pd.DataFrame({"kerb": [None, "lowered"], "crossing": ["zebra", "zebra"]})
    result = tag_completeness(raw, ["kerb", "crossing"])
    assert list(result.index) == ["crossing", "kerb"]
    assert result["kerb"] == pytest.approx(50.0)
