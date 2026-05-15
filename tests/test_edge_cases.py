"""Edge-case and coverage-completion tests."""

import numpy as np
import pandas as pd
import pytest

import puredata
from puredata.core.clean import AutoCleanConfig
from puredata.core.report import FixAction
from puredata.plugins.base import PluginRegistry


class TestUnitNormalisation:
    def test_weight_units_normalised(self):
        df = pd.DataFrame({
            "weight": ["70kg", "154lbs", "80kg", "176lbs", "65kg"] * 4
        })
        clean_df, report = puredata.clean(
            df,
            config=AutoCleanConfig(
                fix_nulls=False, fix_outliers=False, fix_categories=False,
                fix_dates=False, fix_whitespace=False, fix_duplicates=False,
            ),
        )
        unit_fixes = [f for f in report.fixes if f.action == FixAction.UNIT_NORMALISE]
        assert len(unit_fixes) > 0

    def test_distance_units_normalised(self):
        df = pd.DataFrame({
            "distance": ["5km", "3mi", "10km", "2mi", "7km"] * 4
        })
        clean_df, report = puredata.clean(
            df,
            config=AutoCleanConfig(
                fix_nulls=False, fix_outliers=False, fix_categories=False,
                fix_dates=False, fix_whitespace=False, fix_duplicates=False,
            ),
        )
        unit_fixes = [f for f in report.fixes if f.action == FixAction.UNIT_NORMALISE]
        assert len(unit_fixes) > 0
        assert pd.api.types.is_numeric_dtype(clean_df["distance"])

    def test_single_unit_column_not_changed(self):
        df = pd.DataFrame({"dist": ["5km", "10km", "15km", "20km", "25km"] * 4})
        clean_df, report = puredata.clean(
            df,
            config=AutoCleanConfig(
                fix_nulls=False, fix_outliers=False, fix_categories=False,
                fix_dates=False, fix_whitespace=False, fix_duplicates=False,
            ),
        )
        unit_fixes = [f for f in report.fixes if f.action == FixAction.UNIT_NORMALISE]
        assert len(unit_fixes) == 0  # only one unit type — no normalisation needed


class TestDatetimeImputation:
    def test_datetime_col_nulls_filled(self):
        dates = pd.to_datetime(["2023-01-01", None, "2023-01-03", None, "2023-01-05"])
        df = pd.DataFrame({"ts": dates, "val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_outliers=False)
        )
        assert clean_df["ts"].isna().sum() == 0

    def test_datetime_nulls_fix_logged(self):
        dates = pd.to_datetime(["2023-01-01", None, "2023-01-03"])
        df = pd.DataFrame({"ts": dates, "val": [1.0, 2.0, 3.0]})
        _, report = puredata.clean(df, config=AutoCleanConfig(fix_outliers=False))
        actions = {f.action for f in report.fixes}
        assert FixAction.NULL_IMPUTE in actions


class TestPluginDiscover:
    def test_discover_runs_without_error(self):
        reg = PluginRegistry()
        reg.discover()  # should not raise even with no plugins installed
        assert isinstance(reg, PluginRegistry)

    def test_repr_shows_counts(self):
        reg = PluginRegistry()
        r = repr(reg)
        assert "cleaners=0" in r
        assert "validators=0" in r


class TestReportEdgeCases:
    def test_clean_report_no_fixes_summary(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        _, report = puredata.clean(
            df,
            config=AutoCleanConfig(
                fix_nulls=False, fix_outliers=False, fix_types=False,
                fix_duplicates=False, fix_encoding=False, fix_categories=False,
                fix_dates=False, fix_whitespace=False, fix_units=False,
            ),
        )
        summary = report.summary()
        assert "No issues detected" in summary

    def test_watch_report_to_json_path(self, tmp_path):
        import puredata as pd_lib
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        contract = pd_lib.watch(df)
        result = pd_lib.check(df, contract, mode="silent")
        path = tmp_path / "watch.json"
        result.to_json(path)
        assert path.exists()

    def test_watch_report_to_html_path(self, tmp_path):
        import puredata as pd_lib
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        contract = pd_lib.watch(df)
        result = pd_lib.check(df, contract, mode="silent")
        path = tmp_path / "watch.html"
        html = result.to_html(path)
        assert path.exists()
        assert "<html" in html

    def test_compatibility_score_100_same_data(self):
        df = pd.DataFrame({"a": np.random.normal(0, 1, 100)})
        contract = puredata.watch(df)
        result = puredata.check(df, contract, mode="silent")
        assert result.compatibility_score > 80  # same data, should be high

    def test_passed_property(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        contract = puredata.watch(df)
        result = puredata.check(df, contract, mode="silent")
        assert result.passed is True


class TestEncodingFixes:
    def test_bom_stripped(self):
        df = pd.DataFrame({"name": ["﻿Alice", "Bob", "﻿Carol"]})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False, fix_categories=False)
        )
        assert not any(clean_df["name"].str.startswith("﻿").dropna())

    def test_zero_width_space_removed(self):
        df = pd.DataFrame({"val": ["hello​world", "normal", "foo​bar"]})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False, fix_categories=False)
        )
        assert not any(("​" in v) for v in clean_df["val"].dropna())
