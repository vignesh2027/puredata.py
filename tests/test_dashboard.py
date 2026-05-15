"""Tests for dashboard generation."""

import pandas as pd
import pytest

from puredata.dashboard import dashboard, _profile_dataframe, _compute_raw_score


class TestDashboard:
    def test_generates_html(self, tmp_path, train_df):
        path = dashboard(
            train_df,
            open_browser=False,
            output_path=tmp_path / "dashboard.html",
        )
        content = open(path).read()
        assert "<html" in content
        assert "puredata" in content

    def test_returns_path_string(self, tmp_path, train_df):
        path = dashboard(train_df, open_browser=False, output_path=tmp_path / "d.html")
        assert isinstance(path, str)

    def test_with_clean_report(self, tmp_path, dirty_df):
        import puredata
        clean_df, report = puredata.clean(dirty_df)
        path = dashboard(
            clean_df,
            clean_report=report,
            open_browser=False,
            output_path=tmp_path / "d.html",
        )
        assert "<html" in open(path).read()

    def test_with_watch_report(self, tmp_path, train_df, prod_df_clean):
        import puredata
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_clean, contract, mode="silent")
        path = dashboard(
            prod_df_clean,
            watch_report=result,
            open_browser=False,
            output_path=tmp_path / "d.html",
        )
        assert "<html" in open(path).read()

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            dashboard({"not": "a dataframe"}, open_browser=False)

    def test_profile_dataframe(self, train_df):
        profile = _profile_dataframe(train_df)
        assert "columns" in profile
        assert len(profile["columns"]) == len(train_df.columns)

    def test_compute_raw_score(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        s = _compute_raw_score(df)
        assert s == 100.0

    def test_raw_score_with_nulls(self):
        df = pd.DataFrame({"a": [None, None, 1.0]})
        s = _compute_raw_score(df)
        assert s < 100.0
