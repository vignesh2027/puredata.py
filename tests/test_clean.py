"""Tests for AutoClean engine."""

import numpy as np
import pandas as pd
import pytest

import puredata
from puredata.core.clean import AutoClean, AutoCleanConfig
from puredata.core.report import FixAction


class TestAutoCleanBasic:
    def test_returns_dataframe_and_report(self, dirty_df):
        clean_df, report = puredata.clean(dirty_df)
        assert isinstance(clean_df, pd.DataFrame)
        assert len(clean_df) <= len(dirty_df)

    def test_empty_dataframe(self, empty_df):
        clean_df, report = puredata.clean(empty_df)
        assert clean_df.empty
        assert report.mend_score == 100.0

    def test_single_column(self, single_col_df):
        clean_df, report = puredata.clean(single_col_df)
        assert clean_df["a"].isna().sum() == 0

    def test_all_null_column(self, all_null_col_df):
        clean_df, report = puredata.clean(all_null_col_df)
        assert isinstance(clean_df, pd.DataFrame)

    def test_all_duplicates(self, all_duplicate_df):
        clean_df, report = puredata.clean(all_duplicate_df)
        assert len(clean_df) == 1
        actions = {f.action for f in report.fixes}
        assert FixAction.DUPLICATE_REMOVE in actions

    def test_no_modifications_on_clean_data(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": ["x", "y", "z", "x", "y"],
        })
        clean_df, report = puredata.clean(df)
        assert len(report.fixes) == 0 or all(
            f.action != FixAction.NULL_IMPUTE for f in report.fixes
        )

    def test_target_col_protected(self, dirty_df):
        """Target column nulls should not be imputed."""
        dirty_df.loc[0, "label"] = np.nan
        clean_df, report = puredata.clean(dirty_df, target_col="label")
        assert clean_df["label"].isna().sum() >= 1

    def test_numpy_array_input(self):
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        clean_df, report = puredata.clean(arr)
        assert isinstance(clean_df, pd.DataFrame)
        assert clean_df.isna().sum().sum() == 0

    def test_file_path_csv(self, tmp_path, dirty_df):
        p = tmp_path / "dirty.csv"
        dirty_df.to_csv(p, index=False)
        clean_df, report = puredata.clean(p)
        assert isinstance(clean_df, pd.DataFrame)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="puredata"):
            puredata.clean("/nonexistent/path/to/file.csv")

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="puredata"):
            puredata.clean({"a": [1, 2, 3]})


class TestNullImputation:
    def test_numeric_nulls_filled(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0] * 10})
        clean_df, report = puredata.clean(df, config=AutoCleanConfig(fix_outliers=False))
        assert clean_df["x"].isna().sum() == 0

    def test_categorical_nulls_filled_with_mode(self):
        df = pd.DataFrame({"cat": ["a", "a", "b", np.nan, "a"] * 10})
        clean_df, report = puredata.clean(df, config=AutoCleanConfig(fix_outliers=False))
        assert clean_df["cat"].isna().sum() == 0
        assert (clean_df["cat"] == "a").any()

    def test_high_null_rate_categorical_uses_unknown(self):
        # Use distinct rows so duplicate removal doesn't collapse the dataset
        rng = np.random.default_rng(0)
        n = 100
        cats = [np.nan if rng.random() < 0.7 else f"x_{i}" for i in range(n)]
        df = pd.DataFrame({"cat": cats, "idx": range(n)})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_outliers=False, fix_categories=False)
        )
        assert (clean_df["cat"] == "__unknown__").any()

    def test_null_imputation_report_logged(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0] * 10})
        _, report = puredata.clean(df, config=AutoCleanConfig(fix_outliers=False))
        actions = [f.action for f in report.fixes]
        assert FixAction.NULL_IMPUTE in actions


class TestOutlierDetection:
    def _make_outlier_df(self) -> pd.DataFrame:
        """100 distinct near-10 values + 5 extreme outliers, no duplicates."""
        rng = np.random.default_rng(7)
        baseline = 10.0 + rng.uniform(-0.5, 0.5, 95)
        outliers = np.array([99999.0, -99999.0, 99998.0, -99998.0, 99997.0])
        return pd.DataFrame({"val": np.concatenate([baseline, outliers])})

    def test_extreme_outliers_clipped(self):
        df = self._make_outlier_df()
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_duplicates=False, outlier_action="clip")
        )
        actions = {f.action for f in report.fixes}
        assert FixAction.OUTLIER in actions

    def test_outlier_action_remove(self):
        df = self._make_outlier_df()
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_duplicates=False, outlier_action="remove")
        )
        assert len(clean_df) < len(df)

    def test_outlier_action_nan(self):
        df = self._make_outlier_df()
        clean_df, report = puredata.clean(
            df,
            config=AutoCleanConfig(
                fix_nulls=False,
                fix_duplicates=False,
                outlier_action="nan",
            ),
        )
        assert clean_df["val"].isna().any()

    def test_no_false_positives_on_normal_data(self):
        np.random.seed(1)
        df = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_categories=False)
        )
        outlier_fixes = [f for f in report.fixes if f.action == FixAction.OUTLIER]
        # Should not aggressively flag values in a normal distribution
        if outlier_fixes:
            assert len(outlier_fixes[0].rows) < 30  # less than 6%


class TestTypeCoercion:
    def test_numeric_strings_converted(self):
        df = pd.DataFrame({"amount": ["1.5", "2.5", "3.0", "4.0", "5.5"] * 20})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False)
        )
        assert pd.api.types.is_numeric_dtype(clean_df["amount"])


class TestCategoryNormalisation:
    def test_gender_normalised(self):
        df = pd.DataFrame({
            "gender": ["Male", "male", "M", "MALE", "Female", "female", "F", "FEMALE"] * 20
        })
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False)
        )
        assert clean_df["gender"].nunique() <= 2
        actions = {f.action for f in report.fixes}
        assert FixAction.CATEGORY_NORMALISE in actions

    def test_high_cardinality_not_mangled(self):
        df = pd.DataFrame({"id": [f"user_{i}" for i in range(200)]})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False)
        )
        assert clean_df["id"].nunique() == 200


class TestWhitespaceFixing:
    def test_leading_trailing_stripped(self):
        df = pd.DataFrame({"name": ["  Alice  ", "  Bob", "Carol   "] * 10})
        clean_df, _ = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False, fix_categories=False)
        )
        assert not any(clean_df["name"].str.startswith(" ").dropna())
        assert not any(clean_df["name"].str.endswith(" ").dropna())

    def test_double_spaces_collapsed(self):
        df = pd.DataFrame({"x": ["hello  world", "foo   bar"] * 10})
        clean_df, _ = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False, fix_categories=False)
        )
        assert not any(clean_df["x"].str.contains("  ").dropna())


class TestDuplicateRemoval:
    def test_exact_duplicates_removed(self):
        df = pd.DataFrame({"a": [1, 2, 1, 3, 2], "b": ["x", "y", "x", "z", "y"]})
        clean_df, report = puredata.clean(
            df, config=AutoCleanConfig(fix_nulls=False, fix_outliers=False, fix_categories=False)
        )
        assert len(clean_df) == 3
        assert FixAction.DUPLICATE_REMOVE in {f.action for f in report.fixes}


class TestReport:
    def test_mend_score_100_clean_data(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        _, report = puredata.clean(
            df,
            config=AutoCleanConfig(
                fix_nulls=False, fix_outliers=False, fix_categories=False,
                fix_whitespace=False, fix_duplicates=False, fix_encoding=False,
                fix_types=False, fix_dates=False, fix_units=False,
            ),
        )
        assert report.mend_score == 100.0

    def test_report_to_json(self, dirty_df):
        _, report = puredata.clean(dirty_df)
        json_str = report.to_json()
        import json
        data = json.loads(json_str)
        assert "mend_score" in data
        assert "fixes" in data

    def test_report_to_csv(self, dirty_df, tmp_path):
        _, report = puredata.clean(dirty_df)
        csv_df = report.to_csv(tmp_path / "report.csv")
        assert isinstance(csv_df, pd.DataFrame)

    def test_report_to_html(self, dirty_df):
        _, report = puredata.clean(dirty_df)
        html = report.to_html()
        assert "<html" in html
        assert "puredata" in html

    def test_summary_string(self, dirty_df):
        _, report = puredata.clean(dirty_df)
        summary = report.summary()
        assert "MendScore" in summary

    def test_report_save_and_load_json(self, dirty_df, tmp_path):
        _, report = puredata.clean(dirty_df)
        path = tmp_path / "report.json"
        report.to_json(path)
        assert path.exists()

    def test_duration_recorded(self, dirty_df):
        _, report = puredata.clean(dirty_df)
        assert report.duration_seconds > 0


class TestConfig:
    def test_disable_all_fixes(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", " y", "z"]})
        config = AutoCleanConfig(
            fix_nulls=False, fix_outliers=False, fix_types=False,
            fix_duplicates=False, fix_encoding=False, fix_categories=False,
            fix_dates=False, fix_whitespace=False, fix_units=False,
        )
        _, report = puredata.clean(df, config=config)
        assert len(report.fixes) == 0


class TestScoreFunction:
    def test_score_returns_float(self, dirty_df):
        s = puredata.score(dirty_df)
        assert isinstance(s, float)
        assert 0 <= s <= 100

    def test_score_clean_data_is_100(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        s = puredata.score(
            df,
            config=AutoCleanConfig(
                fix_nulls=False, fix_outliers=False, fix_types=False,
                fix_duplicates=False, fix_encoding=False, fix_categories=False,
                fix_dates=False, fix_whitespace=False, fix_units=False,
            ),
        )
        assert s == 100.0
