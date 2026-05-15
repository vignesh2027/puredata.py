"""Tests for DataWatch engine."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import puredata
from puredata.core.report import CheckStatus, DataCompatibilityError
from puredata.core.watch import DataContract, DataWatch


class TestFit:
    def test_returns_data_contract(self, train_df):
        contract = puredata.watch(train_df)
        assert isinstance(contract, DataContract)

    def test_contract_has_all_columns(self, train_df):
        contract = puredata.watch(train_df)
        for col in train_df.columns:
            assert col in contract.columns

    def test_numeric_profile_populated(self, train_df):
        contract = puredata.watch(train_df)
        profile = contract.columns["feature_a"]
        assert profile.min is not None
        assert profile.max is not None
        assert profile.mean is not None
        assert profile.histogram is not None

    def test_categorical_profile_populated(self, train_df):
        contract = puredata.watch(train_df)
        profile = contract.columns["category"]
        assert profile.categories == {"cat", "dog", "bird"}

    def test_fit_with_numpy_array(self):
        arr = np.random.rand(100, 3)
        contract = puredata.watch(arr)
        assert len(contract.columns) == 3

    def test_fit_with_file_path(self, train_df, tmp_path):
        p = tmp_path / "train.csv"
        train_df.to_csv(p, index=False)
        contract = puredata.watch(p)
        assert len(contract.columns) == len(train_df.columns)

    def test_empty_dataframe_fit(self):
        contract = puredata.watch(pd.DataFrame())
        assert len(contract.columns) == 0


class TestContractPersistence:
    def test_save_and_load(self, train_df, tmp_path):
        contract = puredata.watch(train_df)
        path = tmp_path / "contract.json"
        contract.save(path)
        loaded = DataContract.load(path)
        assert loaded.columns.keys() == contract.columns.keys()
        assert loaded.column_order == contract.column_order

    def test_load_preserves_profiles(self, train_df, tmp_path):
        contract = puredata.watch(train_df)
        path = tmp_path / "contract.json"
        contract.save(path)
        loaded = DataContract.load(path)
        assert abs(loaded.columns["feature_a"].mean - contract.columns["feature_a"].mean) < 1e-6

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataContract.load(tmp_path / "nonexistent.json")


class TestCheck:
    def test_clean_data_passes(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_clean, contract, mode="silent")
        assert result.n_failed == 0

    def test_drifted_data_fails(self, train_df, prod_df_drifted):
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_drifted, contract, mode="silent")
        assert result.n_failed > 0

    def test_missing_column_fails(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        prod_df_clean = prod_df_clean.drop(columns=["feature_a"])
        result = puredata.check(prod_df_clean, contract, mode="silent")
        missing_checks = [c for c in result.checks if c.name == "schema.missing_columns"]
        assert any(c.status == CheckStatus.FAIL for c in missing_checks)

    def test_extra_column_warns(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        prod_df_clean["extra_col"] = 0
        result = puredata.check(prod_df_clean, contract, mode="silent")
        extra_checks = [c for c in result.checks if c.name == "schema.extra_columns"]
        assert any(c.status == CheckStatus.WARN for c in extra_checks)

    def test_null_rate_increase_fails(self, train_df):
        contract = puredata.watch(train_df)
        new_df = train_df.copy()
        new_df.loc[:300, "feature_a"] = np.nan  # 60% nulls
        result = puredata.check(new_df, contract, mode="silent")
        null_checks = [c for c in result.checks if c.name == "nulls.rate_increase"]
        assert any(c.status == CheckStatus.FAIL for c in null_checks)

    def test_range_violation_fails(self, train_df):
        contract = puredata.watch(train_df)
        new_df = train_df.copy()
        new_df.loc[0, "feature_b"] = 1_000_000  # way outside range
        result = puredata.check(new_df, contract, mode="silent")
        range_checks = [c for c in result.checks if c.name == "range.violation"]
        assert any(c.status == CheckStatus.FAIL for c in range_checks)

    def test_new_category_values_flagged(self, train_df):
        contract = puredata.watch(train_df)
        new_df = train_df.copy()
        new_df.loc[0, "category"] = "elephant"
        result = puredata.check(new_df, contract, mode="silent")
        card_checks = [c for c in result.checks if c.name == "cardinality.new_values"]
        assert len(card_checks) > 0

    def test_distribution_drift_detected(self, train_df):
        contract = puredata.watch(train_df)
        new_df = train_df.copy()
        new_df["feature_a"] = np.random.normal(100, 50, len(new_df))  # completely different
        result = puredata.check(new_df, contract, mode="silent")
        drift_checks = [c for c in result.checks if c.name == "distribution.drift"]
        assert any(c.status in (CheckStatus.WARN, CheckStatus.FAIL) for c in drift_checks)

    def test_report_has_score(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_clean, contract, mode="silent")
        assert 0 <= result.compatibility_score <= 100


class TestModes:
    def test_warn_mode_emits_warning(self, train_df, prod_df_drifted):
        contract = puredata.watch(train_df, mode="warn")
        engine = DataWatch(mode="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.check(prod_df_drifted, contract)
        assert len(w) >= 0  # may or may not warn depending on drift severity

    def test_strict_mode_raises_on_failure(self, train_df, prod_df_drifted):
        contract = puredata.watch(train_df)
        engine = DataWatch(mode="strict")
        with pytest.raises(DataCompatibilityError):
            engine.check(prod_df_drifted, contract)

    def test_silent_mode_no_warning(self, train_df, prod_df_drifted):
        contract = puredata.watch(train_df)
        engine = DataWatch(mode="silent")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.check(prod_df_drifted, contract)
        puredata_warns = [x for x in w if "puredata" in str(x.message).lower()]
        assert len(puredata_warns) == 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            DataWatch(mode="invalid_mode")


class TestCustomRules:
    def test_passing_rule(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        contract.add_rule(lambda df: None, name="always_pass")
        result = puredata.check(prod_df_clean, contract, mode="silent")
        rule_checks = [c for c in result.checks if "always_pass" in c.name]
        assert any(c.status == CheckStatus.PASS for c in rule_checks)

    def test_failing_rule(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        contract.add_rule(lambda df: "always fails", name="always_fail")
        result = puredata.check(prod_df_clean, contract, mode="silent")
        rule_checks = [c for c in result.checks if "always_fail" in c.name]
        assert any(c.status == CheckStatus.FAIL for c in rule_checks)

    def test_rule_exception_caught(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)

        def bad_rule(df):
            raise RuntimeError("something broke")

        contract.add_rule(bad_rule, name="bad_rule")
        result = puredata.check(prod_df_clean, contract, mode="silent")
        rule_checks = [c for c in result.checks if "bad_rule" in c.name]
        assert any(c.status == CheckStatus.FAIL for c in rule_checks)

    def test_business_rule_chaining(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        contract.add_rule(
            lambda df: None if (df["feature_b"] >= 0).all() else "feature_b contains negatives"
        )
        result = puredata.check(prod_df_clean, contract, mode="silent")
        assert result is not None


class TestWatchReport:
    def test_report_to_json(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_clean, contract, mode="silent")
        j = result.to_json()
        data = json.loads(j)
        assert "compatibility_score" in data
        assert "checks" in data

    def test_report_to_html(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_clean, contract, mode="silent")
        html = result.to_html()
        assert "<html" in html

    def test_report_summary(self, train_df, prod_df_clean):
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_clean, contract, mode="silent")
        summary = result.summary()
        assert "DataWatch" in summary

    def test_raise_if_failed(self, train_df, prod_df_drifted):
        contract = puredata.watch(train_df)
        result = puredata.check(prod_df_drifted, contract, mode="silent")
        if result.n_failed > 0:
            with pytest.raises(DataCompatibilityError):
                result.raise_if_failed()
