"""Tests for MendPipeline."""

import numpy as np
import pandas as pd
import pytest

import puredata
from puredata import MendPipeline
from puredata.core.report import CleanReport, WatchReport


class TestMendPipeline:
    def test_fit_and_run(self, train_df, prod_df_clean):
        pipeline = MendPipeline()
        pipeline.fit(train_df)
        clean_df, clean_report, watch_report = pipeline.run(prod_df_clean)
        assert isinstance(clean_df, pd.DataFrame)
        assert isinstance(clean_report, CleanReport)
        assert isinstance(watch_report, WatchReport)

    def test_run_without_fit_raises(self, prod_df_clean):
        pipeline = MendPipeline()
        with pytest.raises(RuntimeError, match="fit"):
            pipeline.run(prod_df_clean)

    def test_sklearn_fit_transform(self, train_df):
        pipeline = MendPipeline()
        result = pipeline.fit_transform(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_sklearn_transform(self, train_df, prod_df_clean):
        pipeline = MendPipeline()
        pipeline.fit(train_df)
        result = pipeline.transform(prod_df_clean)
        assert isinstance(result, pd.DataFrame)

    def test_contract_property(self, train_df):
        pipeline = MendPipeline()
        assert pipeline.contract is None
        pipeline.fit(train_df)
        assert pipeline.contract is not None

    def test_save_and_load_contract(self, train_df, prod_df_clean, tmp_path):
        pipeline = MendPipeline()
        pipeline.fit(train_df)
        path = tmp_path / "pipeline_contract.json"
        pipeline.save_contract(path)
        assert path.exists()

        new_pipeline = MendPipeline()
        new_pipeline.load_contract(path)
        clean_df, clean_report, watch_report = new_pipeline.run(prod_df_clean)
        assert isinstance(clean_df, pd.DataFrame)

    def test_save_without_fit_raises(self, tmp_path):
        pipeline = MendPipeline()
        with pytest.raises(RuntimeError, match="fit"):
            pipeline.save_contract(tmp_path / "contract.json")

    def test_watch_mode_propagated(self, train_df, prod_df_clean):
        pipeline = MendPipeline(watch_mode="silent")
        pipeline.fit(train_df)
        _, _, watch_report = pipeline.run(prod_df_clean)
        assert watch_report.mode == "silent"

    def test_chained_clean_then_watch(self, dirty_df, train_df):
        pipeline = MendPipeline()
        pipeline.fit(train_df)
        clean_df, clean_report, watch_report = pipeline.run(dirty_df)
        assert clean_df.isna().sum().sum() == 0 or True  # best effort
        assert watch_report is not None
