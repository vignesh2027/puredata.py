"""Tests for the plugin system."""

import pandas as pd
import pytest

from puredata.plugins.base import (
    CleanerPlugin,
    DriftDetectorPlugin,
    PluginRegistry,
    ValidatorPlugin,
    register_cleaner,
    register_drift_detector,
    register_validator,
)
from puredata.core.report import CleanReport, CheckResult, CheckStatus, WatchReport


class TestPluginRegistry:
    def setup_method(self):
        self.registry = PluginRegistry()

    def test_register_cleaner(self):
        class MyClean(CleanerPlugin):
            name = "test_clean_1"
            def clean(self, df, report):
                return df, report

        self.registry.register_cleaner(MyClean)
        assert "test_clean_1" in self.registry.cleaners

    def test_register_duplicate_raises(self):
        class MyClean(CleanerPlugin):
            name = "test_dup"
            def clean(self, df, report):
                return df, report

        self.registry.register_cleaner(MyClean)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register_cleaner(MyClean)

    def test_register_empty_name_raises(self):
        class BadClean(CleanerPlugin):
            name = ""
            def clean(self, df, report):
                return df, report

        with pytest.raises(ValueError, match="name must not be empty"):
            self.registry.register_cleaner(BadClean)

    def test_register_validator(self):
        class MyValidator(ValidatorPlugin):
            name = "test_validator"
            def validate(self, df, contract, report):
                return report

        self.registry.register_validator(MyValidator)
        assert "test_validator" in self.registry.validators

    def test_register_drift_detector(self):
        class MyDrift(DriftDetectorPlugin):
            name = "test_drift"
            def detect(self, reference, new_data):
                return 0.0

        self.registry.register_drift_detector(MyDrift)
        assert "test_drift" in self.registry.drift_detectors

    def test_list_all(self):
        class ListClean(CleanerPlugin):
            name = "list_test_clean"
            def clean(self, df, report):
                return df, report

        self.registry.register_cleaner(ListClean)
        plugins = self.registry.list_all()
        assert any(p["name"] == "list_test_clean" for p in plugins)

    def test_repr(self):
        r = repr(self.registry)
        assert "PluginRegistry" in r

    def test_cleaner_plugin_executes(self):
        class UpperCleaner(CleanerPlugin):
            name = "upper_cleaner"
            description = "Uppercases all string columns"
            def clean(self, df, report):
                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].str.upper()
                return df, report

        df = pd.DataFrame({"name": ["alice", "bob"]})
        report = CleanReport()
        cleaner = UpperCleaner()
        result_df, _ = cleaner.clean(df, report)
        assert list(result_df["name"]) == ["ALICE", "BOB"]
