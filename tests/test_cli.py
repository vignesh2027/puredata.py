"""Tests for the CLI (puredata clean / watch / check / score)."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from typer.testing import CliRunner

from puredata.cli import app

runner = CliRunner()


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({
        "age": [25.0, np.nan, 30.0, 35.0, 28.0],
        "gender": ["Male", "male", "M", "Female", "female"],
        "score": [80.0, 90.0, 75.0, 85.0, 95.0],
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def clean_csv(tmp_path):
    df = pd.DataFrame({
        "a": np.random.normal(0, 1, 50),
        "b": np.random.normal(5, 2, 50),
    })
    p = tmp_path / "clean.csv"
    df.to_csv(p, index=False)
    return p


class TestCleanCommand:
    def test_basic_clean(self, csv_file):
        result = runner.invoke(app, ["clean", str(csv_file)])
        assert result.exit_code == 0

    def test_clean_with_output(self, csv_file, tmp_path):
        out = tmp_path / "out.csv"
        result = runner.invoke(app, ["clean", str(csv_file), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_clean_with_json_report(self, csv_file, tmp_path):
        report = tmp_path / "report.json"
        result = runner.invoke(app, ["clean", str(csv_file), "--report-json", str(report)])
        assert result.exit_code == 0
        assert report.exists()
        data = json.loads(report.read_text())
        assert "mend_score" in data

    def test_clean_with_html_report(self, csv_file, tmp_path):
        report = tmp_path / "report.html"
        result = runner.invoke(app, ["clean", str(csv_file), "--report-html", str(report)])
        assert result.exit_code == 0
        assert report.exists()
        assert "<html" in report.read_text()

    def test_clean_with_csv_report(self, csv_file, tmp_path):
        report = tmp_path / "report.csv"
        result = runner.invoke(app, ["clean", str(csv_file), "--report-csv", str(report)])
        assert result.exit_code == 0
        assert report.exists()

    def test_clean_nonexistent_file(self):
        result = runner.invoke(app, ["clean", "/does/not/exist.csv"])
        assert result.exit_code == 1

    def test_clean_no_nulls_flag(self, csv_file):
        result = runner.invoke(app, ["clean", str(csv_file), "--no-nulls"])
        assert result.exit_code == 0

    def test_clean_with_target(self, csv_file):
        result = runner.invoke(app, ["clean", str(csv_file), "--target", "score"])
        assert result.exit_code == 0


class TestWatchCommand:
    def test_watch_creates_contract(self, clean_csv, tmp_path):
        contract = tmp_path / "contract.json"
        result = runner.invoke(app, ["watch", str(clean_csv), "--contract", str(contract)])
        assert result.exit_code == 0
        assert contract.exists()
        data = json.loads(contract.read_text())
        assert "columns" in data

    def test_watch_nonexistent(self):
        result = runner.invoke(app, ["watch", "/does/not/exist.csv"])
        assert result.exit_code == 1


class TestCheckCommand:
    def test_check_passes_on_same_data(self, clean_csv, tmp_path):
        contract = tmp_path / "contract.json"
        runner.invoke(app, ["watch", str(clean_csv), "--contract", str(contract)])
        result = runner.invoke(app, ["check", str(clean_csv), str(contract)])
        assert result.exit_code == 0

    def test_check_with_json_report(self, clean_csv, tmp_path):
        contract = tmp_path / "contract.json"
        report = tmp_path / "report.json"
        runner.invoke(app, ["watch", str(clean_csv), "--contract", str(contract)])
        result = runner.invoke(app, ["check", str(clean_csv), str(contract), "--report-json", str(report)])
        assert result.exit_code == 0
        assert report.exists()

    def test_check_nonexistent_data(self, clean_csv, tmp_path):
        contract = tmp_path / "contract.json"
        runner.invoke(app, ["watch", str(clean_csv), "--contract", str(contract)])
        result = runner.invoke(app, ["check", "/no/file.csv", str(contract)])
        assert result.exit_code == 1

    def test_check_nonexistent_contract(self, clean_csv, tmp_path):
        result = runner.invoke(app, ["check", str(clean_csv), str(tmp_path / "no.json")])
        assert result.exit_code == 1


class TestScoreCommand:
    def test_score_output(self, csv_file):
        result = runner.invoke(app, ["score", str(csv_file)])
        assert result.exit_code == 0
        assert "MendScore" in result.output or "score" in result.output.lower()

    def test_score_nonexistent(self):
        result = runner.invoke(app, ["score", "/no/file.csv"])
        assert result.exit_code == 1


class TestDashboardCommand:
    def test_dashboard_creates_file(self, csv_file, tmp_path):
        out = tmp_path / "dashboard.html"
        result = runner.invoke(app, ["dashboard", str(csv_file), "--no-browser", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "<html" in out.read_text()

    def test_dashboard_nonexistent(self):
        result = runner.invoke(app, ["dashboard", "/no/file.csv"])
        assert result.exit_code == 1
