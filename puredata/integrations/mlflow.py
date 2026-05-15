"""MLflow integration for puredata data quality tracking."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def log_clean_report(report: Any, run_id: Optional[str] = None) -> None:
    """Log an AutoClean report as MLflow metrics and artifacts.

    Parameters
    ----------
    report:
        A :class:`~puredata.core.report.CleanReport` from :func:`puredata.clean`.
    run_id:
        MLflow run ID. Uses the active run if ``None``.

    Raises
    ------
    ImportError
        If ``mlflow`` is not installed.

    Examples
    --------
    >>> import mlflow, puredata
    >>> from puredata.integrations.mlflow import log_clean_report
    >>> with mlflow.start_run():
    ...     clean_df, report = puredata.clean(df)
    ...     log_clean_report(report)
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "puredata: install mlflow to use this integration — pip install mlflow"
        ) from exc

    ctx = mlflow.start_run(run_id=run_id) if run_id else _noop_context()
    with ctx:
        mlflow.log_metric("puredata.mend_score", report.mend_score)
        mlflow.log_metric("puredata.n_fixes", len(report.fixes))
        mlflow.log_metric("puredata.original_rows", report.original_shape[0])
        mlflow.log_metric("puredata.cleaned_rows", report.cleaned_shape[0])
        mlflow.log_metric("puredata.duration_seconds", report.duration_seconds)
        for i, fix in enumerate(report.fixes):
            mlflow.log_metric(f"puredata.fix.{fix.action.value}.count", len(fix.rows))
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, prefix="puredata_clean_"
        ) as f:
            f.write(report.to_json().encode())
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="puredata")
        os.unlink(tmp_path)


def log_watch_report(report: Any, run_id: Optional[str] = None) -> None:
    """Log a DataWatch compatibility report as MLflow metrics and artifacts.

    Parameters
    ----------
    report:
        A :class:`~puredata.core.report.WatchReport` from :func:`puredata.check`.
    run_id:
        MLflow run ID. Uses the active run if ``None``.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "puredata: install mlflow to use this integration — pip install mlflow"
        ) from exc

    ctx = mlflow.start_run(run_id=run_id) if run_id else _noop_context()
    with ctx:
        mlflow.log_metric("puredata.compatibility_score", report.compatibility_score)
        mlflow.log_metric("puredata.checks_passed", report.n_passed)
        mlflow.log_metric("puredata.checks_warned", report.n_warned)
        mlflow.log_metric("puredata.checks_failed", report.n_failed)
        for chk in report.checks:
            safe_name = chk.name.replace(".", "_").replace(" ", "_")
            status_val = {"pass": 1, "warn": 0.5, "fail": 0, "skip": -1}.get(
                chk.status.value, -1
            )
            mlflow.log_metric(f"puredata.check.{safe_name}", status_val)
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, prefix="puredata_watch_"
        ) as f:
            f.write(report.to_json().encode())
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="puredata")
        os.unlink(tmp_path)


class _noop_context:
    def __enter__(self): return self
    def __exit__(self, *a): pass
