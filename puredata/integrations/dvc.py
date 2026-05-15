"""DVC integration for puredata data quality tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union


def log_clean_report(
    report: Any,
    metrics_path: Union[str, Path] = "puredata_metrics.json",
) -> None:
    """Write AutoClean metrics to a DVC-compatible JSON metrics file.

    Parameters
    ----------
    report:
        A :class:`~puredata.core.report.CleanReport`.
    metrics_path:
        Path to write the JSON metrics file (add to ``dvc.yaml`` metrics).

    Examples
    --------
    >>> from puredata.integrations.dvc import log_clean_report
    >>> clean_df, report = puredata.clean(df)
    >>> log_clean_report(report, "metrics/clean.json")
    """
    metrics = {
        "mend_score": report.mend_score,
        "n_fixes": len(report.fixes),
        "original_rows": report.original_shape[0],
        "cleaned_rows": report.cleaned_shape[0],
        "duration_seconds": report.duration_seconds,
        "fixes_by_type": {},
    }
    for fix in report.fixes:
        key = fix.action.value
        metrics["fixes_by_type"][key] = metrics["fixes_by_type"].get(key, 0) + len(fix.rows)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def log_watch_report(
    report: Any,
    metrics_path: Union[str, Path] = "puredata_watch_metrics.json",
) -> None:
    """Write DataWatch metrics to a DVC-compatible JSON metrics file.

    Parameters
    ----------
    report:
        A :class:`~puredata.core.report.WatchReport`.
    metrics_path:
        Path to write the JSON metrics file.
    """
    metrics = {
        "compatibility_score": report.compatibility_score,
        "n_passed": report.n_passed,
        "n_warned": report.n_warned,
        "n_failed": report.n_failed,
        "checks": {},
    }
    for chk in report.checks:
        metrics["checks"][chk.name.replace(".", "_")] = chk.status.value
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
