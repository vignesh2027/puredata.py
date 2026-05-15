"""Weights & Biases integration for puredata data quality tracking."""

from __future__ import annotations

from typing import Any, Optional


def log_clean_report(report: Any, run: Optional[Any] = None) -> None:
    """Log an AutoClean report to an active W&B run.

    Parameters
    ----------
    report:
        A :class:`~puredata.core.report.CleanReport`.
    run:
        A ``wandb.Run`` object. Uses ``wandb.run`` if ``None``.

    Examples
    --------
    >>> import wandb, puredata
    >>> from puredata.integrations.wandb import log_clean_report
    >>> wandb.init(project="myproject")
    >>> clean_df, report = puredata.clean(df)
    >>> log_clean_report(report)
    """
    try:
        import wandb as _wandb
    except ImportError as exc:
        raise ImportError(
            "puredata: install wandb to use this integration — pip install wandb"
        ) from exc

    active_run = run or _wandb.run
    if active_run is None:
        raise RuntimeError(
            "puredata: no active W&B run. Call wandb.init() first."
        )
    payload: dict = {
        "puredata/mend_score": report.mend_score,
        "puredata/n_fixes": len(report.fixes),
        "puredata/original_rows": report.original_shape[0],
        "puredata/cleaned_rows": report.cleaned_shape[0],
        "puredata/duration_seconds": report.duration_seconds,
    }
    for fix in report.fixes:
        key = f"puredata/fixes/{fix.action.value}"
        payload[key] = payload.get(key, 0) + len(fix.rows)
    active_run.log(payload)

    import json, tempfile, os
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, prefix="puredata_clean_", mode="w"
    ) as f:
        f.write(report.to_json())
        tmp_path = f.name
    active_run.save(tmp_path)
    os.unlink(tmp_path)


def log_watch_report(report: Any, run: Optional[Any] = None) -> None:
    """Log a DataWatch compatibility report to an active W&B run."""
    try:
        import wandb as _wandb
    except ImportError as exc:
        raise ImportError(
            "puredata: install wandb — pip install wandb"
        ) from exc

    active_run = run or _wandb.run
    if active_run is None:
        raise RuntimeError("puredata: no active W&B run. Call wandb.init() first.")

    payload: dict = {
        "puredata/compatibility_score": report.compatibility_score,
        "puredata/checks_passed": report.n_passed,
        "puredata/checks_warned": report.n_warned,
        "puredata/checks_failed": report.n_failed,
    }
    for chk in report.checks:
        safe_name = chk.name.replace(".", "/")
        status_val = {"pass": 1, "warn": 0.5, "fail": 0}.get(chk.status.value, -1)
        payload[f"puredata/check/{safe_name}"] = status_val
    active_run.log(payload)
