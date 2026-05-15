"""puredata unified top-level API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from puredata.core.clean import AutoClean, AutoCleanConfig, clean as _clean
from puredata.core.report import (
    CheckResult,
    CheckStatus,
    CleanReport,
    DataCompatibilityError,
    Fix,
    FixAction,
    WatchReport,
)
from puredata.core.watch import (
    DataContract,
    DataWatch,
    check as _check,
    watch as _watch,
)
from puredata.dashboard import dashboard as _dashboard
from puredata.pipeline import MendPipeline
from puredata.plugins.base import (
    CleanerPlugin,
    DriftDetectorPlugin,
    PluginRegistry,
    ValidatorPlugin,
    registry,
    register_cleaner,
    register_drift_detector,
    register_validator,
)


def clean(
    data: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    *,
    config: Optional[AutoCleanConfig] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, CleanReport]:
    """Clean *data* automatically. The main AutoClean entry point.

    Parameters
    ----------
    data:
        Any pandas or polars DataFrame, numpy array, or path to
        CSV / Excel / Parquet / JSON.
    config:
        Optional :class:`AutoCleanConfig` to override defaults.
    target_col:
        Column to protect from modification (ML label/target column).

    Returns
    -------
    tuple[pd.DataFrame, CleanReport]
        Cleaned DataFrame and full repair report.

    Examples
    --------
    >>> import puredata
    >>> clean_df, report = puredata.clean(dirty_df)
    >>> print(report.summary())
    """
    return _clean(data, config=config, target_col=target_col)


def watch(
    reference: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    *,
    mode: str = "warn",
    metadata: Optional[Dict[str, Any]] = None,
) -> DataContract:
    """Fit a :class:`DataContract` on reference/training data.

    Parameters
    ----------
    reference:
        Training DataFrame, array, or file path.
    mode:
        ``"warn"`` (default), ``"strict"``, or ``"silent"``.
    metadata:
        Arbitrary metadata to store in the contract.

    Returns
    -------
    DataContract

    Examples
    --------
    >>> import puredata
    >>> contract = puredata.watch(train_df)
    """
    return _watch(reference, mode=mode, metadata=metadata)


def check(
    new_data: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    contract: DataContract,
    *,
    mode: Optional[str] = None,
) -> WatchReport:
    """Validate *new_data* against a fitted :class:`DataContract`.

    Parameters
    ----------
    new_data:
        Incoming DataFrame, array, or file path.
    contract:
        A :class:`DataContract` returned by :func:`watch`.
    mode:
        Override the validation mode.

    Returns
    -------
    WatchReport

    Examples
    --------
    >>> import puredata
    >>> result = puredata.check(prod_df, contract)
    >>> print(result.summary())
    """
    return _check(new_data, contract, mode=mode)


def dashboard(
    df: Union[pd.DataFrame, Any],
    clean_report: Optional[CleanReport] = None,
    watch_report: Optional[WatchReport] = None,
    open_browser: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Open the interactive dataset health dashboard.

    Parameters
    ----------
    df:
        DataFrame to visualise.
    clean_report:
        Optional AutoClean report to embed.
    watch_report:
        Optional DataWatch report to embed.
    open_browser:
        Open the dashboard in the browser automatically.
    output_path:
        Save the HTML to this path.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    return _dashboard(
        df,
        clean_report=clean_report,
        watch_report=watch_report,
        open_browser=open_browser,
        output_path=output_path,
    )


def score(
    data: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    *,
    config: Optional[AutoCleanConfig] = None,
) -> float:
    """Return the MendScore (0–100) for *data* without modifying it.

    Parameters
    ----------
    data:
        Input DataFrame, array, or file path.
    config:
        Optional :class:`AutoCleanConfig`.

    Returns
    -------
    float
        MendScore between 0 and 100.
    """
    _, report = _clean(data, config=config)
    return report.mend_score


__all__ = [
    # Core functions
    "clean",
    "watch",
    "check",
    "dashboard",
    "score",
    # Classes
    "AutoClean",
    "AutoCleanConfig",
    "DataWatch",
    "DataContract",
    "MendPipeline",
    # Reports
    "CleanReport",
    "WatchReport",
    "Fix",
    "FixAction",
    "CheckResult",
    "CheckStatus",
    "DataCompatibilityError",
    # Plugins
    "CleanerPlugin",
    "ValidatorPlugin",
    "DriftDetectorPlugin",
    "PluginRegistry",
    "registry",
    "register_cleaner",
    "register_validator",
    "register_drift_detector",
]
