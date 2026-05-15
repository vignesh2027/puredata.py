"""puredata.core — AutoClean and DataWatch engines."""

from puredata.core.clean import AutoClean, AutoCleanConfig, clean
from puredata.core.report import (
    CheckResult,
    CheckStatus,
    CleanReport,
    DataCompatibilityError,
    Fix,
    FixAction,
    WatchReport,
)
from puredata.core.watch import DataContract, DataWatch, check, watch

__all__ = [
    "AutoClean",
    "AutoCleanConfig",
    "clean",
    "DataWatch",
    "DataContract",
    "watch",
    "check",
    "CleanReport",
    "WatchReport",
    "Fix",
    "FixAction",
    "CheckResult",
    "CheckStatus",
    "DataCompatibilityError",
]
