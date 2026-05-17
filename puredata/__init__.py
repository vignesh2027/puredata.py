"""puredata — automatic data cleaning and silent incompatibility detection.

Two problems. Solved perfectly.

    >>> import puredata
    >>> clean_df, report = puredata.clean(dirty_df)
    >>> contract = puredata.watch(train_df)
    >>> result = puredata.check(prod_df, contract)
"""

from puredata.api import (
    AutoClean,
    AutoCleanConfig,
    CheckResult,
    CheckStatus,
    CleanReport,
    CleanerPlugin,
    DataCompatibilityError,
    DataContract,
    DataWatch,
    DriftDetectorPlugin,
    Fix,
    FixAction,
    MendPipeline,
    PluginRegistry,
    ValidatorPlugin,
    WatchReport,
    check,
    clean,
    dashboard,
    register_cleaner,
    register_drift_detector,
    register_validator,
    registry,
    score,
    watch,
)

__version__ = "0.2.0"
__author__ = "Vignesh"
__license__ = "MIT"

__all__ = [
    "__version__",
    "clean",
    "watch",
    "check",
    "dashboard",
    "score",
    "AutoClean",
    "AutoCleanConfig",
    "DataWatch",
    "DataContract",
    "MendPipeline",
    "CleanReport",
    "WatchReport",
    "Fix",
    "FixAction",
    "CheckResult",
    "CheckStatus",
    "DataCompatibilityError",
    "CleanerPlugin",
    "ValidatorPlugin",
    "DriftDetectorPlugin",
    "PluginRegistry",
    "registry",
    "register_cleaner",
    "register_validator",
    "register_drift_detector",
]
