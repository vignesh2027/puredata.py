"""DataWatch: data compatibility and drift detection engine."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from puredata.core.report import (
    CheckResult,
    CheckStatus,
    DataCompatibilityError,
    WatchReport,
)


# ---------------------------------------------------------------------------
# Contract definition
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """Statistical profile of a single column captured at fit time.

    Attributes:
        name: Column name.
        dtype: pandas dtype string.
        null_rate: Fraction of null values (0–1).
        n_unique: Number of unique non-null values.
        categories: Set of unique values (categorical only, capped at 500).
        min: Minimum numeric value.
        max: Maximum numeric value.
        mean: Mean of numeric values.
        std: Standard deviation of numeric values.
        percentiles: Dict mapping percentile key to value (p5, p25, p50, p75, p95).
        histogram: (counts, bin_edges) for numeric drift computation.
    """
    name: str
    dtype: str
    null_rate: float = 0.0
    n_unique: int = 0
    categories: Optional[set] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)
    histogram: Optional[Tuple[List[float], List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "dtype": self.dtype,
            "null_rate": self.null_rate,
            "n_unique": self.n_unique,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "percentiles": self.percentiles,
            "categories": list(self.categories) if self.categories else None,
        }
        return d


@dataclass
class DataContract:
    """Fitted reference profile used by DataWatch for validation.

    Attributes:
        columns: Dict of column name → :class:`ColumnProfile`.
        column_order: Ordered column names as fitted.
        shape: ``(rows, cols)`` of the reference data.
        rules: List of custom rule callables (not serialisable).
        fitted_at: UTC timestamp when the contract was fitted.
        metadata: Arbitrary user metadata.
    """
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    column_order: List[str] = field(default_factory=list)
    shape: Tuple[int, int] = (0, 0)
    rules: List[Callable[[pd.DataFrame], Optional[str]]] = field(default_factory=list)
    fitted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_rule(
        self,
        fn: Callable[[pd.DataFrame], Optional[str]],
        name: str = "",
    ) -> "DataContract":
        """Add a custom business rule.

        The callable receives the new DataFrame and returns either ``None``
        (rule passes) or a human-readable error string (rule fails).

        Parameters
        ----------
        fn:
            Rule function: ``(df) -> Optional[str]``.
        name:
            Optional display name for reports.

        Returns
        -------
        DataContract
            Self, for chaining.

        Examples
        --------
        >>> contract.add_rule(lambda df: None if (df["age"] >= 0).all() else "age < 0 found")
        """
        fn._rule_name = name or fn.__name__  # type: ignore[attr-defined]
        self.rules.append(fn)
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Persist the contract to a JSON file (rules are not saved).

        Parameters
        ----------
        path:
            Destination file path.
        """
        data = {
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "column_order": self.column_order,
            "shape": list(self.shape),
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
            "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataContract":
        """Load a contract previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Source JSON file path.

        Returns
        -------
        DataContract
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        contract = cls()
        contract.column_order = raw.get("column_order", [])
        contract.shape = tuple(raw.get("shape", [0, 0]))  # type: ignore[assignment]
        if raw.get("fitted_at"):
            contract.fitted_at = datetime.fromisoformat(raw["fitted_at"])
        contract.metadata = raw.get("metadata", {})
        for name, col_data in raw.get("columns", {}).items():
            cats = col_data.get("categories")
            profile = ColumnProfile(
                name=col_data["name"],
                dtype=col_data["dtype"],
                null_rate=col_data["null_rate"],
                n_unique=col_data["n_unique"],
                categories=set(cats) if cats is not None else None,
                min=col_data.get("min"),
                max=col_data.get("max"),
                mean=col_data.get("mean"),
                std=col_data.get("std"),
                percentiles=col_data.get("percentiles", {}),
            )
            contract.columns[name] = profile
        return contract


# ---------------------------------------------------------------------------
# DataWatch engine
# ---------------------------------------------------------------------------


class DataWatch:
    """Fit on reference data; validate any new data instantly.

    Parameters
    ----------
    mode:
        ``"warn"`` (default) raises :mod:`warnings` on failures.
        ``"strict"`` raises :exc:`~puredata.core.report.DataCompatibilityError`.
        ``"silent"`` logs only to the returned report.
    null_rate_tolerance:
        Maximum allowed absolute increase in null rate before flagging.
    range_tolerance:
        Fractional extension beyond reference min/max before flagging.
    drift_threshold:
        PSI threshold above which distribution drift is flagged (0.2 = industry standard).
    cardinality_tolerance:
        Fraction of new unique values allowed above reference before flagging.
    """

    def __init__(
        self,
        mode: str = "warn",
        null_rate_tolerance: float = 0.10,
        range_tolerance: float = 0.05,
        drift_threshold: float = 0.2,
        cardinality_tolerance: float = 0.10,
    ) -> None:
        if mode not in ("warn", "strict", "silent"):
            raise ValueError(f"puredata: mode must be 'warn', 'strict', or 'silent'; got {mode!r}")
        self.mode = mode
        self.null_rate_tolerance = null_rate_tolerance
        self.range_tolerance = range_tolerance
        self.drift_threshold = drift_threshold
        self.cardinality_tolerance = cardinality_tolerance

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: Union[pd.DataFrame, Any, np.ndarray, str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataContract:
        """Profile *reference* data and return a :class:`DataContract`.

        Parameters
        ----------
        reference:
            Training / reference DataFrame, array, or file path.
        metadata:
            Arbitrary key-value pairs attached to the contract.

        Returns
        -------
        DataContract
        """
        df = self._coerce_input(reference)
        contract = DataContract(
            column_order=list(df.columns),
            shape=df.shape,
            fitted_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        for col in df.columns:
            contract.columns[col] = self._profile_column(df[col])
        return contract

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            null_rate=series.isna().mean(),
            n_unique=series.nunique(),
        )
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                profile.min = float(non_null.min())
                profile.max = float(non_null.max())
                profile.mean = float(non_null.mean())
                profile.std = float(non_null.std())
                profile.percentiles = {
                    "p5": float(non_null.quantile(0.05)),
                    "p25": float(non_null.quantile(0.25)),
                    "p50": float(non_null.quantile(0.50)),
                    "p75": float(non_null.quantile(0.75)),
                    "p95": float(non_null.quantile(0.95)),
                }
                n_bins = min(20, max(5, int(np.sqrt(len(non_null)))))
                counts, bin_edges = np.histogram(non_null, bins=n_bins)
                profile.histogram = (counts.tolist(), bin_edges.tolist())
        elif series.dtype == object or pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
            cats = series.dropna().unique()
            profile.categories = set(str(c) for c in cats[:500])
        return profile

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(
        self,
        new_data: Union[pd.DataFrame, Any, np.ndarray, str, Path],
        contract: DataContract,
    ) -> WatchReport:
        """Validate *new_data* against *contract*.

        Parameters
        ----------
        new_data:
            Incoming DataFrame, array, or file path.
        contract:
            A :class:`DataContract` produced by :meth:`fit`.

        Returns
        -------
        WatchReport
            Full compatibility report.

        Raises
        ------
        DataCompatibilityError
            In ``strict`` mode when any checks fail.
        """
        df = self._coerce_input(new_data)
        report = WatchReport(
            reference_shape=contract.shape,
            new_shape=df.shape,
            checked_at=datetime.now(timezone.utc),
            mode=self.mode,
        )

        self._check_schema(df, contract, report)
        for col in contract.columns:
            if col not in df.columns:
                continue
            profile = contract.columns[col]
            self._check_null_rate(df[col], profile, report)
            self._check_dtype(df[col], profile, report)
            if pd.api.types.is_numeric_dtype(df[col]):
                self._check_range(df[col], profile, report)
                self._check_drift(df[col], profile, report)
            elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                self._check_cardinality(df[col], profile, report)
        self._check_custom_rules(df, contract, report)

        self._emit(report)
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_schema(
        self, df: pd.DataFrame, contract: DataContract, report: WatchReport
    ) -> None:
        ref_cols = set(contract.column_order)
        new_cols = set(df.columns)

        missing = ref_cols - new_cols
        if missing:
            report.add(CheckResult(
                name="schema.missing_columns",
                column="<schema>",
                status=CheckStatus.FAIL,
                message=f"Missing columns: {sorted(missing)}",
                details={"missing": sorted(missing)},
            ))
        else:
            report.add(CheckResult(
                name="schema.missing_columns",
                column="<schema>",
                status=CheckStatus.PASS,
                message="All reference columns present",
            ))

        extra = new_cols - ref_cols
        if extra:
            report.add(CheckResult(
                name="schema.extra_columns",
                column="<schema>",
                status=CheckStatus.WARN,
                message=f"Unexpected new columns: {sorted(extra)}",
                details={"extra": sorted(extra)},
            ))
        else:
            report.add(CheckResult(
                name="schema.extra_columns",
                column="<schema>",
                status=CheckStatus.PASS,
                message="No unexpected columns",
            ))

    def _check_dtype(
        self, series: pd.Series, profile: ColumnProfile, report: WatchReport
    ) -> None:
        new_dtype = str(series.dtype)
        if new_dtype != profile.dtype:
            report.add(CheckResult(
                name="dtype.changed",
                column=series.name,
                status=CheckStatus.FAIL,
                message=(
                    f"Type changed from '{profile.dtype}' to '{new_dtype}'"
                ),
                details={"expected": profile.dtype, "actual": new_dtype},
            ))
        else:
            report.add(CheckResult(
                name="dtype.changed",
                column=series.name,
                status=CheckStatus.PASS,
                message=f"dtype '{new_dtype}' unchanged",
            ))

    def _check_null_rate(
        self, series: pd.Series, profile: ColumnProfile, report: WatchReport
    ) -> None:
        new_rate = series.isna().mean()
        delta = new_rate - profile.null_rate
        if delta > self.null_rate_tolerance:
            report.add(CheckResult(
                name="nulls.rate_increase",
                column=series.name,
                status=CheckStatus.FAIL,
                message=(
                    f"Null rate jumped from {profile.null_rate:.1%} "
                    f"to {new_rate:.1%} (Δ {delta:+.1%})"
                ),
                details={
                    "reference_null_rate": profile.null_rate,
                    "new_null_rate": new_rate,
                    "delta": delta,
                },
            ))
        elif delta > self.null_rate_tolerance / 2:
            report.add(CheckResult(
                name="nulls.rate_increase",
                column=series.name,
                status=CheckStatus.WARN,
                message=(
                    f"Null rate increased from {profile.null_rate:.1%} "
                    f"to {new_rate:.1%} (Δ {delta:+.1%})"
                ),
                details={
                    "reference_null_rate": profile.null_rate,
                    "new_null_rate": new_rate,
                    "delta": delta,
                },
            ))
        else:
            report.add(CheckResult(
                name="nulls.rate_increase",
                column=series.name,
                status=CheckStatus.PASS,
                message=f"Null rate {new_rate:.1%} within tolerance",
            ))

    def _check_range(
        self, series: pd.Series, profile: ColumnProfile, report: WatchReport
    ) -> None:
        if profile.min is None or profile.max is None:
            return
        non_null = series.dropna()
        if len(non_null) == 0:
            return
        span = profile.max - profile.min if profile.max != profile.min else abs(profile.max)
        tol = span * self.range_tolerance + 1e-9
        new_min = float(non_null.min())
        new_max = float(non_null.max())
        violations: List[str] = []
        if new_min < profile.min - tol:
            violations.append(
                f"min {new_min:.4g} < reference min {profile.min:.4g}"
            )
        if new_max > profile.max + tol:
            violations.append(
                f"max {new_max:.4g} > reference max {profile.max:.4g}"
            )
        if violations:
            out_rows = series.index[
                (series < profile.min - tol) | (series > profile.max + tol)
            ].tolist()
            report.add(CheckResult(
                name="range.violation",
                column=series.name,
                status=CheckStatus.FAIL,
                message="; ".join(violations),
                details={
                    "reference_min": profile.min,
                    "reference_max": profile.max,
                    "new_min": new_min,
                    "new_max": new_max,
                    "out_of_range_rows": out_rows[:50],
                },
            ))
        else:
            report.add(CheckResult(
                name="range.violation",
                column=series.name,
                status=CheckStatus.PASS,
                message=f"Range [{new_min:.4g}, {new_max:.4g}] within reference",
            ))

    def _check_drift(
        self, series: pd.Series, profile: ColumnProfile, report: WatchReport
    ) -> None:
        non_null = series.dropna()
        if len(non_null) < 10 or profile.histogram is None:
            return

        ref_counts, bin_edges = profile.histogram
        ref_counts_arr = np.array(ref_counts, dtype=float)
        new_counts, _ = np.histogram(non_null, bins=bin_edges)
        new_counts_arr = new_counts.astype(float)

        n_bins = len(ref_counts_arr)
        ref_n = ref_counts_arr.sum()
        new_n = new_counts_arr.sum()

        # Laplace smoothing: add 0.5 per bin so empty bins don't explode the log
        ref_smooth = (ref_counts_arr + 0.5) / (ref_n + 0.5 * n_bins)
        new_smooth = (new_counts_arr + 0.5) / (new_n + 0.5 * n_bins)

        # PSI (Population Stability Index) with Laplace-smoothed proportions
        psi = float(np.sum((new_smooth - ref_smooth) * np.log(new_smooth / ref_smooth)))

        # KS test directly on raw values
        ks_stat, ks_p = stats.ks_2samp(
            np.repeat(
                [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(ref_counts))],
                np.maximum(ref_counts, 1).astype(int),
            ),
            non_null.to_numpy(dtype=float, na_value=np.nan),
        )

        # Jensen–Shannon divergence using smoothed proportions
        m = 0.5 * (ref_smooth + new_smooth)
        js_div = float(
            0.5 * np.sum(ref_smooth * np.log(ref_smooth / m))
            + 0.5 * np.sum(new_smooth * np.log(new_smooth / m))
        )

        drift_score = min(100, round((psi / self.drift_threshold) * 50 + ks_stat * 50, 1))
        # Require BOTH high PSI and statistically significant KS test to FAIL;
        # PSI alone is too noisy for small samples (< 1 000).
        if psi > self.drift_threshold and ks_p < 0.05:
            drift_status = CheckStatus.FAIL
        elif psi > self.drift_threshold / 2 or (ks_p < 0.05 and ks_stat > 0.15):
            drift_status = CheckStatus.WARN
        else:
            drift_status = CheckStatus.PASS
        report.add(CheckResult(
            name="distribution.drift",
            column=series.name,
            status=drift_status,
            message=(
                f"Drift score {drift_score}/100 — "
                f"PSI={psi:.3f}, KS={ks_stat:.3f} (p={ks_p:.3f}), JS={js_div:.3f}"
            ),
            details={
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "js_divergence": js_div,
                "drift_score": drift_score,
            },
        ))

    def _check_cardinality(
        self, series: pd.Series, profile: ColumnProfile, report: WatchReport
    ) -> None:
        if profile.categories is None:
            return
        new_cats = set(str(v) for v in series.dropna().unique())
        new_vals = new_cats - profile.categories
        if not new_vals:
            report.add(CheckResult(
                name="cardinality.new_values",
                column=series.name,
                status=CheckStatus.PASS,
                message=f"No unexpected category values (reference has {len(profile.categories)})",
            ))
            return
        ratio = len(new_vals) / max(len(profile.categories), 1)
        status = (
            CheckStatus.FAIL if ratio > self.cardinality_tolerance
            else CheckStatus.WARN
        )
        sample = sorted(str(v) for v in list(new_vals)[:20])
        report.add(CheckResult(
            name="cardinality.new_values",
            column=series.name,
            status=status,
            message=(
                f"{len(new_vals)} unexpected category value(s): {sample}"
            ),
            details={
                "n_new_values": len(new_vals),
                "sample_new_values": sample,
                "reference_n_categories": len(profile.categories),
            },
        ))

    def _check_custom_rules(
        self, df: pd.DataFrame, contract: DataContract, report: WatchReport
    ) -> None:
        for rule_fn in contract.rules:
            rule_name = getattr(rule_fn, "_rule_name", rule_fn.__name__)
            try:
                error = rule_fn(df)
            except Exception as exc:
                report.add(CheckResult(
                    name=f"rule.{rule_name}",
                    column="<custom>",
                    status=CheckStatus.FAIL,
                    message=f"Rule raised exception: {exc}",
                    details={"exception": str(exc)},
                ))
                continue
            if error is None:
                report.add(CheckResult(
                    name=f"rule.{rule_name}",
                    column="<custom>",
                    status=CheckStatus.PASS,
                    message="Rule passed",
                ))
            else:
                report.add(CheckResult(
                    name=f"rule.{rule_name}",
                    column="<custom>",
                    status=CheckStatus.FAIL,
                    message=str(error),
                ))

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def _emit(self, report: WatchReport) -> None:
        if report.n_failed == 0:
            return
        msg = (
            f"puredata DataWatch: {report.n_failed} check(s) failed. "
            f"Call report.summary() for details."
        )
        if self.mode == "strict":
            report.raise_if_failed()
        elif self.mode == "warn":
            warnings.warn(msg, UserWarning, stacklevel=4)

    # ------------------------------------------------------------------
    # Input coercion (mirrors AutoClean)
    # ------------------------------------------------------------------

    def _coerce_input(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        path = Path(data) if isinstance(data, str) else data
        if isinstance(path, Path):
            if not path.exists():
                raise FileNotFoundError(f"puredata: file not found — {path}")
            suffix = path.suffix.lower()
            readers = {
                ".csv": pd.read_csv,
                ".xlsx": pd.read_excel,
                ".parquet": pd.read_parquet,
                ".json": pd.read_json,
            }
            if suffix not in readers:
                raise ValueError(f"puredata: unsupported file format '{suffix}'")
            return readers[suffix](path)
        try:
            import polars as pl
            if isinstance(data, pl.DataFrame):
                return data.to_pandas()
        except ImportError:
            pass
        raise TypeError(
            f"puredata: unsupported input type {type(data).__name__}"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


_default_engine = DataWatch()


def watch(
    reference: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    *,
    mode: str = "warn",
    metadata: Optional[Dict[str, Any]] = None,
) -> DataContract:
    """Fit a :class:`DataContract` on *reference* data.

    Parameters
    ----------
    reference:
        Training / reference DataFrame, array, or file path.
    mode:
        Validation mode: ``"warn"`` (default), ``"strict"``, or ``"silent"``.
    metadata:
        Arbitrary metadata to attach to the contract.

    Returns
    -------
    DataContract

    Examples
    --------
    >>> import puredata
    >>> contract = puredata.watch(train_df)
    """
    engine = DataWatch(mode=mode)
    return engine.fit(reference, metadata=metadata)


def check(
    new_data: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    contract: DataContract,
    *,
    mode: Optional[str] = None,
) -> WatchReport:
    """Validate *new_data* against *contract*.

    Parameters
    ----------
    new_data:
        Incoming DataFrame, array, or file path.
    contract:
        A :class:`DataContract` produced by :func:`watch`.
    mode:
        Override the mode from the contract's DataWatch engine.

    Returns
    -------
    WatchReport

    Examples
    --------
    >>> import puredata
    >>> result = puredata.check(prod_df, contract)
    >>> print(result.summary())
    """
    engine = DataWatch(mode=mode or "warn")
    return engine.check(new_data, contract)
