"""AutoClean: intelligent automatic data cleaning engine."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from rapidfuzz import fuzz, process as fuzz_process

from puredata.core.report import CleanReport, Fix, FixAction


# ---------------------------------------------------------------------------
# Data-type detection helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS: List[str] = [
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y.%m.%d", "%d.%m.%Y", "%m.%d.%Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y%m%d", "%d%m%Y",
    "%a, %d %b %Y %H:%M:%S", "%A, %B %d, %Y", "%m-%d-%y", "%d-%m-%y",
    "%y/%m/%d", "%d/%m/%y", "%m/%d/%y",
]

_UNIT_PATTERNS: Dict[str, Dict[str, float]] = {
    "weight": {
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592,
        "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
    },
    "distance": {
        "km": 1.0, "kilometer": 1.0, "kilometers": 1.0,
        "m": 0.001, "meter": 0.001, "meters": 0.001,
        "mi": 1.60934, "mile": 1.60934, "miles": 1.60934,
        "ft": 0.0003048, "foot": 0.0003048, "feet": 0.0003048,
        "in": 0.0000254, "inch": 0.0000254, "inches": 0.0000254,
        "yd": 0.0009144, "yard": 0.0009144, "yards": 0.0009144,
    },
    "temperature": {
        "c": 1.0, "celsius": 1.0,
        "f": None,  # requires formula, handled specially
        "fahrenheit": None,
        "k": None,  # requires formula
        "kelvin": None,
    },
}


def _try_parse_date(val: str, formats: List[str]) -> Optional[datetime]:
    for fmt in formats:
        try:
            return datetime.strptime(val.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(val, dayfirst=False, yearfirst=False)
    except Exception:
        return None


def _col_is_numeric_strings(series: pd.Series) -> bool:
    """Return True when >80 % of non-null values are numeric strings."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    numeric_count = non_null.str.match(r"^\s*-?\d+(\.\d+)?\s*$").sum()
    return numeric_count / len(non_null) > 0.80


def _col_is_date_strings(series: pd.Series) -> bool:
    """Return True when >70 % of non-null string values look like dates."""
    non_null = series.dropna().astype(str)
    sample = non_null.head(50)
    if len(sample) == 0:
        return False
    # Skip columns where most values are plain numbers — they are not dates
    numeric_count = sample.str.match(r"^\s*-?\d+(\.\d+)?\s*$").sum()
    if numeric_count / len(sample) > 0.50:
        return False
    parsed = sum(1 for v in sample if _try_parse_date(v, _DATE_FORMATS))
    return parsed / len(sample) > 0.70


def _col_is_bool_integers(series: pd.Series) -> bool:
    """Return True when column contains only 0 and 1."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return set(non_null.unique()).issubset({0, 1, True, False})


# ---------------------------------------------------------------------------
# Core AutoClean engine
# ---------------------------------------------------------------------------


@dataclass
class AutoCleanConfig:
    """Configuration for AutoClean behaviour.

    Attributes:
        fix_nulls: Whether to impute missing values.
        fix_outliers: Whether to detect and handle outliers.
        fix_types: Whether to coerce columns to inferred types.
        fix_duplicates: Whether to remove duplicate rows.
        fix_encoding: Whether to repair encoding artefacts.
        fix_categories: Whether to normalise inconsistent categories.
        fix_dates: Whether to normalise date formats.
        fix_whitespace: Whether to strip/normalise whitespace in strings.
        fix_units: Whether to attempt unit normalisation.
        outlier_action: One of ``"clip"``, ``"remove"``, or ``"nan"``.
        outlier_threshold: Minimum fraction of methods that must agree.
        date_output_format: ``strftime`` format for output dates.
        n_neighbors: K for KNN imputation.
        mode: ``"auto"`` applies fixes immediately; ``"review"`` returns
            pending fixes for manual approval before application.
        n_jobs: Parallel jobs for heavy operations (-1 = all cores).
    """

    fix_nulls: bool = True
    fix_outliers: bool = True
    fix_types: bool = True
    fix_duplicates: bool = True
    fix_encoding: bool = True
    fix_categories: bool = True
    fix_dates: bool = True
    fix_whitespace: bool = True
    fix_units: bool = True
    outlier_action: str = "clip"
    outlier_threshold: float = 0.5
    date_output_format: str = "%Y-%m-%d"
    n_neighbors: int = 5
    mode: str = "auto"
    n_jobs: int = -1


class AutoClean:
    """Intelligent automatic data cleaning engine.

    Parameters
    ----------
    config:
        Optional :class:`AutoCleanConfig` to override defaults.
    """

    def __init__(self, config: Optional[AutoCleanConfig] = None) -> None:
        self.config = config or AutoCleanConfig()
        self._report = CleanReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(
        self,
        data: Union[pd.DataFrame, "pl.DataFrame", np.ndarray, str, Path],
        *,
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, CleanReport]:
        """Clean *data* automatically and return the result plus a repair report.

        Parameters
        ----------
        data:
            A pandas DataFrame, polars DataFrame, numpy array, or path to a
            CSV / Excel / Parquet / JSON file.
        target_col:
            Name of the target column. Outlier removal on this column is skipped
            to avoid accidentally corrupting labels.

        Returns
        -------
        tuple[pd.DataFrame, CleanReport]
            The cleaned DataFrame and the full repair report.
        """
        self._report = CleanReport()
        df = self._coerce_input(data)
        original_shape = df.shape

        self._report.original_shape = original_shape
        self._report.started_at = datetime.now(timezone.utc)

        if self.config.fix_encoding:
            df = self._fix_encoding(df)
        if self.config.fix_whitespace:
            df = self._fix_whitespace(df)
        if self.config.fix_types:
            df = self._fix_types(df)
        if self.config.fix_dates:
            df = self._fix_dates(df)
        if self.config.fix_duplicates:
            df = self._fix_duplicates(df)
        if self.config.fix_categories:
            df = self._fix_categories(df)
        if self.config.fix_units:
            df = self._fix_units(df)
        if self.config.fix_nulls:
            df = self._fix_nulls(df, target_col=target_col)
        if self.config.fix_outliers:
            df = self._fix_outliers(df, target_col=target_col)

        self._report.cleaned_shape = df.shape
        self._report.ended_at = datetime.now(timezone.utc)
        return df, self._report

    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------

    def _coerce_input(
        self, data: Union[pd.DataFrame, Any, np.ndarray, str, Path]
    ) -> pd.DataFrame:
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
                ".xls": pd.read_excel,
                ".parquet": pd.read_parquet,
                ".json": pd.read_json,
            }
            if suffix not in readers:
                raise ValueError(
                    f"puredata: unsupported file format '{suffix}'. "
                    "Supported: csv, xlsx, xls, parquet, json"
                )
            return readers[suffix](path)
        # polars support
        try:
            import polars as pl
            if isinstance(data, pl.DataFrame):
                return data.to_pandas()
        except ImportError:
            pass
        raise TypeError(
            f"puredata: unsupported input type {type(data).__name__}. "
            "Pass a pandas DataFrame, polars DataFrame, numpy array, or file path."
        )

    # ------------------------------------------------------------------
    # Encoding fixes
    # ------------------------------------------------------------------

    def _fix_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            series = df[col]
            fixed: List[Any] = []
            changed_rows: List[int] = []
            for idx, val in series.items():
                if not isinstance(val, str):
                    fixed.append(val)
                    continue
                cleaned = self._clean_string_encoding(val)
                fixed.append(cleaned)
                if cleaned != val:
                    changed_rows.append(int(idx))  # type: ignore[arg-type]
            if changed_rows:
                df[col] = fixed
                self._report.add_fix(Fix(
                    column=col,
                    action=FixAction.ENCODING,
                    rows=changed_rows,
                    details=f"Repaired encoding artefacts in {len(changed_rows)} cells",
                ))
        return df

    def _clean_string_encoding(self, val: str) -> str:
        # Strip BOM
        val = val.lstrip("﻿")
        # Remove zero-width and invisible Unicode characters
        val = re.sub(r"[​‌‍﻿­]", "", val)
        # Normalise to NFC (fixes mojibake artefacts from latin-1/utf-8 confusion)
        val = unicodedata.normalize("NFC", val)
        return val

    # ------------------------------------------------------------------
    # Whitespace fixes
    # ------------------------------------------------------------------

    def _fix_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            original = df[col].copy()
            df[col] = (
                df[col]
                .astype(str)
                .where(df[col].notna(), other=np.nan)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .where(df[col].notna(), other=np.nan)
            )
            diff_mask = (df[col] != original) & original.notna()
            changed = diff_mask.sum()
            if changed:
                self._report.add_fix(Fix(
                    column=col,
                    action=FixAction.WHITESPACE,
                    rows=df.index[diff_mask].tolist(),
                    details=f"Stripped/normalised whitespace in {changed} cells",
                ))
        return df

    # ------------------------------------------------------------------
    # Type coercion
    # ------------------------------------------------------------------

    def _fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            series = df[col]
            if (series.dtype == object or pd.api.types.is_string_dtype(series)) and _col_is_numeric_strings(series):
                original = series.copy()
                df[col] = pd.to_numeric(series, errors="coerce")
                changed = (df[col].notna() != original.notna()).sum() + (
                    (df[col] != original.apply(lambda x: float(x) if pd.notna(x) else x)).sum()
                )
                self._report.add_fix(Fix(
                    column=col,
                    action=FixAction.TYPE_COERCE,
                    rows=[],
                    details=f"Converted numeric strings to float64",
                ))
            elif series.dtype in (np.int64, np.float64) and _col_is_bool_integers(series):
                pass  # leave as numeric; bool coercion is lossy for ML
        return df

    # ------------------------------------------------------------------
    # Date normalisation
    # ------------------------------------------------------------------

    def _fix_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        fmt = self.config.date_output_format
        for col in str_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if not _col_is_date_strings(df[col]):
                continue
            parsed_values: List[Any] = []
            changed_rows: List[int] = []
            for idx, val in df[col].items():
                if pd.isna(val):
                    parsed_values.append(np.nan)
                    continue
                dt = _try_parse_date(str(val), _DATE_FORMATS)
                if dt:
                    normalised = dt.strftime(fmt)
                    if normalised != str(val).strip():
                        changed_rows.append(int(idx))  # type: ignore[arg-type]
                    parsed_values.append(normalised)
                else:
                    parsed_values.append(val)
            if changed_rows:
                df[col] = parsed_values
                self._report.add_fix(Fix(
                    column=col,
                    action=FixAction.DATE_NORMALISE,
                    rows=changed_rows,
                    details=f"Normalised {len(changed_rows)} dates to {fmt}",
                ))
        return df

    # ------------------------------------------------------------------
    # Duplicate removal
    # ------------------------------------------------------------------

    def _fix_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        removed = n_before - len(df)
        if removed:
            df = df.reset_index(drop=True)
            self._report.add_fix(Fix(
                column="<all>",
                action=FixAction.DUPLICATE_REMOVE,
                rows=[],
                details=f"Removed {removed} exact duplicate rows",
            ))
        return df

    # ------------------------------------------------------------------
    # Category normalisation
    # ------------------------------------------------------------------

    def _fix_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            n_unique = series.nunique()
            # Only normalise low-cardinality columns that are likely categorical
            if n_unique > 50 or n_unique <= 1:
                continue
            unique_vals: List[str] = [str(v) for v in series.unique().tolist()]
            mapping: Dict[str, str] = {}
            # Cluster similar values using fuzzy matching + prefix matching
            canonical: Dict[str, List[str]] = {}
            used: set = set()
            for val in unique_vals:
                if val in used:
                    continue
                group = [val]
                used.add(val)
                val_lower = val.lower()
                for other in unique_vals:
                    if other in used:
                        continue
                    other_lower = other.lower()
                    ratio = fuzz.ratio(val_lower, other_lower)
                    # Cluster if one value is a short abbreviation/prefix of the other
                    is_abbreviation = (
                        (len(other_lower) <= 3 and val_lower.startswith(other_lower))
                        or (len(val_lower) <= 3 and other_lower.startswith(val_lower))
                    )
                    if ratio >= 85 or is_abbreviation:
                        group.append(other)
                        used.add(other)
                # pick the most frequent as canonical
                freq = {v: series.value_counts().get(v, 0) for v in group}
                canon = max(freq, key=freq.get)  # type: ignore[arg-type]
                for v in group:
                    if v != canon:
                        canonical.setdefault(canon, []).append(v)
                        mapping[v] = canon

            if mapping:
                changed_rows = df.index[df[col].isin(mapping)].tolist()
                df[col] = df[col].map(lambda x: mapping.get(x, x) if isinstance(x, str) else x)
                details = "; ".join(f"{v!r}→{k!r}" for k, vs in canonical.items() for v in vs)
                self._report.add_fix(Fix(
                    column=col,
                    action=FixAction.CATEGORY_NORMALISE,
                    rows=changed_rows,
                    details=f"Normalised categories: {details[:200]}",
                ))
        return df

    # ------------------------------------------------------------------
    # Unit normalisation
    # ------------------------------------------------------------------

    def _fix_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect columns with mixed units and normalise to SI base unit."""
        pattern = re.compile(
            r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$"
        )
        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna().astype(str)
            if len(series) < 5:
                continue
            detected: Dict[str, int] = {}
            for val in series:
                m = pattern.match(val)
                if m:
                    unit = m.group(2).lower()
                    detected[unit] = detected.get(unit, 0) + 1
            if len(detected) < 2:
                continue
            # Determine which unit family this column belongs to
            matched_family: Optional[str] = None
            matched_units: Dict[str, float] = {}
            for family, units in _UNIT_PATTERNS.items():
                if family == "temperature":
                    continue  # formula-based, skip
                overlap = set(detected.keys()) & set(units.keys())
                if len(overlap) >= 2:
                    matched_family = family
                    matched_units = units
                    break
            if not matched_family:
                continue
            numeric_values: List[Any] = []
            changed_rows: List[int] = []
            for idx, val in df[col].items():
                if pd.isna(val):
                    numeric_values.append(np.nan)
                    continue
                m = pattern.match(str(val))
                if m:
                    num = float(m.group(1))
                    unit = m.group(2).lower()
                    factor = matched_units.get(unit)
                    if factor is not None:
                        numeric_values.append(num * factor)
                        changed_rows.append(int(idx))  # type: ignore[arg-type]
                        continue
                numeric_values.append(val)
            if changed_rows:
                df[col] = numeric_values
                self._report.add_fix(Fix(
                    column=col,
                    action=FixAction.UNIT_NORMALISE,
                    rows=changed_rows,
                    details=(
                        f"Normalised {len(changed_rows)} values to "
                        f"{matched_family} SI base unit from mixed units: "
                        + ", ".join(detected.keys())
                    ),
                ))
        return df

    # ------------------------------------------------------------------
    # Null imputation
    # ------------------------------------------------------------------

    def _fix_nulls(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # ---- numeric imputation ----
        null_numeric = [c for c in numeric_cols if df[c].isna().any() and c != target_col]
        if null_numeric:
            df = self._impute_numeric(df, null_numeric)

        # ---- categorical imputation ----
        for col in categorical_cols:
            if col == target_col:
                continue
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue
            null_rate = null_count / len(df)
            if null_rate > 0.50:
                fill_val = "__unknown__"
            else:
                mode_series = df[col].mode()
                fill_val = mode_series.iloc[0] if len(mode_series) > 0 else "__unknown__"
            changed_rows = df.index[df[col].isna()].tolist()
            df[col] = df[col].fillna(fill_val)
            self._report.add_fix(Fix(
                column=col,
                action=FixAction.NULL_IMPUTE,
                rows=changed_rows,
                details=(
                    f"Imputed {null_count} nulls ({null_rate:.1%}) with "
                    f"{'__unknown__' if null_rate > 0.50 else 'mode'} = {fill_val!r}"
                ),
            ))

        # ---- datetime imputation ----
        for col in datetime_cols:
            if col == target_col:
                continue
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue
            changed_rows = df.index[df[col].isna()].tolist()
            df[col] = df[col].ffill().bfill()
            self._report.add_fix(Fix(
                column=col,
                action=FixAction.NULL_IMPUTE,
                rows=changed_rows,
                details=f"Imputed {null_count} datetime nulls via forward/backward fill",
            ))

        return df

    def _impute_numeric(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        sub = df[cols]
        null_rates = sub.isnull().mean()

        # Skip columns that are entirely null — imputers can't help them
        # Fall back to column mean (0 if all null) for those.
        all_null_cols = [c for c in cols if null_rates[c] >= 1.0]
        for col in all_null_cols:
            df[col] = 0.0
            self._report.add_fix(Fix(
                column=col,
                action=FixAction.NULL_IMPUTE,
                rows=df.index.tolist(),
                details="All values null — filled with 0",
            ))

        cols = [c for c in cols if c not in all_null_cols]
        if not cols:
            return df

        null_rates = df[cols].isnull().mean()

        # KNN imputation for moderate missingness
        knn_cols = [c for c in cols if 0 < null_rates[c] <= 0.40]
        iter_cols = [c for c in cols if null_rates[c] > 0.40]

        if knn_cols:
            before = df[knn_cols].copy()
            imputer = KNNImputer(n_neighbors=min(self.config.n_neighbors, len(df) - 1))
            fitted = imputer.fit_transform(df[knn_cols])
            for i, col in enumerate(knn_cols):
                df[col] = fitted[:, i]
                changed_rows = df.index[before[col].isna()].tolist()
                if changed_rows:
                    self._report.add_fix(Fix(
                        column=col,
                        action=FixAction.NULL_IMPUTE,
                        rows=changed_rows,
                        details=(
                            f"KNN-imputed {len(changed_rows)} nulls "
                            f"({null_rates[col]:.1%} missing)"
                        ),
                    ))

        if iter_cols:
            before = df[iter_cols].copy()
            imputer = IterativeImputer(max_iter=10, random_state=42)
            fitted = imputer.fit_transform(df[iter_cols])
            for i, col in enumerate(iter_cols):
                df[col] = fitted[:, i]
                changed_rows = df.index[before[col].isna()].tolist()
                if changed_rows:
                    self._report.add_fix(Fix(
                        column=col,
                        action=FixAction.NULL_IMPUTE,
                        rows=changed_rows,
                        details=(
                            f"Iterative-imputed {len(changed_rows)} nulls "
                            f"({null_rates[col]:.1%} missing)"
                        ),
                    ))

        return df

    # ------------------------------------------------------------------
    # Outlier detection & handling
    # ------------------------------------------------------------------

    def _fix_outliers(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col and df[c].notna().sum() >= 10
        ]
        if not numeric_cols:
            return df

        for col in numeric_cols:
            series = df[col].dropna()
            outlier_mask = self._detect_outliers_ensemble(series, df.index)
            n_outliers = outlier_mask.sum()
            if n_outliers == 0:
                continue

            action = self.config.outlier_action
            changed_rows = df.index[outlier_mask].tolist()

            if action == "remove":
                df = df[~outlier_mask].reset_index(drop=True)
                detail = f"Removed {n_outliers} outlier rows"
            elif action == "nan":
                df.loc[outlier_mask, col] = np.nan
                detail = f"Set {n_outliers} outliers to NaN"
            else:  # clip
                q_low = df[col].quantile(0.01)
                q_high = df[col].quantile(0.99)
                df.loc[outlier_mask, col] = df.loc[outlier_mask, col].clip(q_low, q_high)
                detail = f"Clipped {n_outliers} outliers to [{q_low:.3g}, {q_high:.3g}]"

            self._report.add_fix(Fix(
                column=col,
                action=FixAction.OUTLIER,
                rows=changed_rows,
                details=detail,
            ))

        return df

    def _detect_outliers_ensemble(
        self, series: pd.Series, full_index: pd.Index
    ) -> pd.Series:
        """Return a boolean mask over *full_index* indicating ensemble-detected outliers."""
        votes = pd.Series(0, index=series.index)

        # Method 1: IQR
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            iqr_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            votes += iqr_mask.astype(int)

        # Method 2: Z-score
        z = np.abs(stats.zscore(series, nan_policy="omit"))
        zscore_mask = pd.Series(z > 3, index=series.index)
        votes += zscore_mask.astype(int)

        # Method 3: Isolation Forest (min 20 samples)
        if len(series) >= 20:
            X = series.values.reshape(-1, 1)
            iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=self.config.n_jobs)
            preds = iso.fit_predict(X)
            iso_mask = pd.Series(preds == -1, index=series.index)
            votes += iso_mask.astype(int)

        # Method 4: LOF (min 20 samples)
        if len(series) >= 20:
            X = series.values.reshape(-1, 1)
            lof = LocalOutlierFactor(n_neighbors=min(20, len(series) - 1))
            preds = lof.fit_predict(X)
            lof_mask = pd.Series(preds == -1, index=series.index)
            votes += lof_mask.astype(int)

        max_votes = min(4, 2 + (len(series) >= 20) * 2)
        threshold_votes = max(1, int(max_votes * self.config.outlier_threshold))
        outlier_series = votes >= threshold_votes

        # Align to full_index
        full_mask = pd.Series(False, index=full_index)
        full_mask.loc[outlier_series.index] = outlier_series
        return full_mask


def clean(
    data: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    *,
    config: Optional[AutoCleanConfig] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, CleanReport]:
    """Clean *data* automatically. One-line entry point.

    Parameters
    ----------
    data:
        Pandas/polars DataFrame, numpy array, or path to CSV/Excel/Parquet/JSON.
    config:
        Optional :class:`AutoCleanConfig` to override defaults.
    target_col:
        Column to protect from cleaning (typically the ML target/label).

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
    engine = AutoClean(config=config)
    return engine.clean(data, target_col=target_col)
