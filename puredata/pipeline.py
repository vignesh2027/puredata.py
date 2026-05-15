"""MendPipeline: unified AutoClean + DataWatch workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from puredata.core.clean import AutoClean, AutoCleanConfig
from puredata.core.report import CleanReport, WatchReport
from puredata.core.watch import DataContract, DataWatch


class MendPipeline:
    """Chain AutoClean and DataWatch into a single reusable pipeline.

    Fit the pipeline once on your training data; then call it on any
    new data to clean and validate in a single step forever.

    Parameters
    ----------
    clean_config:
        Optional :class:`~puredata.core.clean.AutoCleanConfig`.
    watch_mode:
        DataWatch validation mode: ``"warn"`` (default), ``"strict"``,
        or ``"silent"``.
    target_col:
        Column to protect during cleaning (ML label column).

    Examples
    --------
    >>> from puredata import MendPipeline
    >>> pipeline = MendPipeline()
    >>> pipeline.fit(train_df)
    >>> clean_df, clean_report, watch_report = pipeline.run(new_df)
    """

    def __init__(
        self,
        clean_config: Optional[AutoCleanConfig] = None,
        watch_mode: str = "warn",
        target_col: Optional[str] = None,
    ) -> None:
        self._cleaner = AutoClean(config=clean_config)
        self._watcher = DataWatch(mode=watch_mode)
        self._contract: Optional[DataContract] = None
        self._target_col = target_col

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: Union[pd.DataFrame, Any, np.ndarray, str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MendPipeline":
        """Fit on *reference* data (your training set).

        Cleans the reference data and builds a :class:`~puredata.core.watch.DataContract`
        from the clean result.

        Parameters
        ----------
        reference:
            Training DataFrame, array, or file path.
        metadata:
            Arbitrary metadata stored in the contract.

        Returns
        -------
        MendPipeline
            Self, for chaining.
        """
        clean_ref, _ = self._cleaner.clean(reference, target_col=self._target_col)
        self._contract = self._watcher.fit(clean_ref, metadata=metadata)
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        data: Union[pd.DataFrame, Any, np.ndarray, str, Path],
    ) -> Tuple[pd.DataFrame, CleanReport, WatchReport]:
        """Clean *data* and validate it against the fitted contract.

        Parameters
        ----------
        data:
            Incoming DataFrame, array, or file path.

        Returns
        -------
        tuple[pd.DataFrame, CleanReport, WatchReport]
            Cleaned data, cleaning report, and compatibility report.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._contract is None:
            raise RuntimeError(
                "puredata: call MendPipeline.fit(reference_data) before .run()"
            )
        clean_df, clean_report = self._cleaner.clean(data, target_col=self._target_col)
        watch_report = self._watcher.check(clean_df, self._contract)
        return clean_df, clean_report, watch_report

    # ------------------------------------------------------------------
    # sklearn-compatible transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one call (sklearn API compatibility).

        Parameters
        ----------
        X:
            Training data.
        y:
            Ignored (kept for sklearn compatibility).

        Returns
        -------
        pd.DataFrame
            Cleaned training data.
        """
        self.fit(X)
        clean_df, _, _ = self.run(X)
        return clean_df

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Transform *X* using the fitted pipeline (sklearn API compatibility).

        Parameters
        ----------
        X:
            New data.
        y:
            Ignored.

        Returns
        -------
        pd.DataFrame
        """
        clean_df, _, _ = self.run(X)
        return clean_df

    # ------------------------------------------------------------------
    # Contract persistence
    # ------------------------------------------------------------------

    def save_contract(self, path: Union[str, Path]) -> None:
        """Save the fitted contract to disk.

        Parameters
        ----------
        path:
            Destination JSON file path.
        """
        if self._contract is None:
            raise RuntimeError("puredata: no contract to save — call .fit() first")
        self._contract.save(path)

    def load_contract(self, path: Union[str, Path]) -> "MendPipeline":
        """Load a previously saved contract from disk.

        Parameters
        ----------
        path:
            Source JSON file path.

        Returns
        -------
        MendPipeline
            Self, for chaining.
        """
        self._contract = DataContract.load(path)
        return self

    @property
    def contract(self) -> Optional[DataContract]:
        """The fitted :class:`~puredata.core.watch.DataContract`, or ``None``."""
        return self._contract
