"""Plugin base classes and registry for puredata extensions."""

from __future__ import annotations

import importlib
import importlib.metadata
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd

from puredata.core.report import CleanReport, WatchReport


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class CleanerPlugin(ABC):
    """Base class for custom data cleaning plugins.

    Subclass this and implement :meth:`clean` to add a custom cleaning
    strategy to the puredata AutoClean pipeline.

    Examples
    --------
    >>> class PhoneNormaliser(CleanerPlugin):
    ...     name = "phone_normaliser"
    ...     description = "Normalise phone numbers to E.164 format"
    ...
    ...     def clean(self, df, report):
    ...         # your logic here
    ...         return df, report
    ...
    >>> from puredata.plugins import register_cleaner
    >>> register_cleaner(PhoneNormaliser)
    """

    name: str = ""
    description: str = ""
    version: str = "0.0.1"

    @abstractmethod
    def clean(
        self, df: pd.DataFrame, report: CleanReport
    ) -> Tuple[pd.DataFrame, CleanReport]:
        """Apply cleaning logic to *df*.

        Parameters
        ----------
        df:
            Input DataFrame.
        report:
            The in-progress :class:`~puredata.core.report.CleanReport`.

        Returns
        -------
        tuple[pd.DataFrame, CleanReport]
            Modified DataFrame and updated report.
        """
        ...

    def __repr__(self) -> str:
        return f"<CleanerPlugin name={self.name!r} version={self.version!r}>"


class ValidatorPlugin(ABC):
    """Base class for custom DataWatch validation plugins.

    Examples
    --------
    >>> class RevenueRule(ValidatorPlugin):
    ...     name = "revenue_positive"
    ...     description = "Revenue must be non-negative"
    ...
    ...     def validate(self, df, contract, report):
    ...         if (df["revenue"] < 0).any():
    ...             from puredata.core.report import CheckResult, CheckStatus
    ...             report.add(CheckResult("revenue_positive", "revenue",
    ...                                    CheckStatus.FAIL, "Negative revenue found"))
    ...         return report
    """

    name: str = ""
    description: str = ""
    version: str = "0.0.1"

    @abstractmethod
    def validate(
        self,
        df: pd.DataFrame,
        contract: Any,
        report: WatchReport,
    ) -> WatchReport:
        """Run a custom validation check.

        Parameters
        ----------
        df:
            New data being validated.
        contract:
            The fitted :class:`~puredata.core.watch.DataContract`.
        report:
            The in-progress :class:`~puredata.core.report.WatchReport`.

        Returns
        -------
        WatchReport
            Updated report.
        """
        ...


class DriftDetectorPlugin(ABC):
    """Base class for custom distribution drift detectors.

    Examples
    --------
    >>> class WassersteinDetector(DriftDetectorPlugin):
    ...     name = "wasserstein"
    ...     description = "Wasserstein distance drift detection"
    ...
    ...     def detect(self, reference, new_data):
    ...         from scipy.stats import wasserstein_distance
    ...         return float(wasserstein_distance(reference, new_data))
    """

    name: str = ""
    description: str = ""
    version: str = "0.0.1"

    @abstractmethod
    def detect(
        self,
        reference: pd.Series,
        new_data: pd.Series,
    ) -> float:
        """Compute a drift score between *reference* and *new_data*.

        Parameters
        ----------
        reference:
            Reference/training column values.
        new_data:
            New/production column values.

        Returns
        -------
        float
            Drift score (higher = more drift). Scale is plugin-defined.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Central registry for all puredata plugins.

    Use the module-level :func:`register_cleaner`, :func:`register_validator`,
    and :func:`register_drift_detector` helpers instead of instantiating this
    directly.
    """

    def __init__(self) -> None:
        self._cleaners: Dict[str, Type[CleanerPlugin]] = {}
        self._validators: Dict[str, Type[ValidatorPlugin]] = {}
        self._drift_detectors: Dict[str, Type[DriftDetectorPlugin]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_cleaner(self, cls: Type[CleanerPlugin]) -> Type[CleanerPlugin]:
        """Register a :class:`CleanerPlugin` class."""
        if not cls.name:
            raise ValueError(
                f"puredata plugin: {cls.__name__}.name must not be empty"
            )
        if cls.name in self._cleaners:
            raise ValueError(
                f"puredata plugin: cleaner '{cls.name}' is already registered"
            )
        self._cleaners[cls.name] = cls
        return cls

    def register_validator(self, cls: Type[ValidatorPlugin]) -> Type[ValidatorPlugin]:
        """Register a :class:`ValidatorPlugin` class."""
        if not cls.name:
            raise ValueError(
                f"puredata plugin: {cls.__name__}.name must not be empty"
            )
        self._validators[cls.name] = cls
        return cls

    def register_drift_detector(
        self, cls: Type[DriftDetectorPlugin]
    ) -> Type[DriftDetectorPlugin]:
        """Register a :class:`DriftDetectorPlugin` class."""
        if not cls.name:
            raise ValueError(
                f"puredata plugin: {cls.__name__}.name must not be empty"
            )
        self._drift_detectors[cls.name] = cls
        return cls

    # ------------------------------------------------------------------
    # Discovery via entry points
    # ------------------------------------------------------------------

    def discover(self) -> None:
        """Auto-discover plugins via ``puredata.plugins`` entry point group.

        Any installed package that declares an entry point under the group
        ``puredata.plugins`` will be loaded automatically.

        Example ``pyproject.toml`` for a plugin package::

            [project.entry-points."puredata.plugins"]
            my_plugin = "my_package.plugin:register"
        """
        try:
            eps = importlib.metadata.entry_points(group="puredata.plugins")
        except TypeError:
            eps = importlib.metadata.entry_points().get("puredata.plugins", [])
        for ep in eps:
            try:
                register_fn = ep.load()
                register_fn(self)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"puredata: failed to load plugin '{ep.name}': {exc}",
                    UserWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    @property
    def cleaners(self) -> Dict[str, Type[CleanerPlugin]]:
        """All registered cleaner plugins."""
        return dict(self._cleaners)

    @property
    def validators(self) -> Dict[str, Type[ValidatorPlugin]]:
        """All registered validator plugins."""
        return dict(self._validators)

    @property
    def drift_detectors(self) -> Dict[str, Type[DriftDetectorPlugin]]:
        """All registered drift detector plugins."""
        return dict(self._drift_detectors)

    def list_all(self) -> List[Dict[str, str]]:
        """Return a list of all registered plugins with metadata."""
        result = []
        for kind, store in [
            ("cleaner", self._cleaners),
            ("validator", self._validators),
            ("drift_detector", self._drift_detectors),
        ]:
            for name, cls in store.items():
                result.append({
                    "kind": kind,
                    "name": name,
                    "description": cls.description,
                    "version": cls.version,
                    "class": cls.__name__,
                })
        return result

    def __repr__(self) -> str:
        return (
            f"<PluginRegistry cleaners={len(self._cleaners)} "
            f"validators={len(self._validators)} "
            f"drift_detectors={len(self._drift_detectors)}>"
        )


# ---------------------------------------------------------------------------
# Module-level singleton and convenience decorators
# ---------------------------------------------------------------------------

registry = PluginRegistry()


def register_cleaner(cls: Type[CleanerPlugin]) -> Type[CleanerPlugin]:
    """Register a :class:`CleanerPlugin` in the global registry.

    Can be used as a class decorator.

    Parameters
    ----------
    cls:
        Subclass of :class:`CleanerPlugin`.

    Returns
    -------
    Type[CleanerPlugin]
        The class, unchanged.
    """
    return registry.register_cleaner(cls)


def register_validator(cls: Type[ValidatorPlugin]) -> Type[ValidatorPlugin]:
    """Register a :class:`ValidatorPlugin` in the global registry.

    Can be used as a class decorator.
    """
    return registry.register_validator(cls)


def register_drift_detector(
    cls: Type[DriftDetectorPlugin],
) -> Type[DriftDetectorPlugin]:
    """Register a :class:`DriftDetectorPlugin` in the global registry.

    Can be used as a class decorator.
    """
    return registry.register_drift_detector(cls)
