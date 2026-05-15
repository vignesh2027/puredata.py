# Plugin Development Guide

puredata has a first-class plugin system. You can add custom cleaning strategies, validators, and drift detectors as standalone Python packages.

## CleanerPlugin

```python
from puredata.plugins import CleanerPlugin, register_cleaner
from puredata.core.report import Fix, FixAction

@register_cleaner
class PhoneNormaliser(CleanerPlugin):
    name = "phone_normaliser"
    description = "Normalise phone numbers to E.164 format"
    version = "1.0.0"

    def clean(self, df, report):
        import re
        for col in df.select_dtypes(include="object").columns:
            if "phone" in col.lower():
                original = df[col].copy()
                df[col] = df[col].str.replace(r"[^\d+]", "", regex=True)
                changed = df.index[df[col] != original].tolist()
                if changed:
                    report.add_fix(Fix(
                        column=col,
                        action=FixAction.ENCODING,  # closest category
                        rows=changed,
                        details=f"Normalised {len(changed)} phone numbers",
                    ))
        return df, report
```

## ValidatorPlugin

```python
from puredata.plugins import ValidatorPlugin, register_validator
from puredata.core.report import CheckResult, CheckStatus

@register_validator
class RevenueRule(ValidatorPlugin):
    name = "revenue_positive"
    description = "Revenue must be non-negative"

    def validate(self, df, contract, report):
        if "revenue" not in df.columns:
            return report
        negatives = df[df["revenue"] < 0]
        if len(negatives) > 0:
            report.add(CheckResult(
                name="revenue_positive",
                column="revenue",
                status=CheckStatus.FAIL,
                message=f"{len(negatives)} rows have negative revenue",
                details={"rows": negatives.index.tolist()[:50]},
            ))
        return report
```

## DriftDetectorPlugin

```python
from puredata.plugins import DriftDetectorPlugin, register_drift_detector

@register_drift_detector
class WassersteinDetector(DriftDetectorPlugin):
    name = "wasserstein"
    description = "Wasserstein distance drift detection"

    def detect(self, reference, new_data):
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(reference.dropna(), new_data.dropna()))
```

## Distributing as a package

Add an entry point in your `pyproject.toml`:

```toml
[project.entry-points."puredata.plugins"]
my_company_plugins = "my_package.puredata_plugin:register"
```

Where `register` is a function that accepts the registry:

```python
# my_package/puredata_plugin.py
def register(registry):
    registry.register_cleaner(PhoneNormaliser)
    registry.register_validator(RevenueRule)
```

puredata will auto-discover and load your plugins when imported.
