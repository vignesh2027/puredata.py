# Advanced Tutorial

## Custom AutoClean configuration

Override defaults with `AutoCleanConfig`:

```python
from puredata import AutoClean, AutoCleanConfig

config = AutoCleanConfig(
    fix_nulls=True,
    fix_outliers=True,
    outlier_action="clip",       # "clip", "remove", or "nan"
    outlier_threshold=0.5,       # fraction of methods that must agree
    fix_categories=True,
    fix_dates=True,
    date_output_format="%d/%m/%Y",  # custom output format
    n_neighbors=10,              # KNN imputation neighbours
    n_jobs=-1,                   # use all CPU cores
)

clean_df, report = puredata.clean(dirty_df, config=config, target_col="label")
```

## Protecting your target column

Pass `target_col` to prevent the label column from being modified:

```python
clean_df, report = puredata.clean(df, target_col="churn")
```

## Custom business rules in DataWatch

```python
contract = puredata.watch(train_df)

# Revenue must always be positive
contract.add_rule(
    lambda df: None if (df["revenue"] >= 0).all() else "Negative revenue detected",
    name="revenue_positive",
)

# Age must be between 0 and 120
contract.add_rule(
    lambda df: None if df["age"].between(0, 120).all() else f"Ages out of range: {df['age'][~df['age'].between(0,120)].tolist()}",
    name="age_valid_range",
)

result = puredata.check(prod_df, contract)
```

## MendPipeline for production workflows

```python
from puredata import MendPipeline, AutoCleanConfig

pipeline = MendPipeline(
    clean_config=AutoCleanConfig(outlier_action="clip"),
    watch_mode="strict",   # raises exception if production data fails
    target_col="churn",
)

pipeline.fit(train_df)
pipeline.save_contract("production_contract.json")

# Later, in production:
new_pipeline = MendPipeline()
new_pipeline.load_contract("production_contract.json")

clean_df, clean_report, watch_report = new_pipeline.run(incoming_df)
print(f"MendScore: {clean_report.mend_score}")
print(f"Compatibility: {watch_report.compatibility_score}")
```

## sklearn Pipeline integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

ml_pipeline = Pipeline([
    ("puredata", puredata.MendPipeline()),
    ("scaler",   StandardScaler()),
    ("model",    GradientBoostingClassifier()),
])

ml_pipeline.fit(X_train, y_train)
predictions = ml_pipeline.predict(X_test)
```

## Async support for large datasets

```python
import asyncio
import puredata

async def clean_large_dataset(path):
    loop = asyncio.get_event_loop()
    clean_df, report = await loop.run_in_executor(
        None, puredata.clean, path
    )
    return clean_df, report

clean_df, report = asyncio.run(clean_large_dataset("huge_dataset.parquet"))
```

## MLflow integration

```python
import mlflow
import puredata
from puredata.integrations.mlflow import log_clean_report, log_watch_report

with mlflow.start_run():
    clean_df, clean_report = puredata.clean(raw_df)
    log_clean_report(clean_report)

    contract = puredata.watch(clean_df)
    result = puredata.check(new_df, contract)
    log_watch_report(result)
```

## Performance tips

- For datasets under 1M rows: use defaults.
- For 1–100M rows: pass `n_jobs=-1` in `AutoCleanConfig` to use all cores.
- For CSV/Parquet files: pass the file path directly — puredata reads in chunks automatically.
- For repeated validation: serialise your contract once with `contract.save()` and load it with `DataContract.load()`.
