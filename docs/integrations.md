# Integrations Guide

## MLflow

Track data quality alongside model experiments automatically.

```bash
pip install "puredata[mlflow]"
```

```python
import mlflow
import puredata
from puredata.integrations.mlflow import log_clean_report, log_watch_report

with mlflow.start_run():
    clean_df, clean_report = puredata.clean(raw_df)
    log_clean_report(clean_report)
    # Logs: mend_score, n_fixes, fix counts per type, JSON artifact

    contract = puredata.watch(clean_df)
    result = puredata.check(new_df, contract)
    log_watch_report(result)
    # Logs: compatibility_score, n_passed/warned/failed, per-check status
```

## Weights & Biases

```bash
pip install "puredata[wandb]"
```

```python
import wandb
import puredata
from puredata.integrations.wandb import log_clean_report, log_watch_report

wandb.init(project="my-ml-project")

clean_df, report = puredata.clean(raw_df)
log_clean_report(report)

contract = puredata.watch(clean_df)
result = puredata.check(prod_df, contract)
log_watch_report(result)
```

## DVC

Track data quality metrics alongside data versioning.

```bash
pip install "puredata[dvc]"
```

```python
from puredata.integrations.dvc import log_clean_report, log_watch_report

clean_df, report = puredata.clean(raw_df)
log_clean_report(report, "metrics/clean_metrics.json")

result = puredata.check(prod_df, contract)
log_watch_report(result, "metrics/watch_metrics.json")
```

Then add to `dvc.yaml`:

```yaml
metrics:
  - metrics/clean_metrics.json
  - metrics/watch_metrics.json
```

## Polars

Pass polars DataFrames directly — puredata converts automatically:

```bash
pip install "puredata[polars]"
```

```python
import polars as pl
import puredata

df = pl.read_csv("data.csv")
clean_df, report = puredata.clean(df)    # clean_df is a pandas DataFrame
contract = puredata.watch(df)
result = puredata.check(new_df, contract)
```
