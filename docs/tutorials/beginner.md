# Beginner Tutorial

This tutorial walks you through the two core features of puredata from zero.

## Setup

```bash
pip install puredata
```

## Step 1: Clean dirty data

Create a messy DataFrame:

```python
import pandas as pd
import numpy as np
import puredata

df = pd.DataFrame({
    "age":    [25, np.nan, 30, 9999, 28, np.nan],
    "gender": ["Male", "male", "M", "Female", "FEMALE", "female"],
    "score":  ["85.5", "90.0", "75.5", "88.0", "92.5", "70.0"],
    "date":   ["2020-01-15", "15/01/2020", "January 15, 2020", "2020-01-15", "01-15-2020", "2020-01-15"],
    "name":   ["  Alice  ", "Bob  ", "  Carol", "dave", "Eve", "frank"],
})
```

Clean it in one line:

```python
clean_df, report = puredata.clean(df)
print(report.summary())
```

The report tells you exactly what was fixed:

```
╔══════════════════════════════════════════════╗
║          puredata AutoClean Report           ║
╚══════════════════════════════════════════════╝
  Original shape : 6 rows × 5 cols
  Cleaned shape  : 6 rows × 5 cols
  MendScore      : 62.5/100
  Duration       : 0.43s
  Total fixes    : 5

  ⚑ [gender] category_normalise: Normalised categories: 'male'→'Male', 'M'→'Male'...
  ⚑ [score] type_coerce: Converted numeric strings to float64
  ⚑ [date] date_normalise: Normalised 4 dates to %Y-%m-%d
  ⚑ [name] whitespace: Stripped/normalised whitespace in 3 cells
  ⚑ [age] null_impute: KNN-imputed 2 nulls (33.3% missing)
```

## Step 2: Export reports

```python
report.to_html("report.html")   # open in browser
report.to_json("report.json")   # for pipelines
report.to_csv("report.csv")     # one row per fix
```

## Step 3: Check your MendScore

```python
score = puredata.score(df)
print(f"Dataset health: {score}/100")
```

## Step 4: Validate production data

Fit a contract on your training data:

```python
contract = puredata.watch(train_df)
contract.save("contract.json")   # save for reuse
```

Validate new data against it:

```python
result = puredata.check(prod_df, contract)
print(result.summary())

if not result.passed:
    print("WARNING: Production data has compatibility issues!")
```

## Step 5: Open the dashboard

```python
puredata.dashboard(df)   # opens in your browser
```

## Next steps

See the [Advanced Tutorial](advanced.md) for custom rules, pipeline integration, and performance tuning.
