# puredata

**Automatic data cleaning and silent incompatibility detection.**

Two problems. Solved perfectly.

## Quick start

```bash
pip install puredata
```

```python
import puredata

# Clean dirty data automatically
clean_df, report = puredata.clean(dirty_df)
print(report.summary())

# Catch silent incompatibility between training and production
contract = puredata.watch(train_df)
result   = puredata.check(prod_df, contract)
print(result.summary())
```

## The two problems

**Problem 1 — Dirty data.** Every project wastes 60–80% of time cleaning data manually. Nulls, outliers, wrong types, duplicates, encoding issues, inconsistent categories, bad dates, mixed units. puredata fixes all of them in one line automatically.

**Problem 2 — Silent data incompatibility.** When you clean data in training and then pass new data to your model, the new data is always slightly different. Your model gives wrong predictions silently. puredata catches every difference and explains exactly what is wrong.

See the [Beginner Tutorial](tutorials/beginner.md) to get started.
