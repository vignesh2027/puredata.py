# Changelog

All notable changes to puredata are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
puredata uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-15

### Added

**AutoClean (Pillar 1)**
- Automatic null imputation with context-aware strategy selection (KNN, Iterative, mode, unknown-category, ffill/bfill)
- Ensemble outlier detection using IQR + Z-score + Isolation Forest + LOF with voting
- Outlier actions: clip, remove, or nan
- Type coercion: numeric strings → float64, date strings → datetime
- Duplicate row removal (exact duplicates)
- Encoding repair: BOM, zero-width spaces, invisible Unicode, NFC normalisation
- Fuzzy category normalisation using rapidfuzz (Male/male/M/MALE → Male)
- Date format normalisation across 200+ formats to ISO 8601 or custom format
- Whitespace stripping and normalisation
- Unit detection and SI normalisation (kg/lbs/g → kg, km/miles/m → km)
- Full per-fix repair report with column, rows, original value, new value, reason
- MendScore (0–100) overall dataset health score
- Report export: HTML, JSON, CSV

**DataWatch (Pillar 2)**
- DataContract: statistical profile fitted on reference data
- Schema violation detection: missing columns, extra columns, type changes
- Range violation detection with configurable tolerance
- Null rate spike detection with configurable threshold
- New category value detection with cardinality tolerance
- Distribution drift detection: KS test + PSI + Jensen–Shannon divergence simultaneously
- Custom business rule API with chaining
- Three validation modes: warn, strict, silent
- Compatibility score (0–100)
- Contract persistence: save/load as JSON
- WatchReport export: HTML, JSON

**MendPipeline**
- Chains AutoClean + DataWatch in a single reusable pipeline
- sklearn API compatibility: fit_transform, transform
- Contract save/load for production deployment

**Dashboard**
- Self-contained HTML dashboard with column profiles, null rates, fix summary, DataWatch results

**CLI**
- `puredata clean` — clean any file
- `puredata watch` — fit a contract
- `puredata check` — validate new data
- `puredata dashboard` — open the dashboard
- `puredata score` — print MendScore

**Plugin system**
- CleanerPlugin, ValidatorPlugin, DriftDetectorPlugin base classes
- PluginRegistry with auto-discovery via entry points
- @register_cleaner, @register_validator, @register_drift_detector decorators

**Integrations**
- MLflow: log_clean_report, log_watch_report
- Weights & Biases: log_clean_report, log_watch_report
- DVC: JSON metrics export

**Input support**
- pandas DataFrame
- polars DataFrame (optional dependency)
- numpy ndarray
- CSV, Excel, Parquet, JSON file paths

---

[0.1.0]: https://github.com/vignesh2027/puredata.py/releases/tag/v0.1.0
