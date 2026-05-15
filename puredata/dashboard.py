"""Live interactive HTML dataset health dashboard."""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from puredata.core.report import CleanReport, WatchReport


def dashboard(
    df: Union[pd.DataFrame, Any],
    clean_report: Optional[CleanReport] = None,
    watch_report: Optional[WatchReport] = None,
    open_browser: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a live interactive HTML health dashboard for *df*.

    Parameters
    ----------
    df:
        DataFrame to profile.
    clean_report:
        Optional :class:`~puredata.core.report.CleanReport` to include.
    watch_report:
        Optional :class:`~puredata.core.report.WatchReport` to include.
    open_browser:
        Whether to open the dashboard in the default browser automatically.
    output_path:
        If provided, save the HTML to this path.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    if not isinstance(df, pd.DataFrame):
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
        except ImportError:
            pass
    if not isinstance(df, pd.DataFrame):
        raise TypeError("puredata.dashboard: pass a pandas or polars DataFrame")

    html = _build_dashboard_html(df, clean_report, watch_report)

    if output_path:
        dest = Path(output_path)
    else:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="puredata_dashboard_"
        )
        dest = Path(tmp.name)

    dest.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open(dest.as_uri())

    return str(dest)


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _build_dashboard_html(
    df: pd.DataFrame,
    clean_report: Optional[CleanReport],
    watch_report: Optional[WatchReport],
) -> str:
    profile = _profile_dataframe(df)
    col_data_json = json.dumps(profile["columns"], default=str)
    mend_score = clean_report.mend_score if clean_report else _compute_raw_score(df)
    compat_score = watch_report.compatibility_score if watch_report else None

    fixes_html = ""
    if clean_report and clean_report.fixes:
        for fix in clean_report.fixes:
            fixes_html += (
                f"<tr><td>{fix.column}</td>"
                f"<td><span class='badge'>{fix.action.value}</span></td>"
                f"<td>{len(fix.rows)}</td>"
                f"<td>{fix.details}</td></tr>"
            )

    watch_html = ""
    if watch_report:
        for chk in watch_report.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(chk.status.value, "?")
            cls = {"pass": "green", "warn": "orange", "fail": "red"}.get(chk.status.value, "")
            watch_html += (
                f"<tr class='{cls}'><td>{chk.column}</td>"
                f"<td>{chk.name}</td>"
                f"<td>{icon} {chk.status.value}</td>"
                f"<td>{chk.message}</td></tr>"
            )

    compat_block = ""
    if compat_score is not None:
        compat_block = f"""
        <div class="stat">
          <div class="val" id="compat-score">{compat_score}</div>
          <div class="lbl">Compatibility Score</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>puredata Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#0f172a;color:#e2e8f0;min-height:100vh}}
  header{{background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:32px 48px;
          border-bottom:1px solid #1e293b}}
  header h1{{font-size:2rem;color:#38bdf8;font-weight:800;letter-spacing:-.02em}}
  header p{{color:#94a3b8;margin-top:6px}}
  .main{{max-width:1200px;margin:0 auto;padding:32px 24px}}
  .stat-row{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:32px}}
  .stat{{background:#1e293b;border-radius:12px;padding:20px 28px;flex:1;
         min-width:150px;text-align:center;border:1px solid #334155}}
  .stat .val{{font-size:2.4rem;font-weight:700;color:#38bdf8}}
  .stat .lbl{{font-size:0.8rem;color:#64748b;margin-top:4px;text-transform:uppercase;
              letter-spacing:.06em}}
  .score-ring{{position:relative;display:inline-flex;align-items:center;
               justify-content:center}}
  .card{{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:24px;
         border:1px solid #334155}}
  .card h2{{color:#94a3b8;font-size:0.85rem;text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-size:0.88rem}}
  th{{background:#0f172a;color:#64748b;padding:10px 12px;text-align:left;
      font-size:0.75rem;text-transform:uppercase;letter-spacing:.05em}}
  td{{padding:9px 12px;border-bottom:1px solid #0f172a}}
  tr:hover td{{background:rgba(255,255,255,.03)}}
  tr.green td{{color:#4ade80}}
  tr.orange td{{color:#fb923c}}
  tr.red td{{color:#f87171}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.72rem;
           font-weight:600;background:#0f172a;color:#94a3b8}}
  .bar-wrap{{background:#0f172a;border-radius:999px;height:8px;overflow:hidden;
             margin:4px 0}}
  .bar-fill{{height:100%;background:linear-gradient(90deg,#38bdf8,#818cf8);
              border-radius:999px}}
  .null-fill{{background:#f87171}}
  .col-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
             gap:16px}}
  .col-card{{background:#0f172a;border-radius:8px;padding:16px;border:1px solid #1e293b}}
  .col-name{{font-weight:600;color:#e2e8f0;margin-bottom:8px;font-size:0.9rem}}
  .col-meta{{font-size:0.78rem;color:#64748b;display:flex;justify-content:space-between}}
  .tabs{{display:flex;gap:4px;margin-bottom:24px;background:#0f172a;
         border-radius:8px;padding:4px;width:fit-content}}
  .tab{{padding:8px 20px;border-radius:6px;cursor:pointer;font-size:0.85rem;
        color:#64748b;transition:all .15s}}
  .tab.active{{background:#1e293b;color:#e2e8f0}}
  #histogram-panel{{display:none}}
</style>
</head>
<body>
<header>
  <h1>puredata</h1>
  <p>Dataset Health Dashboard — {df.shape[0]:,} rows × {df.shape[1]} columns</p>
</header>
<div class="main">
  <div class="stat-row">
    <div class="stat">
      <div class="val">{df.shape[0]:,}</div>
      <div class="lbl">Rows</div>
    </div>
    <div class="stat">
      <div class="val">{df.shape[1]}</div>
      <div class="lbl">Columns</div>
    </div>
    <div class="stat">
      <div class="val" style="color:{'#4ade80' if mend_score>=80 else '#fb923c' if mend_score>=50 else '#f87171'}">{mend_score}</div>
      <div class="lbl">MendScore</div>
    </div>
    <div class="stat">
      <div class="val">{df.isnull().mean().mean():.1%}</div>
      <div class="lbl">Avg Null Rate</div>
    </div>
    <div class="stat">
      <div class="val">{df.duplicated().sum():,}</div>
      <div class="lbl">Duplicates</div>
    </div>
    {compat_block}
  </div>

  <div class="card">
    <h2>Column Profiles</h2>
    <div class="col-grid" id="col-grid"></div>
  </div>

  {'<div class="card"><h2>AutoClean Fixes</h2><table><thead><tr><th>Column</th><th>Fix Type</th><th>Rows Affected</th><th>Details</th></tr></thead><tbody>' + fixes_html + '</tbody></table></div>' if fixes_html else ''}

  {'<div class="card"><h2>DataWatch Results</h2><table><thead><tr><th>Column</th><th>Check</th><th>Status</th><th>Message</th></tr></thead><tbody>' + watch_html + '</tbody></table></div>' if watch_html else ''}

</div>
<script>
const colData = {col_data_json};
const grid = document.getElementById('col-grid');
colData.forEach(col => {{
  const nullPct = (col.null_rate * 100).toFixed(1);
  const card = document.createElement('div');
  card.className = 'col-card';
  let extra = '';
  if (col.is_numeric && col.min !== null) {{
    extra = `<div class="col-meta"><span>min: ${{col.min?.toFixed?.(3) ?? col.min}}</span><span>max: ${{col.max?.toFixed?.(3) ?? col.max}}</span><span>mean: ${{col.mean?.toFixed?.(3) ?? col.mean}}</span></div>`;
  }} else if (col.n_unique !== null) {{
    extra = `<div class="col-meta"><span>${{col.n_unique}} unique values</span><span>${{col.dtype}}</span></div>`;
  }}
  card.innerHTML = `
    <div class="col-name">${{col.name}}</div>
    <div class="col-meta"><span>${{col.dtype}}</span><span>${{nullPct}}% null</span></div>
    <div class="bar-wrap"><div class="bar-fill null-fill" style="width:${{nullPct}}%"></div></div>
    ${{extra}}`;
  grid.appendChild(card);
}});
</script>
</body>
</html>"""


def _profile_dataframe(df: pd.DataFrame) -> dict:
    columns = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        non_null = series.dropna()
        col_info: dict = {
            "name": col,
            "dtype": str(series.dtype),
            "null_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique()),
            "is_numeric": is_numeric,
            "min": None,
            "max": None,
            "mean": None,
        }
        if is_numeric and len(non_null) > 0:
            col_info["min"] = float(non_null.min())
            col_info["max"] = float(non_null.max())
            col_info["mean"] = float(non_null.mean())
        columns.append(col_info)
    return {"columns": columns}


def _compute_raw_score(df: pd.DataFrame) -> float:
    """Rough MendScore without running AutoClean."""
    null_penalty = float(df.isnull().mean().mean())
    dup_penalty = df.duplicated().sum() / max(len(df), 1)
    score = max(0.0, 1.0 - null_penalty - dup_penalty) * 100
    return round(score, 1)
