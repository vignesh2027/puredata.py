"""Report generation for AutoClean and DataWatch."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class FixAction(str, Enum):
    """Category of a cleaning fix."""
    ENCODING = "encoding"
    WHITESPACE = "whitespace"
    TYPE_COERCE = "type_coerce"
    DATE_NORMALISE = "date_normalise"
    DUPLICATE_REMOVE = "duplicate_remove"
    CATEGORY_NORMALISE = "category_normalise"
    UNIT_NORMALISE = "unit_normalise"
    NULL_IMPUTE = "null_impute"
    OUTLIER = "outlier"


class CheckStatus(str, Enum):
    """Result status of a DataWatch check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class Fix:
    """A single data cleaning action.

    Attributes:
        column: Column name affected.
        action: Category of fix applied.
        rows: List of row indices changed.
        details: Human-readable description of the fix.
        timestamp: When the fix was applied.
    """
    column: str
    action: FixAction
    rows: List[int]
    details: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "column": self.column,
            "action": self.action.value,
            "n_rows_affected": len(self.rows),
            "rows": self.rows[:100],  # cap for serialisation
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CleanReport:
    """Complete repair report produced by :func:`puredata.clean`.

    Attributes:
        fixes: Ordered list of all :class:`Fix` objects applied.
        original_shape: ``(rows, cols)`` before cleaning.
        cleaned_shape: ``(rows, cols)`` after cleaning.
        started_at: UTC time cleaning began.
        ended_at: UTC time cleaning finished.
    """
    fixes: List[Fix] = field(default_factory=list)
    original_shape: tuple = (0, 0)
    cleaned_shape: tuple = (0, 0)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    def add_fix(self, fix: Fix) -> None:
        """Append a fix to the report."""
        self.fixes.append(fix)

    @property
    def duration_seconds(self) -> float:
        """Elapsed cleaning time in seconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return 0.0

    @property
    def mend_score(self) -> float:
        """MendScore: overall dataset health (0–100) before cleaning.

        A score of 100 means no fixes were needed. Each fix category
        costs points proportionally to cells affected.
        """
        if self.original_shape[0] == 0 or self.original_shape[1] == 0:
            return 100.0
        total_cells = self.original_shape[0] * self.original_shape[1]
        affected_cells = sum(max(1, len(f.rows)) for f in self.fixes)
        raw = 1.0 - min(affected_cells / total_cells, 1.0)
        return round(raw * 100, 1)

    def summary(self) -> str:
        """Return a compact human-readable summary of all fixes."""
        lines: List[str] = [
            "╔══════════════════════════════════════════════╗",
            "║          puredata AutoClean Report           ║",
            "╚══════════════════════════════════════════════╝",
            f"  Original shape : {self.original_shape[0]} rows × {self.original_shape[1]} cols",
            f"  Cleaned shape  : {self.cleaned_shape[0]} rows × {self.cleaned_shape[1]} cols",
            f"  MendScore      : {self.mend_score}/100",
            f"  Duration       : {self.duration_seconds:.2f}s",
            f"  Total fixes    : {len(self.fixes)}",
            "",
        ]
        if not self.fixes:
            lines.append("  ✓ No issues detected — data is already clean.")
        else:
            for fix in self.fixes:
                icon = "✓" if fix.action == FixAction.DUPLICATE_REMOVE else "⚑"
                lines.append(f"  {icon} [{fix.column}] {fix.action.value}: {fix.details}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full report to a dictionary."""
        return {
            "original_shape": list(self.original_shape),
            "cleaned_shape": list(self.cleaned_shape),
            "mend_score": self.mend_score,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "fixes": [f.to_dict() for f in self.fixes],
        }

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export report as JSON.

        Parameters
        ----------
        path:
            If provided, writes the JSON to this file path.

        Returns
        -------
        str
            JSON string.
        """
        data = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(data, encoding="utf-8")
        return data

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Export fixes as a CSV-ready DataFrame.

        Parameters
        ----------
        path:
            If provided, writes CSV to this file path.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per fix.
        """
        rows = [f.to_dict() for f in self.fixes]
        result_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["column", "action", "n_rows_affected", "rows", "details", "timestamp"]
        )
        if path:
            result_df.to_csv(path, index=False)
        return result_df

    def to_html(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export report as a self-contained HTML page.

        Parameters
        ----------
        path:
            If provided, writes HTML to this file path.

        Returns
        -------
        str
            HTML string.
        """
        html = _render_clean_html(self)
        if path:
            Path(path).write_text(html, encoding="utf-8")
        return html


# ---------------------------------------------------------------------------
# DataWatch report types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single DataWatch check.

    Attributes:
        name: Check identifier.
        column: Column this check applies to (``"<schema>"`` for schema checks).
        status: :class:`CheckStatus` outcome.
        message: Human-readable description of the outcome.
        details: Machine-readable payload for pipeline integration.
    """
    name: str
    column: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "column": self.column,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class WatchReport:
    """Complete compatibility report produced by :func:`puredata.check`.

    Attributes:
        checks: All :class:`CheckResult` objects from the validation run.
        reference_shape: Shape of the training/reference data.
        new_shape: Shape of the new data being validated.
        checked_at: UTC time validation ran.
        mode: DataWatch mode used (``"warn"``, ``"strict"``, ``"silent"``).
    """
    checks: List[CheckResult] = field(default_factory=list)
    reference_shape: tuple = (0, 0)
    new_shape: tuple = (0, 0)
    checked_at: Optional[datetime] = None
    mode: str = "warn"

    def add(self, result: CheckResult) -> None:
        """Append a check result."""
        self.checks.append(result)

    @property
    def passed(self) -> bool:
        """True when every check passed or warned (none failed)."""
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def n_warned(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def compatibility_score(self) -> float:
        """Compatibility score from 0–100 (100 = fully compatible)."""
        if not self.checks:
            return 100.0
        passed = self.n_passed + self.n_warned * 0.5
        return round(passed / len(self.checks) * 100, 1)

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        lines: List[str] = [
            "╔══════════════════════════════════════════════╗",
            "║         puredata DataWatch Report            ║",
            "╚══════════════════════════════════════════════╝",
            f"  Reference shape     : {self.reference_shape[0]} × {self.reference_shape[1]}",
            f"  New data shape      : {self.new_shape[0]} × {self.new_shape[1]}",
            f"  Compatibility score : {self.compatibility_score}/100",
            f"  Checks: {self.n_passed} ✓  {self.n_warned} ⚠  {self.n_failed} ✗",
            "",
        ]
        for chk in self.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "–"}.get(
                chk.status.value, "?"
            )
            lines.append(f"  {icon} [{chk.column}] {chk.name}: {chk.message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "compatibility_score": self.compatibility_score,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_warned": self.n_warned,
            "reference_shape": list(self.reference_shape),
            "new_shape": list(self.new_shape),
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "mode": self.mode,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export to JSON."""
        data = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(data, encoding="utf-8")
        return data

    def to_html(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export to HTML report."""
        html = _render_watch_html(self)
        if path:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def raise_if_failed(self) -> None:
        """Raise :exc:`DataCompatibilityError` if any checks failed."""
        if self.n_failed > 0:
            failed = [c for c in self.checks if c.status == CheckStatus.FAIL]
            msgs = "\n".join(f"  • [{c.column}] {c.name}: {c.message}" for c in failed)
            raise DataCompatibilityError(
                f"puredata DataWatch found {self.n_failed} compatibility failure(s):\n{msgs}"
            )


class DataCompatibilityError(Exception):
    """Raised by DataWatch in strict mode when checks fail."""


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_BASE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         margin: 0; background: #0f172a; color: #e2e8f0; }}
  .container {{ max-width: 960px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #38bdf8; font-size: 2rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat {{ background: #0f172a; border-radius: 8px; padding: 16px 24px; flex: 1;
           min-width: 140px; text-align: center; }}
  .stat .val {{ font-size: 2rem; font-weight: 700; color: #38bdf8; }}
  .stat .lbl {{ font-size: 0.8rem; color: #64748b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #0f172a; color: #94a3b8; padding: 12px; text-align: left;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; font-size: 0.9rem; }}
  tr:hover td {{ background: #0f172a; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; }}
  .badge-pass {{ background: #052e16; color: #4ade80; }}
  .badge-fail {{ background: #450a0a; color: #f87171; }}
  .badge-warn {{ background: #431407; color: #fb923c; }}
  .badge-encoding {{ background: #1e1b4b; color: #a5b4fc; }}
  .badge-outlier {{ background: #450a0a; color: #fca5a5; }}
  .badge-null_impute {{ background: #052e16; color: #86efac; }}
  .badge-default {{ background: #1e293b; color: #94a3b8; }}
  .score-bar {{ background: #0f172a; border-radius: 999px; height: 12px; overflow: hidden; }}
  .score-fill {{ height: 100%; background: linear-gradient(90deg,#38bdf8,#818cf8);
                 border-radius: 999px; transition: width .4s; }}
  footer {{ text-align: center; color: #334155; font-size: 0.75rem; margin-top: 40px; }}
</style>
</head>
<body>
<div class="container">
  <h1>puredata</h1>
  <p class="subtitle">{subtitle}</p>
  {body}
  <footer>Generated by puredata · {ts}</footer>
</div>
</body>
</html>"""


def _render_clean_html(report: CleanReport) -> str:
    score = report.mend_score
    score_color = "#4ade80" if score >= 80 else "#fb923c" if score >= 50 else "#f87171"
    rows_html = ""
    for fix in report.fixes:
        badge_class = f"badge-{fix.action.value}" if f"badge-{fix.action.value}" else "badge-default"
        rows_html += (
            f"<tr><td>{fix.column}</td>"
            f"<td><span class='badge {badge_class}'>{fix.action.value}</span></td>"
            f"<td>{len(fix.rows)}</td>"
            f"<td>{fix.details}</td></tr>"
        )
    body = f"""
    <div class="stat-row">
      <div class="stat"><div class="val">{report.original_shape[0]}</div><div class="lbl">Original Rows</div></div>
      <div class="stat"><div class="val">{report.cleaned_shape[0]}</div><div class="lbl">Clean Rows</div></div>
      <div class="stat"><div class="val">{len(report.fixes)}</div><div class="lbl">Fixes Applied</div></div>
      <div class="stat"><div class="val" style="color:{score_color}">{score}</div><div class="lbl">MendScore</div></div>
    </div>
    <div class="card">
      <div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Column</th><th>Fix Type</th><th>Rows Affected</th><th>Details</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    return _HTML_BASE.format(
        title="puredata AutoClean Report",
        subtitle="AutoClean Repair Report",
        body=body,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def _render_watch_html(report: WatchReport) -> str:
    score = report.compatibility_score
    score_color = "#4ade80" if score >= 80 else "#fb923c" if score >= 50 else "#f87171"
    rows_html = ""
    for chk in report.checks:
        badge_class = f"badge-{chk.status.value}"
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "–"}.get(chk.status.value, "?")
        rows_html += (
            f"<tr><td>{chk.column}</td>"
            f"<td>{chk.name}</td>"
            f"<td><span class='badge {badge_class}'>{icon} {chk.status.value}</span></td>"
            f"<td>{chk.message}</td></tr>"
        )
    body = f"""
    <div class="stat-row">
      <div class="stat"><div class="val">{report.n_passed}</div><div class="lbl">Passed</div></div>
      <div class="stat"><div class="val" style="color:#fb923c">{report.n_warned}</div><div class="lbl">Warnings</div></div>
      <div class="stat"><div class="val" style="color:#f87171">{report.n_failed}</div><div class="lbl">Failed</div></div>
      <div class="stat"><div class="val" style="color:{score_color}">{score}</div><div class="lbl">Compat Score</div></div>
    </div>
    <div class="card">
      <div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Column</th><th>Check</th><th>Status</th><th>Message</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    return _HTML_BASE.format(
        title="puredata DataWatch Report",
        subtitle="DataWatch Compatibility Report",
        body=body,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
