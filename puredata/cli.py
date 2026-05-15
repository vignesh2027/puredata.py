"""puredata CLI — clean, check, and dashboard from the terminal."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

app = typer.Typer(
    name="puredata",
    help="Automatic data cleaning and silent incompatibility detection.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _load_df(path: Path):
    import pandas as pd
    suffix = path.suffix.lower()
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }
    if suffix not in readers:
        console.print(f"[red]Error:[/] Unsupported file format '{suffix}'")
        raise typer.Exit(1)
    return readers[suffix](path)


# ---------------------------------------------------------------------------
# clean command
# ---------------------------------------------------------------------------

@app.command()
def clean(
    file: Path = typer.Argument(..., help="Path to input data file (CSV/Excel/Parquet/JSON)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save cleaned data here"),
    report_json: Optional[Path] = typer.Option(None, "--report-json", help="Save JSON report here"),
    report_html: Optional[Path] = typer.Option(None, "--report-html", help="Save HTML report here"),
    report_csv: Optional[Path] = typer.Option(None, "--report-csv", help="Save CSV report here"),
    no_nulls: bool = typer.Option(False, "--no-nulls", help="Skip null imputation"),
    no_outliers: bool = typer.Option(False, "--no-outliers", help="Skip outlier detection"),
    no_duplicates: bool = typer.Option(False, "--no-duplicates", help="Skip duplicate removal"),
    target_col: Optional[str] = typer.Option(None, "--target", "-t", help="Target/label column to protect"),
) -> None:
    """[bold cyan]Clean[/] any dirty dataset automatically.

    Examples:

      puredata clean mydata.csv
      puredata clean mydata.csv -o clean.csv --report-html report.html
    """
    if not file.exists():
        console.print(f"[red]Error:[/] File not found: {file}")
        raise typer.Exit(1)

    from puredata.core.clean import AutoClean, AutoCleanConfig

    config = AutoCleanConfig(
        fix_nulls=not no_nulls,
        fix_outliers=not no_outliers,
        fix_duplicates=not no_duplicates,
    )
    engine = AutoClean(config=config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(f"Cleaning [cyan]{file.name}[/]…", total=None)
        df = _load_df(file)
        clean_df, report = engine.clean(df, target_col=target_col)

    # Print summary table
    table = Table(box=box.ROUNDED, border_style="dim", title="AutoClean Summary")
    table.add_column("Column", style="cyan")
    table.add_column("Fix Type", style="magenta")
    table.add_column("Rows", justify="right")
    table.add_column("Details")
    for fix in report.fixes:
        table.add_row(fix.column, fix.action.value, str(len(fix.rows)), fix.details[:80])
    console.print(table)

    score_color = "green" if report.mend_score >= 80 else "yellow" if report.mend_score >= 50 else "red"
    console.print(Panel(
        f"MendScore [bold {score_color}]{report.mend_score}/100[/]  ·  "
        f"{len(report.fixes)} fix(es) applied  ·  "
        f"{report.original_shape[0]}→{report.cleaned_shape[0]} rows  ·  "
        f"{report.duration_seconds:.2f}s",
        title="[bold]puredata AutoClean[/]",
        border_style="cyan",
    ))

    # Outputs
    if output:
        suffix = output.suffix.lower()
        if suffix == ".csv":
            clean_df.to_csv(output, index=False)
        elif suffix in (".xlsx",):
            clean_df.to_excel(output, index=False)
        elif suffix == ".parquet":
            clean_df.to_parquet(output, index=False)
        else:
            clean_df.to_csv(output, index=False)
        console.print(f"[green]✓[/] Saved cleaned data → {output}")

    if report_json:
        report.to_json(report_json)
        console.print(f"[green]✓[/] Saved JSON report → {report_json}")
    if report_html:
        report.to_html(report_html)
        console.print(f"[green]✓[/] Saved HTML report → {report_html}")
    if report_csv:
        report.to_csv(report_csv)
        console.print(f"[green]✓[/] Saved CSV report → {report_csv}")


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------

@app.command()
def watch(
    reference: Path = typer.Argument(..., help="Reference/training data file"),
    contract_out: Path = typer.Option(
        Path("contract.json"), "--contract", "-c", help="Where to save the contract"
    ),
) -> None:
    """[bold cyan]Fit[/] a DataWatch contract on reference data.

    Examples:

      puredata watch train.csv --contract mycontract.json
    """
    if not reference.exists():
        console.print(f"[red]Error:[/] File not found: {reference}")
        raise typer.Exit(1)

    from puredata.core.watch import DataWatch

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as p:
        p.add_task(f"Profiling [cyan]{reference.name}[/]…", total=None)
        df = _load_df(reference)
        watcher = DataWatch()
        contract = watcher.fit(df)
        contract.save(contract_out)

    console.print(Panel(
        f"Profiled [bold cyan]{len(contract.columns)}[/] columns from {reference.name}\n"
        f"Contract saved → [bold]{contract_out}[/]",
        title="[bold]puredata DataWatch[/]",
        border_style="blue",
    ))


# ---------------------------------------------------------------------------
# check command
# ---------------------------------------------------------------------------

@app.command()
def check(
    file: Path = typer.Argument(..., help="New data file to validate"),
    contract: Path = typer.Argument(..., help="Contract JSON file (from puredata watch)"),
    report_json: Optional[Path] = typer.Option(None, "--report-json"),
    report_html: Optional[Path] = typer.Option(None, "--report-html"),
    strict: bool = typer.Option(False, "--strict", help="Exit with code 1 if any checks fail"),
) -> None:
    """[bold cyan]Validate[/] new data against a saved contract.

    Examples:

      puredata check prod.csv contract.json
      puredata check prod.csv contract.json --strict
    """
    for p, label in [(file, "data file"), (contract, "contract file")]:
        if not p.exists():
            console.print(f"[red]Error:[/] {label} not found: {p}")
            raise typer.Exit(1)

    from puredata.core.watch import DataContract, DataWatch

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as prog:
        prog.add_task(f"Validating [cyan]{file.name}[/]…", total=None)
        df = _load_df(file)
        loaded_contract = DataContract.load(contract)
        watcher = DataWatch(mode="silent")
        result = watcher.check(df, loaded_contract)

    table = Table(box=box.ROUNDED, border_style="dim", title="DataWatch Results")
    table.add_column("Column", style="cyan")
    table.add_column("Check", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("Message")
    for chk in result.checks:
        icon_style = {"pass": "green", "warn": "yellow", "fail": "red"}.get(
            chk.status.value, "dim"
        )
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(chk.status.value, "–")
        table.add_row(
            chk.column,
            chk.name,
            f"[{icon_style}]{icon} {chk.status.value}[/]",
            chk.message[:100],
        )
    console.print(table)

    score_color = "green" if result.compatibility_score >= 80 else "yellow" if result.compatibility_score >= 50 else "red"
    console.print(Panel(
        f"Compatibility score [{score_color}]{result.compatibility_score}/100[/]  ·  "
        f"[green]{result.n_passed} passed[/]  [yellow]{result.n_warned} warned[/]  "
        f"[red]{result.n_failed} failed[/]",
        title="[bold]puredata DataWatch[/]",
        border_style="blue",
    ))

    if report_json:
        result.to_json(report_json)
        console.print(f"[green]✓[/] Saved JSON report → {report_json}")
    if report_html:
        result.to_html(report_html)
        console.print(f"[green]✓[/] Saved HTML report → {report_html}")

    if strict and result.n_failed > 0:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# dashboard command
# ---------------------------------------------------------------------------

@app.command()
def show_dashboard(
    file: Path = typer.Argument(..., help="Data file to visualise"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save HTML here"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open the browser"),
) -> None:
    """[bold cyan]Open[/] the interactive dataset health dashboard.

    Examples:

      puredata dashboard mydata.csv
    """
    if not file.exists():
        console.print(f"[red]Error:[/] File not found: {file}")
        raise typer.Exit(1)

    from puredata.dashboard import dashboard as open_dashboard

    df = _load_df(file)
    path = open_dashboard(df, open_browser=not no_browser, output_path=output)
    console.print(f"[green]✓[/] Dashboard ready → {path}")


app.command(name="dashboard")(show_dashboard)


# ---------------------------------------------------------------------------
# score command
# ---------------------------------------------------------------------------

@app.command()
def score(
    file: Path = typer.Argument(..., help="Data file to score"),
) -> None:
    """Print the MendScore health score for a dataset (0–100)."""
    if not file.exists():
        console.print(f"[red]Error:[/] File not found: {file}")
        raise typer.Exit(1)

    from puredata.core.clean import AutoClean

    df = _load_df(file)
    _, report = AutoClean().clean(df)
    score_color = "green" if report.mend_score >= 80 else "yellow" if report.mend_score >= 50 else "red"
    console.print(f"MendScore: [{score_color}]{report.mend_score}/100[/]")


if __name__ == "__main__":
    app()
