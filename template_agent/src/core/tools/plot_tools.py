"""Matplotlib/seaborn plotting tools for the Snowflake analyst agent.

Follows the data-viz-plots skill: publication-style defaults, tight layout,
300 DPI PNG export, plt.close after save.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

from template_agent.src.core.plot_artifacts import (
    get_plot_artifacts_dir,
    plot_public_url,
    register_plot_artifact,
)
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

PlotType = Literal["bar", "line", "scatter", "heatmap", "box", "violin", "histogram"]

_SUPPORTED = {"bar", "line", "scatter", "heatmap", "box", "violin", "histogram"}


def _apply_plot_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
        }
    )


def _tool_error(message: str, error_type: str = "validation_error") -> dict[str, Any]:
    return {"error": message, "error_type": error_type, "retryable": False}


def _rows_to_dataframe(columns: list[str], rows: list[list[Any]]) -> pd.DataFrame:
    if not columns:
        raise ValueError("columns must not be empty")
    if len(rows) > settings.PLOT_MAX_ROWS:
        rows = rows[: settings.PLOT_MAX_ROWS]
    return pd.DataFrame(rows, columns=columns)


def _validate_columns(df: pd.DataFrame, *names: Optional[str]) -> Optional[str]:
    for name in names:
        if name and name not in df.columns:
            return f"Column '{name}' not found. Available: {list(df.columns)}"
    return None


def _save_figure(fig: plt.Figure, *, title: str, plot_type: str) -> dict[str, Any]:
    from uuid import uuid4

    out_dir = get_plot_artifacts_dir()
    filename = f"chart_{plot_type}_{uuid4().hex[:10]}.png"
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    artifact = register_plot_artifact(
        file_path=path,
        title=title,
        plot_type=plot_type,
    )
    url = plot_public_url(artifact.plot_id)
    return {
        "status": "ok",
        "plot_id": artifact.plot_id,
        "plot_type": plot_type,
        "title": title,
        "filename": artifact.filename,
        "url": url,
        "message": (
            f"Created {plot_type} chart '{title}'. "
            f"Reference this plot in your answer (plot_id={artifact.plot_id})."
        ),
    }


def _render_chart(
    plot_type: str,
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: Optional[str],
    hue_column: Optional[str],
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, Any]:
    _apply_plot_style()
    plot_type = plot_type.lower().strip()
    if plot_type not in _SUPPORTED:
        return _tool_error(
            f"Unsupported plot_type '{plot_type}'. Use one of: {sorted(_SUPPORTED)}"
        )

    err = _validate_columns(df, x_column, y_column, hue_column)
    if err:
        return _tool_error(err)

    display_title = title or f"{plot_type.title()} chart"
    xlab = x_label or x_column
    ylab = y_label or (y_column or "")

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        if plot_type == "bar":
            if not y_column:
                return _tool_error("bar plots require y_column")
            if hue_column:
                sns.barplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)
            else:
                sns.barplot(data=df, x=x_column, y=y_column, ax=ax)
        elif plot_type == "line":
            if not y_column:
                return _tool_error("line plots require y_column")
            if hue_column:
                for label, group in df.groupby(hue_column):
                    ax.plot(
                        group[x_column],
                        group[y_column],
                        marker="o",
                        label=str(label),
                        linewidth=2,
                    )
                ax.legend(loc="best", frameon=True)
            else:
                ax.plot(df[x_column], df[y_column], marker="o", linewidth=2)
        elif plot_type == "scatter":
            if not y_column:
                return _tool_error("scatter plots require y_column")
            if hue_column:
                sns.scatterplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue_column,
                    s=40,
                    alpha=0.7,
                    ax=ax,
                )
            else:
                ax.scatter(df[x_column], df[y_column], s=40, alpha=0.7)
        elif plot_type == "box":
            if not y_column:
                return _tool_error("box plots require y_column")
            sns.boxplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue_column,
                palette="Set2",
                ax=ax,
            )
            ax.tick_params(axis="x", rotation=45)
        elif plot_type == "violin":
            if not y_column:
                return _tool_error("violin plots require y_column")
            sns.violinplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue_column,
                palette="muted",
                inner="quartile",
                ax=ax,
            )
            ax.tick_params(axis="x", rotation=45)
        elif plot_type == "histogram":
            series = df[y_column or x_column]
            ax.hist(series.dropna(), bins=min(30, max(10, len(series) // 5)), edgecolor="black")
            xlab = xlab or (y_column or x_column)
            ylab = ylab or "Count"
        elif plot_type == "heatmap":
            if y_column and y_column in df.columns:
                pivot = df.pivot_table(
                    index=x_column, columns=y_column, aggfunc="mean"
                )
            else:
                numeric = df.select_dtypes(include="number")
                if numeric.empty:
                    return _tool_error(
                        "heatmap requires numeric columns or y_column for pivot"
                    )
                pivot = numeric
            sns.heatmap(pivot, cmap="viridis", ax=ax, cbar_kws={"label": ylab or "Value"})
            xlab = ""
            ylab = ""

        ax.set_title(display_title, fontweight="bold")
        if xlab:
            ax.set_xlabel(xlab)
        if ylab:
            ax.set_ylabel(ylab)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return _save_figure(fig, title=display_title, plot_type=plot_type)
    except Exception as exc:
        plt.close(fig)
        logger.exception("plot.render failed type=%s", plot_type)
        return _tool_error(f"Failed to create chart: {exc}", error_type="render_error")


@tool
def create_chart(
    plot_type: PlotType,
    columns: list[str],
    rows: list[list[Any]],
    x_column: str,
    y_column: Optional[str] = None,
    hue_column: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> dict[str, Any]:
    """Create a publication-style chart from tabular data (e.g. SQL query results).

    Use after ``run_select_query`` when the user asks for a plot, chart, or visualization.
    Pass the same ``columns`` and ``rows`` returned by the query.

    Args:
        plot_type: One of bar, line, scatter, heatmap, box, violin, histogram.
        columns: Column names (same order as SQL result).
        rows: Row values as list of lists.
        x_column: Column for x-axis (or category axis).
        y_column: Column for y-axis (required for most types except heatmap pivot).
        hue_column: Optional grouping/color column.
        title: Chart title.
        x_label: Optional x-axis label override.
        y_label: Optional y-axis label override.

    Returns:
        Dict with plot_id, url, and message on success; ``error`` on failure.
    """
    if not settings.PLOT_ENABLED:
        return _tool_error("Plotting is disabled (PLOT_ENABLED=false)")

    try:
        df = _rows_to_dataframe(columns, rows)
    except Exception as exc:
        return _tool_error(f"Invalid tabular data: {exc}")

    if df.empty:
        return _tool_error("No rows to plot")

    return _render_chart(
        plot_type,
        df,
        x_column=x_column,
        y_column=y_column,
        hue_column=hue_column,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )


@tool
def create_chart_from_query(
    plot_type: PlotType,
    query_result: dict[str, Any],
    x_column: str,
    y_column: Optional[str] = None,
    hue_column: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> dict[str, Any]:
    """Create a chart directly from a ``run_select_query`` result dict.

    Args:
        plot_type: bar, line, scatter, heatmap, box, violin, or histogram.
        query_result: Dict with ``columns`` and ``rows`` keys (Snowflake tool output).
        x_column: X-axis column name.
        y_column: Y-axis column name.
        hue_column: Optional group/color column.
        title: Chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.
    """
    if query_result.get("error"):
        return _tool_error(
            f"Cannot plot failed query: {query_result.get('error')}",
            error_type="query_error",
        )
    columns = query_result.get("columns")
    rows = query_result.get("rows")
    if not columns or rows is None:
        return _tool_error(
            "query_result must include 'columns' and 'rows' from run_select_query"
        )
    try:
        df = _rows_to_dataframe(columns, rows)
    except Exception as exc:
        return _tool_error(f"Invalid tabular data: {exc}")
    if df.empty:
        return _tool_error("No rows to plot")
    return _render_chart(
        plot_type,
        df,
        x_column=x_column,
        y_column=y_column,
        hue_column=hue_column,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )


PLOT_TOOLS = [create_chart, create_chart_from_query]
