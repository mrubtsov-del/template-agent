"""Matplotlib/seaborn plotting tools for the Snowflake analyst agent.

Follows the data-viz-plots skill: publication-style defaults, tight layout,
300 DPI PNG export, plt.close after save.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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

_PALETTES = frozenset(
    {"viridis", "muted", "Set2", "husl", "deep", "pastel", "colorblind", "flare"}
)
_HEX_COLOR = re.compile(r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")


@dataclass(frozen=True)
class ChartStyle:
    """Small, safe chart styling options users may request in chat."""

    show_grid: bool = False
    palette: Optional[str] = None
    color: Optional[str] = None
    fig_width: float = 8.0
    fig_height: float = 5.0
    rotate_x_labels: bool = False


def _parse_chart_style(
    *,
    show_grid: bool = False,
    palette: Optional[str] = None,
    color: Optional[str] = None,
    fig_width: float = 8.0,
    fig_height: float = 5.0,
    rotate_x_labels: bool = False,
) -> ChartStyle | dict[str, Any]:
    if fig_width < 4 or fig_width > 16 or fig_height < 3 or fig_height > 12:
        return _tool_error("fig_width must be 4–16 and fig_height must be 3–12")
    normalized_palette: Optional[str] = None
    if palette and str(palette).strip():
        key = str(palette).strip()
        if key not in _PALETTES:
            return _tool_error(
                f"palette must be one of: {', '.join(sorted(_PALETTES))}"
            )
        normalized_palette = key
    normalized_color: Optional[str] = None
    if color and str(color).strip():
        candidate = str(color).strip()
        if not _HEX_COLOR.match(candidate):
            return _tool_error("color must be a hex code like #EE0000 or #c00")
        normalized_color = candidate
    return ChartStyle(
        show_grid=show_grid,
        palette=normalized_palette,
        color=normalized_color,
        fig_width=float(fig_width),
        fig_height=float(fig_height),
        rotate_x_labels=rotate_x_labels,
    )


def _apply_plot_style(style: ChartStyle) -> None:
    sns.set_style("whitegrid" if style.show_grid else "white")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.grid": style.show_grid,
        }
    )


def _normalize_query_result(raw: Any) -> dict[str, Any]:
    """Accept dict or JSON string (Gemini sometimes passes tool output as string)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("query_result is empty")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("query_result JSON must be an object with columns and rows")
        return parsed
    raise ValueError(
        f"query_result must be a dict or JSON string, got {type(raw).__name__}"
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

    plot_id = uuid4().hex
    out_dir = get_plot_artifacts_dir()
    filename = f"{plot_id}.png"
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    artifact = register_plot_artifact(
        file_path=path,
        title=title,
        plot_type=plot_type,
        plot_id=plot_id,
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


def _style_dict(style: ChartStyle) -> dict[str, Any]:
    return {
        "show_grid": style.show_grid,
        "palette": style.palette,
        "color": style.color,
        "fig_width": style.fig_width,
        "fig_height": style.fig_height,
        "rotate_x_labels": style.rotate_x_labels,
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
    style: ChartStyle | None = None,
) -> dict[str, Any]:
    chart_style = style or ChartStyle()
    _apply_plot_style(chart_style)
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

    fig, ax = plt.subplots(figsize=(chart_style.fig_width, chart_style.fig_height))
    bar_color = chart_style.color if not hue_column else None
    cmap = chart_style.palette or "viridis"

    try:
        if plot_type == "bar":
            if not y_column:
                return _tool_error("bar plots require y_column")
            if hue_column:
                sns.barplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue_column,
                    palette=chart_style.palette or "Set2",
                    ax=ax,
                )
            else:
                sns.barplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    color=bar_color,
                    ax=ax,
                )
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
                ax.plot(
                    df[x_column],
                    df[y_column],
                    marker="o",
                    linewidth=2,
                    color=bar_color,
                )
        elif plot_type == "scatter":
            if not y_column:
                return _tool_error("scatter plots require y_column")
            if hue_column:
                sns.scatterplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue_column,
                    palette=chart_style.palette or "deep",
                    s=40,
                    alpha=0.7,
                    ax=ax,
                )
            else:
                ax.scatter(
                    df[x_column],
                    df[y_column],
                    s=40,
                    alpha=0.7,
                    color=bar_color,
                )
        elif plot_type == "box":
            if not y_column:
                return _tool_error("box plots require y_column")
            sns.boxplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue_column,
                palette=chart_style.palette or "Set2",
                ax=ax,
            )
        elif plot_type == "violin":
            if not y_column:
                return _tool_error("violin plots require y_column")
            sns.violinplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue_column,
                palette=chart_style.palette or "muted",
                inner="quartile",
                ax=ax,
            )
        elif plot_type == "histogram":
            series = df[y_column or x_column]
            ax.hist(
                series.dropna(),
                bins=min(30, max(10, len(series) // 5)),
                edgecolor="black",
                color=bar_color,
            )
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
            sns.heatmap(
                pivot,
                cmap=cmap,
                ax=ax,
                cbar_kws={"label": ylab or "Value"},
            )
            xlab = ""
            ylab = ""

        ax.set_title(display_title, fontweight="bold")
        if xlab:
            ax.set_xlabel(xlab)
        if ylab:
            ax.set_ylabel(ylab)
        if chart_style.rotate_x_labels:
            ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        result = _save_figure(fig, title=display_title, plot_type=plot_type)
        if result.get("status") == "ok":
            result["style"] = _style_dict(chart_style)
        return result
    except Exception as exc:
        plt.close(fig)
        logger.exception("plot.render failed type=%s", plot_type)
        return _tool_error(f"Failed to create chart: {exc}", error_type="render_error")


@tool
def create_chart_from_query(
    plot_type: PlotType,
    query_result: Any,
    x_column: str,
    y_column: Optional[str] = None,
    hue_column: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    show_grid: bool = False,
    palette: Optional[str] = None,
    color: Optional[str] = None,
    fig_width: float = 8.0,
    fig_height: float = 5.0,
    rotate_x_labels: bool = False,
) -> dict[str, Any]:
    """Create a chart from a ``run_select_query`` result (dict or JSON string).

    Args:
        plot_type: bar, line, scatter, heatmap, box, violin, or histogram.
        query_result: Output of ``run_select_query`` — object with ``columns`` and
            ``rows``, not a bare SQL string. JSON string is also accepted.
        x_column: X-axis column name.
        y_column: Y-axis column name (required for bar/line; use count column for aggregates).
        hue_column: Optional group/color column.
        title: Chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.
        show_grid: Show light background grid (default off).
        palette: Seaborn/matplotlib palette for grouped plots or heatmaps
            (viridis, muted, Set2, husl, deep, pastel, colorblind, flare).
        color: Single-series hex color, e.g. #EE0000 (ignored when hue_column is set).
        fig_width: Figure width in inches (4–16).
        fig_height: Figure height in inches (3–12).
        rotate_x_labels: Rotate x tick labels 45° for long category names.
    """
    try:
        query_result = _normalize_query_result(query_result)
    except (ValueError, json.JSONDecodeError) as exc:
        return _tool_error(
            f"Invalid query_result: {exc}. Pass the full dict from run_select_query "
            "(columns + rows), or call create_chart_from_sql with your SQL.",
            error_type="validation_error",
        )

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
    parsed = _parse_chart_style(
        show_grid=show_grid,
        palette=palette,
        color=color,
        fig_width=fig_width,
        fig_height=fig_height,
        rotate_x_labels=rotate_x_labels,
    )
    if isinstance(parsed, dict):
        return parsed
    return _render_chart(
        plot_type,
        df,
        x_column=x_column,
        y_column=y_column,
        hue_column=hue_column,
        title=title,
        x_label=x_label,
        y_label=y_label,
        style=parsed,
    )


@tool
def create_chart_from_sql(
    plot_type: PlotType,
    sql: str,
    x_column: str,
    y_column: Optional[str] = None,
    hue_column: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    show_grid: bool = False,
    palette: Optional[str] = None,
    color: Optional[str] = None,
    fig_width: float = 8.0,
    fig_height: float = 5.0,
    rotate_x_labels: bool = False,
) -> dict[str, Any]:
    """Run read-only SQL and create a chart in one step (preferred for charts).

    Use aggregated SQL for bar charts (e.g. COUNT by USER_ID), not raw high-cardinality IDs.

    Args:
        plot_type: bar, line, scatter, heatmap, box, violin, or histogram.
        sql: Read-only SELECT/WITH executed via ``run_select_query``.
        x_column: Column for x-axis.
        y_column: Column for y-axis (e.g. ENROLMENT_COUNT).
        hue_column: Optional grouping column.
        title: Chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.
        show_grid: Show light background grid (default off).
        palette: Palette name for grouped plots or heatmaps.
        color: Hex color for single-series charts (e.g. #CC0000).
        fig_width: Figure width in inches (4–16).
        fig_height: Figure height in inches (3–12).
        rotate_x_labels: Rotate x tick labels 45°.
    """
    return {
        "error": "create_chart_from_sql is not available when using Snowflake MCP. "
        "Use execute_sql_query first, then pass the result to create_chart_from_query.",
        "error_type": "not_available",
    }


# create_chart_from_sql is disabled — Snowflake queries now go through MCP.
# LLM should call execute_sql_query (Snowflake MCP) then create_chart_from_query.
PLOT_TOOLS = [create_chart_from_query]
