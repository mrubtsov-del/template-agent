"""Tests for matplotlib plotting tools."""

import pytest

from template_agent.src.core.plot_artifacts import plot_request_scope
from template_agent.src.core.tools import plot_tools


@pytest.fixture(autouse=True)
def _plot_scope():
    with plot_request_scope():
        yield


def test_create_bar_chart(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_tools.settings, "PLOT_ENABLED", True)
    monkeypatch.setattr(plot_tools.settings, "PLOT_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(plot_tools.settings, "AGENT_PUBLIC_BASE_URL", "https://agent.test")

    result = plot_tools.create_chart.invoke(
        {
            "plot_type": "bar",
            "columns": ["category", "value"],
            "rows": [["A", 10], ["B", 25], ["C", 15]],
            "x_column": "category",
            "y_column": "value",
            "title": "Sample counts",
        }
    )

    assert result["status"] == "ok"
    assert result["plot_type"] == "bar"
    assert result["url"].startswith("https://agent.test/api/v1/plots/")
    assert (tmp_path / result["filename"]).is_file()


def test_create_chart_rejects_missing_column(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_tools.settings, "PLOT_ENABLED", True)
    monkeypatch.setattr(plot_tools.settings, "PLOT_ARTIFACT_DIR", str(tmp_path))

    result = plot_tools.create_chart.invoke(
        {
            "plot_type": "line",
            "columns": ["x", "y"],
            "rows": [[1, 2]],
            "x_column": "missing",
            "y_column": "y",
        }
    )
    assert "error" in result


def test_create_chart_from_query_result(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_tools.settings, "PLOT_ENABLED", True)
    monkeypatch.setattr(plot_tools.settings, "PLOT_ARTIFACT_DIR", str(tmp_path))

    query_result = {
        "columns": ["month", "total"],
        "rows": [["Jan", 100], ["Feb", 120]],
        "row_count": 2,
        "truncated": False,
    }
    result = plot_tools.create_chart_from_query.invoke(
        {
            "plot_type": "line",
            "query_result": query_result,
            "x_column": "month",
            "y_column": "total",
            "title": "Monthly total",
        }
    )
    assert result["status"] == "ok"
    assert result["plot_type"] == "line"
