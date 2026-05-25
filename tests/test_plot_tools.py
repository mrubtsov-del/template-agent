"""Tests for matplotlib plotting tools."""

import json

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

    result = plot_tools.create_chart_from_query.invoke(
        {
            "plot_type": "bar",
            "query_result": {
                "columns": ["category", "value"],
                "rows": [["A", 10], ["B", 25], ["C", 15]],
            },
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

    result = plot_tools.create_chart_from_query.invoke(
        {
            "plot_type": "line",
            "query_result": {"columns": ["x", "y"], "rows": [[1, 2]]},
            "x_column": "missing",
            "y_column": "y",
        }
    )
    assert "error" in result


def test_plot_png_route_served(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from template_agent.src.api import app
    from template_agent.src.core.plot_artifacts import register_plot_artifact

    monkeypatch.setattr(
        "template_agent.src.core.plot_artifacts.settings.PLOT_ARTIFACT_DIR",
        str(tmp_path),
    )
    path = tmp_path / "abc123.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    register_plot_artifact(
        file_path=path,
        title="Test chart",
        plot_type="bar",
        plot_id="abc123",
    )

    client = TestClient(app)
    response = client.get("/api/v1/plots/abc123.png")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_create_chart_from_query_accepts_json_string(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_tools.settings, "PLOT_ENABLED", True)
    monkeypatch.setattr(plot_tools.settings, "PLOT_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(plot_tools.settings, "AGENT_PUBLIC_BASE_URL", "https://agent.test")

    payload = json.dumps(
        {
            "columns": ["user_id", "n"],
            "rows": [[1, 10], [2, 20]],
        }
    )
    result = plot_tools.create_chart_from_query.invoke(
        {
            "plot_type": "bar",
            "query_result": payload,
            "x_column": "user_id",
            "y_column": "n",
        }
    )
    assert result["status"] == "ok"


def test_plot_tools_gemini_compatible_schema():
    """Gemini rejects nested list[list]; sql-based chart tool has no rows param."""
    sql_schema = plot_tools.create_chart_from_sql.args_schema.model_json_schema()
    assert "sql" in sql_schema.get("properties", {})
    assert "rows" not in sql_schema.get("properties", {})

    query_schema = plot_tools.create_chart_from_query.args_schema.model_json_schema()
    assert "query_result" in query_schema.get("properties", {})
    assert "rows" not in query_schema.get("properties", {})


def test_chart_style_grid_and_color(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_tools.settings, "PLOT_ENABLED", True)
    monkeypatch.setattr(plot_tools.settings, "PLOT_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(plot_tools.settings, "AGENT_PUBLIC_BASE_URL", "https://agent.test")

    result = plot_tools.create_chart_from_query.invoke(
        {
            "plot_type": "bar",
            "query_result": {
                "columns": ["category", "value"],
                "rows": [["A", 10], ["B", 25]],
            },
            "x_column": "category",
            "y_column": "value",
            "show_grid": True,
            "color": "#EE0000",
            "rotate_x_labels": True,
        }
    )
    assert result["status"] == "ok"
    assert result["style"]["show_grid"] is True
    assert result["style"]["color"] == "#EE0000"
    assert result["style"]["rotate_x_labels"] is True


def test_chart_style_rejects_invalid_color(monkeypatch, tmp_path):
    monkeypatch.setattr(plot_tools.settings, "PLOT_ENABLED", True)
    monkeypatch.setattr(plot_tools.settings, "PLOT_ARTIFACT_DIR", str(tmp_path))

    result = plot_tools.create_chart_from_query.invoke(
        {
            "plot_type": "bar",
            "query_result": {"columns": ["x", "y"], "rows": [["a", 1]]},
            "x_column": "x",
            "y_column": "y",
            "color": "red",
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
