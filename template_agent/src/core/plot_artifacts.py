"""In-memory plot artifact registry and per-request plot collection."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
from uuid import uuid4

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

_plot_session: contextvars.ContextVar[List["PlotArtifact"]] = contextvars.ContextVar(
    "plot_session_artifacts",
    default=[],
)

_plot_store: dict[str, "PlotArtifact"] = {}


@dataclass
class PlotArtifact:
    plot_id: str
    file_path: Path
    filename: str
    title: str
    plot_type: str


def get_plot_artifacts_dir() -> Path:
    base = Path(settings.PLOT_ARTIFACT_DIR or "/tmp/snowflake-bot-plots")
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_public_url(plot_id: str) -> str:
    path = f"/api/v1/plots/{plot_id}.png"
    base = (settings.AGENT_PUBLIC_BASE_URL or "").rstrip("/")
    return f"{base}{path}" if base else path


def register_plot_artifact(
    *,
    file_path: Path,
    title: str,
    plot_type: str,
    plot_id: str | None = None,
) -> PlotArtifact:
    plot_id = plot_id or uuid4().hex
    filename = file_path.name
    artifact = PlotArtifact(
        plot_id=plot_id,
        file_path=file_path,
        filename=filename,
        title=title or filename,
        plot_type=plot_type,
    )
    _plot_store[plot_id] = artifact
    session = list(_plot_session.get())
    session.append(artifact)
    _plot_session.set(session)
    logger.info(
        "plot.registered plot_id=%s type=%s path=%s",
        plot_id,
        plot_type,
        file_path,
    )
    return artifact


def get_plot_artifact(plot_id: str) -> Optional[PlotArtifact]:
    artifact = _plot_store.get(plot_id)
    if artifact is not None:
        return artifact
    # Fallback: PNG saved as {plot_id}.png (survives in-memory eviction on same pod).
    path = get_plot_artifacts_dir() / f"{plot_id}.png"
    if path.is_file():
        return PlotArtifact(
            plot_id=plot_id,
            file_path=path,
            filename=path.name,
            title=path.stem,
            plot_type="chart",
        )
    return None


def get_session_plot_artifacts() -> List[PlotArtifact]:
    return list(_plot_session.get())


@contextmanager
def plot_request_scope() -> Iterator[None]:
    """Reset per-request plot list; collect artifacts created during the run."""
    reset = _plot_session.set([])
    try:
        yield
    finally:
        _plot_session.reset(reset)
