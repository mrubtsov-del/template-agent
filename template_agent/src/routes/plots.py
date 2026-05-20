"""Serve generated plot PNG artifacts."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from template_agent.src.core.plot_artifacts import get_plot_artifact

router = APIRouter(prefix="/api/v1/plots", tags=["plots"])


@router.get("/{plot_id}.png")
async def get_plot_png(plot_id: str) -> FileResponse:
    """Return a chart PNG created by ``create_chart`` during an agent run."""
    artifact = get_plot_artifact(plot_id)
    if artifact is None or not artifact.file_path.is_file():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(
        path=artifact.file_path,
        media_type="image/png",
        filename=artifact.filename,
    )
