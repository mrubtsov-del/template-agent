"""Source package for template MCP server."""

from pathlib import Path
import sys


def _add_vendor_to_path() -> None:
    """Add local vendor directory to Python path if present.

    This allows private dependencies (e.g. vendored shadowbot_agent_api)
    to be imported both locally and in container builds without external
    package registry access.
    """
    repo_root = Path(__file__).resolve().parents[2]
    vendor_path = repo_root / "vendor"
    if vendor_path.exists():
        vendor_path_str = str(vendor_path)
        if vendor_path_str not in sys.path:
            sys.path.insert(0, vendor_path_str)


_add_vendor_to_path()
