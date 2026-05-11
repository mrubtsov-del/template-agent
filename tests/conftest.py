"""Pytest fixtures and shared test setup.

This conftest exists so the vendored `shadowbot_agent_api` package is reachable
from pytest's collection phase. Normally `_add_vendor_to_path()` runs when the
agent's `template_agent.src` package is imported, but several test modules
import `shadowbot_agent_api` directly at module level (before any
`template_agent.src.*` import has had a chance to mutate `sys.path`), so we
add the path here too.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_PATH = _REPO_ROOT / "vendor"
if _VENDOR_PATH.exists():
    vendor_str = str(_VENDOR_PATH)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
