"""Shadowbot V2 data sources handler.

# /api/v2/conversations/data/sources

Describes the data collections this agent has access to. Used by Shadowbot's
UI to display which sources back the answers.
"""

from datetime import datetime, timezone
from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_data_sources_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import (
    DataCollection,
    DataSourcesResponseV2,
)

from template_agent.src.settings import settings


# /api/v2/conversations/data/sources
@get_data_sources_handler_v2()
@require_auth
async def handle_get_data_sources_v2(
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> DataSourcesResponseV2:
    """Report Snowflake schema/tables as a live data collection."""
    db = settings.SNOWFLAKE_DATABASE or ""
    schema = settings.SNOWFLAKE_SCHEMA or ""
    label = f"Snowflake {db}.{schema}".strip(". ") or "Snowflake"

    # Snowflake data is live so we surface "Today" as the freshness signal.
    # Shadowbot UI renders this string directly under each collection card.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    live = [DataCollection(name=label, last_updated=today)]
    upcoming: list[DataCollection] = []

    return DataSourcesResponseV2(
        live_collections=live,
        upcoming_collections=upcoming,
    )
