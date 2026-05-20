"""Shadowbot V2 data sources handler.

# GET /api/v2/conversations/data/sources
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_data_sources_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import DataCollection, DataSourcesResponseV2

from template_agent.src.routes.common import (
    logger,
    resolve_user_label,
    snowflake_auth_present,
    utc_iso_z,
)
from template_agent.src.settings import settings


@get_data_sources_handler_v2()
@require_auth
async def handle_get_data_sources_v2(
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> DataSourcesResponseV2:
    """Report Snowflake databases/schemas as live data collections."""
    targets = settings.snowflake_allowed_schema_targets
    today = utc_iso_z()[:10]
    if targets:
        collections = [
            DataCollection(name=f"Snowflake {target}", last_updated=today)
            for target in targets
        ]
    else:
        db = settings.SNOWFLAKE_DATABASE or ""
        schema = settings.SNOWFLAKE_SCHEMA or ""
        label = f"Snowflake {db}.{schema}".strip(". ") or "Snowflake"
        collections = [DataCollection(name=label, last_updated=today)]

    logger.info(
        "[V2] Data sources",
        user_id=resolve_user_label(user),
        collections=len(collections),
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    return DataSourcesResponseV2(
        live_collections=collections,
        upcoming_collections=[],
    )
