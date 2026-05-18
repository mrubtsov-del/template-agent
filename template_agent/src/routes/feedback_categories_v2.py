"""Shadowbot V2 feedback categories handler.

# GET /api/v2/conversations/feedback/categories
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_feedback_categories_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import (
    FeedbackCategoriesResponseV2,
    FeedbackCategoryItem,
)

from template_agent.src.routes.common import (
    logger,
    resolve_user_label,
    snowflake_auth_present,
)

# Standard Shadowbot categories plus Snowflake SQL feedback.
_V2_FEEDBACK_CATEGORIES: tuple[FeedbackCategoryItem, ...] = (
    FeedbackCategoryItem(
        code="missing_content",
        label="Missing content",
        comment_required=False,
    ),
    FeedbackCategoryItem(
        code="incorrect_or_outdated_info",
        label="Incorrect or outdated info",
        comment_required=False,
    ),
    FeedbackCategoryItem(
        code="error_or_outage",
        label="Error or outage",
        comment_required=False,
    ),
    FeedbackCategoryItem(
        code="unclear_or_low_quality_answer",
        label="Unclear or low-quality answer",
        comment_required=False,
    ),
    FeedbackCategoryItem(
        code="wrong_sql",
        label="Wrong SQL",
        comment_required=True,
    ),
    FeedbackCategoryItem(
        code="feature_request",
        label="Feature request",
        comment_required=True,
    ),
    FeedbackCategoryItem(
        code="other",
        label="Other",
        comment_required=True,
    ),
)


@get_feedback_categories_handler_v2()
@require_auth
async def handle_get_feedback_categories_v2(
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> FeedbackCategoriesResponseV2:
    """Return the standard Shadowbot categories plus ``wrong_sql``."""
    logger.info(
        "[V2] Feedback categories",
        user_id=resolve_user_label(user),
        count=len(_V2_FEEDBACK_CATEGORIES),
        snowflake_auth=snowflake_auth_present(custom_auth),
    )
    return FeedbackCategoriesResponseV2(
        feedback_categories=list(_V2_FEEDBACK_CATEGORIES),
    )
