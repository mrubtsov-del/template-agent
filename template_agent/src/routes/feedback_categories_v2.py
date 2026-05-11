"""Shadowbot V2 feedback categories handler.

# /api/v2/conversations/feedback/categories

Returns the list of feedback categories this agent supports, so the Shadowbot
UI can render the right dropdown when a user reacts to a message.

Uses the standard Shadowbot category set so the UI rendering matches what
other agents emit, plus `wrong_sql` which is specific to a SQL-generating
agent like ours.
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


# /api/v2/conversations/feedback/categories
@get_feedback_categories_handler_v2()
@require_auth
async def handle_get_feedback_categories_v2(
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> FeedbackCategoriesResponseV2:
    """Return the standard Shadowbot categories plus our SQL-specific one."""
    return FeedbackCategoriesResponseV2(
        feedback_categories=[
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
        ]
    )
