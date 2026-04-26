"""Tests for the prompt module."""

from unittest.mock import patch

from template_agent.src.core.prompt import get_current_date, get_system_prompt


class TestPrompt:
    """Test cases for prompt functions."""

    def test_get_current_date(self):
        """Test get_current_date returns formatted date string."""
        date_str = get_current_date()
        assert isinstance(date_str, str)
        # Should be in format "Month Day, Year" (e.g., "December 25, 2024")
        assert len(date_str.split()) == 3

    def test_get_system_prompt(self):
        """Test get_system_prompt returns non-empty string."""
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Snowflake Data Analyst Agent" in prompt
        assert "Today's date is" in prompt
        assert "## Tools available" in prompt
        assert "## Behaviour rules" in prompt
        assert "## Output format" in prompt
        assert "What I ran" in prompt
        assert "Next step" in prompt

    @patch("template_agent.src.core.prompt.get_current_date")
    def test_get_system_prompt_includes_date(self, mock_get_date):
        """Test that get_system_prompt includes the current date."""
        mock_get_date.return_value = "December 25, 2024"
        prompt = get_system_prompt()
        assert "Today's date is December 25, 2024" in prompt

    def test_get_system_prompt_mentions_read_only_and_grounding(self):
        """Prompt should explicitly enforce safe and grounded behavior."""
        prompt = get_system_prompt()
        assert "read-only" in prompt
        assert "Never issue `INSERT`, `UPDATE`, `DELETE`" in prompt
        assert "Every final answer must be grounded in tool observations" in prompt
        assert "Do not state numeric results until the final answer" in prompt
        assert "For simple aggregate questions" in prompt
