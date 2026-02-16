"""Tests for dataset loading and formatting."""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import format_prompt, SYSTEM_PROMPT


class TestFormatPrompt:
    """Tests for prompt formatting."""

    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        sample = {"question": "What is 2+2?"}
        prompt = format_prompt(sample)

        assert "What is 2+2?" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt

    def test_format_prompt_includes_system_prompt(self):
        """Test that system prompt is included."""
        sample = {"question": "Test question"}
        prompt = format_prompt(sample)

        assert "<thought>" in prompt  # Part of system prompt format
        assert "<answer>" in prompt

    def test_system_prompt_structure(self):
        """Test system prompt has required elements."""
        assert "<thought>" in SYSTEM_PROMPT
        assert "<call>" in SYSTEM_PROMPT
        assert "<obs>" in SYSTEM_PROMPT
        assert "<answer>" in SYSTEM_PROMPT
        assert "search_wiki" in SYSTEM_PROMPT


class TestSystemPrompt:
    """Tests for the system prompt."""

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_has_instructions(self):
        """Test that system prompt has numbered instructions."""
        assert "1." in SYSTEM_PROMPT
        assert "2." in SYSTEM_PROMPT

    def test_system_prompt_format_example(self):
        """Test that system prompt shows the format."""
        assert "Format:" in SYSTEM_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
