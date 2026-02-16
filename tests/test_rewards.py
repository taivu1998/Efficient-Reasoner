"""Tests for the reward function module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rewards import (
    extract_xml_tag,
    is_correct,
    validate_xml_structure,
    count_tool_calls,
    accuracy_efficiency_reward_func,
)


class TestExtractXmlTag:
    """Tests for XML tag extraction."""

    def test_basic_extraction(self):
        text = "<answer>Paris</answer>"
        assert extract_xml_tag(text, "answer") == "Paris"

    def test_extraction_with_spaces(self):
        """Test fuzzy matching with spaces in tags."""
        text = "< answer >Paris< /answer >"
        assert extract_xml_tag(text, "answer") == "Paris"

    def test_extraction_multiline(self):
        text = "<thought>\nThis is a thought.\nWith multiple lines.\n</thought>"
        result = extract_xml_tag(text, "thought")
        assert "This is a thought" in result

    def test_extraction_nested_content(self):
        text = '<call>search_wiki("Paris")</call>'
        assert extract_xml_tag(text, "call") == 'search_wiki("Paris")'

    def test_extraction_not_found(self):
        text = "No tags here"
        assert extract_xml_tag(text, "answer") == ""

    def test_case_insensitive(self):
        text = "<ANSWER>Paris</ANSWER>"
        assert extract_xml_tag(text, "answer") == "Paris"


class TestIsCorrect:
    """Tests for answer correctness checking."""

    def test_exact_match(self):
        assert is_correct("Paris", "Paris")

    def test_case_insensitive(self):
        assert is_correct("paris", "Paris")
        assert is_correct("PARIS", "paris")

    def test_containment(self):
        assert is_correct("The answer is Paris", "Paris")
        assert is_correct("Paris", "The capital is Paris")

    def test_word_subset(self):
        assert is_correct("Sam Altman is the CEO", "Sam Altman")

    def test_empty_strings(self):
        assert not is_correct("", "Paris")
        assert not is_correct("Paris", "")
        assert not is_correct("", "")

    def test_wrong_answer(self):
        assert not is_correct("London", "Paris")


class TestValidateXmlStructure:
    """Tests for XML structure validation."""

    def test_valid_structure(self):
        text = "<thought>Thinking</thought><answer>Result</answer>"
        assert validate_xml_structure(text)

    def test_valid_with_tool_call(self):
        text = '<thought>Need search</thought><call>search_wiki("test")</call><obs>Found</obs><answer>Result</answer>'
        assert validate_xml_structure(text)

    def test_missing_answer(self):
        text = "<thought>Thinking</thought>"
        assert not validate_xml_structure(text)

    def test_fuzzy_match(self):
        text = "< answer >Result< /answer >"
        assert validate_xml_structure(text)


class TestCountToolCalls:
    """Tests for counting tool calls."""

    def test_no_calls(self):
        text = "<thought>Thinking</thought><answer>Result</answer>"
        assert count_tool_calls(text) == 0

    def test_single_call(self):
        text = '<call>search_wiki("test")</call>'
        assert count_tool_calls(text) == 1

    def test_multiple_calls(self):
        text = "<call>search1</call><obs>result1</obs><call>search2</call><obs>result2</obs>"
        assert count_tool_calls(text) == 2

    def test_fuzzy_match(self):
        text = "< call >search< /call >"
        assert count_tool_calls(text) == 1


class TestRewardFunction:
    """Tests for the main reward function."""

    def test_correct_no_calls(self):
        """Correct answer without tool calls should get high reward."""
        completions = ["<thought>Easy</thought><answer>Paris</answer>"]
        ground_truth = ["Paris"]
        rewards = accuracy_efficiency_reward_func(completions, ground_truth)
        # 1.0 (correct) + 0.1 (format) - 0 (no calls) + 0.1 (efficient bonus) = 1.2
        assert rewards[0] == pytest.approx(1.2)

    def test_correct_with_calls(self):
        """Correct answer with tool calls should have efficiency penalty."""
        completions = [
            "<thought>Need info</thought><call>search</call><obs>info</obs><answer>Paris</answer>"
        ]
        ground_truth = ["Paris"]
        rewards = accuracy_efficiency_reward_func(completions, ground_truth)
        # 1.0 (correct) + 0.1 (format) - 0.05 (1 call) = 1.05
        assert rewards[0] == pytest.approx(1.05)

    def test_wrong_answer(self):
        """Wrong answer should get penalty."""
        completions = ["<thought>Guess</thought><answer>London</answer>"]
        ground_truth = ["Paris"]
        rewards = accuracy_efficiency_reward_func(completions, ground_truth)
        # -0.5 (wrong) + 0.1 (format) = -0.4
        assert rewards[0] == pytest.approx(-0.4)

    def test_invalid_format(self):
        """Invalid format should get format penalty."""
        completions = ["Just Paris, no tags"]
        ground_truth = ["Paris"]
        rewards = accuracy_efficiency_reward_func(completions, ground_truth)
        # -0.5 (wrong, can't extract) + (-0.5) (bad format) = -1.0
        assert rewards[0] == pytest.approx(-1.0)

    def test_batch_processing(self):
        """Test processing multiple completions."""
        completions = [
            "<answer>Paris</answer>",
            "<answer>London</answer>",
        ]
        ground_truth = ["Paris", "Paris"]
        rewards = accuracy_efficiency_reward_func(completions, ground_truth)
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]  # First is correct, second is wrong


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
