"""Tests for dataset fallback functionality."""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (
    get_fallback_dataset,
    FALLBACK_DATASET,
    create_synthetic_sample,
    format_prompt,
)


class TestFallbackDataset:
    """Tests for fallback dataset functionality."""

    def test_fallback_dataset_not_empty(self):
        """Test that fallback dataset is not empty."""
        assert len(FALLBACK_DATASET) > 0

    def test_fallback_train_split(self):
        """Test getting train split of fallback data."""
        data = get_fallback_dataset("train")
        assert len(data) > 0
        assert all("question" in item for item in data)
        assert all("answer" in item for item in data)

    def test_fallback_validation_split(self):
        """Test getting validation split of fallback data."""
        data = get_fallback_dataset("validation")
        assert len(data) > 0
        assert len(data) <= len(FALLBACK_DATASET)

    def test_fallback_data_has_questions_and_answers(self):
        """Test that fallback data has proper structure."""
        for item in FALLBACK_DATASET:
            assert "question" in item
            assert "answer" in item
            assert isinstance(item["question"], str)
            assert isinstance(item["answer"], str)
            assert len(item["question"]) > 0
            assert len(item["answer"]) > 0


class TestSyntheticSamples:
    """Tests for synthetic sample creation."""

    def test_create_sample_without_search(self):
        """Test creating a sample without search."""
        sample = create_synthetic_sample(
            question="What is 2+2?", answer="4", use_search=False
        )

        assert sample["question"] == "What is 2+2?"
        assert sample["answer"] == "4"
        assert "trace" in sample
        assert "<answer>4</answer>" in sample["trace"]
        assert "<call>" not in sample["trace"]

    def test_create_sample_with_search(self):
        """Test creating a sample with search."""
        sample = create_synthetic_sample(
            question="Who directed Inception?",
            answer="Christopher Nolan",
            use_search=True,
        )

        assert sample["question"] == "Who directed Inception?"
        assert sample["answer"] == "Christopher Nolan"
        assert "trace" in sample
        assert "<call>" in sample["trace"]
        assert "<obs>" in sample["trace"]
        assert "<answer>Christopher Nolan</answer>" in sample["trace"]


class TestFormatPrompt:
    """Tests for prompt formatting."""

    def test_format_prompt_includes_system_prompt(self):
        """Test that formatted prompt includes system prompt."""
        sample = {"question": "What is AI?"}
        prompt = format_prompt(sample)

        assert "What is AI?" in prompt
        assert "<thought>" in prompt
        assert "<answer>" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt


class TestConstants:
    """Tests for constants module."""

    def test_import_constants(self):
        """Test that constants can be imported."""
        from src.constants import (
            DEFAULT_MAX_SEQ_LENGTH,
            DEFAULT_LEARNING_RATE,
            DEFAULT_BATCH_SIZE,
            DEFAULT_TOOL_COST,
            DEFAULT_CORRECT_REWARD,
        )

        assert DEFAULT_MAX_SEQ_LENGTH == 2048
        assert DEFAULT_LEARNING_RATE == 2.0e-5
        assert DEFAULT_BATCH_SIZE == 4
        assert DEFAULT_TOOL_COST == 0.05
        assert DEFAULT_CORRECT_REWARD == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
