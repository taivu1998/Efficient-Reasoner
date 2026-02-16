"""Tests for the mock search environment."""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mock_env import MockSearchEnv


class TestMockSearchEnv:
    """Tests for MockSearchEnv."""

    @pytest.fixture
    def env(self):
        """Create a mock environment with fallback data."""
        env = MockSearchEnv()
        # Force fallback data for predictable testing
        env._load_fallback_data()
        return env

    def test_initialization(self, env):
        """Test environment initializes with data."""
        assert len(env) > 0

    def test_exact_match(self, env):
        """Test exact title lookup."""
        result = env.search("inception")
        assert "Inception" in result or "Christopher Nolan" in result

    def test_case_insensitive(self, env):
        """Test case-insensitive search."""
        result1 = env.search("INCEPTION")
        result2 = env.search("inception")
        assert result1 == result2

    def test_partial_match(self, env):
        """Test partial title matching."""
        result = env.search("nolan")
        assert "Nolan" in result or "Christopher" in result

    def test_no_match(self, env):
        """Test handling of no matches."""
        result = env.search("xyznonexistent123")
        assert "No relevant information found" in result

    def test_quote_stripping(self, env):
        """Test that quotes are stripped from queries."""
        result1 = env.search('"inception"')
        result2 = env.search("'inception'")
        result3 = env.search("inception")
        assert result1 == result3
        assert result2 == result3

    def test_contains_operator(self, env):
        """Test the __contains__ operator."""
        assert "inception" in env
        assert "xyznonexistent" not in env

    def test_truncation(self, env):
        """Test that results are truncated to 500 chars."""
        # Add a long document
        env.knowledge_base["longdoc"] = "x" * 1000
        result = env.search("longdoc")
        assert len(result) <= 500


class TestMockSearchEnvMulti:
    """Tests for multi-result search."""

    @pytest.fixture
    def env(self):
        env = MockSearchEnv()
        env._load_fallback_data()
        return env

    def test_search_multi_returns_list(self, env):
        """Test that search_multi returns a list."""
        results = env.search_multi("inception")
        assert isinstance(results, list)

    def test_search_multi_respects_top_k(self, env):
        """Test that search_multi respects top_k parameter."""
        results = env.search_multi("the", top_k=2)
        assert len(results) <= 2

    def test_search_multi_no_results(self, env):
        """Test search_multi with no matches."""
        results = env.search_multi("xyznonexistent123")
        assert len(results) == 1
        assert "No relevant information found" in results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
