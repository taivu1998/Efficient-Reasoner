"""
Tool functions for GRPO training.

These functions are used by the GRPOTrainer's native tool support
to execute tools during the generation loop.
"""

from typing import List, Dict, Any, Optional
import os
import json

_mock_env = None


def _get_mock_env():
    """Lazy initialization of mock environment."""
    global _mock_env
    if _mock_env is None:
        from src.mock_env import MockSearchEnv

        _mock_env = MockSearchEnv()
    return _mock_env


def search_wiki(query: str) -> str:
    """
    Search Wikipedia for information.

    This tool searches a pre-indexed knowledge base for relevant information.
    Use this when you need factual information that you may not know.

    Args:
        query: The search query string. Can be a topic, question, or keywords.

    Returns:
        Search results as a string, or "No relevant information found." if no match.

    Example:
        >>> search_wiki("Christopher Nolan")
        "Christopher Nolan is a British-American filmmaker..."
    """
    env = _get_mock_env()
    result = env.search(query)
    return result


def search_wiki_multi(queries: List[str]) -> List[str]:
    """
    Search Wikipedia for multiple queries at once.

    Use this when you need to gather information from multiple sources
    in a single step for efficiency.

    Args:
        queries: A list of search query strings.

    Returns:
        A list of search results corresponding to each query.

    Example:
        >>> search_wiki_multi(["Inception", "Christopher Nolan"])
        ["Inception is a 2010 science fiction film...", "Christopher Nolan is..."]
    """
    env = _get_mock_env()
    results = []
    for query in queries:
        results.append(env.search(query))
    return results


def get_entity_info(entity: str) -> Dict[str, Any]:
    """
    Get detailed information about an entity.

    This tool retrieves structured information about a specific entity
    like a person, place, organization, or concept.

    Args:
        entity: The name of the entity to look up.

    Returns:
        A dictionary with entity information, or {"error": "Not found"} if unknown.
    """
    env = _get_mock_env()
    result = env.search(entity)

    if "No relevant information found" in result:
        return {"error": "Not found", "entity": entity}

    return {"entity": entity, "information": result, "source": "wiki_search"}


def count_tokens_estimate(text: str) -> int:
    """
    Estimate the number of tokens in text.

    This is a rough approximation for token counting (~4 chars per token).

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


TOOL_FUNCTIONS = [
    search_wiki,
    search_wiki_multi,
    get_entity_info,
]


def get_tools() -> List:
    """
    Get all available tool functions for GRPO training.

    Returns:
        List of callable tool functions.
    """
    return TOOL_FUNCTIONS


def reset_mock_env():
    """Reset the mock environment (useful for testing)."""
    global _mock_env
    _mock_env = None
