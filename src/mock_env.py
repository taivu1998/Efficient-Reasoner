import json
import os
from typing import Dict, Optional, List
from datasets import load_dataset

# Lazy logger initialization to avoid circular imports
_logger = None


def get_logger():
    global _logger
    if _logger is None:
        from src.utils import setup_logging
        _logger = setup_logging()
    return _logger


class MockSearchEnv:
    """
    A Static Mock Environment for RL Training.

    Optimized for speed:
    1. Pre-loads 'Gold Paragraphs' from HotpotQA.
    2. Returns findings instantly (dict lookup).
    3. Latency: <1ms (vs 1.5s for real API).

    This simulates a "Perfect Search Engine" but allows the RL loop
    to run at thousands of steps per hour.
    """

    def __init__(self, data_path: Optional[str] = None, max_docs: int = 1000):
        """
        Initialize the mock search environment.

        Args:
            data_path: Optional path to pre-built index JSON file
            max_docs: Maximum number of documents to index from HotpotQA
        """
        self.knowledge_base: Dict[str, str] = {}
        self.max_docs = max_docs

        if data_path and os.path.exists(data_path):
            self._load_index(data_path)
        else:
            self._build_index()

    def _load_index(self, path: str):
        """Load pre-built index from JSON file."""
        logger = get_logger()
        logger.info(f"Loading index from {path}...")
        with open(path, 'r') as f:
            self.knowledge_base = json.load(f)
        logger.info(f"Loaded {len(self.knowledge_base)} documents.")

    def _build_index(self):
        """
        Loads HotpotQA subset and maps Entity -> Supporting Fact.

        HotpotQA context structure:
        - context['title']: List of document titles
        - context['sentences']: List of list of sentences for each document
        """
        logger = get_logger()
        logger.info("Building Mock Search Index (Distractor Subset)...")

        try:
            ds = load_dataset(
                "hotpot_qa",
                "distractor",
                split="train",
                trust_remote_code=True
            )
            ds = ds.select(range(min(self.max_docs, len(ds))))

            for item in ds:
                context = item['context']
                titles = context['title']
                sentences_list = context['sentences']

                # Pair each title with its sentences
                for title, sentences in zip(titles, sentences_list):
                    # Join sentences into a paragraph
                    text = " ".join(sentences)
                    # Store with lowercase key for case-insensitive lookup
                    self.knowledge_base[title.lower()] = text

            logger.info(f"Index built with {len(self.knowledge_base)} documents.")

        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}. Using fallback data.")
            self._load_fallback_data()

    def _load_fallback_data(self):
        """Load minimal fallback data for testing."""
        self.knowledge_base = {
            "inception": "Inception is a 2010 science fiction film written and directed by Christopher Nolan.",
            "christopher nolan": "Christopher Nolan is a British-American filmmaker known for directing The Dark Knight trilogy and Inception.",
            "openai": "OpenAI is an AI research company. Sam Altman is the CEO of OpenAI.",
            "paris": "Paris is the capital and largest city of France.",
            "test": "This is a test document for fallback testing."
        }

    def save_index(self, path: str):
        """Save the index to a JSON file for faster loading."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.knowledge_base, f)
        get_logger().info(f"Index saved to {path}")

    def search(self, query: str) -> str:
        """
        Simulates a search engine.
        Returns the paragraph if keyword match found, else generic 'No result'.

        Args:
            query: Search query string

        Returns:
            Retrieved paragraph or 'No relevant information found' message
        """
        # Normalize query
        query_lower = query.lower().strip()
        # Remove quotes that model might include
        query_lower = query_lower.strip("'\"")

        # O(1) Exact title lookup
        if query_lower in self.knowledge_base:
            return self.knowledge_base[query_lower][:500]  # Truncate for context length

        # O(N) Partial match on titles
        for key, value in self.knowledge_base.items():
            if query_lower in key or key in query_lower:
                return value[:500]

        # O(N) Content search as last resort
        for key, value in self.knowledge_base.items():
            if query_lower in value.lower():
                return value[:500]

        return "No relevant information found."

    def search_multi(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search and return multiple relevant documents.

        Args:
            query: Search query string
            top_k: Number of documents to return

        Returns:
            List of retrieved paragraphs
        """
        query_lower = query.lower().strip().strip("'\"")
        results = []

        # Exact match first
        if query_lower in self.knowledge_base:
            results.append(self.knowledge_base[query_lower][:500])

        # Partial title matches
        for key, value in self.knowledge_base.items():
            if len(results) >= top_k:
                break
            if (query_lower in key or key in query_lower) and value[:500] not in results:
                results.append(value[:500])

        # Content matches
        for key, value in self.knowledge_base.items():
            if len(results) >= top_k:
                break
            if query_lower in value.lower() and value[:500] not in results:
                results.append(value[:500])

        if not results:
            return ["No relevant information found."]

        return results

    def __len__(self):
        return len(self.knowledge_base)

    def __contains__(self, key: str):
        return key.lower() in self.knowledge_base
