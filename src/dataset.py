from datasets import load_dataset, Dataset
from typing import Dict, Optional, List
import json
import os
import logging

logger = logging.getLogger("EfficientReasoning")

SYSTEM_PROMPT = """You are an efficient, reasoning-focused AI assistant.
Your goal is to answer the user's question accurately while minimizing computational cost.

1. Think step-by-step inside <thought> tags.
2. If you are confident, output the answer immediately in <answer> tags.
3. If you need information, use the <call> tag to search.
4. Do not search if the answer is common knowledge.

Format:
<thought>Reasoning...</thought>
<call>search_wiki("query")</call>
<obs>Observation...</obs>
<answer>Final Answer</answer>
"""

FALLBACK_DATASET = [
    {"question": "Who directed the movie Inception?", "answer": "Christopher Nolan"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote the novel 1984?", "answer": "George Orwell"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "Who is the CEO of OpenAI?", "answer": "Sam Altman"},
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
    },
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "In what year was the iPhone first released?", "answer": "2007"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {
        "question": "What is the speed of light?",
        "answer": "299,792,458 meters per second",
    },
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    {"question": "What is the tallest mountain on Earth?", "answer": "Mount Everest"},
    {"question": "Who invented the printing press?", "answer": "Johannes Gutenberg"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {
        "question": "Who was the first person to walk on the moon?",
        "answer": "Neil Armstrong",
    },
    {"question": "What year did the Titanic sink?", "answer": "1912"},
    {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
    {
        "question": "What is the smallest country in the world?",
        "answer": "Vatican City",
    },
    {
        "question": "Who developed the theory of relativity?",
        "answer": "Albert Einstein",
    },
    {"question": "What is the longest river in the world?", "answer": "Nile"},
    {"question": "Who painted Starry Night?", "answer": "Vincent van Gogh"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
    {"question": "What is the hardest natural substance?", "answer": "Diamond"},
]


def format_prompt(sample: Dict) -> str:
    """Formats the input prompt for the model."""
    question = sample["question"]
    return f"{SYSTEM_PROMPT}\n\nUser: {question}\nAssistant:"


def get_fallback_dataset(split: str = "train") -> List[Dict]:
    """
    Returns a fallback dataset for when HotpotQA cannot be loaded.

    Args:
        split: Dataset split (train or validation)

    Returns:
        List of sample dictionaries
    """
    if split == "validation":
        return FALLBACK_DATASET[:10]
    return FALLBACK_DATASET


def load_grpo_dataset(config: dict, split: str = "train"):
    """
    Loads HotpotQA and formats it for GRPO.
    Returns a dataset object compatible with GRPOTrainer.

    Falls back to synthetic data if HotpotQA cannot be loaded.

    GRPOTrainer expects:
    - 'prompt': The input prompt for the model
    - Additional columns are passed to reward function via **kwargs

    Args:
        config: Configuration dictionary
        split: Dataset split to load ('train' or 'validation')

    Returns:
        Dataset formatted for GRPOTrainer
    """
    fallback_on_error = config.get("data", {}).get("fallback_on_error", True)

    try:
        ds = load_dataset(
            config["data"]["dataset_name"],
            config["data"]["subset"],
            split=split,
            trust_remote_code=True,
        )
        logger.info(f"Loaded {len(ds)} samples from {config['data']['dataset_name']}")
    except Exception as e:
        if fallback_on_error:
            logger.warning(f"Failed to load dataset: {e}. Using fallback data.")
            samples = get_fallback_dataset(split)
            ds = Dataset.from_list(samples)
        else:
            raise

    max_samples = config["data"].get("max_samples")
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    def process(x):
        return {"prompt": format_prompt(x), "answer_ground_truth": x["answer"]}

    original_columns = ds.column_names
    ds = ds.map(process, remove_columns=original_columns)

    return ds


def load_sft_dataset(config: dict, data_path: Optional[str] = None):
    """
    Loads SFT dataset from generated traces.

    Args:
        config: Configuration dictionary
        data_path: Path to SFT data JSON file

    Returns:
        Dataset formatted for SFT training
    """
    if data_path is None:
        data_path = "data/processed/sft_data.json"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"SFT data not found at {data_path}. "
            "Run 'python scripts/generate_sft.py' first."
        )

    with open(data_path, "r") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        formatted_data.append(
            {
                "prompt": format_prompt({"question": item["question"]}),
                "completion": item["trace"],
                "answer_ground_truth": item["answer"],
            }
        )

    return Dataset.from_list(formatted_data)


def load_evaluation_dataset(config: dict, max_samples: Optional[int] = None):
    """
    Loads HotpotQA validation set for evaluation.

    Falls back to synthetic data if HotpotQA cannot be loaded.

    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to load

    Returns:
        Dataset with questions and ground truth answers
    """
    fallback_on_error = config.get("data", {}).get("fallback_on_error", True)

    try:
        ds = load_dataset(
            config["data"]["dataset_name"],
            config["data"]["subset"],
            split="validation",
            trust_remote_code=True,
        )
        logger.info(f"Loaded {len(ds)} validation samples")
    except Exception as e:
        if fallback_on_error:
            logger.warning(
                f"Failed to load validation dataset: {e}. Using fallback data."
            )
            samples = get_fallback_dataset("validation")
            ds = Dataset.from_list(samples)
        else:
            raise

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    def process(x):
        return {
            "question": x["question"],
            "prompt": format_prompt(x),
            "answer_ground_truth": x["answer"],
            "supporting_facts": x.get("supporting_facts", []),
            "context": x.get("context", {}),
        }

    original_columns = ds.column_names
    ds = ds.map(process, remove_columns=original_columns)

    return ds


def create_synthetic_sample(
    question: str, answer: str, use_search: bool = False
) -> Dict:
    """
    Creates a synthetic training sample for SFT.

    Args:
        question: The question string
        answer: The ground truth answer
        use_search: Whether to include a search call in the trace

    Returns:
        Dictionary with question, answer, and trace
    """
    if use_search:
        trace = (
            f"<thought>This question requires specific information that I should search for.</thought>"
            f'<call>search_wiki("{question}")</call>'
            f"<obs>Based on search results: {answer}</obs>"
            f"<answer>{answer}</answer>"
        )
    else:
        trace = f"<thought>This is a straightforward question I can answer directly.</thought><answer>{answer}</answer>"

    return {"question": question, "answer": answer, "trace": trace}
