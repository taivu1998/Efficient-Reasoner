"""
Phase 1: Cold Start Data Generation

Generates synthetic reasoning traces for SFT (Supervised Fine-Tuning).
This bootstraps the model to reliably output valid XML tool calls.

In production, this would use GPT-4o to generate "perfect" reasoning traces.
For this implementation, we create synthetic traces based on HotpotQA data.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import random
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm

from src.config_parser import load_config
from src.mock_env import MockSearchEnv


def generate_direct_answer_trace(question: str, answer: str) -> str:
    """
    Generate a trace for questions that can be answered directly (System 1).
    These are simple factual questions or common knowledge.
    """
    return f"<thought>This is a straightforward question that I can answer from my knowledge.</thought><answer>{answer}</answer>"


def generate_search_trace(
    question: str,
    answer: str,
    env: MockSearchEnv,
    search_query: Optional[str] = None
) -> str:
    """
    Generate a trace for questions requiring search (System 2).
    Demonstrates the tool-use pattern.
    """
    # Use provided query or extract key terms from question
    if search_query is None:
        # Simple heuristic: use question words as search query
        search_query = question.replace("?", "").strip()

    # Get observation from mock environment
    obs = env.search(search_query)

    trace = (
        f"<thought>This question requires me to look up specific information.</thought>"
        f"<call>search_wiki(\"{search_query}\")</call>"
        f"<obs>{obs}</obs>"
        f"<thought>Based on the search results, I can now answer the question.</thought>"
        f"<answer>{answer}</answer>"
    )
    return trace


def generate_multi_hop_trace(
    question: str,
    answer: str,
    env: MockSearchEnv,
    supporting_facts: List = None
) -> str:
    """
    Generate a trace for multi-hop questions requiring multiple searches.
    """
    # For multi-hop, we do two searches
    words = question.replace("?", "").split()
    query1 = " ".join(words[:len(words)//2]) if len(words) > 4 else question
    query2 = " ".join(words[len(words)//2:]) if len(words) > 4 else answer

    obs1 = env.search(query1)
    obs2 = env.search(query2)

    trace = (
        f"<thought>This is a complex question that may require multiple searches.</thought>"
        f"<call>search_wiki(\"{query1}\")</call>"
        f"<obs>{obs1}</obs>"
        f"<thought>I found some information, but I need to search for more details.</thought>"
        f"<call>search_wiki(\"{query2}\")</call>"
        f"<obs>{obs2}</obs>"
        f"<thought>Now I have enough information to answer the question.</thought>"
        f"<answer>{answer}</answer>"
    )
    return trace


def is_simple_question(question: str) -> bool:
    """
    Heuristic to determine if a question is simple enough for direct answer.
    Simple questions: math, common facts, definitions.
    """
    simple_patterns = [
        "what is",
        "2+2",
        "capital of",
        "how many",
        "what color",
        "what year was",
    ]
    question_lower = question.lower()
    return any(pattern in question_lower for pattern in simple_patterns)


def generate_sft_data(
    config: dict,
    num_samples: int = 500,
    output_path: str = "data/processed/sft_data.json"
) -> List[Dict]:
    """
    Generate SFT training data from HotpotQA.

    Creates a mix of:
    - Direct answer traces (for simple questions)
    - Single-search traces (for most questions)
    - Multi-hop traces (for complex questions)

    Args:
        config: Configuration dictionary
        num_samples: Number of samples to generate
        output_path: Path to save the generated data

    Returns:
        List of training examples
    """
    print(f"Generating {num_samples} SFT training examples...")

    # Initialize mock environment
    env = MockSearchEnv()

    # Load HotpotQA
    ds = load_dataset(
        config['data']['dataset_name'],
        config['data']['subset'],
        split='train',
        trust_remote_code=True
    )

    # Sample questions
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))

    data = []
    direct_count = 0
    search_count = 0
    multi_hop_count = 0

    for idx in tqdm(indices, desc="Generating traces"):
        item = ds[idx]
        question = item['question']
        answer = item['answer']
        q_type = item.get('type', 'unknown')

        # Decide trace type based on question complexity
        # Mix: 20% direct, 60% single search, 20% multi-hop
        rand = random.random()

        if rand < 0.2 or is_simple_question(question):
            trace = generate_direct_answer_trace(question, answer)
            direct_count += 1
        elif rand < 0.8 or q_type == 'comparison':
            trace = generate_search_trace(question, answer, env)
            search_count += 1
        else:
            trace = generate_multi_hop_trace(question, answer, env)
            multi_hop_count += 1

        data.append({
            "question": question,
            "answer": answer,
            "trace": trace,
            "type": q_type
        })

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nGenerated SFT data saved to {output_path}")
    print(f"Statistics:")
    print(f"  - Direct answer traces: {direct_count}")
    print(f"  - Single search traces: {search_count}")
    print(f"  - Multi-hop traces: {multi_hop_count}")
    print(f"  - Total: {len(data)}")

    return data


def main():
    """
    Phase 1: Cold Start Data Generation.
    Generates synthetic reasoning traces for SFT training.
    """
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of training samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/sft_data.json",
        help="Output path for generated data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Generate data
    generate_sft_data(
        config=config,
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
