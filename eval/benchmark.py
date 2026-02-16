"""
Phase 3: Evaluation Benchmark

Runs the test set through different model configurations and collects metrics
for the Pareto frontier analysis.
"""

import sys
import os
import gc
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import torch

from src.config_parser import load_config
from src.dataset import load_evaluation_dataset
from src.model_utils import (
    load_model_for_inference,
    load_model_and_tokenizer,
    cleanup_model,
)
from src.rewards import is_correct, extract_xml_tag, count_tool_calls
from src.mock_env import MockSearchEnv
from scripts.inference import run_agentic_inference

logger = logging.getLogger("EfficientReasoning")


@dataclass
class EvaluationResult:
    """Results from evaluating a single example."""

    question: str
    ground_truth: str
    prediction: str
    is_correct: bool
    num_tool_calls: int
    total_tokens: int
    trace: str


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    model_name: str
    accuracy: float
    avg_tool_calls: float
    avg_tokens: float
    total_samples: int
    correct_samples: int
    results: List[Dict]


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def get_memory_usage() -> Optional[Dict[str, float]]:
    """Get current GPU memory usage if available."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
        }
    return None


def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_cached_results(checkpoint_path: str, output_dir: str) -> Optional[Dict]:
    """Load cached benchmark results if available."""
    cache_file = os.path.join(
        output_dir, f"{checkpoint_path.replace('/', '_')}_cached.json"
    )
    if os.path.exists(cache_file):
        logger.info(f"Loading cached results from {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def save_cached_results(checkpoint_path: str, output_dir: str, results: Dict):
    """Save results to cache."""
    cache_file = os.path.join(
        output_dir, f"{checkpoint_path.replace('/', '_')}_cached.json"
    )
    with open(cache_file, "w") as f:
        json.dump(results, f)


def evaluate_model(
    model,
    tokenizer,
    dataset,
    max_samples: Optional[int] = None,
    max_steps: int = 5,
    verbose: bool = False,
    cache_results_every: int = 10,
) -> BenchmarkResults:
    """
    Evaluate a model on the benchmark dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: Evaluation dataset
        max_samples: Maximum number of samples to evaluate
        max_steps: Maximum agentic steps per question
        verbose: Whether to print progress
        cache_results_every: Save intermediate results every N samples

    Returns:
        BenchmarkResults with aggregated metrics
    """
    results = []
    correct_count = 0
    total_tool_calls = 0
    total_tokens = 0

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    iterator = (
        tqdm(range(num_samples), desc="Evaluating") if verbose else range(num_samples)
    )

    for i in iterator:
        sample = dataset[i]
        question = sample["question"]
        ground_truth = sample["answer_ground_truth"]

        try:
            prediction, trace, num_calls = run_agentic_inference(
                model, tokenizer, question, max_steps=max_steps, verbose=False
            )
        except Exception as e:
            logger.warning(f"Inference failed for sample {i}: {e}")
            prediction = ""
            trace = ""
            num_calls = 0

        correct = is_correct(prediction, ground_truth)
        tokens = count_tokens(trace, tokenizer)

        if correct:
            correct_count += 1
        total_tool_calls += num_calls
        total_tokens += tokens

        result = EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            is_correct=correct,
            num_tool_calls=num_calls,
            total_tokens=tokens,
            trace=trace,
        )
        results.append(asdict(result))

        if verbose and (i + 1) % 10 == 0:
            mem = get_memory_usage()
            if mem:
                logger.info(f"Memory: {mem['allocated']:.2f}GB allocated")

    accuracy = correct_count / num_samples if num_samples > 0 else 0
    avg_tool_calls = total_tool_calls / num_samples if num_samples > 0 else 0
    avg_tokens = total_tokens / num_samples if num_samples > 0 else 0

    return BenchmarkResults(
        model_name="",
        accuracy=accuracy,
        avg_tool_calls=avg_tool_calls,
        avg_tokens=avg_tokens,
        total_samples=num_samples,
        correct_samples=correct_count,
        results=results,
    )


def run_benchmark(
    config: dict,
    checkpoints: Dict[str, str],
    output_dir: str = "eval/results",
    max_samples: int = 100,
    max_steps: int = 5,
    use_cache: bool = True,
) -> Dict[str, BenchmarkResults]:
    """
    Run benchmark on multiple model checkpoints.

    Args:
        config: Configuration dictionary
        checkpoints: Dict mapping model names to checkpoint paths
        output_dir: Directory to save results
        max_samples: Maximum samples per model
        max_steps: Maximum agentic steps
        use_cache: Whether to use cached results

    Returns:
        Dictionary mapping model names to their results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading evaluation dataset...")
    dataset = load_evaluation_dataset(config, max_samples=max_samples)
    print(f"Loaded {len(dataset)} evaluation samples")

    all_results = {}

    eval_config = config.get("evaluation", {})
    cache_enabled = use_cache and eval_config.get("cache_results", True)

    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)

        mem_before = get_memory_usage()
        if mem_before:
            print(f"Memory before: {mem_before['allocated']:.2f}GB")

        # Try to load cached results
        if cache_enabled and checkpoint_path != "base":
            cached = load_cached_results(checkpoint_path, output_dir)
            if cached:
                results = BenchmarkResults(
                    model_name=model_name,
                    accuracy=cached["accuracy"],
                    avg_tool_calls=cached["avg_tool_calls"],
                    avg_tokens=cached["avg_tokens"],
                    total_samples=cached["total_samples"],
                    correct_samples=cached["correct_samples"],
                    results=cached["results"],
                )
                all_results[model_name] = results
                print(f"  Loaded from cache!")
                continue

        # Load model
        try:
            if checkpoint_path == "base":
                model, tokenizer = load_model_and_tokenizer(config, for_training=False)
            else:
                model, tokenizer = load_model_for_inference(checkpoint_path, config)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue

        # Evaluate
        results = evaluate_model(
            model,
            tokenizer,
            dataset,
            max_samples=max_samples,
            max_steps=max_steps,
            verbose=True,
        )
        results.model_name = model_name

        # Print summary
        print(f"\nResults for {model_name}:")
        print(f"  Accuracy: {results.accuracy:.2%}")
        print(f"  Avg Tool Calls: {results.avg_tool_calls:.2f}")
        print(f"  Avg Tokens: {results.avg_tokens:.1f}")

        # Save individual results
        result_path = os.path.join(
            output_dir, f"{model_name.replace('/', '_')}_results.json"
        )
        results_dict = {
            "model_name": results.model_name,
            "accuracy": results.accuracy,
            "avg_tool_calls": results.avg_tool_calls,
            "avg_tokens": results.avg_tokens,
            "total_samples": results.total_samples,
            "correct_samples": results.correct_samples,
            "results": results.results,
        }
        with open(result_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"  Results saved to: {result_path}")

        # Save to cache
        if cache_enabled and checkpoint_path != "base":
            save_cached_results(checkpoint_path, output_dir, results_dict)

        all_results[model_name] = results

        # Cleanup
        cleanup_model(model)
        mem_after = get_memory_usage()
        if mem_after:
            print(f"Memory after cleanup: {mem_after['allocated']:.2f}GB")

    # Save summary
    summary = {
        name: {
            "accuracy": r.accuracy,
            "avg_tool_calls": r.avg_tool_calls,
            "avg_tokens": r.avg_tokens,
            "total_samples": r.total_samples,
        }
        for name, r in all_results.items()
    }

    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=["base"],
        help="Model checkpoints to evaluate (use 'base' for base model)",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Names for each checkpoint (defaults to checkpoint paths)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Maximum agentic steps per question"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable result caching"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    names = args.names if args.names else args.checkpoints
    if len(names) != len(args.checkpoints):
        names = args.checkpoints

    checkpoints = dict(zip(names, args.checkpoints))

    results = run_benchmark(
        config=config,
        checkpoints=checkpoints,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        use_cache=not args.no_cache,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    for name, r in results.items():
        print(
            f"  {name}: Accuracy={r.accuracy:.2%}, AvgCalls={r.avg_tool_calls:.2f}, AvgTokens={r.avg_tokens:.1f}"
        )


if __name__ == "__main__":
    main()
