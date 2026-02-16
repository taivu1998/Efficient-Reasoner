import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import argparse
import re
import gc
from typing import Optional, Tuple, List
from transformers import StoppingCriteria, StoppingCriteriaList

from src.model_utils import load_model_for_inference
from src.config_parser import load_config
from src.mock_env import MockSearchEnv
from src.rewards import extract_xml_tag
from src.dataset import SYSTEM_PROMPT


# Default inference constants
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_TOTAL_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1


class StopOnXMLTags(StoppingCriteria):
    """
    Enhanced stopping criteria that stops on XML tag completion.
    """

    def __init__(self, tokenizer, stop_strings: List[str], decode_length: int = 50):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.decode_length = decode_length

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last few tokens and check for stop strings
        generated_text = self.tokenizer.decode(
            input_ids[0][-self.decode_length :], skip_special_tokens=False
        )
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True
        return False


class MaxTokensStoppingCriteria(StoppingCriteria):
    """Stop when max tokens generated."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.current_tokens = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.current_tokens = input_ids.shape[1]
        return self.current_tokens >= self.max_tokens


def detect_stuck_generation(
    generated_text: str, previous_text: str, threshold: int = 50
) -> bool:
    """
    Detect if generation is stuck (repeating the same text).

    Args:
        generated_text: Current generated text
        previous_text: Text from previous iteration
        threshold: Minimum new characters to not be considered stuck

    Returns:
        True if generation appears stuck
    """
    if not previous_text:
        return False

    # Check if we have significant new content
    new_content = generated_text[len(previous_text) :]
    if len(new_content.strip()) < threshold:
        return True

    # Check for repetitive patterns (more sophisticated)
    # If the same n-gram appears too frequently
    words = new_content.split()
    if len(words) < 10:
        return True

    # Simple repetition check
    if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
        return True

    return False


def find_unclosed_tag(text: str) -> Optional[str]:
    """
    Find unclosed XML tags in text.

    Returns:
        The tag name if an unclosed tag is found, None otherwise
    """
    # Check for unclosed call tags
    call_starts = len(re.findall(r"<\s*call\s*>", text, re.IGNORECASE))
    call_ends = len(re.findall(r"<\s*/\s*call\s*>", text, re.IGNORECASE))

    if call_starts > call_ends:
        return "call"

    # Check for unclosed obs tags
    obs_starts = len(re.findall(r"<\s*obs\s*>", text, re.IGNORECASE))
    obs_ends = len(re.findall(r"<\s*/\s*obs\s*>", text, re.IGNORECASE))

    if obs_starts > obs_ends:
        return "obs"

    return None


def run_agentic_inference(
    model,
    tokenizer,
    prompt,
    max_steps: int = 5,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    verbose: bool = True,
    env: Optional[MockSearchEnv] = None,
) -> Tuple[str, str, int]:
    """
    Executes the End-to-End Agentic Loop with proper loop protection:
    Think -> Call? -> MockExec -> Append Obs -> Resume -> Answer

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: User question/query
        max_steps: Maximum number of agentic steps (tool calls)
        max_new_tokens: Maximum new tokens per generation step
        max_total_tokens: Maximum total tokens in context
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        repetition_penalty: Repetition penalty
        verbose: Whether to print intermediate steps
        env: Optional pre-initialized MockSearchEnv

    Returns:
        Tuple of (final_answer, full_trace, num_tool_calls)
    """
    if env is None:
        env = MockSearchEnv()

    # Use consistent system prompt from dataset.py
    current_context = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Question: {prompt}")
        print("=" * 60)

    num_tool_calls = 0
    full_trace = ""
    total_tokens_generated = 0
    previous_content = ""

    # Calculate effective max tokens (leave room for context)
    effective_max_length = min(tokenizer.model_max_length, max_total_tokens)
    max_new_per_step = min(max_new_tokens, effective_max_length // (max_steps + 1))

    # Create stopping criteria - stop on complete XML tags
    stop_strings = ["</call>", "</answer>", "</thought>"]
    stopping_criteria_list = [
        StopOnXMLTags(tokenizer, stop_strings, decode_length=30),
        MaxTokensStoppingCriteria(max_tokens=effective_max_length),
    ]
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)

    for step in range(max_steps):
        # Check context length before generation
        context_tokens = len(tokenizer.encode(current_context))
        if context_tokens >= effective_max_length - 100:
            if verbose:
                print(f"\n[Warning] Context length limit reached at step {step}")
            break

        # Check if we're stuck
        if detect_stuck_generation(full_trace, previous_content):
            if verbose:
                print(
                    f"\n[Warning] Generation appears stuck at step {step}. Ending loop."
                )
            break

        inputs = tokenizer(
            current_context,
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_length - max_new_per_step,
        ).to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_per_step,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
        except Exception as e:
            if verbose:
                print(f"\n[Error] Generation failed: {e}")
            break

        # Extract only the newly generated text
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        new_content = tokenizer.decode(new_tokens, skip_special_tokens=False)

        total_tokens_generated += len(new_tokens)

        if verbose:
            print(new_content, end="", flush=True)

        # Store previous content for stuck detection
        previous_content = full_trace
        full_trace += new_content
        current_context += new_content

        # Track stopping criteria state
        stopped_early = False

        # Check for Tool Call
        if "</call>" in new_content:
            num_tool_calls += 1
            call_content = extract_xml_tag(new_content, "call")

            # Parse query: search_wiki("query") or just the query text
            query_match = re.search(
                r'search_wiki\(["\']?(.*?)["\']?\)', call_content, re.IGNORECASE
            )
            if query_match:
                query = query_match.group(1)
            else:
                query = call_content.strip()

            # Execute in Mock Env
            obs = env.search(query)
            obs_str = f"\n<obs>{obs}</obs>\n"

            if verbose:
                print(obs_str, end="", flush=True)

            full_trace += obs_str
            current_context += obs_str

            # Reset stuck detection after tool execution
            previous_content = ""
            continue

        # Check for unclosed tags - if found, continue to get completion
        unclosed = find_unclosed_tag(new_content)
        if unclosed and "</answer>" not in new_content:
            if verbose:
                print(f"\n[Info] Found unclosed <{unclosed}> tag, continuing...")
            continue

        # Check for Finish
        if "</answer>" in new_content:
            break

        # Check if we've generated a lot without stopping
        if total_tokens_generated > max_total_tokens * 0.8:
            if verbose:
                print(f"\n[Warning] Approaching token limit. Ending loop.")
            break

    # Extract final answer
    final_answer = extract_xml_tag(full_trace, "answer")

    # If no answer found, try to extract anything reasonable
    if not final_answer:
        # Try to get the last bit of text as fallback
        if full_trace:
            final_answer = full_trace.split("</")[-1].split(">")[-1].strip()
            if not final_answer:
                final_answer = full_trace.strip().split("\n")[-1][:200]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Final Answer: {final_answer}")
        print(f"Tool Calls: {num_tool_calls}")
        print(f"Total Tokens Generated: {total_tokens_generated}")
        print("=" * 60)

    return final_answer, full_trace, num_tool_calls


def cleanup_model(model):
    """Clean up model from memory."""
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_batch_inference(
    model, tokenizer, questions: List[str], max_steps: int = 5, cleanup: bool = True
) -> List[dict]:
    """
    Run inference on a batch of questions with proper memory management.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of question strings
        max_steps: Maximum agentic steps per question
        cleanup: Whether to clean up memory after each inference

    Returns:
        List of result dictionaries
    """
    env = MockSearchEnv()
    results = []

    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] Processing...")

        answer, trace, num_calls = run_agentic_inference(
            model, tokenizer, question, max_steps, verbose=False, env=env
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "trace": trace,
                "num_tool_calls": num_calls,
            }
        )

        print(
            f"  Answer: {answer[:100]}..."
            if len(answer) > 100
            else f"  Answer: {answer}"
        )
        print(f"  Tool calls: {num_calls}")

        # Periodic cleanup
        if cleanup and i % 10 == 5:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run agentic inference with trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained LoRA adapter or model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Who is the director of the movie Inception?",
        help="Question to ask the model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Maximum number of agentic steps"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens per generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (0 to disable sampling)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model for inference (handles LoRA adapter loading)
    print(f"Loading model from {args.checkpoint}...")
    try:
        model, tokenizer = load_model_for_inference(args.checkpoint, config)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference
    try:
        run_agentic_inference(
            model,
            tokenizer,
            args.query,
            args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
