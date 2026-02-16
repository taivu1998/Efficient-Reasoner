import re
import json
from typing import List, Dict, Optional, Any

# Default reward constants (can be overridden by config)
DEFAULT_CORRECT_REWARD = 1.0
DEFAULT_WRONG_PENALTY = -0.5
DEFAULT_FORMAT_REWARD = 0.1
DEFAULT_FORMAT_PENALTY = -0.5
DEFAULT_TOOL_COST = 0.05
DEFAULT_INCOMPLETE_CALL_PENALTY = -0.2
DEFAULT_EFFICIENT_BONUS = 0.1


def extract_xml_tag(text: str, tag: str) -> str:
    """
    Robust regex extraction for content between <tag> and </tag>.
    Handles common typos like extra spaces: < tag>, <tag >, < tag >.
    """
    # Fuzzy pattern that handles spaces around tag names
    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_xml_tag_strict(text: str, tag: str) -> str:
    """Strict regex extraction without fuzzy matching."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def is_correct(prediction: str, ground_truth: str) -> bool:
    """
    Checks if the prediction matches the ground truth using fuzzy matching.
    Uses normalized string comparison with containment check.
    """
    if not prediction or not ground_truth:
        return False

    # Normalize: lowercase, strip whitespace, remove punctuation for comparison
    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()

    # Exact match
    if pred == gt:
        return True

    # Containment match (either direction)
    if gt in pred or pred in gt:
        return True

    # Word-level match for short answers
    pred_words = set(pred.split())
    gt_words = set(gt.split())
    if gt_words and gt_words.issubset(pred_words):
        return True

    return False


def validate_xml_structure(text: str) -> bool:
    """
    Checks if the text contains valid XML structure for agentic workflow.
    Uses fuzzy matching to handle model typos.
    """
    # Must contain answer tags (with fuzzy matching for spaces)
    answer_pattern = r"<\s*answer\s*>.*?<\s*/\s*answer\s*>"
    if not re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE):
        return False
    return True


def count_tool_calls(text: str) -> int:
    """
    Counts the number of tool calls in the text.
    Uses fuzzy matching for <call> tags.
    """
    # Fuzzy pattern for <call> tags
    pattern = r"<\s*call\s*>"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


def verify_tool_execution(completion: str) -> Dict[str, Any]:
    """
    Verifies that tool calls are properly executed (obs follows call).

    Returns:
        Dict with:
        - 'num_calls': number of <call> tags
        - 'num_obs': number of <obs> tags
        - 'complete_calls': number of calls that have corresponding obs
        - 'has_incomplete_calls': whether there are calls without obs
    """
    # Count call tags
    call_pattern = r"<\s*call\s*>.*?<\s*/\s*call\s*>"
    calls = re.findall(call_pattern, completion, re.DOTALL | re.IGNORECASE)
    num_calls = len(calls)

    # Count observation tags
    obs_pattern = r"<\s*obs\s*>.*?<\s*/\s*obs\s*>"
    obs_tags = re.findall(obs_pattern, completion, re.DOTALL | re.IGNORECASE)
    num_obs = len(obs_tags)

    # Check for incomplete calls (call without obs after it)
    # Split by calls and check if obs follows each call
    has_incomplete = False
    if num_calls > 0:
        # Simple heuristic: if more calls than observations, likely incomplete
        # More sophisticated: check ordering
        if num_calls > num_obs:
            has_incomplete = True
        else:
            # Check if all calls have obs after them
            parts = re.split(r"<\s*call\s*>", completion, flags=re.IGNORECASE)
            for i, part in enumerate(parts[1:], 1):  # Skip first part (before any call)
                if "<" not in part or "obs" not in part.lower():
                    has_incomplete = True
                    break

    return {
        "num_calls": num_calls,
        "num_obs": num_obs,
        "complete_calls": min(num_calls, num_obs),
        "has_incomplete_calls": has_incomplete,
    }


def get_tool_execution_from_metadata(
    completion: str, tool_calls: List[Dict] = None
) -> int:
    """
    Get actual tool execution count from TRL metadata or text.

    TRL passes tool execution results in the completion metadata.
    Falls back to counting <obs> tags if no metadata available.
    """
    if tool_calls is not None and len(tool_calls) > 0:
        return len(tool_calls)

    # Fallback: count <obs> tags as proxy for executed calls
    obs_pattern = r"<\s*obs\s*>.*?<\s*/\s*obs\s*>"
    obs_tags = re.findall(obs_pattern, completion, re.DOTALL | re.IGNORECASE)
    return len(obs_tags)


def accuracy_efficiency_reward_func(
    completions: List[str],
    answer_ground_truth: List[str],
    correctness_weight: float = DEFAULT_CORRECT_REWARD,
    wrong_penalty: float = DEFAULT_WRONG_PENALTY,
    format_weight: float = DEFAULT_FORMAT_REWARD,
    format_penalty: float = DEFAULT_FORMAT_PENALTY,
    efficiency_penalty: float = DEFAULT_TOOL_COST,
    incomplete_call_penalty: float = DEFAULT_INCOMPLETE_CALL_PENALTY,
    efficient_bonus: float = DEFAULT_EFFICIENT_BONUS,
    **kwargs,
) -> List[float]:
    """
    Enhanced Reward Function for EfficientReasoning.

    Formula: R = Correctness + Format - Cost - IncompletePenalty + EfficientBonus

    This creates the key trade-off:
    - Is the +1.0 (Correctness) worth the -0.05 (Cost)?
    - If I know the answer, Cost is waste. If I don't, Cost is investment.

    New features:
    - Verifies tool execution (checks for obs after call)
    - Penalizes incomplete tool sequences
    - Rewards efficient correct answers (no tools needed)

    Args:
        completions: List of model completions
        answer_ground_truth: List of ground truth answers
        correctness_weight: Reward for correct answer (default: 1.0)
        wrong_penalty: Penalty for wrong answer (default: -0.5)
        format_weight: Reward for valid XML format (default: 0.1)
        format_penalty: Penalty for invalid format (default: -0.5)
        efficiency_penalty: Cost per tool call (default: 0.05)
        incomplete_call_penalty: Penalty for calls without observations (default: -0.2)
        efficient_bonus: Bonus for correct answer without tool calls (default: 0.1)

    Returns:
        List of reward values for each completion
    """
    rewards = []

    # Extract tool_calls from kwargs if provided by TRL
    # TRL passes additional metadata in kwargs
    tool_calls_list = kwargs.get("tool_calls", None)

    for i, (completion, gt) in enumerate(zip(completions, answer_ground_truth)):
        current_reward = 0.0

        # 1. Format Check (Guardrail)
        # Heavy penalty for breaking format to maintain parseable outputs
        if validate_xml_structure(completion):
            current_reward += format_weight
        else:
            current_reward += format_penalty

        # 2. Tool Execution Verification
        # Check if tool calls are properly executed
        exec_info = verify_tool_execution(completion)
        num_calls = exec_info["num_calls"]

        # Get actual executed calls (from obs tags or metadata)
        executed_calls = get_tool_execution_from_metadata(
            completion,
            tool_calls_list[i]
            if tool_calls_list and i < len(tool_calls_list)
            else None,
        )

        # 3. Efficiency Cost (The Optimization Objective)
        # Use actual executed calls for efficiency penalty
        current_reward -= efficiency_penalty * executed_calls

        # 4. Penalty for incomplete tool sequences
        if exec_info["has_incomplete_calls"]:
            current_reward += incomplete_call_penalty

        # 5. Correctness (The Anchor)
        # This must significantly outweigh the cost to prevent "Ostrich" behavior
        pred = extract_xml_tag(completion, "answer")
        if is_correct(pred, gt):
            current_reward += correctness_weight

            # 6. Efficient Bonus: Extra reward for correct answer without tools
            # This strongly incentivizes the model to answer when it knows
            if num_calls == 0:
                current_reward += efficient_bonus
        else:
            current_reward += wrong_penalty

        rewards.append(current_reward)

    return rewards


def create_reward_func_from_config(config: Dict) -> callable:
    """
    Factory function to create a reward function with config-specified weights.

    Args:
        config: Configuration dictionary with 'rewards' section

    Returns:
        Configured reward function
    """
    rewards_config = config.get("rewards", {})

    def configured_reward_func(completions, answer_ground_truth, **kwargs):
        return accuracy_efficiency_reward_func(
            completions=completions,
            answer_ground_truth=answer_ground_truth,
            correctness_weight=rewards_config.get(
                "correctness_weight", DEFAULT_CORRECT_REWARD
            ),
            wrong_penalty=rewards_config.get("wrong_penalty", DEFAULT_WRONG_PENALTY),
            format_weight=rewards_config.get("format_weight", DEFAULT_FORMAT_REWARD),
            format_penalty=rewards_config.get("format_penalty", DEFAULT_FORMAT_PENALTY),
            efficiency_penalty=rewards_config.get(
                "efficiency_penalty", DEFAULT_TOOL_COST
            ),
            incomplete_call_penalty=rewards_config.get(
                "incomplete_call_penalty", DEFAULT_INCOMPLETE_CALL_PENALTY
            ),
            efficient_bonus=rewards_config.get(
                "efficient_bonus", DEFAULT_EFFICIENT_BONUS
            ),
            **kwargs,
        )

    return configured_reward_func
