# EfficientReasoning Source Package
# Note: Some imports require optional dependencies (peft, trl, unsloth)


def __getattr__(name):
    """Lazy import for modules with heavy dependencies."""
    if name == "MockSearchEnv":
        from .mock_env import MockSearchEnv

        return MockSearchEnv
    elif name in (
        "search_wiki",
        "search_wiki_multi",
        "get_entity_info",
        "get_tools",
        "reset_mock_env",
    ):
        from . import tools

        return getattr(tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .utils import setup_logging, seed_everything, get_device
from .config_parser import load_config, parse_args, ConfigValidator
from .rewards import (
    accuracy_efficiency_reward_func,
    create_reward_func_from_config,
    extract_xml_tag,
    extract_xml_tag_strict,
    is_correct,
    validate_xml_structure,
    count_tool_calls,
    verify_tool_execution,
)
from .dataset import (
    load_grpo_dataset,
    load_sft_dataset,
    load_evaluation_dataset,
    SYSTEM_PROMPT,
    format_prompt,
)

__all__ = [
    # Utils
    "setup_logging",
    "seed_everything",
    "get_device",
    # Config
    "load_config",
    "parse_args",
    "ConfigValidator",
    # Rewards
    "accuracy_efficiency_reward_func",
    "create_reward_func_from_config",
    "extract_xml_tag",
    "extract_xml_tag_strict",
    "is_correct",
    "validate_xml_structure",
    "count_tool_calls",
    "verify_tool_execution",
    # Dataset
    "load_grpo_dataset",
    "load_sft_dataset",
    "load_evaluation_dataset",
    "SYSTEM_PROMPT",
    "format_prompt",
    # Mock Environment (lazy)
    "MockSearchEnv",
    # Tools (lazy)
    "search_wiki",
    "search_wiki_multi",
    "get_entity_info",
    "get_tools",
    "reset_mock_env",
]
