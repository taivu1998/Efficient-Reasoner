"""
Constants used throughout the EfficientReasoning project.

This module centralizes magic numbers and configuration defaults
to make the codebase more maintainable.
"""

# ============================================================================
# Model Configuration
# ============================================================================

DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_PROMPT_LENGTH = 1024
DEFAULT_MAX_COMPLETION_LENGTH = 1024

# ============================================================================
# LoRA Configuration
# ============================================================================

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ============================================================================
# Training Configuration
# ============================================================================

DEFAULT_LEARNING_RATE = 2.0e-5
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_NUM_GENERATIONS = 8  # G in GRPO
DEFAULT_MAX_STEPS = 500
DEFAULT_BETA = 0.04  # KL penalty
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 100
DEFAULT_SAVE_TOTAL_LIMIT = 3

# LR Scheduler
DEFAULT_LR_SCHEDULER = "cosine"
DEFAULT_WARMUP_RATIO = 0.1

# ============================================================================
# Reward Function Configuration
# ============================================================================

DEFAULT_CORRECT_REWARD = 1.0
DEFAULT_WRONG_PENALTY = -0.5
DEFAULT_FORMAT_REWARD = 0.1
DEFAULT_FORMAT_PENALTY = -0.5
DEFAULT_TOOL_COST = 0.05
DEFAULT_INCOMPLETE_CALL_PENALTY = -0.2
DEFAULT_EFFICIENT_BONUS = 0.1

# ============================================================================
# Inference Configuration
# ============================================================================

DEFAULT_INFERENCE_MAX_STEPS = 5
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1

# ============================================================================
# Evaluation Configuration
# ============================================================================

DEFAULT_EVAL_MAX_SAMPLES = 100
DEFAULT_EVAL_MAX_STEPS = 5

# ============================================================================
# Tool Configuration
# ============================================================================

DEFAULT_MAX_TOOL_CALLS = 5
DEFAULT_TOOL_TIMEOUT_SECONDS = 30

# ============================================================================
# XML Tag Names
# ============================================================================

XML_TAGS = {
    "thought": ("<thought>", "</thought>"),
    "call": ("<call>", "</call>"),
    "obs": ("<obs>", "</obs>"),
    "answer": ("<answer>", "</answer>"),
}

# ============================================================================
# Data Configuration
# ============================================================================

DEFAULT_DATASET_NAME = "hotpot_qa"
DEFAULT_DATASET_SUBSET = "distractor"
DEFAULT_MAX_SAMPLES = 1000

# ============================================================================
# Special Tokens
# ============================================================================

SPECIAL_TOKENS = [
    "<thought>",
    "</thought>",
    "<call>",
    "</call>",
    "<obs>",
    "</obs>",
    "<answer>",
    "</answer>",
]

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are an efficient, reasoning-focused AI assistant.
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

# ============================================================================
# Memory Management
# ============================================================================

MEMORY_CLEANUP_INTERVAL = 10  # Cleanup every N samples
TOKEN_TRUNCATION_MARGIN = 256  # Reserve tokens for truncation

# ============================================================================
# Generation Safety Limits
# ============================================================================

MAX_TOTAL_TOKENS_DEFAULT = 2048
STUCK_DETECTION_THRESHOLD = 50  # Min new chars to not be stuck
MAX_CONTEXT_RATIO = 0.8  # Max ratio of context to max length
