import yaml
import argparse
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("EfficientReasoning")


@dataclass
class ConfigValidationError:
    """Represents a configuration validation error."""

    field: str
    message: str
    severity: str  # 'error', 'warning'


class ConfigValidator:
    """Validates configuration values."""

    REQUIRED_SECTIONS = ["project", "model", "training", "rewards"]
    REQUIRED_MODEL_FIELDS = ["name_or_path", "max_seq_length"]
    REQUIRED_TRAINING_FIELDS = [
        "learning_rate",
        "batch_size",
        "max_steps",
        "num_generations",
        "beta",
    ]
    REQUIRED_PROJECT_FIELDS = ["name", "output_dir"]

    def __init__(self):
        self.errors: List[ConfigValidationError] = []

    def validate(self, config: Dict[str, Any]) -> List[ConfigValidationError]:
        """Run all validations and return list of errors/warnings."""
        self.errors = []

        self._validate_required_sections(config)
        self._validate_project(config.get("project", {}))
        self._validate_model(config.get("model", {}))
        self._validate_training(config.get("training", {}))
        self._validate_rewards(config.get("rewards", {}))
        self._validate_lora(config.get("lora", {}))
        self._validate_tools(config.get("tools", {}))

        return self.errors

    def _add_error(self, field: str, message: str, severity: str = "error"):
        self.errors.append(ConfigValidationError(field, message, severity))

    def _validate_required_sections(self, config: Dict):
        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                self._add_error(section, f"Missing required section '{section}'")

    def _validate_project(self, config: Dict):
        for field in self.REQUIRED_PROJECT_FIELDS:
            if field not in config:
                self._add_error(f"project.{field}", f"Missing required field '{field}'")

        # Validate seed
        if "seed" in config and not isinstance(config["seed"], int):
            self._add_error("project.seed", "Seed must be an integer", "warning")

        # Validate output_dir
        if "output_dir" in config:
            output_dir = config["output_dir"]
            if not isinstance(output_dir, str):
                self._add_error("project.output_dir", "output_dir must be a string")
            elif output_dir:
                parent_dir = os.path.dirname(output_dir) or "."
                if not os.path.exists(parent_dir):
                    self._add_error(
                        "project.output_dir",
                        f"Parent directory '{parent_dir}' does not exist",
                        "warning",
                    )

    def _validate_model(self, config: Dict):
        for field in self.REQUIRED_MODEL_FIELDS:
            if field not in config:
                self._add_error(f"model.{field}", f"Missing required field '{field}'")

        # Validate max_seq_length
        if "max_seq_length" in config:
            ms = config["max_seq_length"]
            if not isinstance(ms, int):
                self._add_error("model.max_seq_length", "Must be an integer")
            elif ms < 256:
                self._add_error("model.max_seq_length", "Too small (<256)", "warning")
            elif ms > 16384:
                self._add_error(
                    "model.max_seq_length", "Very large (>16384)", "warning"
                )

        # Validate load_in_4bit
        if "load_in_4bit" in config:
            if not isinstance(config["load_in_4bit"], bool):
                self._add_error("model.load_in_4bit", "Must be boolean")

    def _validate_training(self, config: Dict):
        for field in self.REQUIRED_TRAINING_FIELDS:
            if field not in config:
                self._add_error(
                    f"training.{field}", f"Missing required field '{field}'"
                )

        # Validate learning_rate
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if not isinstance(lr, (int, float)):
                self._add_error("training.learning_rate", "Must be a number")
            elif lr <= 0:
                self._add_error("training.learning_rate", "Must be positive")
            elif lr > 1e-2:
                self._add_error(
                    "training.learning_rate", "Very high (>0.01)", "warning"
                )

        # Validate batch_size
        if "batch_size" in config:
            bs = config["batch_size"]
            if not isinstance(bs, int):
                self._add_error("training.batch_size", "Must be an integer")
            elif bs < 1:
                self._add_error("training.batch_size", "Must be >= 1")
            elif bs > 32:
                self._add_error("training.batch_size", "Very large (>32)", "warning")

        # Validate num_generations
        if "num_generations" in config:
            ng = config["num_generations"]
            if not isinstance(ng, int):
                self._add_error("training.num_generations", "Must be an integer")
            elif ng < 1:
                self._add_error("training.num_generations", "Must be >= 1")
            elif ng > 32:
                self._add_error(
                    "training.num_generations", "Very large (>32)", "warning"
                )

        # Validate beta (KL penalty)
        if "beta" in config:
            beta = config["beta"]
            if not isinstance(beta, (int, float)):
                self._add_error("training.beta", "Must be a number")
            elif beta < 0:
                self._add_error("training.beta", "Must be non-negative")
            elif beta > 1.0:
                self._add_error("training.beta", "Very high (>1.0)", "warning")

        # Validate max_steps
        if "max_steps" in config:
            ms = config["max_steps"]
            if not isinstance(ms, int):
                self._add_error("training.max_steps", "Must be an integer")
            elif ms < 1:
                self._add_error("training.max_steps", "Must be >= 1")

        # Validate logging_steps and save_steps
        for field in ["logging_steps", "save_steps"]:
            if field in config:
                val = config[field]
                if not isinstance(val, int):
                    self._add_error(f"training.{field}", "Must be an integer")
                elif val < 1:
                    self._add_error(f"training.{field}", "Must be >= 1")

        # Validate lr_scheduler
        if "lr_scheduler" in config:
            valid_schedulers = [
                "linear",
                "cosine",
                "constant",
                "polynomial",
                "inverse_sqrt",
            ]
            if config["lr_scheduler"] not in valid_schedulers:
                self._add_error(
                    "training.lr_scheduler",
                    f"Must be one of {valid_schedulers}",
                    "warning",
                )

        # Validate report_to
        if "report_to" in config:
            valid_options = ["none", "wandb", "tensorboard", "mlflow"]
            if config["report_to"] not in valid_options:
                self._add_error(
                    "training.report_to", f"Must be one of {valid_options}", "warning"
                )

    def _validate_rewards(self, config: Dict):
        # Validate reward weights
        reward_fields = [
            "correctness_weight",
            "wrong_penalty",
            "format_weight",
            "format_penalty",
            "efficiency_penalty",
            "incomplete_call_penalty",
            "efficient_bonus",
        ]

        for field in reward_fields:
            if field in config:
                val = config[field]
                if not isinstance(val, (int, float)):
                    self._add_error(f"rewards.{field}", "Must be a number")

        # Warn if efficiency penalty is too high or low
        if "efficiency_penalty" in config:
            ep = config["efficiency_penalty"]
            if ep < 0:
                self._add_error("rewards.efficiency_penalty", "Should be non-negative")
            elif ep > 0.5:
                self._add_error(
                    "rewards.efficiency_penalty",
                    "Very high (>0.5), may discourage all tool use",
                    "warning",
                )

    def _validate_lora(self, config: Dict):
        if not config:
            return

        # Validate r (rank)
        if "r" in config:
            r = config["r"]
            if not isinstance(r, int):
                self._add_error("lora.r", "Must be an integer")
            elif r < 1:
                self._add_error("lora.r", "Must be >= 1")
            elif r > 128:
                self._add_error("lora.r", "Very large (>128)", "warning")

        # Validate lora_alpha
        if "lora_alpha" in config:
            la = config["lora_alpha"]
            if not isinstance(la, (int, float)):
                self._add_error("lora.lora_alpha", "Must be a number")

        # Validate lora_dropout
        if "lora_dropout" in config:
            ld = config["lora_dropout"]
            if not isinstance(ld, (int, float)):
                self._add_error("lora.lora_dropout", "Must be a number")
            elif ld < 0 or ld > 1:
                self._add_error("lora.lora_dropout", "Must be between 0 and 1")

        # Validate target_modules
        if "target_modules" in config:
            if not isinstance(config["target_modules"], list):
                self._add_error("lora.target_modules", "Must be a list")

    def _validate_tools(self, config: Dict):
        if not config:
            return

        # Validate enabled
        if "enabled" in config:
            if not isinstance(config["enabled"], bool):
                self._add_error("tools.enabled", "Must be boolean")

        # Validate max_tool_calls
        if "max_tool_calls" in config:
            mtc = config["max_tool_calls"]
            if not isinstance(mtc, int):
                self._add_error("tools.max_tool_calls", "Must be an integer")
            elif mtc < 1:
                self._add_error("tools.max_tool_calls", "Must be >= 1")
            elif mtc > 10:
                self._add_error("tools.max_tool_calls", "Very high (>10)", "warning")


def load_config(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    Loads a YAML configuration file with optional validation.

    Args:
        config_path: Path to the YAML config file
        validate: Whether to run validation (default: True)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If validation fails with errors
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if validate:
        validator = ConfigValidator()
        errors = validator.validate(config)

        # Log warnings
        warnings = [e for e in errors if e.severity == "warning"]
        for w in warnings:
            logger.warning(f"Config warning ({w.field}): {w.message}")

        # Raise on errors
        critical_errors = [e for e in errors if e.severity == "error"]
        if critical_errors:
            error_msgs = [f"{e.field}: {e.message}" for e in critical_errors]
            raise ValueError(f"Config validation failed:\n" + "\n".join(error_msgs))

    return config


def parse_args(validate: bool = True) -> Dict[str, Any]:
    """
    Parses CLI arguments and allows overriding config values using dot notation.
    Example: --training.batch_size 8

    Args:
        validate: Whether to validate the config (default: True)

    Returns:
        Configuration dictionary with overrides applied
    """
    parser = argparse.ArgumentParser(description="EfficientReasoning GRPO Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip config validation"
    )

    # Allow arbitrary overrides
    args, unknown = parser.parse_known_args()

    should_validate = validate and not args.no_validate

    try:
        config = load_config(args.config, validate=should_validate)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    # Process overrides
    i = 0
    overrides_applied = []
    while i < len(unknown):
        key = unknown[i]
        if key.startswith("--"):
            key = key[2:]
            if i + 1 >= len(unknown):
                logger.warning(f"Missing value for override: {key}")
                i += 1
                continue

            value = unknown[i + 1]

            # Type casting with better handling
            value = _parse_override_value(value)

            keys = key.split(".")
            sub_config = config
            for k in keys[:-1]:
                if k not in sub_config:
                    sub_config[k] = {}
                sub_config = sub_config[k]
            sub_config[keys[-1]] = value
            overrides_applied.append(f"{key}={value}")
            i += 2
        else:
            i += 1

    if overrides_applied:
        logger.info(f"Applied config overrides: {', '.join(overrides_applied)}")

    return config


def _parse_override_value(value: str) -> Any:
    """Parse a command line override value with proper type detection."""
    # Handle lists (comma-separated)
    if "," in value:
        return [_parse_override_value(v.strip()) for v in value.split(",")]

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # None
    if value.lower() == "none" or value.lower() == "null":
        return None

    # Integer
    if value.isdigit():
        return int(value)

    # Float
    try:
        if "." in value:
            return float(value)
    except ValueError:
        pass

    # String (default)
    return value
