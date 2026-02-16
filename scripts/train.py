import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from trl import GRPOTrainer, GRPOConfig

from src.config_parser import parse_args
from src.utils import setup_logging, seed_everything
from src.model_utils import load_model_and_tokenizer
from src.dataset import load_grpo_dataset
from src.rewards import accuracy_efficiency_reward_func, create_reward_func_from_config
from src.tools import get_tools


def setup_logging_dest(config: dict, logger):
    """Setup logging destination based on config."""
    report_to = config["training"].get("report_to", "none")

    if report_to != "none" and report_to != "none":
        logger.info(f"Logging to: {report_to}")

    return report_to


def get_lr_scheduler(config: dict, training_args):
    """Get learning rate scheduler based on config."""
    scheduler_name = config["training"].get("lr_scheduler", "linear")
    warmup_ratio = config["training"].get("warmup_ratio", 0.1)
    num_training_steps = config["training"].get("max_steps", 500)
    warmup_steps = int(num_training_steps * warmup_ratio)

    from transformers import get_scheduler

    scheduler_type_map = {
        "linear": "linear",
        "cosine": "cosine",
        "constant": "constant",
        "polynomial": "polynomial",
        "inverse_sqrt": "inverse_sqrt",
    }

    scheduler_type = scheduler_type_map.get(scheduler_name, "linear")

    return get_scheduler(
        scheduler_type,
        training_args.optimizer,
        warmup_steps,
        num_training_steps,
    )


def main():
    # 1. Setup
    config = parse_args()
    logger = setup_logging()
    seed_everything(config["project"]["seed"])

    logger.info("=" * 60)
    logger.info("Phase 2: EfficientReasoning GRPO Training")
    logger.info("=" * 60)
    logger.info(f"Output directory: {config['project']['output_dir']}")

    # 2. Data
    logger.info("Loading Datasets...")
    try:
        dataset = load_grpo_dataset(config)
        logger.info(f"Loaded {len(dataset)} training samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Please ensure HotpotQA is available or check data config.")
        raise

    # 3. Model
    logger.info("Loading Model...")
    try:
        model, tokenizer = load_model_and_tokenizer(config)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Ensure tokenizer has pad token (critical for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # 4. Setup Tools for native GRPO tool support
    tools_config = config.get("tools", {})
    use_tools = tools_config.get("enabled", True)
    max_tool_calls = tools_config.get("max_tool_calls", 5)

    tools = None
    if use_tools:
        logger.info("Loading tools for GRPO training...")
        try:
            tools = get_tools()
            logger.info(f"Loaded {len(tools)} tools: {[t.__name__ for t in tools]}")
        except Exception as e:
            logger.warning(f"Failed to load tools: {e}. Continuing without tools.")
            tools = None

    # 5. Create reward function with config weights
    reward_func = create_reward_func_from_config(config)
    logger.info(f"Reward weights: {config.get('rewards', {})}")

    # 6. Determine resume checkpoint
    output_dir = config["project"]["output_dir"]
    resume_from_checkpoint = config["training"].get("resume_from_checkpoint", None)

    # Check for auto-resume (latest checkpoint)
    if resume_from_checkpoint is None:
        # Check if there are existing checkpoints
        checkpoint_path = os.path.join(output_dir, "checkpoint-*")
        if os.path.exists(output_dir):
            checkpoints = [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(output_dir, d))
            ]
            if checkpoints:
                # Find the latest checkpoint
                checkpoint_nums = []
                for ckpt in checkpoints:
                    try:
                        checkpoint_nums.append(int(ckpt.replace("checkpoint-", "")))
                    except ValueError:
                        continue
                if checkpoint_nums:
                    resume_from_checkpoint = os.path.join(
                        output_dir, f"checkpoint-{max(checkpoint_nums)}"
                    )
                    logger.info(
                        f"Auto-resuming from checkpoint: {resume_from_checkpoint}"
                    )

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    # 7. Training Config
    training_config = config["training"]
    model_config = config["model"]
    max_seq_length = model_config.get("max_seq_length", 2048)

    # Determine report_to
    report_to = training_config.get("report_to", "none")

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        num_generations=training_config["num_generations"],
        max_steps=training_config["max_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config.get("save_total_limit", 3),
        beta=training_config["beta"],
        # Critical: Set prompt and completion length limits
        max_prompt_length=max_seq_length // 2,
        max_completion_length=max_seq_length // 2,
        # Use bf16 if available (better for modern GPUs), else fp16
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        # Logging and reporting
        report_to=report_to,
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
        # Tool configuration
        max_tool_calling_iterations=max_tool_calls if tools else 0,
        # Resume checkpoint
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # 8. Initialize GRPOTrainer with native tool support
    logger.info("Initializing GRPOTrainer...")
    logger.info(f"Tool support: {'enabled' if tools else 'disabled'}")

    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "reward_funcs": [reward_func],
        "args": training_args,
        "train_dataset": dataset,
    }

    # Add tools if available
    if tools:
        trainer_kwargs["tools"] = tools

    trainer = GRPOTrainer(**train_kwargs)

    # 9. Train
    logger.info("=" * 60)
    logger.info("Starting GRPO Training Loop")
    logger.info("=" * 60)
    logger.info(f"Training for {training_config['max_steps']} steps")
    logger.info(
        f"Batch size: {training_config['batch_size']} x {training_config['gradient_accumulation_steps']} = {training_config['batch_size'] * training_config['gradient_accumulation_steps']}"
    )
    logger.info(f"Num generations (G): {training_config['num_generations']}")
    logger.info(f"KL penalty (beta): {training_config['beta']}")
    logger.info(f"Learning rate: {training_config['learning_rate']}")
    logger.info(f"LR scheduler: {training_config.get('lr_scheduler', 'linear')}")
    logger.info(f"Max tool calls per response: {max_tool_calls if tools else 'N/A'}")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        trainer.save_model(os.path.join(output_dir, "checkpoint-interrupted"))
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    # 10. Save final model
    final_path = os.path.join(output_dir, "checkpoint-final")
    logger.info(f"Saving Final Model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training config with the model for reproducibility
    import yaml

    config_path = os.path.join(final_path, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {final_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
