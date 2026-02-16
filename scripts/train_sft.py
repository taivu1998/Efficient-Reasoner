"""
Phase 1: Supervised Fine-Tuning (SFT)

Trains the model on synthetic reasoning traces to learn the XML tool-use format.
This is the "Cold Start" phase that bootstraps the model before GRPO training.
"""

import sys
import os
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

from src.config_parser import parse_args
from src.utils import setup_logging, seed_everything
from src.model_utils import load_model_and_tokenizer
from src.dataset import load_sft_dataset, SYSTEM_PROMPT


def formatting_prompts_func(examples):
    """
    Format examples for SFT training.
    Combines prompt and completion into full training sequences.
    """
    texts = []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        # Full sequence: prompt + completion
        text = f"{prompt}{completion}"
        texts.append(text)
    return {"text": texts}


def main():
    # 1. Setup
    config = parse_args()
    logger = setup_logging()
    seed_everything(config["project"]["seed"])

    logger.info("=" * 60)
    logger.info("Phase 1: SFT Training (Cold Start)")
    logger.info("=" * 60)

    # 2. Load SFT data
    logger.info("Loading SFT dataset...")
    sft_data_path = "data/processed/sft_data.json"

    if not os.path.exists(sft_data_path):
        logger.error(f"SFT data not found at {sft_data_path}")
        logger.error(
            "Run 'python scripts/generate_sft.py' first to generate training data."
        )
        return

    try:
        dataset = load_sft_dataset(config, data_path=sft_data_path)
        logger.info(f"Loaded {len(dataset)} SFT training examples")
    except Exception as e:
        logger.error(f"Failed to load SFT dataset: {e}")
        raise

    # Format dataset for SFT
    dataset = dataset.map(
        formatting_prompts_func, batched=True, remove_columns=dataset.column_names
    )

    # 3. Load model
    logger.info("Loading Model...")
    try:
        model, tokenizer = load_model_and_tokenizer(config, for_training=True)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # 4. Determine resume checkpoint
    sft_config = config.get("sft", {})
    output_dir = os.path.join(config["project"]["output_dir"], "sft-checkpoint")

    resume_from_checkpoint = sft_config.get("resume_from_checkpoint")

    # Check for auto-resume
    if resume_from_checkpoint is None:
        if os.path.exists(output_dir):
            checkpoints = [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(output_dir, d))
            ]
            if checkpoints:
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
    elif resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        if resume_from_checkpoint:
            logger.warning(
                f"Checkpoint not found: {resume_from_checkpoint}. Starting fresh."
            )

    # 5. SFT Configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=sft_config.get("learning_rate", 2e-5),
        per_device_train_batch_size=sft_config.get("batch_size", 4),
        gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", 2),
        num_train_epochs=sft_config.get("num_epochs", 3),
        warmup_ratio=sft_config.get("warmup_ratio", 0.1),
        max_seq_length=sft_config.get("max_seq_length", 1024),
        # Use bf16 if available
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        # Logging
        logging_steps=10,
        save_steps=100,
        save_total_limit=sft_config.get("save_total_limit", 3),
        report_to=config["training"].get("report_to", "none"),
        # Optimization
        gradient_checkpointing=True,
        optim="adamw_8bit",
        # Resume
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # 6. Initialize SFTTrainer
    logger.info("Initializing SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            dataset_text_field="text",
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise

    # 7. Train
    logger.info("=" * 60)
    logger.info("Starting SFT Training")
    logger.info("=" * 60)
    logger.info(f"Training for {sft_config.get('num_epochs', 3)} epochs")
    logger.info(
        f"Batch size: {sft_config.get('batch_size', 4)} x {sft_config.get('gradient_accumulation_steps', 2)}"
    )
    logger.info(f"Learning rate: {sft_config.get('learning_rate', 2e-5)}")
    logger.info(f"Max sequence length: {sft_config.get('max_seq_length', 1024)}")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        interrupted_path = os.path.join(output_dir, "checkpoint-interrupted")
        trainer.save_model(interrupted_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 8. Save
    final_path = os.path.join(output_dir, "final")
    logger.info(f"Saving SFT Model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Save config with model for reproducibility
    import yaml

    config_path = os.path.join(final_path, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    logger.info("=" * 60)
    logger.info("SFT Training complete!")
    logger.info(f"Model saved to: {final_path}")
    logger.info("You can now proceed to Phase 2: GRPO Training")
    logger.info(f"Run: python scripts/train.py --config configs/default_config.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
