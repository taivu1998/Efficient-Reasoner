import os
import gc
import logging
import torch
from typing import Optional, Tuple, Dict, Any
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("EfficientReasoning")

UNSLOTH_AVAILABLE = False
FastLanguageModel = None

try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    logger.info(
        "Unsloth not available. Using standard transformers (slower but works on CPU/Mac)."
    )

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


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""

    pass


class DeviceError(Exception):
    """Exception raised for invalid device specifications."""

    pass


def get_device() -> str:
    """
    Get the best available device for model inference/training.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def validate_config_for_loading(config: dict) -> None:
    """
    Validate configuration before model loading.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["model", "lora", "project"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")

    model_config = config["model"]
    if "name_or_path" not in model_config:
        raise ValueError("Missing required field: model.name_or_path")

    if "max_seq_length" not in model_config:
        raise ValueError("Missing required field: model.max_seq_length")


def cleanup_model(model) -> None:
    """
    Clean up model from memory.

    Args:
        model: The model to clean up
    """
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_model_and_tokenizer(
    config: dict, for_training: bool = True, device: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Loads the model using Unsloth if available, otherwise falls back to standard transformers.
    Adds special agentic tokens to the tokenizer and resizes model embeddings.

    Args:
        config: Configuration dictionary
        for_training: If True, applies LoRA for training. If False, loads for inference.
        device: Specific device to load on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ModelLoadError: If model loading fails
    """
    validate_config_for_loading(config)

    model_name = config["model"]["name_or_path"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"].get("load_in_4bit", True)

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Max sequence length: {max_seq_length}")
    logger.info(f"Load in 4-bit: {load_in_4bit}")
    logger.info(f"Device: {device or get_device()}")

    try:
        if UNSLOTH_AVAILABLE:
            return _load_with_unsloth(config, for_training)
        else:
            return _load_with_transformers(config, for_training, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ModelLoadError(f"Model loading failed: {e}") from e


def _load_with_unsloth(config: dict, for_training: bool = True):
    """Load model using Unsloth (faster, requires CUDA)."""
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is not available")

    model_name = config["model"]["name_or_path"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"].get("load_in_4bit", True)

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA not available. Unsloth works best with CUDA. Falling back may be slow."
        )

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    except Exception as e:
        logger.error(f"Unsloth loading failed: {e}. Trying transformers...")
        return _load_with_transformers(config, for_training)

    _configure_tokenizer(tokenizer, model)

    if for_training:
        lora_config = config.get("lora", {})
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get("r", 16),
            target_modules=lora_config.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config["project"]["seed"],
        )

    return model, tokenizer


def _load_with_transformers(
    config: dict, for_training: bool = True, device: Optional[str] = None
):
    """Load model using standard transformers (works on CPU/Mac)."""
    model_name = config["model"]["name_or_path"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"].get("load_in_4bit", True)

    if device is None:
        device = get_device()

    if device == "cuda":
        device_map = "auto"
        torch_dtype = torch.float16
    elif device == "mps":
        device_map = "mps"
        torch_dtype = torch.float16
        load_in_4bit = False
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        load_in_4bit = False

    quantization_config = None
    if load_in_4bit and device == "cuda":
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            logger.warning(
                f"Failed to configure 4-bit quantization: {e}. Using full precision."
            )
            load_in_4bit = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            model_max_length=max_seq_length,
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to load tokenizer: {e}") from e

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}") from e

    _configure_tokenizer(tokenizer, model)

    if for_training:
        lora_config = config.get("lora", {})
        lora_cfg = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def _configure_tokenizer(tokenizer, model) -> None:
    """Configure tokenizer with special tokens."""
    special_tokens = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added {num_added} special tokens to tokenizer")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token")


def load_model_for_inference(
    checkpoint_path: str, config: dict, device: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Loads a trained model (with LoRA adapter) for inference.

    Args:
        checkpoint_path: Path to the saved LoRA adapter checkpoint
        config: Configuration dictionary
        device: Device to load model on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer) ready for inference

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ModelLoadError: If model loading fails
    """
    if device is None:
        device = get_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    logger.info(f"Device: {device}")

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    is_adapter = os.path.exists(adapter_config_path)
    logger.info(f"LoRA adapter detected: {is_adapter}")

    try:
        if UNSLOTH_AVAILABLE:
            return _load_inference_unsloth(checkpoint_path, config, is_adapter)
        else:
            return _load_inference_transformers(
                checkpoint_path, config, device, is_adapter
            )
    except Exception as e:
        raise ModelLoadError(f"Failed to load model for inference: {e}") from e


def _load_inference_unsloth(checkpoint_path: str, config: dict, is_adapter: bool):
    """Load model for inference using Unsloth."""
    if not UNSLOTH_AVAILABLE:
        return _load_inference_transformers(
            checkpoint_path, config, get_device(), is_adapter
        )

    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"].get("load_in_4bit", True)

    if is_adapter:
        base_model_name = config["model"]["name_or_path"]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        _configure_tokenizer(tokenizer, model)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        _configure_tokenizer(tokenizer, model)

    model.eval()
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_inference_transformers(
    checkpoint_path: str, config: dict, device: str, is_adapter: bool
):
    """Load model for inference using standard transformers."""
    torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    if is_adapter:
        base_model_name = config["model"]["name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            model = model.to(device)

        _configure_tokenizer(tokenizer, model)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            model = model.to(device)

        _configure_tokenizer(tokenizer, model)

    model.eval()
    return model, tokenizer


def get_generation_config(config: dict = None) -> Dict[str, Any]:
    """
    Returns default generation configuration for agentic inference.

    Args:
        config: Optional configuration dictionary

    Returns:
        Dictionary of generation parameters
    """
    default_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": None,
        "eos_token_id": None,
    }

    if config and "inference" in config:
        default_config.update(config["inference"])

    return default_config
