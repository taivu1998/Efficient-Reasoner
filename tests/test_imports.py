"""Test that all imports work correctly."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestImports:
    """Test all module imports."""

    def test_import_utils(self):
        """Test utils module imports."""
        from src.utils import setup_logging, seed_everything, get_device
        assert callable(setup_logging)
        assert callable(seed_everything)
        assert callable(get_device)

    def test_import_config_parser(self):
        """Test config_parser module imports."""
        from src.config_parser import load_config, parse_args
        assert callable(load_config)
        assert callable(parse_args)

    def test_import_rewards(self):
        """Test rewards module imports."""
        from src.rewards import (
            extract_xml_tag,
            is_correct,
            validate_xml_structure,
            count_tool_calls,
            accuracy_efficiency_reward_func,
            create_reward_func_from_config,
        )
        assert callable(extract_xml_tag)
        assert callable(is_correct)
        assert callable(validate_xml_structure)
        assert callable(count_tool_calls)
        assert callable(accuracy_efficiency_reward_func)
        assert callable(create_reward_func_from_config)

    def test_import_dataset(self):
        """Test dataset module imports."""
        from src.dataset import (
            format_prompt,
            load_grpo_dataset,
            load_sft_dataset,
            load_evaluation_dataset,
            SYSTEM_PROMPT,
        )
        assert callable(format_prompt)
        assert callable(load_grpo_dataset)
        assert callable(load_sft_dataset)
        assert callable(load_evaluation_dataset)
        assert isinstance(SYSTEM_PROMPT, str)

    def test_import_mock_env(self):
        """Test mock_env module imports."""
        from src.mock_env import MockSearchEnv
        assert callable(MockSearchEnv)

    def test_import_model_utils(self):
        """Test model_utils module imports."""
        from src.model_utils import (
            load_model_and_tokenizer,
            load_model_for_inference,
            get_generation_config,
            SPECIAL_TOKENS,
        )
        assert callable(load_model_and_tokenizer)
        assert callable(load_model_for_inference)
        assert callable(get_generation_config)
        assert isinstance(SPECIAL_TOKENS, list)

    def test_import_src_package(self):
        """Test importing from src package."""
        from src import (
            setup_logging,
            seed_everything,
            load_config,
            accuracy_efficiency_reward_func,
            MockSearchEnv,
            SYSTEM_PROMPT,
        )
        assert callable(setup_logging)
        assert callable(accuracy_efficiency_reward_func)

    def test_import_eval_package(self):
        """Test importing from eval package."""
        from eval.benchmark import run_benchmark, evaluate_model
        from eval.plot_pareto import plot_pareto_frontier, compute_pareto_frontier
        assert callable(run_benchmark)
        assert callable(evaluate_model)
        assert callable(plot_pareto_frontier)
        assert callable(compute_pareto_frontier)


class TestDependencies:
    """Test that required dependencies are installed."""

    def test_torch_available(self):
        """Test PyTorch is available."""
        import torch
        assert hasattr(torch, 'tensor')

    def test_transformers_available(self):
        """Test transformers is available."""
        import transformers
        assert hasattr(transformers, 'AutoTokenizer')

    def test_datasets_available(self):
        """Test datasets is available."""
        import datasets
        assert hasattr(datasets, 'load_dataset')

    def test_trl_available(self):
        """Test TRL is available."""
        import trl
        assert hasattr(trl, 'GRPOTrainer')

    def test_peft_available(self):
        """Test PEFT is available."""
        import peft
        assert hasattr(peft, 'PeftModel')

    def test_yaml_available(self):
        """Test PyYAML is available."""
        import yaml
        assert hasattr(yaml, 'safe_load')

    def test_tqdm_available(self):
        """Test tqdm is available."""
        import tqdm
        assert hasattr(tqdm, 'tqdm')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
