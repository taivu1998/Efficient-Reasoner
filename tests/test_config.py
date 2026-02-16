"""Tests for configuration parsing."""

import pytest
import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_parser import load_config


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_default_config(self):
        """Test loading the default config file."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default_config.yaml"
        )
        config = load_config(config_path)

        # Check required sections exist
        assert "project" in config
        assert "model" in config
        assert "lora" in config
        assert "data" in config
        assert "training" in config
        assert "rewards" in config

    def test_config_values(self):
        """Test specific config values."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default_config.yaml"
        )
        config = load_config(config_path)

        # Check specific values
        assert config["model"]["max_seq_length"] == 2048
        assert config["training"]["beta"] == 0.04
        assert config["rewards"]["efficiency_penalty"] == 0.05

    def test_load_custom_config(self):
        """Test loading a custom config file."""
        custom_config = {
            "project": {"name": "test", "seed": 123, "output_dir": "logs/"},
            "model": {"name_or_path": "test-model", "max_seq_length": 2048},
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 4,
                "num_generations": 8,
                "max_steps": 100,
                "beta": 0.1,
            },
            "rewards": {"correctness_weight": 1.0},
            "lora": {"r": 16, "lora_alpha": 32},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["project"]["name"] == "test"
            assert config["project"]["seed"] == 123
        finally:
            os.unlink(temp_path)

    def test_load_custom_config_no_validation(self):
        """Test loading a custom config file without validation."""
        custom_config = {
            "project": {"name": "test", "seed": 123},
            "model": {"name_or_path": "test-model"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name

        try:
            config = load_config(temp_path, validate=False)
            assert config["project"]["name"] == "test"
            assert config["project"]["seed"] == 123
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
