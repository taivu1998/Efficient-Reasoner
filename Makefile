.PHONY: install clean generate-sft train-sft train inference evaluate plot lint all

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf logs/ outputs/ eval/results/
	rm -rf .pytest_cache/

# Phase 1: Generate SFT Data
generate-sft:
	python scripts/generate_sft.py --config configs/default_config.yaml --num-samples 500

# Phase 1: Train SFT Model (Cold Start)
train-sft:
	python scripts/train_sft.py --config configs/default_config.yaml

# Phase 2: Train GRPO Model
train:
	python scripts/train.py --config configs/default_config.yaml

# Phase 3: Run Inference
inference:
	python scripts/inference.py --config configs/default_config.yaml --checkpoint logs/checkpoint-final

# Phase 3: Run Benchmark Evaluation
evaluate:
	python eval/benchmark.py --config configs/default_config.yaml --checkpoints base logs/sft-checkpoint/final logs/checkpoint-final --names "Base Model" "SFT Model" "GRPO Model"

# Phase 3: Generate Pareto Plot
plot:
	python eval/plot_pareto.py --results-dir eval/results --report

# Linting
lint:
	flake8 src scripts eval --max-line-length 100

# Run the full pipeline
all: generate-sft train-sft train evaluate plot
	@echo "Full pipeline complete!"

# Quick test of the inference loop
test-inference:
	python scripts/inference.py --config configs/default_config.yaml --checkpoint logs/checkpoint-final --query "Who directed the movie Inception?"

# Help
help:
	@echo "EfficientReasoning Makefile Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make generate-sft - Generate SFT training data (Phase 1)"
	@echo "  make train-sft    - Train SFT model (Phase 1)"
	@echo "  make train        - Train GRPO model (Phase 2)"
	@echo "  make inference    - Run inference with trained model (Phase 3)"
	@echo "  make evaluate     - Run benchmark evaluation (Phase 3)"
	@echo "  make plot         - Generate Pareto frontier plot (Phase 3)"
	@echo "  make all          - Run full pipeline"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make lint         - Run linting"
