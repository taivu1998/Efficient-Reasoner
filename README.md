# EfficientReasoning: Optimizing Agentic Compute-Accuracy Trade-offs via GRPO

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/GRPO-Optimized-green.svg" alt="GRPO">
  <img src="https://img.shields.io/badge/Tests-54%20Passing-brightgreen.svg" alt="Tests">
</p>

> **Research Goal:** Demonstrate that "Tool Use" and "Stopping Criteria" are learnable policies optimized via GRPO, enabling a 3B parameter model to dynamically allocate compute only when necessary—achieving the optimal balance between accuracy and computational efficiency.

---

## 🔬 Research Problem

Current LLM agents suffer from **computational inelasticity**—they invoke expensive tool calls (like web search) regardless of question difficulty. This leads to:

- **Wasted compute** on trivial questions that can be answered directly
- **High latency** due to unnecessary tool invocations  
- **Poor efficiency** in production deployments

### Our Solution

We treat the **"Decision to Search"** as a learnable policy optimized through **Group Relative Policy Optimization (GRPO)**. The model learns to answer directly when confident and invoke search tools only when necessary—naturally forming a **System 1 (fast) / System 2 (slow)** dual process architecture.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EfficientReasoning Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │   Phase 1   │───▶│   Phase 2   │───▶│   Phase 3   │───▶│  Results  │ │
│  │    SFT      │    │    GRPO     │    │ Inference   │    │           │ │
│  │ Cold Start  │    │   Training  │    │ + Evaluation │    │ Pareto    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    │ Frontier  │ │
│       │                  │                  │               └───────────┘ │
│       ▼                  ▼                  ▼                                  │
│  ┌─────────┐       ┌─────────┐       ┌─────────────┐                         │
│  │ Synthetic│       │ Policy  │       │  Agentic    │                         │
│  │  Traces │       │ Update  │       │   Loop     │                         │
│  └─────────┘       └─────────┘       └─────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **GRPOTrainer** | Hugging Face TRL's GRPO implementation with native tool support |
| **Custom Reward Function** | Multi-component reward balancing accuracy, format, and efficiency |
| **Mock Search Environment** | O(1) knowledge base simulation for fast RL training |
| **Agentic Inference Loop** | Multi-step reasoning with tool execution |
| **Pareto Analysis** | Visualize compute-accuracy trade-offs |

---

## 🎯 The Reward Function

The core innovation is the **Value of Information (VOI)** reward that implicitly learns when to search:

```python
R = Correctness ± 1.0 + Format ± 0.1 - Cost(efficiency_penalty × calls) + Bonus(efficient)
```

### Reward Components

| Component | Value | Purpose |
|----------|-------|---------|
| **Correctness** | +1.0 / -0.5 | Anchor the reward to accuracy |
| **Format** | +0.1 / -0.5 | Enforce valid XML output |
| **Efficiency Cost** | -0.05 per call | Penalize unnecessary tool use |
| **Incomplete Call** | -0.2 | Penalize unexecuted tool calls |
| **Efficient Bonus** | +0.1 | Reward correct answers without tools |

This creates a natural trade-off: **Is the +1.0 correctness worth the -0.05 cost?** The model learns that when it already knows the answer, searching is wasteful. But when uncertain, the investment pays off.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EfficientReasoning-GRPO.git
cd EfficientReasoning-GRPO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
make install
# Or manually:
pip install -r requirements.txt

# Install Unsloth for faster training (optional but recommended)
pip install unsloth
```

### 2. Run the Full Pipeline

```bash
# Complete pipeline: Generate SFT → Train SFT → Train GRPO → Evaluate → Plot
make all
```

### 3. Individual Phases

```bash
# Phase 1: Generate synthetic SFT data
make generate-sft

# Phase 1: Train SFT model (cold start)
make train-sft

# Phase 2: Train GRPO model
make train

# Phase 3: Run inference
python scripts/inference.py \
    --checkpoint logs/checkpoint-final \
    --query "Who directed the movie Inception?"

# Phase 3: Benchmark evaluation
make evaluate

# Phase 3: Generate Pareto plot
make plot
```

---

## ⚙️ Configuration

All configuration is centralized in `configs/default_config.yaml`:

```yaml
# Model Configuration
model:
  name_or_path: "unsloth/Qwen2.5-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true

# LoRA Fine-tuning
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# GRPO Training
training:
  learning_rate: 2.0e-5
  batch_size: 4
  num_generations: 8       # Group size (G)
  max_steps: 500
  beta: 0.04              # KL penalty
  lr_scheduler: "cosine"  # Learning rate scheduler
  resume_from_checkpoint: null

# Tool Support
tools:
  enabled: true
  max_tool_calls: 5

# Reward Function
rewards:
  correctness_weight: 1.0
  wrong_penalty: -0.5
  format_weight: 0.1
  format_penalty: -0.5
  efficiency_penalty: 0.05
  incomplete_call_penalty: -0.2
  efficient_bonus: 0.1
```

### CLI Overrides

Override any config value via command line:

```bash
python scripts/train.py --config configs/default_config.yaml \
    --training.learning_rate 1e-4 \
    --training.max_steps 1000 \
    --tools.enabled false
```

---

## 📊 Expected Results

After training, you'll observe the emergence of a **Pareto-optimal frontier**:

| Model | Accuracy | Avg Tokens | Avg Tool Calls | Efficiency Gain |
|-------|----------|------------|----------------|-----------------|
| Base (Zero-Shot) | ~40% | ~200 | ~1.5 | - |
| SFT (Always Search) | ~65% | ~450 | ~2.0 | -50% tokens |
| **GRPO (Efficient)** | **~65%** | **~270** | **~0.8** | **+40%** |

The GRPO model achieves **comparable accuracy to SFT while using 40% fewer tokens** by learning to skip unnecessary searches.

---

## 🔧 Advanced Features

### Checkpoint Resumption

Training automatically resumes from the latest checkpoint if interrupted:

```bash
# Manual resume
python scripts/train.py --config configs/default_config.yaml \
    --training.resume_from_checkpoint logs/checkpoint-100
```

### Result Caching

Benchmark evaluation caches results for faster iterative testing:

```bash
# Run with caching (default)
python eval/benchmark.py --checkpoints base logs/checkpoint-final

# Disable caching
python eval/benchmark.py --checkpoints base logs/checkpoint-final --no-cache
```

### Native Tool Support

The GRPOTrainer integrates directly with tool functions:

```python
def search_wiki(query: str) -> str:
    """Search Wikipedia for information."""
    return mock_env.search(query)

trainer = GRPOTrainer(
    model=model,
    tools=[search_wiki],  # Tools execute during generation!
    reward_funcs=[reward_func],
    ...
)
```

---

## 📁 Project Structure

```
EfficientReasoning-GRPO/
├── configs/
│   └── default_config.yaml          # Main configuration
├── src/
│   ├── config_parser.py            # YAML + CLI parsing with validation
│   ├── constants.py                # Centralized magic numbers
│   ├── dataset.py                  # HotpotQA loader + fallback data
│   ├── mock_env.py                 # O(1) search simulation
│   ├── model_utils.py              # Unsloth/Transformers loading
│   ├── rewards.py                  # Reward function (core logic)
│   ├── tools.py                    # Tool function definitions
│   └── utils.py                    # Logging, seeding utilities
├── scripts/
│   ├── generate_sft.py            # Phase 1: Generate synthetic traces
│   ├── train_sft.py               # Phase 1: SFT cold start
│   ├── train.py                   # Phase 2: GRPO training
│   └── inference.py               # Phase 3: Agentic inference
├── eval/
│   ├── benchmark.py                # Evaluation with caching
│   └── plot_pareto.py             # Pareto frontier visualization
├── tests/                          # 54 unit tests
├── Makefile                        # Build automation
├── pyproject.toml                  # Project metadata
└── README.md
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_rewards.py -v
pytest tests/test_config.py -v
pytest tests/test_fallback.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🖥️ Hardware Requirements

| Configuration | RAM | GPU VRAM | Training Time |
|--------------|-----|----------|---------------|
| Minimum | 16GB | 8GB (4-bit) | ~6-8 hours |
| Recommended | 32GB | 24GB (RTX 3090/4090) | ~2-4 hours |
| Optimal | 64GB | 80GB (A100) | ~1-2 hours |

---

## 🔬 Technical Deep Dive

### Why GRPO?

GRPO (Group Relative Policy Optimization) offers several advantages over PPO:

1. **No Value Network**: Computes advantages relative to group mean, reducing memory
2. **Stable Training**: Normalizes rewards within groups for smoother gradients
3. **Efficient**: Lower computational overhead than actor-critic methods

### The VOI Learning Dynamics

```
Step 1: Model learns XML format (from SFT cold start)
Step 2: GRPO encourages correct answers (+1.0)
Step 3: GRPO penalizes tool calls (-0.05 each)
Step 4: Model discovers:
        - "If I know the answer, searching wastes -0.05"
        - "If I'm unsure, the +1.0 reward justifies the -0.05 cost"
Step 5: Emergent behavior: dynamic compute allocation
```

### Tool Execution Verification

The reward function verifies actual tool execution:

```python
def verify_tool_execution(completion):
    # Check that <obs> follows each <call>
    num_calls = count_xml_tags(completion, "call")
    num_obs = count_xml_tags(completion, "obs")
    
    if num_calls > num_obs:
        return {"penalty": -0.2, "executed": num_obs}
    return {"penalty": 0, "executed": num_calls}
```

---

## 🛠️ Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size and sequence length
model:
  load_in_4bit: true
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

### Unsloth Installation

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Dataset Loading Fails

The system automatically falls back to synthetic data (25 built-in samples) if HotpotQA cannot be loaded.

---

## 📈 Logging & Monitoring

Configure external logging:

```yaml
training:
  report_to: "wandb"  # or "tensorboard", "mlflow"
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests (`pytest tests/ -v`)
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{efficientreasoning2024,
  title = {EfficientReasoning: Optimizing Agentic Compute-Accuracy Trade-offs via GRPO},
  author = {EfficientReasoning Team},
  year = {2024},
  url = {https://github.com/yourusername/EfficientReasoning-GRPO}
}
```

---

## 🙏 Acknowledgments

- [Hugging Face TRL](https://github.com/huggingface/trl) - GRPO implementation
- [Unsloth](https://github.com/unslothai/unsloth) - Efficient fine-tuning
- [HotpotQA](https://hotpotqa.github.io/) - Multi-hop QA dataset
- [Qwen](https://huggingface.co/Qwen) - Base model

---

<p align="center">
  <strong>Star us on GitHub if you find this useful!</strong>
</p>
