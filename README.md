# Local LLM Bench

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://www.apple.com/mac/)

Compare local LLM inference performance across **MLX** and **Ollama** backends on Apple Silicon.

Includes automatic **F1 evaluation** for structured extraction tasks and an **interactive HTML report** with dark mode.

## Features

- **Interactive CLI** — Select models from a menu, no flags to memorize
- **Memory-aware** — Auto-detects model sizes and shows what fits in RAM
- **F1 scoring** — Provide ground truth for automatic precision/recall/F1 evaluation
- **Incremental logging** — Results append to CSV, run models across multiple sessions
- **Auto MLX server** — Starts and stops the MLX server as needed
- **Interactive HTML report** — Sortable tables, charts, dark mode, system info
- **Prompt versioning** — Track performance across different prompt iterations

## Report Preview

The HTML report includes:
- System specs (OS, CPU, RAM, GPU)
- Key metrics cards (Best F1, Fastest Model, Lowest Memory)
- Interactive time series charts with metric/model/prompt filters
- Model comparison bubble chart (speed vs accuracy vs memory)
- Sortable results table with search
- Light/dark mode toggle

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [Ollama](https://ollama.com) and/or [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)

## Quick Start

```bash
pip install -r requirements.txt

# Edit config.yaml with your prompt and models
python benchmark.py
```

## Configuration

```yaml
# Your prompt (supports /no_think for Qwen3 models)
prompt: |
  Extract information from this text and return JSON...

# Optional: ground truth for F1 evaluation
ground_truth:
  field1: "expected value"
  items:
    - name: "Item 1"

# Models to benchmark (memory is auto-detected)
models:
  - name: MLX - 8B
    model_id: mlx-community/Qwen3-8B-4bit
    backend: mlx

  - name: Ollama - 8B
    model_id: qwen3:8b
    backend: ollama

# Settings
settings:
  temperature: 0.3
  max_tokens: 8192
  timeout_seconds: 600
  min_free_ram_buffer_gb: 4
```

## Usage

### Interactive (Default)

```bash
python benchmark.py
```

Shows a menu to select which model to run. Best for exploring and one-off runs.

### Run a Specific Model

```bash
python benchmark.py --model 1    # Run the first model in config
python benchmark.py -m 2         # Run the second model
```

### Run All Models

```bash
python benchmark.py --all
```

Runs all models sequentially, properly unloading each before loading the next.

### Fully Automated

```bash
python benchmark.py --all --skip-warnings
```

Runs all models with no prompts. Use this for scripting or unattended runs.

### CLI Reference

| Option | Description |
|--------|-------------|
| `--model N`, `-m N` | Run model N (1-indexed) |
| `--all` | Run all models sequentially |
| `--skip-warnings` | Skip memory/process warning prompts |
| `--config FILE`, `-c FILE` | Use custom config file (default: config.yaml) |

## Output

```
results/
├── benchmark_log.csv    # All runs (append-only)
├── report.md            # Markdown summary table
├── report.html          # Interactive HTML report
├── prompts/             # Stored by content hash
│   └── a1b2c3d4.txt
└── responses/           # Raw model outputs
    └── mlx-8b_2025-01-24T14-30-52.txt
```

## Setup

### MLX

```bash
pip install mlx mlx-lm

# Models download automatically on first use, or pre-download:
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-8B-4bit')"
```

### Ollama

```bash
# Install from https://ollama.com

ollama pull qwen3:8b
ollama pull qwen3-coder:30b-a3b-q4_K_M
```

## Tips

- **Close memory-heavy apps** before running large models
- **Use `/no_think`** in prompts for Qwen3 models to disable chain-of-thought
- **Same models on both backends** ensures fair MLX vs Ollama comparison
- **Ground truth matching** uses fuzzy string matching (handles minor variations)

## License

MIT

---

Built by [eeshans.com](https://eeshans.com) | [Subscribe to my newsletter](https://0to1datascience.substack.com)
