# PEFT Vision Transformer Research Framework

A comprehensive framework for Parameter-Efficient Fine-Tuning (PEFT) of Vision Transformers using LoRA and quantization techniques.

## Overview

This research project implements a systematic study of PEFT techniques for Vision Transformers, focusing on:

- **LoRA (Low-Rank Adaptation)** with various rank configurations
- **Quantization** using 8-bit and 4-bit precision
- **AdaLoRA** for adaptive rank allocation
- **QA-LoRA** for quantization-aware training

The framework is optimized for resource-constrained research environments (M2 MacBook with 96GB memory) while producing publication-quality results.

## Project Structure

```
peft-vision-transformer/
├── src/                    # Core implementation
│   ├── models/            # ViT and LoRA implementations
│   ├── training/          # Training pipelines
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Helper functions
├── experiments/            # Experiment configurations and scripts
│   ├── configs/           # YAML configuration files
│   └── scripts/           # Execution scripts
├── results/               # Experimental results and analysis
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.1.2 or compatible
- CUDA support (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd peft-vision-transformer
```

2. Install dependencies and set up development environment:
```bash
make setup
```

This will:
- Install all required dependencies
- Set up pre-commit hooks for code quality
- Configure development tools

### Manual Installation

If you prefer manual installation:

```bash
# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pre-commit
pre-commit install
```

## Quick Start

### Running Experiments

1. **Configure experiment**: Create or modify YAML files in `experiments/configs/`
2. **Execute experiment**: Use scripts in `experiments/scripts/`
3. **Analyze results**: Results will be saved in `results/`

### Development

- **Format code**: `make format`
- **Run linting**: `make lint`
- **Run tests**: `make test`
- **Run tests with coverage**: `make test-cov`

## Research Objectives

This framework supports research into:

1. **Novel LoRA applications** to Vision Transformers
2. **Quantization techniques** for efficient ViT fine-tuning
3. **Adaptive methods** like AdaLoRA for vision tasks
4. **Resource-constrained training** on consumer hardware

## Target Models and Datasets

### Models
- DeiT-tiny (5M parameters)
- DeiT-small (22M parameters)  
- ViT-small (20M parameters)

### Datasets
- CIFAR-10
- CIFAR-100
- TinyImageNet

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Ensure code passes all checks: `make lint test`
5. Submit a pull request

## Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Document all public APIs
- Use type hints throughout the codebase
- Maintain backwards compatibility when possible

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{peft-vision-transformer,
  title={Parameter-Efficient Fine-Tuning for Vision Transformers: A Comprehensive Study},
  author={Research Team},
  year={2024},
  url={https://github.com/research-team/peft-vision-transformer}
}
```

## Acknowledgments

This work builds upon:
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)