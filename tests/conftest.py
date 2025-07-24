"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "model_name": "deit_tiny_patch16_224",
        "dataset_name": "cifar10",
        "lora_config": {
            "rank": 4,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["qkv"]
        },
        "training_config": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 10
        }
    }


@pytest.fixture
def set_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    return torch.nn.Linear(10, 5)


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "input": torch.randn(4, 3, 224, 224),
        "target": torch.randint(0, 10, (4,))
    }