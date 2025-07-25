#!/usr/bin/env python3
"""
Simple test for configuration system without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Test basic imports and functionality
try:
    # Import directly from the standalone module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "standalone_config", 
        "src/experiments/standalone_config.py"
    )
    standalone_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone_config)
    
    ExperimentConfig = standalone_config.ExperimentConfig
    ModelConfig = standalone_config.ModelConfig
    DatasetConfig = standalone_config.DatasetConfig
    ExperimentMatrix = standalone_config.ExperimentMatrix
    ConfigValidator = standalone_config.ConfigValidator
    
    print("✓ Successfully imported experiment config classes")
    
    # Test model config
    model_config = ModelConfig(name="deit_tiny_patch16_224")
    print(f"✓ Model config created: {model_config.name}")
    
    # Test dataset config
    dataset_config = DatasetConfig(name="cifar10")
    print(f"✓ Dataset config created: {dataset_config.name}, classes={dataset_config.num_classes}")
    
    # Test experiment config
    exp_config = ExperimentConfig(
        name="test_experiment",
        model=model_config,
        dataset=dataset_config
    )
    print(f"✓ Experiment config created: {exp_config.name}")
    print(f"  - Model: {exp_config.model.name}")
    print(f"  - Dataset: {exp_config.dataset.name}")
    print(f"  - Experiment ID: {exp_config.get_experiment_id()}")
    
    # Test experiment matrix
    matrix = ExperimentMatrix(exp_config)
    matrix.add_seed_variation([42, 123])
    print(f"✓ Experiment matrix created with {matrix.count_experiments()} experiments")
    
    # Test config validation
    is_valid, errors = ConfigValidator.validate_config(exp_config)
    print(f"✓ Config validation: valid={is_valid}, errors={len(errors)}")
    
    # Test serialization
    config_dict = exp_config.to_dict()
    restored_config = ExperimentConfig.from_dict(config_dict)
    print(f"✓ Config serialization: {restored_config.name}")
    
    print("\n✓ All experiment configuration tests passed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)