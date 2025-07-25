#!/usr/bin/env python3
"""
Simple test script for baseline experiments to verify the implementation works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_baseline_experiment_creation():
    """Test that we can create baseline experiment configurations."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "experiments" / "scripts"))
        from run_baseline_experiments import BaselineExperimentRunner
        
        runner = BaselineExperimentRunner()
        configs = runner.create_baseline_matrix()
        
        print(f"✓ Created {len(configs)} baseline experiment configurations")
        
        # Test a few configurations
        for i, config in enumerate(configs[:3]):
            print(f"  {i+1}. {config.get_experiment_id()}")
            print(f"     Model: {config.model.name}")
            print(f"     Dataset: {config.dataset.name}")
            print(f"     LoRA: rank={config.lora.rank}, alpha={config.lora.alpha}")
            print(f"     Seed: {config.seed}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create baseline configurations: {e}")
        return False

def test_configuration_validation():
    """Test that configurations pass validation."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "experiments" / "scripts"))
        from run_baseline_experiments import BaselineExperimentRunner
        from experiments.config import ConfigValidator
        
        runner = BaselineExperimentRunner()
        configs = runner.create_baseline_matrix()
        
        valid_count = 0
        invalid_count = 0
        
        for config in configs:
            is_valid, errors = ConfigValidator.validate_config(config)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                print(f"  ✗ Invalid: {config.get_experiment_id()} - {errors}")
        
        print(f"✓ Validation: {valid_count} valid, {invalid_count} invalid configurations")
        return invalid_count == 0
        
    except Exception as e:
        print(f"✗ Failed to validate configurations: {e}")
        return False

def test_experiment_requirements():
    """Test that the experiment meets task 8.1 requirements."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "experiments" / "scripts"))
        from run_baseline_experiments import BaselineExperimentRunner
        
        runner = BaselineExperimentRunner()
        configs = runner.create_baseline_matrix()
        
        # Check requirements
        models = set(config.model.name for config in configs)
        datasets = set(config.dataset.name for config in configs)
        lora_ranks = set(config.lora.rank for config in configs if config.lora)
        seeds = set(config.seed for config in configs)
        
        print("✓ Task 8.1 Requirements Check:")
        print(f"  Models: {sorted(models)}")
        print(f"  Datasets: {sorted(datasets)}")
        print(f"  LoRA ranks: {sorted(lora_ranks)}")
        print(f"  Seeds: {sorted(seeds)}")
        
        # Verify requirements
        required_ranks = {2, 4, 8, 16, 32}
        if not required_ranks.issubset(lora_ranks):
            print(f"  ✗ Missing required LoRA ranks: {required_ranks - lora_ranks}")
            return False
        
        if len(seeds) < 3:
            print(f"  ✗ Need at least 3 seeds for statistical significance, got {len(seeds)}")
            return False
        
        print("  ✓ All task requirements satisfied")
        return True
        
    except Exception as e:
        print(f"✗ Failed to check requirements: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Baseline Experiment Implementation (Task 8.1)")
    print("=" * 60)
    
    tests = [
        ("Configuration Creation", test_baseline_experiment_creation),
        ("Configuration Validation", test_configuration_validation),
        ("Task Requirements", test_experiment_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  ✗ {test_name} failed")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Baseline experiment implementation is ready.")
        print("\nNext steps:")
        print("1. Install PyTorch and dependencies to run actual experiments")
        print("2. Run: python3 experiments/scripts/run_baseline_experiments.py --max-experiments 5")
        print("3. Monitor results in experiments/outputs/baseline/")
    else:
        print("✗ Some tests failed. Please fix issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()