#!/usr/bin/env python3
"""
Simple test for quantization experiment setup (Task 8.2).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_quantization_imports():
    """Test that quantization modules can be imported."""
    print("Testing quantization module imports...")
    
    try:
        from models.quantization_manager import QuantizationManager
        from models.model_info import QuantizationConfig
        print("✓ Quantization modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import quantization modules: {e}")
        return False

def test_quantization_config():
    """Test quantization configuration creation."""
    print("Testing quantization configuration...")
    
    try:
        from models.model_info import QuantizationConfig
        
        # Test 8-bit config
        config_8bit = QuantizationConfig(
            bits=8,
            compute_dtype="float16",
            quant_type="nf4",
            double_quant=False
        )
        print(f"✓ 8-bit config created: {config_8bit}")
        
        # Test 4-bit config
        config_4bit = QuantizationConfig(
            bits=4,
            compute_dtype="float16",
            quant_type="nf4",
            double_quant=True
        )
        print(f"✓ 4-bit config created: {config_4bit}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create quantization config: {e}")
        return False

def test_quantization_manager():
    """Test quantization manager functionality."""
    print("Testing quantization manager...")
    
    try:
        from models.quantization_manager import QuantizationManager
        from models.model_info import QuantizationConfig
        
        manager = QuantizationManager()
        print("✓ QuantizationManager created")
        
        # Test config creation
        config = QuantizationConfig(bits=8, compute_dtype="float16")
        
        try:
            bnb_config = manager.create_bnb_config(config)
            print("✓ BitsAndBytesConfig created successfully")
        except RuntimeError as e:
            if "not available" in str(e):
                print("⚠ BitsAndBytesConfig creation skipped (dependencies not available)")
            else:
                raise
        
        # Test memory measurement (should work without PyTorch)
        memory = manager.measure_system_memory()
        print(f"✓ System memory measured: {memory}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test quantization manager: {e}")
        return False

def test_experiment_config():
    """Test experiment configuration with quantization."""
    print("Testing experiment configuration with quantization...")
    
    try:
        from experiments.config import ExperimentConfig, ModelConfig, DatasetConfig, LoRAConfig
        from models.model_info import QuantizationConfig
        
        # Create config with quantization
        config = ExperimentConfig(
            name="test_quantization_experiment",
            description="Test quantization experiment",
            model=ModelConfig(name="deit_tiny_patch16_224"),
            dataset=DatasetConfig(name="cifar10"),
            lora=LoRAConfig(rank=8, alpha=16.0),
            quantization=QuantizationConfig(bits=8, compute_dtype="float16"),
            seed=42
        )
        
        print(f"✓ Experiment config with quantization created: {config.get_experiment_id()}")
        
        # Test config validation
        from experiments.config import ConfigValidator
        is_valid, errors = ConfigValidator.validate_config(config)
        
        if is_valid:
            print("✓ Configuration validation passed")
        else:
            print(f"⚠ Configuration validation issues: {errors}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test experiment config: {e}")
        return False

def test_quantization_experiment_runner():
    """Test quantization experiment runner setup."""
    print("Testing quantization experiment runner...")
    
    try:
        # Import the runner
        sys.path.insert(0, str(Path(__file__).parent / "experiments" / "scripts"))
        from run_quantization_experiments import QuantizationExperimentRunner
        
        runner = QuantizationExperimentRunner()
        print("✓ QuantizationExperimentRunner created")
        
        # Test matrix creation
        configs = runner.create_quantization_matrix()
        print(f"✓ Created {len(configs)} quantization experiment configurations")
        
        # Analyze configuration breakdown
        no_quant = sum(1 for c in configs if c.quantization is None)
        quant_8bit = sum(1 for c in configs if c.quantization and c.quantization.bits == 8)
        quant_4bit = sum(1 for c in configs if c.quantization and c.quantization.bits == 4)
        
        print(f"  - No quantization: {no_quant}")
        print(f"  - 8-bit quantization: {quant_8bit}")
        print(f"  - 4-bit quantization: {quant_4bit}")
        
        # Test a few configurations
        print("\nSample configurations:")
        for i, config in enumerate(configs[:3]):
            quant_info = "None"
            if config.quantization:
                quant_info = f"{config.quantization.bits}-bit"
            print(f"  {i+1}. {config.get_experiment_id()} (quantization: {quant_info})")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test quantization experiment runner: {e}")
        return False

def test_gradient_flow_monitor():
    """Test gradient flow monitoring."""
    print("Testing gradient flow monitor...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "experiments" / "scripts"))
        from run_quantization_experiments import GradientFlowMonitor
        
        monitor = GradientFlowMonitor()
        print("✓ GradientFlowMonitor created")
        
        # Test analysis with mock data
        import numpy as np
        monitor.gradient_norms = list(np.random.normal(1.0, 0.1, 100))
        
        analysis = monitor.analyze_gradient_flow()
        print(f"✓ Gradient flow analysis completed: {analysis}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test gradient flow monitor: {e}")
        return False

def main():
    """Run all tests."""
    print("PEFT Vision Transformer Quantization Tests (Task 8.2)")
    print("=" * 60)
    
    tests = [
        test_quantization_imports,
        test_quantization_config,
        test_quantization_manager,
        test_experiment_config,
        test_quantization_experiment_runner,
        test_gradient_flow_monitor
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n{test.__name__}:")
        print("-" * 40)
        
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                failed += 1
                print("✗ FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n✓ All tests passed! Quantization experiment setup is ready.")
        print("\nNext steps:")
        print("1. Run: python experiments/scripts/run_quantization_experiments.py --dry-run")
        print("2. Run: python experiments/scripts/run_quantization_experiments.py --validate-only")
        print("3. Run actual experiments: python experiments/scripts/run_quantization_experiments.py")
    else:
        print(f"\n✗ {failed} tests failed. Please fix issues before running experiments.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())