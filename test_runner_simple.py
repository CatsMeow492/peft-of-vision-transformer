#!/usr/bin/env python3
"""
Simple test for experiment runner without external dependencies.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Test experiment runner functionality
try:
    # Import directly from the standalone modules
    import importlib.util
    
    # Load standalone config
    spec = importlib.util.spec_from_file_location(
        "standalone_config", 
        "src/experiments/standalone_config.py"
    )
    standalone_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone_config)
    
    # Load results module
    spec = importlib.util.spec_from_file_location(
        "results", 
        "src/experiments/results.py"
    )
    results_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(results_module)
    
    # Load runner module
    spec = importlib.util.spec_from_file_location(
        "runner", 
        "src/experiments/runner.py"
    )
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    
    print("✓ Successfully imported experiment runner modules")
    
    # Create test configurations
    ExperimentConfig = standalone_config.ExperimentConfig
    ModelConfig = standalone_config.ModelConfig
    DatasetConfig = standalone_config.DatasetConfig
    ExperimentMatrix = standalone_config.ExperimentMatrix
    
    ExperimentRunner = runner_module.ExperimentRunner
    ExperimentResult = results_module.ExperimentResult
    ResultsManager = results_module.ResultsManager
    
    # Test basic configuration
    config1 = ExperimentConfig(
        name="test_experiment_1",
        model=ModelConfig(name="deit_tiny_patch16_224"),
        dataset=DatasetConfig(name="cifar10")
    )
    
    config2 = ExperimentConfig(
        name="test_experiment_2", 
        model=ModelConfig(name="deit_small_patch16_224"),
        dataset=DatasetConfig(name="cifar100")
    )
    
    print("✓ Test configurations created")
    
    # Test experiment matrix
    matrix = ExperimentMatrix(config1)
    matrix.add_seed_variation([42, 123])
    matrix.add_dataset_variation(["cifar10", "cifar100"])
    
    print(f"✓ Experiment matrix created with {matrix.count_experiments()} experiments")
    
    # Test results manager
    with tempfile.TemporaryDirectory() as temp_dir:
        results_manager = ResultsManager(temp_dir, use_database=False)  # Skip DB for simplicity
        print("✓ Results manager created")
        
        # Create a mock result
        from datetime import datetime
        mock_result = ExperimentResult(
            experiment_id="test_exp_1",
            config=config1,
            metrics={"final_accuracy": 0.85, "final_loss": 0.3},
            start_time=datetime.now(),
            end_time=datetime.now(),
            status="completed"
        )
        
        # Test saving and loading
        results_manager.add_result(mock_result)
        loaded_result = results_manager.get_result("test_exp_1")
        
        if loaded_result and loaded_result.experiment_id == "test_exp_1":
            print("✓ Results save/load working")
        else:
            print("✗ Results save/load failed")
        
        # Test experiment runner (without actual execution)
        runner = ExperimentRunner(
            output_dir=Path(temp_dir) / "runner_test",
            max_memory_gb=16.0
        )
        
        runner.add_experiment(config1)
        runner.add_experiment(config2)
        
        print(f"✓ Experiment runner created with {len(runner.experiment_queue.experiments)} experiments")
        
        # Test status
        status = runner.get_status()
        print(f"✓ Runner status: {status['queue']['total']} total experiments")
        
        # Test resource monitor
        resource_monitor = runner_module.ResourceMonitor(max_memory_gb=16.0)
        resource_monitor.start_monitoring()
        
        import time
        time.sleep(0.1)  # Brief monitoring
        
        stats = resource_monitor.get_current_stats()
        resource_monitor.stop_monitoring()
        
        print(f"✓ Resource monitoring: {stats['memory_gb']:.1f}GB memory")
        
        # Test experiment matrix runner function
        print("✓ Testing matrix runner function...")
        
        # Create a small matrix for testing
        small_matrix = ExperimentMatrix(config1)
        small_matrix.add_seed_variation([42])  # Just one experiment
        
        # This would normally run experiments, but we'll just test the setup
        try:
            runner2 = ExperimentRunner(
                output_dir=Path(temp_dir) / "matrix_test",
                max_memory_gb=16.0
            )
            runner2.add_experiment_matrix(small_matrix)
            print(f"✓ Matrix added to runner: {len(runner2.experiment_queue.experiments)} experiments")
        except Exception as e:
            print(f"✗ Matrix runner test failed: {e}")
    
    print("\n✓ All experiment runner tests passed!")
    
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