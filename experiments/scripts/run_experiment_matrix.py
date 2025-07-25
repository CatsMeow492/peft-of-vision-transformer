#!/usr/bin/env python3
"""
Script to run experiment matrix for PEFT Vision Transformer research.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from experiments.standalone_config import (
        ExperimentConfig,
        ModelConfig,
        DatasetConfig,
        LoRAConfig,
        QuantizationConfig,
        TrainingConfig,
        ExperimentMatrix,
        ConfigValidator,
        create_default_experiment_matrix
    )
    print("✓ Configuration system loaded")
except ImportError as e:
    print(f"✗ Failed to import configuration system: {e}")
    sys.exit(1)


def create_small_test_matrix() -> ExperimentMatrix:
    """Create a small test matrix for validation."""
    base_config = ExperimentConfig(
        name="small_test_matrix",
        description="Small test matrix for validation",
        tags=["test", "validation"],
        model=ModelConfig(name="deit_tiny_patch16_224"),
        dataset=DatasetConfig(name="cifar10", batch_size=16),  # Smaller batch for testing
        lora=LoRAConfig(rank=4, alpha=8.0),  # Smaller LoRA for testing
        training=TrainingConfig(
            learning_rate=1e-4,
            num_epochs=2,  # Very short for testing
            batch_size=16,
            save_steps=100,
            eval_steps=50
        ),
        seed=42
    )
    
    matrix = ExperimentMatrix(base_config)
    
    # Add minimal variations for testing
    matrix.add_lora_rank_variation([4, 8])
    matrix.add_seed_variation([42, 123])
    
    return matrix


def create_comprehensive_matrix() -> ExperimentMatrix:
    """Create comprehensive experiment matrix for full research."""
    base_config = ExperimentConfig(
        name="comprehensive_peft_study",
        description="Comprehensive PEFT study on Vision Transformers",
        tags=["peft", "vision_transformer", "lora", "quantization", "research"],
        model=ModelConfig(name="deit_tiny_patch16_224"),
        dataset=DatasetConfig(name="cifar10"),
        lora=LoRAConfig(rank=8, alpha=16.0),
        training=TrainingConfig(
            learning_rate=1e-4,
            num_epochs=10,
            batch_size=32,
            save_steps=500,
            eval_steps=100
        )
    )
    
    matrix = ExperimentMatrix(base_config)
    
    # Model variations
    matrix.add_model_variation([
        "deit_tiny_patch16_224",
        "deit_small_patch16_224"
    ])
    
    # Dataset variations
    matrix.add_dataset_variation([
        "cifar10",
        "cifar100"
    ])
    
    # LoRA rank variations
    matrix.add_lora_rank_variation([2, 4, 8, 16, 32])
    
    # Quantization variations
    matrix.add_quantization_variation([8, 4])
    
    # Multiple seeds for statistical significance
    matrix.add_seed_variation([42, 123, 456])
    
    # PEFT method variations
    matrix.add_method_variation(["lora", "adalora"])
    
    return matrix


def validate_matrix(matrix: ExperimentMatrix) -> bool:
    """Validate all configurations in the matrix."""
    print(f"Validating experiment matrix with {matrix.count_experiments()} experiments...")
    
    valid_count = 0
    invalid_count = 0
    
    for i, config in enumerate(matrix.generate_configs()):
        is_valid, errors = ConfigValidator.validate_config(config)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"  ✗ Config {i+1} invalid: {errors}")
        
        # Show progress for large matrices
        if (i + 1) % 50 == 0:
            print(f"  Validated {i+1} configurations...")
    
    print(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
    return invalid_count == 0


def save_matrix_configs(matrix: ExperimentMatrix, output_dir: Path):
    """Save all matrix configurations to individual YAML files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {matrix.count_experiments()} configurations to {output_dir}")
    
    for i, config in enumerate(matrix.generate_configs()):
        config_file = output_dir / f"{config.get_experiment_id()}.yaml"
        
        try:
            config.save_yaml(config_file)
        except Exception as e:
            print(f"  ✗ Failed to save config {i+1}: {e}")
            # Save as JSON fallback
            import json
            json_file = output_dir / f"{config.get_experiment_id()}.json"
            with open(json_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
            print(f"  → Saved as JSON: {json_file}")
    
    print(f"✓ Configurations saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run PEFT Vision Transformer experiments")
    parser.add_argument(
        "--matrix-type",
        choices=["small", "comprehensive", "default"],
        default="small",
        help="Type of experiment matrix to create"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs"),
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/generated_configs"),
        help="Directory to save generated configurations"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configurations, don't run experiments"
    )
    parser.add_argument(
        "--save-configs",
        action="store_true",
        help="Save all generated configurations to files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    print("PEFT Vision Transformer Experiment Runner")
    print("=" * 50)
    
    # Create experiment matrix
    if args.matrix_type == "small":
        print("Creating small test matrix...")
        matrix = create_small_test_matrix()
    elif args.matrix_type == "comprehensive":
        print("Creating comprehensive research matrix...")
        matrix = create_comprehensive_matrix()
    elif args.matrix_type == "default":
        print("Creating default matrix...")
        matrix = create_default_experiment_matrix()
    else:
        print(f"Unknown matrix type: {args.matrix_type}")
        sys.exit(1)
    
    # Show matrix summary
    summary = matrix.get_summary()
    print(f"✓ Matrix created: {summary}")
    
    # Validate configurations
    if not validate_matrix(matrix):
        print("✗ Matrix validation failed")
        if not args.dry_run:
            sys.exit(1)
    
    # Save configurations if requested
    if args.save_configs:
        if args.dry_run:
            print(f"[DRY RUN] Would save configs to {args.config_dir}")
        else:
            save_matrix_configs(matrix, args.config_dir)
    
    # Stop here if only validating
    if args.validate_only:
        print("✓ Validation complete")
        return
    
    # Show what experiments would be run
    print("\nExperiment Preview:")
    print("-" * 30)
    
    for i, config in enumerate(matrix.generate_configs()):
        if i >= 5:  # Show first 5 experiments
            remaining = matrix.count_experiments() - 5
            print(f"... and {remaining} more experiments")
            break
        
        print(f"{i+1}. {config.get_experiment_id()}")
        print(f"   Model: {config.model.name}")
        print(f"   Dataset: {config.dataset.name}")
        if config.lora:
            print(f"   LoRA: rank={config.lora.rank}, alpha={config.lora.alpha}")
        if config.quantization:
            print(f"   Quantization: {config.quantization.bits}-bit")
        print(f"   Seed: {config.seed}")
        print()
    
    if args.dry_run:
        print("[DRY RUN] Would run experiments with the above configurations")
        return
    
    # Actually run experiments (simulation for now)
    print("Starting experiment execution...")
    print("Note: This is currently a simulation. Actual training integration needed.")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate running experiments
    import time
    import json
    from datetime import datetime
    
    for i, config in enumerate(matrix.generate_configs()):
        print(f"Running experiment {i+1}/{matrix.count_experiments()}: {config.get_experiment_id()}")
        
        # Create experiment directory
        exp_dir = args.output_dir / config.get_experiment_id()
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        
        # Simulate training time
        time.sleep(0.1)
        
        # Create mock results
        mock_results = {
            "experiment_id": config.get_experiment_id(),
            "status": "completed",
            "metrics": {
                "final_accuracy": 0.80 + (hash(config.get_experiment_id()) % 100) / 500,
                "final_loss": 0.5 - (hash(config.get_experiment_id()) % 100) / 1000,
                "training_time_seconds": 120 + (hash(config.get_experiment_id()) % 60)
            },
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "peak_memory_gb": 2.0 + (hash(config.get_experiment_id()) % 100) / 100
        }
        
        # Save results
        results_file = exp_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(mock_results, f, indent=2)
        
        print(f"  ✓ Completed: accuracy={mock_results['metrics']['final_accuracy']:.3f}")
    
    print(f"\n✓ All experiments completed! Results saved to {args.output_dir}")
    print("\nNext steps:")
    print("1. Integrate with actual training pipeline")
    print("2. Add statistical analysis of results")
    print("3. Generate publication-quality figures")


if __name__ == "__main__":
    main()