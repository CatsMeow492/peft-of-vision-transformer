"""
Tests for experiment configuration system.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.experiments.config import (
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    ExperimentMatrix,
    ConfigValidator,
    create_default_experiment_matrix
)
from src.models.lora_config import LoRAConfig
from src.models.model_info import QuantizationConfig
from src.training.peft_trainer import TrainingConfig


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig(name="deit_tiny_patch16_224")
        
        assert config.name == "deit_tiny_patch16_224"
        assert config.source == "timm"
        assert config.pretrained is True
        assert config.image_size == 224
        assert config.patch_size == 16
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid config
        config = ModelConfig(name="deit_tiny_patch16_224", source="timm")
        assert config.source == "timm"
        
        # Invalid source
        with pytest.raises(ValueError, match="Model source must be"):
            ModelConfig(name="test", source="invalid")
        
        # Invalid image size
        with pytest.raises(ValueError, match="Image size must be positive"):
            ModelConfig(name="test", image_size=0)


class TestDatasetConfig:
    """Test DatasetConfig class."""
    
    def test_default_dataset_config(self):
        """Test default dataset configuration."""
        config = DatasetConfig(name="cifar10")
        
        assert config.name == "cifar10"
        assert config.num_classes == 10
        assert config.image_size == 224
        assert config.batch_size == 32
    
    def test_dataset_num_classes(self):
        """Test dataset num_classes property."""
        assert DatasetConfig(name="cifar10").num_classes == 10
        assert DatasetConfig(name="cifar100").num_classes == 100
        assert DatasetConfig(name="tiny_imagenet").num_classes == 200
    
    def test_dataset_validation(self):
        """Test dataset configuration validation."""
        # Valid dataset
        config = DatasetConfig(name="cifar10")
        assert config.name == "cifar10"
        
        # Invalid dataset
        with pytest.raises(ValueError, match="Dataset must be one of"):
            DatasetConfig(name="invalid_dataset")


class TestExperimentConfig:
    """Test ExperimentConfig class."""
    
    def test_default_experiment_config(self):
        """Test default experiment configuration."""
        config = ExperimentConfig(name="test_experiment")
        
        assert config.name == "test_experiment"
        assert config.model.name == "deit_tiny_patch16_224"
        assert config.dataset.name == "cifar10"
        assert config.seed == 42
    
    def test_config_post_init_adjustments(self):
        """Test configuration adjustments in __post_init__."""
        # Test image size synchronization
        model = ModelConfig(name="test", image_size=256)
        dataset = DatasetConfig(name="cifar10", image_size=224)
        
        config = ExperimentConfig(
            name="test",
            model=model,
            dataset=dataset
        )
        
        # Model image size should be adjusted to match dataset
        assert config.model.image_size == 224
        assert config.model.num_classes == 10
    
    def test_config_serialization(self):
        """Test configuration serialization to/from dict."""
        original_config = ExperimentConfig(
            name="test_experiment",
            description="Test description",
            tags=["test", "experiment"],
            lora=LoRAConfig(rank=16, alpha=32.0),
            seed=123
        )
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = ExperimentConfig.from_dict(config_dict)
        
        assert restored_config.name == original_config.name
        assert restored_config.description == original_config.description
        assert restored_config.tags == original_config.tags
        assert restored_config.lora.rank == original_config.lora.rank
        assert restored_config.seed == original_config.seed
    
    def test_config_yaml_serialization(self):
        """Test YAML serialization."""
        config = ExperimentConfig(
            name="yaml_test",
            lora=LoRAConfig(rank=8),
            quantization=QuantizationConfig(bits=8)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            
            # Load back from YAML
            loaded_config = ExperimentConfig.load_yaml(f.name)
            
            assert loaded_config.name == config.name
            assert loaded_config.lora.rank == config.lora.rank
            assert loaded_config.quantization.bits == config.quantization.bits
        
        # Clean up
        Path(f.name).unlink()
    
    def test_experiment_id_generation(self):
        """Test experiment ID generation."""
        config = ExperimentConfig(
            name="test",
            model=ModelConfig(name="deit_tiny_patch16_224"),
            dataset=DatasetConfig(name="cifar10"),
            lora=LoRAConfig(rank=8, alpha=16.0),
            seed=42
        )
        
        exp_id = config.get_experiment_id()
        expected_parts = ["deit_tiny_patch16_224", "cifar10", "seed42", "lora_r8_a16.0"]
        
        for part in expected_parts:
            assert part in exp_id
    
    def test_method_validation(self):
        """Test PEFT method validation."""
        # QA-LoRA without quantization should fail
        with pytest.raises(ValueError, match="QA-LoRA requires quantization"):
            ExperimentConfig(
                name="test",
                use_qa_lora=True,
                quantization=None
            )
        
        # AdaLoRA without LoRA should fail
        with pytest.raises(ValueError, match="AdaLoRA requires LoRA"):
            ExperimentConfig(
                name="test",
                use_adalora=True,
                lora=None
            )


class TestExperimentMatrix:
    """Test ExperimentMatrix class."""
    
    def test_basic_matrix_generation(self):
        """Test basic matrix generation."""
        base_config = ExperimentConfig(name="base")
        matrix = ExperimentMatrix(base_config)
        
        # No variations should yield just the base config
        configs = list(matrix.generate_configs())
        assert len(configs) == 1
        assert configs[0].name == "base"
    
    def test_single_variation(self):
        """Test matrix with single variation."""
        base_config = ExperimentConfig(name="base")
        matrix = ExperimentMatrix(base_config)
        
        matrix.add_seed_variation([42, 123, 456])
        
        configs = list(matrix.generate_configs())
        assert len(configs) == 3
        
        seeds = [config.seed for config in configs]
        assert set(seeds) == {42, 123, 456}
    
    def test_multiple_variations(self):
        """Test matrix with multiple variations."""
        base_config = ExperimentConfig(
            name="base",
            lora=LoRAConfig(rank=8)
        )
        matrix = ExperimentMatrix(base_config)
        
        matrix.add_lora_rank_variation([4, 8, 16])
        matrix.add_seed_variation([42, 123])
        
        configs = list(matrix.generate_configs())
        assert len(configs) == 6  # 3 ranks × 2 seeds
        
        # Check all combinations are present
        combinations = [(config.lora.rank, config.seed) for config in configs]
        expected = [(4, 42), (4, 123), (8, 42), (8, 123), (16, 42), (16, 123)]
        assert set(combinations) == set(expected)
    
    def test_method_variation(self):
        """Test PEFT method variations."""
        base_config = ExperimentConfig(
            name="base",
            lora=LoRAConfig(rank=8),
            quantization=QuantizationConfig(bits=8)
        )
        matrix = ExperimentMatrix(base_config)
        
        matrix.add_method_variation(["lora", "adalora", "qa_lora"])
        
        configs = list(matrix.generate_configs())
        assert len(configs) == 3
        
        # Check method flags
        methods = []
        for config in configs:
            if config.use_adalora:
                methods.append("adalora")
            elif config.use_qa_lora:
                methods.append("qa_lora")
            else:
                methods.append("lora")
        
        assert set(methods) == {"lora", "adalora", "qa_lora"}
    
    def test_matrix_count(self):
        """Test experiment count calculation."""
        base_config = ExperimentConfig(name="base", lora=LoRAConfig())
        matrix = ExperimentMatrix(base_config)
        
        matrix.add_lora_rank_variation([4, 8, 16])  # 3 options
        matrix.add_seed_variation([42, 123])        # 2 options
        matrix.add_dataset_variation(["cifar10", "cifar100"])  # 2 options
        
        assert matrix.count_experiments() == 12  # 3 × 2 × 2
    
    def test_matrix_summary(self):
        """Test matrix summary generation."""
        base_config = ExperimentConfig(name="test_base", lora=LoRAConfig())
        matrix = ExperimentMatrix(base_config)
        
        matrix.add_lora_rank_variation([4, 8])
        matrix.add_seed_variation([42, 123, 456])
        
        summary = matrix.get_summary()
        
        assert summary["total_experiments"] == 6
        assert summary["base_config"] == "test_base"
        assert "lora.rank" in summary["variations"]
        assert summary["variations"]["lora.rank"] == 2
        assert summary["variations"]["seed"] == 3


class TestConfigValidator:
    """Test ConfigValidator class."""
    
    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        config = ExperimentConfig(
            name="valid_test",
            model=ModelConfig(name="deit_tiny_patch16_224"),
            dataset=DatasetConfig(name="cifar10"),
            lora=LoRAConfig(rank=8, alpha=16.0),
            training=TrainingConfig(learning_rate=1e-4, num_epochs=10)
        )
        
        is_valid, errors = ConfigValidator.validate_config(config)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_model_validation(self):
        """Test validation with invalid model."""
        config = ExperimentConfig(
            name="invalid_model_test",
            model=ModelConfig(name="unsupported_model")
        )
        
        is_valid, errors = ConfigValidator.validate_config(config)
        assert not is_valid
        assert any("Unsupported model" in error for error in errors)
    
    def test_method_combination_validation(self):
        """Test validation of method combinations."""
        # QA-LoRA without quantization
        config = ExperimentConfig(
            name="invalid_qa_lora",
            use_qa_lora=True,
            quantization=None
        )
        
        is_valid, errors = ConfigValidator.validate_config(config)
        assert not is_valid
        assert any("QA-LoRA requires quantization" in error for error in errors)
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        config = ExperimentConfig(
            name="memory_test",
            model=ModelConfig(name="deit_tiny_patch16_224"),
            dataset=DatasetConfig(name="cifar10", batch_size=32)
        )
        
        memory_gb = ConfigValidator._estimate_memory_usage(config)
        assert memory_gb > 0
        assert memory_gb < 100  # Reasonable upper bound
    
    def test_config_compatibility(self):
        """Test configuration compatibility checking."""
        config1 = ExperimentConfig(
            name="config1",
            model=ModelConfig(name="deit_tiny_patch16_224"),
            dataset=DatasetConfig(name="cifar10")
        )
        
        config2 = ExperimentConfig(
            name="config2",
            model=ModelConfig(name="deit_tiny_patch16_224"),
            dataset=DatasetConfig(name="cifar10")
        )
        
        # Same model and dataset should be compatible
        is_compatible, reasons = ConfigValidator.check_compatibility(config1, config2)
        assert is_compatible
        assert len(reasons) == 0
        
        # Different models should not be compatible
        config3 = ExperimentConfig(
            name="config3",
            model=ModelConfig(name="deit_small_patch16_224"),
            dataset=DatasetConfig(name="cifar10")
        )
        
        is_compatible, reasons = ConfigValidator.check_compatibility(config1, config3)
        assert not is_compatible
        assert "Different models" in reasons


class TestDefaultMatrix:
    """Test default experiment matrix creation."""
    
    def test_default_matrix_creation(self):
        """Test creation of default experiment matrix."""
        matrix = create_default_experiment_matrix()
        
        assert isinstance(matrix, ExperimentMatrix)
        assert matrix.count_experiments() > 1
        
        # Generate a few configs to test
        configs = list(matrix.generate_configs())
        assert len(configs) > 10  # Should have many combinations
        
        # Check that we have different models, datasets, etc.
        models = {config.model.name for config in configs[:10]}
        datasets = {config.dataset.name for config in configs[:10]}
        
        assert len(models) > 1
        assert len(datasets) > 1
    
    def test_default_matrix_validation(self):
        """Test that default matrix generates valid configurations."""
        matrix = create_default_experiment_matrix()
        
        # Test first few configurations
        for i, config in enumerate(matrix.generate_configs()):
            if i >= 5:  # Test first 5 configs
                break
            
            is_valid, errors = ConfigValidator.validate_config(config)
            if not is_valid:
                print(f"Config {i} errors: {errors}")
            assert is_valid, f"Default matrix generated invalid config: {errors}"


class TestYAMLConfigs:
    """Test loading of YAML configuration files."""
    
    def test_load_base_lora_config(self):
        """Test loading base LoRA configuration."""
        config_path = Path("experiments/configs/base_lora_experiment.yaml")
        
        if config_path.exists():
            config = ExperimentConfig.load_yaml(config_path)
            
            assert config.name == "base_lora_cifar10"
            assert config.model.name == "deit_tiny_patch16_224"
            assert config.dataset.name == "cifar10"
            assert config.lora.rank == 8
            assert not config.use_adalora
            assert not config.use_qa_lora
            
            # Validate the loaded config
            is_valid, errors = ConfigValidator.validate_config(config)
            assert is_valid, f"Base config validation failed: {errors}"
    
    def test_load_quantized_config(self):
        """Test loading quantized LoRA configuration."""
        config_path = Path("experiments/configs/quantized_lora_experiment.yaml")
        
        if config_path.exists():
            config = ExperimentConfig.load_yaml(config_path)
            
            assert config.name == "quantized_lora_cifar10"
            assert config.quantization.bits == 8
            assert config.lora.rank == 8
            
            # Validate the loaded config
            is_valid, errors = ConfigValidator.validate_config(config)
            assert is_valid, f"Quantized config validation failed: {errors}"
    
    def test_load_adalora_config(self):
        """Test loading AdaLoRA configuration."""
        config_path = Path("experiments/configs/adalora_experiment.yaml")
        
        if config_path.exists():
            config = ExperimentConfig.load_yaml(config_path)
            
            assert config.name == "adalora_cifar100"
            assert config.use_adalora is True
            assert config.dataset.name == "cifar100"
            assert config.lora.rank == 16
            
            # Validate the loaded config
            is_valid, errors = ConfigValidator.validate_config(config)
            assert is_valid, f"AdaLoRA config validation failed: {errors}"


if __name__ == "__main__":
    pytest.main([__file__])