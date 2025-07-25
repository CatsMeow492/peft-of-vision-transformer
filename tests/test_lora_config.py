"""
Tests for LoRA configuration and adapter functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.models.lora_config import LoRAConfig, LoRAAdapter


class TestLoRAConfig:
    """Test cases for LoRAConfig."""
    
    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.bias == "none"
        assert config.task_type == "FEATURE_EXTRACTION"
        assert config.inference_mode is False
        assert config.use_rslora is False
        assert config.use_dora is False
        
        # Check default target modules
        assert config.target_modules is not None
        assert "qkv" in config.target_modules
        assert "query" in config.target_modules
    
    def test_custom_config(self):
        """Test custom LoRA configuration."""
        target_modules = ["attention.query", "attention.key"]
        
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.2,
            target_modules=target_modules,
            bias="all",
            use_rslora=True
        )
        
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.2
        assert config.target_modules == target_modules
        assert config.bias == "all"
        assert config.use_rslora is True
    
    def test_validation_errors(self):
        """Test LoRA configuration validation."""
        # Invalid rank
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            LoRAConfig(rank=0)
        
        # Invalid alpha
        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            LoRAConfig(alpha=0)
        
        # Invalid dropout
        with pytest.raises(ValueError, match="LoRA dropout must be between 0 and 1"):
            LoRAConfig(dropout=1.5)
        
        # Invalid bias
        with pytest.raises(ValueError, match="LoRA bias must be"):
            LoRAConfig(bias="invalid")
    
    def test_trainable_params_ratio(self):
        """Test trainable parameters ratio estimation."""
        config = LoRAConfig(rank=8, target_modules=["qkv", "proj"])
        
        # Test with typical ViT parameter count
        total_params = 5_000_000  # 5M parameters (DeiT-tiny)
        ratio = config.get_trainable_params_ratio(total_params)
        
        assert 0 < ratio < 1
        assert ratio < 0.1  # LoRA should be much smaller than base model
    
    def test_to_peft_config_without_peft(self):
        """Test conversion to PEFT config when library is not available."""
        config = LoRAConfig()
        
        # This should raise an error since PEFT is not available in test environment
        with pytest.raises(RuntimeError, match="PEFT library not available"):
            config.to_peft_config()


class TestLoRAAdapter:
    """Test cases for LoRAAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = LoRAAdapter()
    
    def test_init(self):
        """Test adapter initialization."""
        assert isinstance(self.adapter._applied_models, dict)
        assert len(self.adapter._applied_models) == 0
    
    def test_detect_target_modules(self):
        """Test target module detection."""
        # Create a mock model with named modules
        mock_model = MagicMock()
        mock_modules = [
            ("encoder.layer.0.attention.query", MagicMock()),
            ("encoder.layer.0.attention.key", MagicMock()),
            ("encoder.layer.0.attention.value", MagicMock()),
            ("encoder.layer.0.mlp.fc1", MagicMock()),
            ("encoder.layer.1.attention.qkv", MagicMock()),
        ]
        
        # Add weight attribute to modules
        for name, module in mock_modules:
            module.weight = MagicMock()
            module.weight.shape = (768, 768)  # 2D weight for linear layer
        
        mock_model.named_modules.return_value = mock_modules
        
        target_patterns = ["query", "key", "value", "qkv", "fc1"]
        detected = self.adapter._detect_target_modules(mock_model, target_patterns)
        
        expected = [
            "encoder.layer.0.attention.key",
            "encoder.layer.0.attention.query", 
            "encoder.layer.0.attention.value",
            "encoder.layer.0.mlp.fc1",
            "encoder.layer.1.attention.qkv"
        ]
        
        assert sorted(detected) == sorted(expected)
    
    def test_apply_lora_without_peft(self):
        """Test LoRA application when PEFT is not available."""
        mock_model = MagicMock()
        config = LoRAConfig()
        
        # This should raise an error since PEFT is not available
        with pytest.raises(RuntimeError, match="PEFT library not available"):
            self.adapter.apply_lora(mock_model, config)
    
    def test_get_parameter_ratio(self):
        """Test parameter ratio calculation."""
        # Create a mock model with parameters
        mock_model = MagicMock()
        
        # Mock parameters - some trainable, some not
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.requires_grad = True
        
        param2 = MagicMock()
        param2.numel.return_value = 500
        param2.requires_grad = False
        
        param3 = MagicMock()
        param3.numel.return_value = 200
        param3.requires_grad = True
        
        mock_model.parameters.return_value = [param1, param2, param3]
        
        ratio = self.adapter.get_parameter_ratio(mock_model)
        
        # Expected: (1000 + 200) / (1000 + 500 + 200) = 1200 / 1700 ≈ 0.706
        expected_ratio = 1200 / 1700
        assert abs(ratio - expected_ratio) < 0.001
    
    def test_validate_adapter(self):
        """Test adapter validation."""
        # Create a mock model
        mock_model = MagicMock()
        
        # Mock parameters
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.requires_grad = True
        
        param2 = MagicMock()
        param2.numel.return_value = 4000
        param2.requires_grad = False
        
        mock_model.parameters.return_value = [param1, param2]
        
        # Mock named modules with some LoRA modules
        mock_modules = [
            ("base.layer1", MagicMock()),
            ("lora_A.layer1", MagicMock()),
            ("lora_B.layer1", MagicMock()),
        ]
        mock_model.named_modules.return_value = mock_modules
        
        config = LoRAConfig(rank=8, target_modules=["layer1"])
        
        stats = self.adapter.validate_adapter(mock_model, config)
        
        assert stats["validation_passed"] is True
        assert stats["total_parameters"] == 5000
        assert stats["trainable_parameters"] == 1000
        assert stats["trainable_ratio"] == 0.2
        assert stats["lora_modules_count"] == 2  # lora_A and lora_B
        assert stats["lora_rank"] == 8
    
    def test_list_applied_models(self):
        """Test listing applied models."""
        # Initially empty
        models = self.adapter.list_applied_models()
        assert len(models) == 0
        
        # Add a mock applied model
        mock_model = MagicMock()
        mock_config = LoRAConfig(rank=16)
        
        self.adapter._applied_models["test_model"] = {
            "model": mock_model,
            "config": mock_config,
            "base_model": MagicMock()
        }
        
        # Mock parameter ratio calculation
        with patch.object(self.adapter, 'get_parameter_ratio', return_value=0.1):
            models = self.adapter.list_applied_models()
        
        assert len(models) == 1
        assert "test_model" in models
        assert models["test_model"]["config"] == mock_config
        assert models["test_model"]["parameter_ratio"] == 0.1
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add a mock model to cache
        self.adapter._applied_models["test"] = {"model": MagicMock()}
        assert len(self.adapter._applied_models) == 1
        
        # Clear cache
        self.adapter.clear_cache()
        assert len(self.adapter._applied_models) == 0