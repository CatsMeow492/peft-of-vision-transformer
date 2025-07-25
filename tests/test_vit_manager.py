"""
Tests for ViTModelManager class.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from src.models.vit_manager import ViTModelManager
from src.models.model_info import QuantizationConfig, ModelInfo


class TestViTModelManager:
    """Test cases for ViTModelManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ViTModelManager()
    
    def test_init(self):
        """Test manager initialization."""
        assert isinstance(self.manager._loaded_models, dict)
        assert len(self.manager._loaded_models) == 0
    
    def test_supported_models(self):
        """Test that supported models are properly defined."""
        models = self.manager.list_supported_models()
        
        # Check that key models are supported
        assert "deit_tiny_patch16_224" in models
        assert "deit_small_patch16_224" in models
        assert "vit_small_patch16_224" in models
        
        # Check model configuration structure
        for model_name, config in models.items():
            assert "source" in config
            assert "params" in config
            assert "input_size" in config
            assert "description" in config
            assert config["source"] in ["timm", "huggingface"]
    
    def test_unsupported_model(self):
        """Test loading unsupported model raises error."""
        with pytest.raises(ValueError, match="not supported"):
            self.manager.load_model("unsupported_model")
    
    @patch('timm.create_model')
    def test_load_timm_model(self, mock_create_model):
        """Test loading timm model."""
        # Mock timm model
        mock_model = MagicMock(spec=nn.Module)
        mock_model.__class__.__name__ = "VisionTransformer"
        mock_create_model.return_value = mock_model
        
        # Load model
        model = self.manager.load_model("deit_tiny_patch16_224", num_classes=10)
        
        # Verify timm.create_model was called correctly
        mock_create_model.assert_called_once_with(
            "deit_tiny_patch16_224",
            pretrained=True,
            num_classes=10
        )
        
        assert model == mock_model
    
    @patch('src.models.vit_manager.AutoModel')
    @patch('src.models.vit_manager.AutoConfig')
    def test_load_huggingface_model(self, mock_config, mock_model):
        """Test loading HuggingFace model."""
        # Mock HuggingFace components
        mock_config_instance = MagicMock()
        mock_config.from_pretrained.return_value = mock_config_instance
        
        mock_model_instance = MagicMock(spec=nn.Module)
        mock_model_instance.__class__.__name__ = "ViTModel"
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Load model
        model = self.manager.load_model("google/vit-base-patch16-224", num_classes=10)
        
        # Verify calls
        mock_config.from_pretrained.assert_called_once_with("google/vit-base-patch16-224")
        assert mock_config_instance.num_labels == 10
        
        mock_model.from_pretrained.assert_called_once()
        assert model == mock_model_instance
    
    def test_quantization_config_validation(self):
        """Test quantization configuration validation."""
        # Valid config
        config = QuantizationConfig(bits=8, compute_dtype="float16")
        assert config.bits == 8
        assert config.compute_dtype == "float16"
        
        # Invalid bits
        with pytest.raises(ValueError, match="Only 4-bit and 8-bit"):
            QuantizationConfig(bits=16)
        
        # Invalid dtype
        with pytest.raises(ValueError, match="Unsupported compute dtype"):
            QuantizationConfig(compute_dtype="float64")
        
        # Invalid quant type
        with pytest.raises(ValueError, match="Quantization type must be"):
            QuantizationConfig(quant_type="invalid")
    
    def test_model_caching(self):
        """Test that models are cached properly."""
        with patch('timm.create_model') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_create.return_value = mock_model
            
            # Load model twice
            model1 = self.manager.load_model("deit_tiny_patch16_224")
            model2 = self.manager.load_model("deit_tiny_patch16_224")
            
            # Should only call timm.create_model once due to caching
            assert mock_create.call_count == 1
            assert model1 == model2
    
    def test_clear_cache(self):
        """Test cache clearing."""
        with patch('timm.create_model') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_create.return_value = mock_model
            
            # Load model and verify it's cached
            self.manager.load_model("deit_tiny_patch16_224")
            assert len(self.manager._loaded_models) == 1
            
            # Clear cache
            self.manager.clear_cache()
            assert len(self.manager._loaded_models) == 0
    
    def test_get_model_info(self):
        """Test model info extraction."""
        # Create a simple mock model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Add some attributes that ViT models typically have
        model.head = nn.Linear(50, 10)
        
        info = self.manager.get_model_info(model, "test_model")
        
        assert isinstance(info, ModelInfo)
        assert info.name == "test_model"
        assert info.total_params > 0
        assert info.trainable_params > 0
        assert 0 <= info.trainable_ratio <= 1
        assert info.model_size_mb > 0
        assert info.architecture == "Sequential"
    
    def test_validate_model_success(self):
        """Test successful model validation."""
        # Create a simple model that should pass validation
        model = nn.Sequential(
            nn.Linear(3 * 224 * 224, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        # Mock the forward pass to avoid memory issues
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = torch.randn(1, 10)
            
            result = self.manager.validate_model(model, "test_model")
            assert result is True
    
    def test_validate_model_failure(self):
        """Test model validation failure."""
        # Create a model that will fail forward pass
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # This should fail because input size doesn't match
        result = self.manager.validate_model(model, "test_model")
        assert result is False


class TestQuantizationConfig:
    """Test cases for QuantizationConfig."""
    
    def test_valid_config(self):
        """Test valid quantization configuration."""
        config = QuantizationConfig(
            bits=8,
            compute_dtype="float16",
            quant_type="nf4",
            double_quant=True
        )
        
        assert config.bits == 8
        assert config.compute_dtype == "float16"
        assert config.quant_type == "nf4"
        assert config.double_quant is True
    
    def test_default_values(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        
        assert config.bits == 8
        assert config.compute_dtype == "float16"
        assert config.quant_type == "nf4"
        assert config.double_quant is True
    
    def test_invalid_bits(self):
        """Test invalid bits configuration."""
        with pytest.raises(ValueError):
            QuantizationConfig(bits=16)
    
    def test_invalid_dtype(self):
        """Test invalid compute dtype."""
        with pytest.raises(ValueError):
            QuantizationConfig(compute_dtype="invalid")
    
    def test_invalid_quant_type(self):
        """Test invalid quantization type."""
        with pytest.raises(ValueError):
            QuantizationConfig(quant_type="invalid")