"""
Tests for QuantizationManager class.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.models.quantization_manager import QuantizationManager
from src.models.model_info import QuantizationConfig


class TestQuantizationManager:
    """Test cases for QuantizationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = QuantizationManager()
    
    def test_init(self):
        """Test manager initialization."""
        assert isinstance(self.manager._quantized_models, dict)
        assert len(self.manager._quantized_models) == 0
    
    def test_create_bnb_config_4bit(self):
        """Test creating 4-bit BitsAndBytesConfig."""
        config = QuantizationConfig(bits=4, compute_dtype="float16", quant_type="nf4")
        
        # This will fail without transformers, which is expected
        with pytest.raises(RuntimeError, match="transformers library not available"):
            self.manager.create_bnb_config(config)
    
    def test_create_bnb_config_8bit(self):
        """Test creating 8-bit BitsAndBytesConfig."""
        config = QuantizationConfig(bits=8, compute_dtype="float16")
        
        # This will fail without transformers, which is expected
        with pytest.raises(RuntimeError, match="transformers library not available"):
            self.manager.create_bnb_config(config)
    
    def test_quantize_model_without_dependencies(self):
        """Test quantization when dependencies are missing."""
        mock_model = MagicMock()
        config = QuantizationConfig(bits=8)
        
        # Should fail without bitsandbytes
        with pytest.raises(RuntimeError, match="bitsandbytes library not available"):
            self.manager.quantize_model(mock_model, config)
    
    def test_measure_memory_usage_none_model(self):
        """Test memory measurement with None model."""
        memory = self.manager.measure_memory_usage(None)
        assert memory == 0.0
    
    def test_measure_memory_usage_mock_model(self):
        """Test memory measurement with mock model."""
        # Create a mock model with parameters
        mock_model = MagicMock()
        
        # Mock parameters
        param1 = MagicMock()
        param1.nelement.return_value = 1000
        param1.element_size.return_value = 4  # 4 bytes per element
        
        param2 = MagicMock()
        param2.nelement.return_value = 500
        param2.element_size.return_value = 4
        
        mock_model.parameters.return_value = [param1, param2]
        mock_model.buffers.return_value = []  # No buffers
        
        memory_mb = self.manager.measure_memory_usage(mock_model)
        
        # Expected: (1000 + 500) * 4 bytes = 6000 bytes = 6000 / (1024*1024) MB
        expected_mb = 6000 / (1024 * 1024)
        assert abs(memory_mb - expected_mb) < 0.001
    
    def test_measure_system_memory(self):
        """Test system memory measurement."""
        memory_info = self.manager.measure_system_memory()
        
        assert isinstance(memory_info, dict)
        assert "total_mb" in memory_info
        assert "available_mb" in memory_info
        assert "used_mb" in memory_info
        assert "percent_used" in memory_info
        
        # Values should be non-negative
        assert memory_info["total_mb"] >= 0
        assert memory_info["available_mb"] >= 0
        assert memory_info["used_mb"] >= 0
        assert 0 <= memory_info["percent_used"] <= 100
    
    def test_estimate_quantization_savings_8bit(self):
        """Test quantization savings estimation for 8-bit."""
        # Create mock model
        mock_model = MagicMock()
        
        # Mock parameters for memory calculation
        param = MagicMock()
        param.nelement.return_value = 1000000  # 1M parameters
        param.element_size.return_value = 4    # 4 bytes each
        
        mock_model.parameters.return_value = [param]
        mock_model.buffers.return_value = []
        
        # Mock named_modules to return linear layers
        linear_module = MagicMock()
        linear_module.__class__.__name__ = "Linear"
        mock_model.named_modules.return_value = [
            ("layer1", linear_module),
            ("layer2", linear_module)
        ]
        
        config = QuantizationConfig(bits=8)
        
        # Mock isinstance to return True for Linear layers
        with patch('builtins.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            
            savings = self.manager.estimate_quantization_savings(mock_model, config)
        
        assert isinstance(savings, dict)
        assert "current_memory_mb" in savings
        assert "estimated_memory_after_mb" in savings
        assert "estimated_savings_mb" in savings
        assert "estimated_savings_percent" in savings
        assert "linear_layers_count" in savings
        
        # 8-bit should give ~50% reduction
        assert savings["estimated_savings_percent"] == 50.0
        assert savings["linear_layers_count"] == 2
    
    def test_estimate_quantization_savings_4bit(self):
        """Test quantization savings estimation for 4-bit."""
        mock_model = MagicMock()
        
        # Mock parameters
        param = MagicMock()
        param.nelement.return_value = 1000000
        param.element_size.return_value = 4
        
        mock_model.parameters.return_value = [param]
        mock_model.buffers.return_value = []
        mock_model.named_modules.return_value = [("layer1", MagicMock())]
        
        config = QuantizationConfig(bits=4)
        
        with patch('builtins.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            
            savings = self.manager.estimate_quantization_savings(mock_model, config)
        
        # 4-bit should give ~75% reduction
        assert savings["estimated_savings_percent"] == 75.0
    
    def test_verify_quantization_without_dependencies(self):
        """Test quantization verification without dependencies."""
        original_model = MagicMock()
        quantized_model = MagicMock()
        config = QuantizationConfig(bits=8)
        
        # Mock named_modules to return some modules
        original_model.named_modules.return_value = [("layer1", MagicMock())]
        quantized_model.named_modules.return_value = [("layer1", MagicMock())]
        
        with patch('builtins.isinstance') as mock_isinstance:
            # Mock isinstance to return True for Linear layers in original
            mock_isinstance.side_effect = lambda obj, cls: cls.__name__ == "Linear"
            
            result = self.manager.verify_quantization(original_model, quantized_model, config)
        
        assert isinstance(result, dict)
        assert "quantization_applied" in result
        assert "quantized_layers" in result
        assert "total_layers" in result
        
        # Should show no quantization applied (no bitsandbytes layers)
        assert result["quantization_applied"] is False
        assert result["quantized_layers"] == 0
    
    def test_get_quantization_info_not_found(self):
        """Test getting info for non-existent model."""
        info = self.manager.get_quantization_info("nonexistent")
        assert info is None
    
    def test_list_quantized_models_empty(self):
        """Test listing quantized models when none exist."""
        models = self.manager.list_quantized_models()
        assert isinstance(models, dict)
        assert len(models) == 0
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add a mock quantized model
        self.manager._quantized_models["test"] = {"model": MagicMock()}
        assert len(self.manager._quantized_models) == 1
        
        # Clear cache
        self.manager.clear_cache()
        assert len(self.manager._quantized_models) == 0


class TestQuantizationConfig:
    """Test cases for QuantizationConfig validation."""
    
    def test_valid_8bit_config(self):
        """Test valid 8-bit configuration."""
        config = QuantizationConfig(bits=8, compute_dtype="float16")
        
        assert config.bits == 8
        assert config.compute_dtype == "float16"
        assert config.quant_type == "nf4"  # default
        assert config.double_quant is True  # default
    
    def test_valid_4bit_config(self):
        """Test valid 4-bit configuration."""
        config = QuantizationConfig(
            bits=4,
            compute_dtype="bfloat16",
            quant_type="fp4",
            double_quant=False
        )
        
        assert config.bits == 4
        assert config.compute_dtype == "bfloat16"
        assert config.quant_type == "fp4"
        assert config.double_quant is False
    
    def test_invalid_bits(self):
        """Test invalid bit configuration."""
        with pytest.raises(ValueError, match="Only 4-bit and 8-bit quantization supported"):
            QuantizationConfig(bits=16)
    
    def test_invalid_compute_dtype(self):
        """Test invalid compute dtype."""
        with pytest.raises(ValueError, match="Unsupported compute dtype"):
            QuantizationConfig(compute_dtype="float64")
    
    def test_invalid_quant_type(self):
        """Test invalid quantization type."""
        with pytest.raises(ValueError, match="Quantization type must be"):
            QuantizationConfig(quant_type="invalid")