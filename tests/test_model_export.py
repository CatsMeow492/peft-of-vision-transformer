"""
Tests for model export and adapter merging functionality.
"""

import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.models.model_export import (
    ModelExporter, ExportConfig, MergeValidationResult
)
from src.models.model_info import ModelInfo


class SimpleViTModel(nn.Module):
    """Simple Vision Transformer model for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Linear(768, 384)
        self.attention = nn.MultiheadAttention(384, 6, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(384, 1536),
            nn.GELU(),
            nn.Linear(1536, 384)
        )
        self.head = nn.Linear(384, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        # Simplified forward pass
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten
        x = self.patch_embed(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        # MLP
        mlp_out = self.mlp(x)
        x = x + mlp_out
        
        # Classification head
        x = x.squeeze(1)
        return self.head(x)


class MockPeftModel:
    """Mock PEFT model for testing."""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self._modules = base_model._modules
        self.training = base_model.training
    
    def get_base_model(self):
        return self.base_model
    
    def merge_and_unload(self, safe_merge=True):
        # Return a copy of the base model (simulating merge)
        return self.base_model
    
    def eval(self):
        self.training = False
        self.base_model.eval()
        return self
    
    def parameters(self):
        return self.base_model.parameters()
    
    def state_dict(self):
        return self.base_model.state_dict()
    
    def __call__(self, x):
        return self.base_model(x)


class TestExportConfig:
    """Test ExportConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExportConfig()
        assert config.export_format == "pytorch"
        assert config.output_path == "exported_model"
        assert config.merge_adapters is True
        assert config.validate_merged is True
        assert config.precision == "float32"
        assert config.optimize_for_inference is True
        assert config.include_metadata is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExportConfig(
            export_format="onnx",
            output_path="/tmp/model",
            precision="float16",
            optimize_for_inference=False
        )
        assert config.export_format == "onnx"
        assert config.output_path == "/tmp/model"
        assert config.precision == "float16"
        assert config.optimize_for_inference is False
    
    def test_invalid_format(self):
        """Test validation of export format."""
        with pytest.raises(ValueError, match="Export format must be one of"):
            ExportConfig(export_format="invalid")
    
    def test_invalid_precision(self):
        """Test validation of precision."""
        with pytest.raises(ValueError, match="Precision must be one of"):
            ExportConfig(precision="invalid")


class TestModelExporter:
    """Test ModelExporter class."""
    
    @pytest.fixture
    def exporter(self):
        """Create ModelExporter instance."""
        return ModelExporter()
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleViTModel(num_classes=10)
    
    @pytest.fixture
    def mock_peft_model(self, simple_model):
        """Create mock PEFT model."""
        return MockPeftModel(simple_model)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_merge_adapters_success(self, exporter, mock_peft_model):
        """Test successful adapter merging."""
        with patch('src.models.model_export.PeftModel', MockPeftModel):
            merged = exporter.merge_adapters(mock_peft_model, "test_model")
            
            assert merged is not None
            assert "test_model" in exporter._exported_models
            assert exporter._exported_models["test_model"]["merge_method"] == "safe"
    
    def test_merge_adapters_invalid_model(self, exporter, simple_model):
        """Test merging with invalid model type."""
        with pytest.raises(ValueError, match="Model must be a PeftModel instance"):
            exporter.merge_adapters(simple_model)
    
    def test_merge_adapters_peft_unavailable(self, exporter, mock_peft_model):
        """Test merging when PEFT is unavailable."""
        with patch('src.models.model_export.PeftModel', None):
            with pytest.raises(RuntimeError, match="PEFT library not available"):
                exporter.merge_adapters(mock_peft_model)
    
    def test_validate_merged_model_success(self, exporter, simple_model, mock_peft_model):
        """Test successful model validation."""
        result = exporter.validate_merged_model(simple_model, mock_peft_model)
        
        assert isinstance(result, MergeValidationResult)
        assert result.validation_passed is True
        assert result.forward_pass_check is True
        assert result.parameter_count_check is True
        assert result.max_weight_difference < 1e-5
    
    def test_validate_merged_model_forward_pass_difference(self, exporter, mock_peft_model):
        """Test validation with forward pass differences."""
        # Create two different models
        model1 = SimpleViTModel(num_classes=10)
        model2 = SimpleViTModel(num_classes=10)
        
        # Initialize with different weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.data = p1.data + 1.0  # Add significant difference
        
        result = exporter.validate_merged_model(model1, mock_peft_model)
        
        # Should fail due to large differences
        assert result.validation_passed is False
        assert len(result.errors) > 0
    
    def test_export_pytorch_format(self, exporter, simple_model, temp_dir):
        """Test PyTorch format export."""
        config = ExportConfig(
            export_format="pytorch",
            output_path=str(Path(temp_dir) / "test_model")
        )
        
        model_info = ModelInfo(
            name="test_model",
            total_params=1000,
            trainable_params=100,
            trainable_ratio=0.1,
            model_size_mb=1.0,
            architecture="SimpleViTModel",
            input_size=(224, 224),
            num_classes=10,
            attention_layers=["attention"]
        )
        
        result = exporter.export_model(simple_model, config, model_info)
        
        assert result["success"] is True
        assert len(result["files_created"]) == 2
        assert Path(result["files_created"][0]).exists()
        assert Path(result["files_created"][1]).exists()
    
    @patch('torch.onnx.export')
    @patch('onnx.load')
    @patch('onnx.checker.check_model')
    def test_export_onnx_format(self, mock_check, mock_load, mock_export, exporter, simple_model, temp_dir):
        """Test ONNX format export."""
        config = ExportConfig(
            export_format="onnx",
            output_path=str(Path(temp_dir) / "test_model")
        )
        
        # Mock ONNX operations
        mock_export.return_value = None
        mock_load.return_value = Mock()
        mock_check.return_value = None
        
        with patch('os.path.getsize', return_value=1024*1024):  # 1MB
            result = exporter.export_model(simple_model, config)
        
        assert result["success"] is True
        assert len(result["files_created"]) == 1
        assert result["model_size_mb"] == 1.0
        mock_export.assert_called_once()
    
    def test_export_onnx_unavailable(self, exporter, simple_model, temp_dir):
        """Test ONNX export when library is unavailable."""
        config = ExportConfig(
            export_format="onnx",
            output_path=str(Path(temp_dir) / "test_model")
        )
        
        with patch.dict('sys.modules', {'onnx': None}):
            with pytest.raises(RuntimeError, match="ONNX library not available"):
                exporter.export_model(simple_model, config)
    
    @patch('torch.jit.trace')
    def test_export_torchscript_trace(self, mock_trace, exporter, simple_model, temp_dir):
        """Test TorchScript export with tracing."""
        config = ExportConfig(
            export_format="torchscript",
            output_path=str(Path(temp_dir) / "test_model")
        )
        
        # Mock successful tracing
        mock_traced = Mock()
        mock_traced.save = Mock()
        mock_trace.return_value = mock_traced
        
        with patch('os.path.getsize', return_value=1024*1024):
            result = exporter.export_model(simple_model, config)
        
        assert result["success"] is True
        assert result["torchscript_method"] == "trace"
        mock_trace.assert_called_once()
        mock_traced.save.assert_called_once()
    
    @patch('torch.jit.trace')
    @patch('torch.jit.script')
    def test_export_torchscript_script_fallback(self, mock_script, mock_trace, exporter, simple_model, temp_dir):
        """Test TorchScript export with scripting fallback."""
        config = ExportConfig(
            export_format="torchscript",
            output_path=str(Path(temp_dir) / "test_model")
        )
        
        # Mock tracing failure, scripting success
        mock_trace.side_effect = Exception("Tracing failed")
        mock_scripted = Mock()
        mock_scripted.save = Mock()
        mock_script.return_value = mock_scripted
        
        with patch('os.path.getsize', return_value=1024*1024):
            result = exporter.export_model(simple_model, config)
        
        assert result["success"] is True
        assert result["torchscript_method"] == "script"
        mock_script.assert_called_once()
        mock_scripted.save.assert_called_once()
    
    def test_export_unsupported_format(self, exporter, simple_model):
        """Test export with unsupported format."""
        config = ExportConfig(export_format="pytorch")  # Valid format
        config.export_format = "unsupported"  # Change after validation
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            exporter.export_model(simple_model, config)
    
    def test_precision_conversion(self, exporter, simple_model, temp_dir):
        """Test model precision conversion during export."""
        config = ExportConfig(
            export_format="pytorch",
            output_path=str(Path(temp_dir) / "test_model"),
            precision="float16"
        )
        
        original_dtype = next(simple_model.parameters()).dtype
        result = exporter.export_model(simple_model, config)
        
        assert result["success"] is True
        # Model should be converted to half precision
        # Note: This is a simplified test - in practice, we'd check the exported model
    
    def test_detect_quantization(self, exporter, simple_model):
        """Test quantization detection."""
        # Test with regular model (no quantization)
        assert exporter._detect_quantization(simple_model) is False
        
        # Test with mock quantized layer
        class MockQuantizedLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.__class__.__name__ = "bnb.nn.Linear8bitLt"
        
        simple_model.quantized_layer = MockQuantizedLayer()
        assert exporter._detect_quantization(simple_model) is True
    
    def test_preserve_quantization_during_export(self, exporter, simple_model, temp_dir):
        """Test quantization preservation during export."""
        config = ExportConfig(
            export_format="pytorch",
            output_path=str(Path(temp_dir) / "quantized_model")
        )
        
        # Add mock quantized layer
        class MockQuantizedLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.__class__.__name__ = "bnb.nn.Linear8bitLt"
                self.weight = nn.Parameter(torch.randn(10, 10))
        
        simple_model.quantized_layer = MockQuantizedLayer()
        
        result = exporter.preserve_quantization_during_export(simple_model, config)
        
        assert result["success"] is True
        assert result["quantization_preserved"] is True
        assert "quantization_info" in result
    
    def test_preserve_quantization_non_pytorch_format(self, exporter, simple_model, temp_dir):
        """Test quantization preservation warning for non-PyTorch formats."""
        config = ExportConfig(
            export_format="onnx",
            output_path=str(Path(temp_dir) / "quantized_model")
        )
        
        # Add mock quantized layer
        class MockQuantizedLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.__class__.__name__ = "bnb.nn.Linear8bitLt"
        
        simple_model.quantized_layer = MockQuantizedLayer()
        
        with patch('torch.onnx.export'), patch('onnx.load'), patch('onnx.checker.check_model'):
            with patch('os.path.getsize', return_value=1024*1024):
                result = exporter.preserve_quantization_during_export(simple_model, config)
        
        assert result["quantization_preserved"] is False
        assert "quantization_warning" in result
    
    def test_collect_quantization_info(self, exporter, simple_model):
        """Test quantization information collection."""
        # Add mock quantized layers
        class MockQuantizedLayer(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.__class__.__name__ = f"bnb.nn.{name}"
                self.weight = nn.Parameter(torch.randn(10, 10))
        
        simple_model.quant1 = MockQuantizedLayer("Linear8bitLt")
        simple_model.quant2 = MockQuantizedLayer("Linear4bit")
        
        info = exporter._collect_quantization_info(simple_model)
        
        assert len(info["quantized_layers"]) == 2
        assert len(info["quantization_types"]) == 2
        assert info["total_quantized_params"] > 0
    
    def test_get_export_summary(self, exporter, mock_peft_model):
        """Test getting export summary."""
        with patch('src.models.model_export.PeftModel', MockPeftModel):
            exporter.merge_adapters(mock_peft_model, "test_model")
            
            summary = exporter.get_export_summary("test_model")
            assert summary is not None
            assert "merged_model" in summary
            
            # Test non-existent model
            assert exporter.get_export_summary("non_existent") is None
    
    def test_list_exported_models(self, exporter, mock_peft_model):
        """Test listing exported models."""
        with patch('src.models.model_export.PeftModel', MockPeftModel):
            exporter.merge_adapters(mock_peft_model, "model1")
            exporter.merge_adapters(mock_peft_model, "model2")
            
            models = exporter.list_exported_models()
            assert len(models) == 2
            assert "model1" in models
            assert "model2" in models
    
    def test_clear_cache(self, exporter, mock_peft_model):
        """Test clearing export cache."""
        with patch('src.models.model_export.PeftModel', MockPeftModel):
            exporter.merge_adapters(mock_peft_model, "test_model")
            assert len(exporter._exported_models) == 1
            
            exporter.clear_cache()
            assert len(exporter._exported_models) == 0


class TestMergeValidationResult:
    """Test MergeValidationResult dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        result = MergeValidationResult(
            validation_passed=True,
            numerical_precision_check=True,
            forward_pass_check=True,
            parameter_count_check=True,
            size_comparison={},
            max_weight_difference=0.0,
            mean_weight_difference=0.0
        )
        
        assert result.validation_passed is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_with_errors_and_warnings(self):
        """Test with errors and warnings."""
        result = MergeValidationResult(
            validation_passed=False,
            numerical_precision_check=False,
            forward_pass_check=False,
            parameter_count_check=False,
            size_comparison={},
            max_weight_difference=1.0,
            mean_weight_difference=0.5,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert result.validation_passed is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__])