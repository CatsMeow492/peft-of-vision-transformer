#!/usr/bin/env python3
"""
Simple test script for model export functionality.
"""

import torch
import torch.nn as nn
import tempfile
from pathlib import Path

# Import our modules
from src.models.model_export import ModelExporter, ExportConfig
from src.models.model_info import ModelInfo


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


def test_basic_export():
    """Test basic model export functionality."""
    print("Testing basic model export...")
    
    # Create model and exporter
    model = SimpleModel()
    exporter = ModelExporter()
    
    # Test PyTorch export
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExportConfig(
            export_format="pytorch",
            output_path=str(Path(tmpdir) / "test_model")
        )
        
        model_info = ModelInfo(
            name="simple_model",
            total_params=sum(p.numel() for p in model.parameters()),
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainable_ratio=1.0,
            model_size_mb=1.0,
            architecture="SimpleModel",
            input_size=(100,),
            num_classes=10,
            attention_layers=[]
        )
        
        result = exporter.export_model(model, config, model_info)
        
        print(f"Export result: {result}")
        assert result["success"] is True
        assert len(result["files_created"]) == 2
        
        # Check files exist
        for file_path in result["files_created"]:
            assert Path(file_path).exists(), f"File {file_path} does not exist"
        
        print("✓ PyTorch export test passed")


def test_export_config():
    """Test export configuration."""
    print("Testing export configuration...")
    
    # Test default config
    config = ExportConfig()
    assert config.export_format == "pytorch"
    assert config.precision == "float32"
    print("✓ Default config test passed")
    
    # Test custom config
    config = ExportConfig(
        export_format="onnx",
        precision="float16",
        optimize_for_inference=False
    )
    assert config.export_format == "onnx"
    assert config.precision == "float16"
    assert config.optimize_for_inference is False
    print("✓ Custom config test passed")
    
    # Test invalid config
    try:
        ExportConfig(export_format="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Invalid config validation test passed")


def test_quantization_detection():
    """Test quantization detection."""
    print("Testing quantization detection...")
    
    model = SimpleModel()
    exporter = ModelExporter()
    
    # Test with regular model (no quantization)
    has_quant = exporter._detect_quantization(model)
    assert has_quant is False
    print("✓ No quantization detection test passed")
    
    # Test with mock quantized layer
    class MockQuantizedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.__class__.__name__ = "bnb.nn.Linear8bitLt"
    
    model.quantized_layer = MockQuantizedLayer()
    has_quant = exporter._detect_quantization(model)
    assert has_quant is True
    print("✓ Quantization detection test passed")


if __name__ == "__main__":
    print("Running model export tests...")
    
    try:
        test_export_config()
        test_basic_export()
        test_quantization_detection()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()