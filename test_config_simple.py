#!/usr/bin/env python3
"""
Simple test script for export configuration without torch dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_export_config_import():
    """Test that we can import the export configuration."""
    print("Testing export configuration import...")
    
    try:
        from models.model_export import ExportConfig, MergeValidationResult
        print("✓ Import successful")
        
        # Test default config
        config = ExportConfig()
        assert config.export_format == "pytorch"
        assert config.precision == "float32"
        assert config.merge_adapters is True
        print("✓ Default config creation successful")
        
        # Test custom config
        config = ExportConfig(
            export_format="onnx",
            precision="float16",
            optimize_for_inference=False
        )
        assert config.export_format == "onnx"
        assert config.precision == "float16"
        assert config.optimize_for_inference is False
        print("✓ Custom config creation successful")
        
        # Test validation
        try:
            ExportConfig(export_format="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✓ Validation works: {e}")
        
        # Test MergeValidationResult
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
        print("✓ MergeValidationResult creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import or basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running basic configuration tests...")
    
    success = test_export_config_import()
    
    if success:
        print("\n✅ Basic configuration tests passed!")
    else:
        print("\n❌ Basic configuration tests failed!")
        sys.exit(1)