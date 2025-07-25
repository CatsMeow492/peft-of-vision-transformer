#!/usr/bin/env python3
"""
Simple test script for reproducibility functionality.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_reproducibility_imports():
    """Test that we can import reproducibility modules."""
    print("Testing reproducibility imports...")
    
    try:
        from utils.reproducibility import (
            ReproducibilityManager, EnvironmentSpec, DatasetChecksum, ReproductionTest
        )
        print("✓ Import successful")
        
        # Test basic class creation
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReproducibilityManager(project_root=tmpdir)
            assert manager.project_root == Path(tmpdir)
            assert manager.reproducibility_dir.exists()
            print("✓ ReproducibilityManager creation successful")
        
        # Test EnvironmentSpec
        spec = EnvironmentSpec(
            python_version="3.10.0",
            platform_info={"system": "Darwin"},
            hardware_info={"cpu_count": 8},
            package_versions={"torch": "2.1.2"}
        )
        assert spec.python_version == "3.10.0"
        print("✓ EnvironmentSpec creation successful")
        
        # Test DatasetChecksum
        checksum = DatasetChecksum(
            dataset_name="test_dataset",
            file_path="/path/to/dataset",
            checksum_value="abc123",
            file_size_bytes=1024
        )
        assert checksum.dataset_name == "test_dataset"
        assert checksum.checksum_type == "sha256"  # default
        print("✓ DatasetChecksum creation successful")
        
        # Test ReproductionTest
        test = ReproductionTest(
            test_name="test_model_accuracy",
            test_type="integration",
            expected_results={"accuracy": 0.85}
        )
        assert test.test_name == "test_model_accuracy"
        assert test.tolerance["default"] == 1e-5
        print("✓ ReproductionTest creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import or basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_capture():
    """Test environment capture functionality."""
    print("Testing environment capture...")
    
    try:
        from utils.reproducibility import ReproducibilityManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReproducibilityManager(project_root=tmpdir)
            
            # Test hardware info gathering (should work without external dependencies)
            hardware_info = manager._get_hardware_info()
            assert isinstance(hardware_info, dict)
            print(f"✓ Hardware info gathered: {len(hardware_info)} items")
            
            # Test package version gathering
            versions = manager._get_package_versions()
            assert isinstance(versions, dict)
            assert len(versions) > 0
            print(f"✓ Package versions gathered: {len(versions)} packages")
            
            # Test environment variables
            env_vars = manager._get_relevant_env_vars()
            assert isinstance(env_vars, dict)
            print(f"✓ Environment variables gathered: {len(env_vars)} variables")
            
        return True
        
    except Exception as e:
        print(f"❌ Environment capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checksum_computation():
    """Test checksum computation functionality."""
    print("Testing checksum computation...")
    
    try:
        from utils.reproducibility import ReproducibilityManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReproducibilityManager(project_root=tmpdir)
            
            # Create test file
            test_file = Path(tmpdir) / "test.txt"
            test_content = b"Hello, world!"
            test_file.write_bytes(test_content)
            
            # Test file checksum
            checksum, size = manager._compute_file_checksum(test_file, "sha256")
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA256 hex length
            assert size == len(test_content)
            print(f"✓ File checksum computed: {checksum[:16]}...")
            
            # Test directory checksum
            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            (test_dir / "file1.txt").write_bytes(b"content1")
            (test_dir / "file2.txt").write_bytes(b"content2")
            
            dir_checksum, dir_size = manager._compute_directory_checksum(test_dir, "sha256")
            assert isinstance(dir_checksum, str)
            assert len(dir_checksum) == 64
            assert dir_size == len(b"content1") + len(b"content2")
            print(f"✓ Directory checksum computed: {dir_checksum[:16]}...")
            
            # Test dataset checksum computation
            dataset_checksum = manager.compute_dataset_checksum(
                test_file, "test_dataset", "sha256"
            )
            assert dataset_checksum.dataset_name == "test_dataset"
            assert dataset_checksum.checksum_value == checksum
            print("✓ Dataset checksum computation successful")
            
        return True
        
    except Exception as e:
        print(f"❌ Checksum computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reproduction_test_creation():
    """Test reproduction test creation."""
    print("Testing reproduction test creation...")
    
    try:
        from utils.reproducibility import ReproducibilityManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReproducibilityManager(project_root=tmpdir)
            
            # Create reproduction test
            test = manager.create_reproduction_test(
                test_name="test_accuracy",
                test_type="integration",
                expected_results={"accuracy": 0.85, "loss": 0.15},
                test_command="python test_model.py",
                tolerance={"accuracy": 0.01}
            )
            
            assert test.test_name == "test_accuracy"
            assert test.expected_results["accuracy"] == 0.85
            assert test.tolerance["accuracy"] == 0.01
            assert "test_accuracy" in manager._reproduction_tests
            print("✓ Reproduction test creation successful")
            
            # Test result validation
            actual_results = {
                "accuracy": 0.84,  # Within tolerance
                "loss": 0.16       # Within default tolerance
            }
            expected_results = {
                "accuracy": 0.85,
                "loss": 0.15
            }
            tolerance = {"accuracy": 0.02, "default": 0.02}
            
            validation = manager._validate_test_results(
                actual_results, expected_results, tolerance
            )
            
            assert validation["passed"] is True
            assert len(validation["errors"]) == 0
            print("✓ Test result validation successful")
            
        return True
        
    except Exception as e:
        print(f"❌ Reproduction test creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation():
    """Test report generation functionality."""
    print("Testing report generation...")
    
    try:
        from utils.reproducibility import ReproducibilityManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ReproducibilityManager(project_root=tmpdir)
            
            # Add some test data
            test_file = Path(tmpdir) / "dataset.txt"
            test_file.write_bytes(b"test data")
            manager.compute_dataset_checksum(test_file, "test_dataset")
            
            manager.create_reproduction_test(
                "test_model", "unit", {"accuracy": 0.85}
            )
            
            # Generate report
            report = manager.generate_reproducibility_report()
            
            assert isinstance(report, dict)
            assert "report_timestamp" in report
            assert "dataset_checksums" in report
            assert "reproduction_tests" in report
            assert "summary" in report
            
            summary = report["summary"]
            assert summary["total_datasets"] == 1
            assert summary["total_tests"] == 1
            print("✓ Report generation successful")
            
            # Test setup instructions
            instructions = manager.create_setup_instructions()
            assert isinstance(instructions, str)
            assert "# Reproducibility Setup Instructions" in instructions
            assert "test_dataset" in instructions
            print("✓ Setup instructions generation successful")
            
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running reproducibility tests...")
    
    tests = [
        test_reproducibility_imports,
        test_environment_capture,
        test_checksum_computation,
        test_reproduction_test_creation,
        test_report_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All reproducibility tests passed!")
    else:
        print("❌ Some reproducibility tests failed!")
        sys.exit(1)