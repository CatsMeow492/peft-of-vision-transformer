"""
Tests for reproducibility functionality.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.utils.reproducibility import (
    ReproducibilityManager, EnvironmentSpec, DatasetChecksum, ReproductionTest
)


class TestEnvironmentSpec:
    """Test EnvironmentSpec dataclass."""
    
    def test_default_creation(self):
        """Test creating environment spec with required fields."""
        spec = EnvironmentSpec(
            python_version="3.10.0",
            platform_info={"system": "Darwin"},
            hardware_info={"cpu_count": 8},
            package_versions={"torch": "2.1.2"}
        )
        
        assert spec.python_version == "3.10.0"
        assert spec.platform_info["system"] == "Darwin"
        assert spec.hardware_info["cpu_count"] == 8
        assert spec.package_versions["torch"] == "2.1.2"
        assert spec.environment_variables == {}
        assert spec.conda_environment is None
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        spec = EnvironmentSpec(
            python_version="3.10.0",
            platform_info={"system": "Darwin"},
            hardware_info={"cpu_count": 8},
            package_versions={"torch": "2.1.2"}
        )
        
        spec_dict = spec.to_dict()
        assert isinstance(spec_dict, dict)
        assert spec_dict["python_version"] == "3.10.0"
        assert "timestamp" in spec_dict
    
    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        data = {
            "python_version": "3.10.0",
            "platform_info": {"system": "Darwin"},
            "hardware_info": {"cpu_count": 8},
            "package_versions": {"torch": "2.1.2"},
            "environment_variables": {},
            "conda_environment": None,
            "pip_freeze_output": None,
            "git_commit_hash": None,
            "timestamp": "2024-01-01 00:00:00 UTC"
        }
        
        spec = EnvironmentSpec.from_dict(data)
        assert spec.python_version == "3.10.0"
        assert spec.platform_info["system"] == "Darwin"


class TestDatasetChecksum:
    """Test DatasetChecksum dataclass."""
    
    def test_creation(self):
        """Test creating dataset checksum."""
        checksum = DatasetChecksum(
            dataset_name="test_dataset",
            file_path="/path/to/dataset",
            checksum_value="abc123",
            file_size_bytes=1024
        )
        
        assert checksum.dataset_name == "test_dataset"
        assert checksum.file_path == "/path/to/dataset"
        assert checksum.checksum_type == "sha256"  # default
        assert checksum.checksum_value == "abc123"
        assert checksum.file_size_bytes == 1024
        assert checksum.num_samples is None


class TestReproductionTest:
    """Test ReproductionTest dataclass."""
    
    def test_creation(self):
        """Test creating reproduction test."""
        test = ReproductionTest(
            test_name="test_model_accuracy",
            test_type="integration",
            expected_results={"accuracy": 0.85}
        )
        
        assert test.test_name == "test_model_accuracy"
        assert test.test_type == "integration"
        assert test.expected_results["accuracy"] == 0.85
        assert test.tolerance["default"] == 1e-5
        assert test.dependencies == []


class TestReproducibilityManager:
    """Test ReproducibilityManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create ReproducibilityManager instance."""
        return ReproducibilityManager(project_root=temp_dir)
    
    def test_initialization(self, temp_dir):
        """Test manager initialization."""
        manager = ReproducibilityManager(project_root=temp_dir)
        
        assert manager.project_root == temp_dir
        assert manager.reproducibility_dir == temp_dir / ".reproducibility"
        assert manager.reproducibility_dir.exists()
        assert len(manager._environment_specs) == 0
        assert len(manager._dataset_checksums) == 0
        assert len(manager._reproduction_tests) == 0
    
    @patch('platform.system')
    @patch('platform.release')
    @patch('sys.version', '3.10.0 (main, Oct  5 2021, 10:15:23)')
    def test_capture_environment(self, mock_release, mock_system, manager):
        """Test environment capture."""
        mock_system.return_value = "Darwin"
        mock_release.return_value = "21.0.0"
        
        with patch.object(manager, '_get_hardware_info', return_value={"cpu_count": 8}):
            with patch.object(manager, '_get_package_versions', return_value={"torch": "2.1.2"}):
                with patch.object(manager, '_get_relevant_env_vars', return_value={}):
                    with patch.object(manager, '_get_conda_environment', return_value=None):
                        with patch.object(manager, '_get_pip_freeze', return_value=None):
                            with patch.object(manager, '_get_git_commit_hash', return_value=None):
                                
                                spec = manager.capture_environment("test_env")
                                
                                assert isinstance(spec, EnvironmentSpec)
                                assert spec.platform_info["system"] == "Darwin"
                                assert spec.hardware_info["cpu_count"] == 8
                                assert spec.package_versions["torch"] == "2.1.2"
                                assert "test_env" in manager._environment_specs
    
    def test_get_hardware_info(self, manager):
        """Test hardware information gathering."""
        with patch('os.cpu_count', return_value=8):
            hardware_info = manager._get_hardware_info()
            
            assert "cpu_count" in hardware_info
            assert hardware_info["cpu_count"] == 8
    
    def test_get_package_versions(self, manager):
        """Test package version gathering."""
        with patch('builtins.__import__') as mock_import:
            # Mock torch module
            mock_torch = Mock()
            mock_torch.__version__ = "2.1.2"
            mock_import.return_value = mock_torch
            
            versions = manager._get_package_versions()
            
            # Should contain torch version
            assert "torch" in versions
    
    def test_get_relevant_env_vars(self, manager):
        """Test environment variable gathering."""
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0', 'OTHER_VAR': 'value'}):
            env_vars = manager._get_relevant_env_vars()
            
            assert 'CUDA_VISIBLE_DEVICES' in env_vars
            assert env_vars['CUDA_VISIBLE_DEVICES'] == '0'
            assert 'OTHER_VAR' not in env_vars  # Not in relevant list
    
    def test_compute_file_checksum(self, manager, temp_dir):
        """Test file checksum computation."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_content = b"Hello, world!"
        test_file.write_bytes(test_content)
        
        checksum, size = manager._compute_file_checksum(test_file, "sha256")
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length
        assert size == len(test_content)
    
    def test_compute_directory_checksum(self, manager, temp_dir):
        """Test directory checksum computation."""
        # Create test files
        (temp_dir / "file1.txt").write_bytes(b"content1")
        (temp_dir / "file2.txt").write_bytes(b"content2")
        
        checksum, size = manager._compute_directory_checksum(temp_dir, "sha256")
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length
        assert size == len(b"content1") + len(b"content2")
    
    def test_compute_dataset_checksum_file(self, manager, temp_dir):
        """Test dataset checksum computation for file."""
        # Create test dataset file
        dataset_file = temp_dir / "dataset.txt"
        dataset_file.write_bytes(b"dataset content")
        
        checksum = manager.compute_dataset_checksum(
            dataset_file, "test_dataset", "sha256"
        )
        
        assert isinstance(checksum, DatasetChecksum)
        assert checksum.dataset_name == "test_dataset"
        assert checksum.file_path == str(dataset_file)
        assert checksum.checksum_type == "sha256"
        assert len(checksum.checksum_value) == 64
        assert checksum.file_size_bytes == len(b"dataset content")
    
    def test_compute_dataset_checksum_directory(self, manager, temp_dir):
        """Test dataset checksum computation for directory."""
        # Create test dataset directory
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "data1.txt").write_bytes(b"data1")
        (dataset_dir / "data2.txt").write_bytes(b"data2")
        
        checksum = manager.compute_dataset_checksum(
            dataset_dir, "test_dataset", "sha256"
        )
        
        assert isinstance(checksum, DatasetChecksum)
        assert checksum.dataset_name == "test_dataset"
        assert checksum.file_path == str(dataset_dir)
        assert checksum.file_size_bytes == len(b"data1") + len(b"data2")
    
    def test_compute_dataset_checksum_nonexistent(self, manager, temp_dir):
        """Test dataset checksum computation for nonexistent path."""
        nonexistent_path = temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            manager.compute_dataset_checksum(nonexistent_path, "test_dataset")
    
    def test_estimate_dataset_size(self, manager, temp_dir):
        """Test dataset size estimation."""
        # Test known dataset
        size = manager._estimate_dataset_size(temp_dir, "cifar10")
        assert size == 60000
        
        # Test directory with images
        dataset_dir = temp_dir / "images"
        dataset_dir.mkdir()
        (dataset_dir / "img1.jpg").write_bytes(b"fake image 1")
        (dataset_dir / "img2.png").write_bytes(b"fake image 2")
        (dataset_dir / "other.txt").write_bytes(b"not an image")
        
        size = manager._estimate_dataset_size(dataset_dir, "custom_dataset")
        assert size == 2  # Only image files counted
    
    def test_validate_dataset_success(self, manager, temp_dir):
        """Test successful dataset validation."""
        # Create and compute checksum for dataset
        dataset_file = temp_dir / "dataset.txt"
        dataset_file.write_bytes(b"dataset content")
        
        original_checksum = manager.compute_dataset_checksum(
            dataset_file, "test_dataset", "sha256"
        )
        
        # Validate the same dataset
        result = manager.validate_dataset("test_dataset", dataset_file)
        assert result is True
    
    def test_validate_dataset_failure(self, manager, temp_dir):
        """Test dataset validation failure."""
        # Create and compute checksum for original dataset
        dataset_file = temp_dir / "dataset.txt"
        dataset_file.write_bytes(b"original content")
        
        original_checksum = manager.compute_dataset_checksum(
            dataset_file, "test_dataset", "sha256"
        )
        
        # Modify the dataset
        dataset_file.write_bytes(b"modified content")
        
        # Validation should fail
        result = manager.validate_dataset("test_dataset", dataset_file)
        assert result is False
    
    def test_validate_dataset_no_checksum(self, manager, temp_dir):
        """Test dataset validation with no stored checksum."""
        dataset_file = temp_dir / "dataset.txt"
        dataset_file.write_bytes(b"content")
        
        result = manager.validate_dataset("nonexistent_dataset", dataset_file)
        assert result is False
    
    def test_create_reproduction_test(self, manager):
        """Test reproduction test creation."""
        test = manager.create_reproduction_test(
            test_name="test_accuracy",
            test_type="integration",
            expected_results={"accuracy": 0.85, "loss": 0.15},
            test_command="python test_model.py",
            tolerance={"accuracy": 0.01}
        )
        
        assert isinstance(test, ReproductionTest)
        assert test.test_name == "test_accuracy"
        assert test.test_type == "integration"
        assert test.expected_results["accuracy"] == 0.85
        assert test.tolerance["accuracy"] == 0.01
        assert test.test_command == "python test_model.py"
        assert "test_accuracy" in manager._reproduction_tests
    
    @patch('subprocess.run')
    def test_run_test_command(self, mock_run, manager):
        """Test running test command."""
        # Mock successful command execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Test passed"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = manager._run_test_command("python test.py")
        
        assert result["success"] is True
        assert result["returncode"] == 0
        assert result["stdout"] == "Test passed"
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_test_command_failure(self, mock_run, manager):
        """Test running test command with failure."""
        # Mock failed command execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Test failed"
        mock_run.return_value = mock_result
        
        result = manager._run_test_command("python test.py")
        
        assert result["success"] is False
        assert result["returncode"] == 1
        assert result["stderr"] == "Test failed"
    
    def test_validate_test_results_success(self, manager):
        """Test successful test result validation."""
        actual_results = {
            "accuracy": 0.85,
            "loss": 0.15,
            "status": "passed"
        }
        expected_results = {
            "accuracy": 0.84,
            "loss": 0.16,
            "status": "passed"
        }
        tolerance = {"accuracy": 0.02, "loss": 0.02, "default": 1e-5}
        
        validation = manager._validate_test_results(
            actual_results, expected_results, tolerance
        )
        
        assert validation["passed"] is True
        assert len(validation["errors"]) == 0
        assert validation["details"]["accuracy"]["passed"] is True
        assert validation["details"]["loss"]["passed"] is True
        assert validation["details"]["status"]["passed"] is True
    
    def test_validate_test_results_failure(self, manager):
        """Test test result validation failure."""
        actual_results = {
            "accuracy": 0.80,  # Too different from expected
            "status": "failed"
        }
        expected_results = {
            "accuracy": 0.90,
            "status": "passed"
        }
        tolerance = {"default": 0.01}  # Tight tolerance
        
        validation = manager._validate_test_results(
            actual_results, expected_results, tolerance
        )
        
        assert validation["passed"] is False
        assert len(validation["errors"]) == 2  # Both accuracy and status failed
        assert validation["details"]["accuracy"]["passed"] is False
        assert validation["details"]["status"]["passed"] is False
    
    def test_validate_test_results_missing_key(self, manager):
        """Test test result validation with missing key."""
        actual_results = {
            "accuracy": 0.85
            # Missing "loss" key
        }
        expected_results = {
            "accuracy": 0.85,
            "loss": 0.15
        }
        tolerance = {"default": 1e-5}
        
        validation = manager._validate_test_results(
            actual_results, expected_results, tolerance
        )
        
        assert validation["passed"] is False
        assert len(validation["errors"]) == 1
        assert "Missing result key: loss" in validation["errors"][0]
    
    def test_generate_reproducibility_report(self, manager, temp_dir):
        """Test reproducibility report generation."""
        # Add some test data
        manager.capture_environment = Mock(return_value=EnvironmentSpec(
            python_version="3.10.0",
            platform_info={"system": "Darwin"},
            hardware_info={"cpu_count": 8},
            package_versions={"torch": "2.1.2"}
        ))
        
        # Create test dataset checksum
        dataset_file = temp_dir / "dataset.txt"
        dataset_file.write_bytes(b"test data")
        manager.compute_dataset_checksum(dataset_file, "test_dataset")
        
        # Create test reproduction test
        manager.create_reproduction_test(
            "test_model", "unit", {"accuracy": 0.85}
        )
        
        report = manager.generate_reproducibility_report()
        
        assert isinstance(report, dict)
        assert "report_timestamp" in report
        assert "environment_specifications" in report
        assert "dataset_checksums" in report
        assert "reproduction_tests" in report
        assert "summary" in report
        
        summary = report["summary"]
        assert summary["total_datasets"] == 1
        assert summary["total_tests"] == 1
        assert 0.0 <= summary["reproducibility_score"] <= 1.0
    
    def test_create_setup_instructions(self, manager, temp_dir):
        """Test setup instructions creation."""
        # Add some test data
        manager._environment_specs["test"] = EnvironmentSpec(
            python_version="3.10.0 (main, Oct  5 2021, 10:15:23)",
            platform_info={"system": "Darwin", "release": "21.0.0"},
            hardware_info={"cpu_count": 8, "total_memory_gb": 16.0},
            package_versions={"torch": "2.1.2", "transformers": "4.36.2"},
            conda_environment="name: test-env\ndependencies:\n  - python=3.10"
        )
        
        dataset_file = temp_dir / "dataset.txt"
        dataset_file.write_bytes(b"test data")
        manager.compute_dataset_checksum(dataset_file, "test_dataset")
        
        manager.create_reproduction_test(
            "test_model", "unit", {"accuracy": 0.85},
            test_command="python test_model.py"
        )
        
        instructions = manager.create_setup_instructions()
        
        assert isinstance(instructions, str)
        assert "# Reproducibility Setup Instructions" in instructions
        assert "## Environment Setup" in instructions
        assert "## Dataset Setup" in instructions
        assert "## Running Reproduction Tests" in instructions
        assert "python=3.10" in instructions  # From conda environment
        assert "torch==2.1.2" in instructions  # From package versions
        assert "test_dataset" in instructions  # Dataset name
        assert "python test_model.py" in instructions  # Test command
    
    def test_load_all_specifications(self, manager, temp_dir):
        """Test loading all specifications from files."""
        # Create test files
        env_data = {
            "python_version": "3.10.0",
            "platform_info": {"system": "Darwin"},
            "hardware_info": {"cpu_count": 8},
            "package_versions": {"torch": "2.1.2"},
            "environment_variables": {},
            "conda_environment": None,
            "pip_freeze_output": None,
            "git_commit_hash": None,
            "timestamp": "2024-01-01 00:00:00 UTC"
        }
        
        env_file = manager.reproducibility_dir / "environment_test.json"
        with open(env_file, 'w') as f:
            json.dump(env_data, f)
        
        checksum_data = {
            "dataset_name": "test_dataset",
            "file_path": "/path/to/dataset",
            "checksum_type": "sha256",
            "checksum_value": "abc123",
            "file_size_bytes": 1024,
            "num_samples": None,
            "preprocessing_config": None,
            "validation_timestamp": "2024-01-01 00:00:00 UTC"
        }
        
        checksum_file = manager.reproducibility_dir / "dataset_test_dataset_checksum.json"
        with open(checksum_file, 'w') as f:
            json.dump(checksum_data, f)
        
        test_data = {
            "test_name": "test_model",
            "test_type": "unit",
            "expected_results": {"accuracy": 0.85},
            "tolerance": {"default": 1e-5},
            "test_duration_seconds": None,
            "test_command": "python test.py",
            "test_script_path": None,
            "dependencies": []
        }
        
        test_file = manager.reproducibility_dir / "test_test_model.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load specifications
        manager.load_all_specifications()
        
        assert len(manager._environment_specs) == 1
        assert "test" in manager._environment_specs
        assert len(manager._dataset_checksums) == 1
        assert "test_dataset" in manager._dataset_checksums
        assert len(manager._reproduction_tests) == 1
        assert "test_model" in manager._reproduction_tests
    
    def test_clear_all_data(self, manager):
        """Test clearing all data."""
        # Add some test data
        manager._environment_specs["test"] = Mock()
        manager._dataset_checksums["test"] = Mock()
        manager._reproduction_tests["test"] = Mock()
        
        assert len(manager._environment_specs) == 1
        assert len(manager._dataset_checksums) == 1
        assert len(manager._reproduction_tests) == 1
        
        manager.clear_all_data()
        
        assert len(manager._environment_specs) == 0
        assert len(manager._dataset_checksums) == 0
        assert len(manager._reproduction_tests) == 0


if __name__ == "__main__":
    pytest.main([__file__])