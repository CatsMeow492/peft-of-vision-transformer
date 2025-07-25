"""
Comprehensive reproducibility package for PEFT Vision Transformer research.

This module provides functionality to ensure reproducible research by:
- Creating environment specifications with exact dependency versions
- Implementing dataset checksums and preprocessing validation
- Adding automated reproduction testing and validation
- Creating detailed documentation and setup instructions
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import numpy as np
else:
    try:
        import torch
        import numpy as np
    except ImportError:
        torch = None
        np = None

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentSpec:
    """Specification of the computational environment."""
    
    python_version: str
    platform_info: Dict[str, str]
    hardware_info: Dict[str, Any]
    package_versions: Dict[str, str]
    environment_variables: Dict[str, str] = field(default_factory=dict)
    conda_environment: Optional[str] = None
    pip_freeze_output: Optional[str] = None
    git_commit_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "hardware_info": self.hardware_info,
            "package_versions": self.package_versions,
            "environment_variables": self.environment_variables,
            "conda_environment": self.conda_environment,
            "pip_freeze_output": self.pip_freeze_output,
            "git_commit_hash": self.git_commit_hash,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentSpec":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DatasetChecksum:
    """Checksum information for dataset validation."""
    
    dataset_name: str
    file_path: str
    checksum_type: str = "sha256"
    checksum_value: str = ""
    file_size_bytes: int = 0
    num_samples: Optional[int] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    validation_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))


@dataclass
class ReproductionTest:
    """Configuration for reproduction testing."""
    
    test_name: str
    test_type: str  # "unit", "integration", "end_to_end"
    expected_results: Dict[str, Any]
    tolerance: Dict[str, float] = field(default_factory=lambda: {"default": 1e-5})
    test_duration_seconds: Optional[float] = None
    test_command: Optional[str] = None
    test_script_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class ReproducibilityManager:
    """
    Manager for ensuring research reproducibility.
    
    Handles environment specification, dataset validation, and reproduction testing.
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize reproducibility manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.reproducibility_dir = self.project_root / ".reproducibility"
        self.reproducibility_dir.mkdir(exist_ok=True)
        
        # Initialize storage
        self._environment_specs: Dict[str, EnvironmentSpec] = {}
        self._dataset_checksums: Dict[str, DatasetChecksum] = {}
        self._reproduction_tests: Dict[str, ReproductionTest] = {}
    
    def capture_environment(self, name: str = "default") -> EnvironmentSpec:
        """
        Capture current environment specification.
        
        Args:
            name: Name for this environment specification
            
        Returns:
            Environment specification
        """
        logger.info(f"Capturing environment specification: {name}")
        
        try:
            # Python version
            python_version = sys.version
            
            # Platform information
            platform_info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "platform": platform.platform()
            }
            
            # Hardware information
            hardware_info = self._get_hardware_info()
            
            # Package versions
            package_versions = self._get_package_versions()
            
            # Environment variables (filtered for relevant ones)
            env_vars = self._get_relevant_env_vars()
            
            # Conda environment
            conda_env = self._get_conda_environment()
            
            # Pip freeze output
            pip_freeze = self._get_pip_freeze()
            
            # Git commit hash
            git_hash = self._get_git_commit_hash()
            
            env_spec = EnvironmentSpec(
                python_version=python_version,
                platform_info=platform_info,
                hardware_info=hardware_info,
                package_versions=package_versions,
                environment_variables=env_vars,
                conda_environment=conda_env,
                pip_freeze_output=pip_freeze,
                git_commit_hash=git_hash
            )
            
            # Store and save
            self._environment_specs[name] = env_spec
            self._save_environment_spec(name, env_spec)
            
            logger.info(f"Environment specification captured: {name}")
            return env_spec
            
        except Exception as e:
            logger.error(f"Failed to capture environment: {str(e)}")
            raise RuntimeError(f"Environment capture failed: {str(e)}") from e
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        hardware_info = {}
        
        try:
            # CPU information
            if hasattr(os, 'cpu_count'):
                hardware_info["cpu_count"] = os.cpu_count()
            
            # Memory information
            try:
                import psutil
                memory = psutil.virtual_memory()
                hardware_info["total_memory_gb"] = memory.total / (1024**3)
                hardware_info["available_memory_gb"] = memory.available / (1024**3)
            except ImportError:
                logger.warning("psutil not available for memory information")
            
            # GPU information
            if torch is not None:
                hardware_info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    hardware_info["cuda_version"] = torch.version.cuda
                    hardware_info["gpu_count"] = torch.cuda.device_count()
                    hardware_info["gpu_names"] = [
                        torch.cuda.get_device_name(i) 
                        for i in range(torch.cuda.device_count())
                    ]
                
                # MPS (Apple Silicon) support
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    hardware_info["mps_available"] = True
            
        except Exception as e:
            logger.warning(f"Could not gather complete hardware info: {str(e)}")
        
        return hardware_info
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = [
            'torch', 'torchvision', 'torchaudio', 'transformers', 'peft',
            'timm', 'bitsandbytes', 'datasets', 'numpy', 'pandas',
            'matplotlib', 'seaborn', 'scipy', 'scikit-learn', 'pillow'
        ]
        
        versions = {}
        for package in packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                versions[package] = version
            except ImportError:
                versions[package] = 'not_installed'
            except Exception as e:
                versions[package] = f'error: {str(e)}'
        
        return versions
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDNN_VERSION',
            'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'TOKENIZERS_PARALLELISM', 'TRANSFORMERS_CACHE',
            'HF_HOME', 'WANDB_PROJECT', 'WANDB_ENTITY'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value is not None:
                env_vars[var] = value
        
        return env_vars
    
    def _get_conda_environment(self) -> Optional[str]:
        """Get conda environment information."""
        try:
            result = subprocess.run(
                ['conda', 'env', 'export'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _get_pip_freeze(self) -> Optional[str]:
        """Get pip freeze output."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except subprocess.TimeoutExpired:
            pass
        return None
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _save_environment_spec(self, name: str, spec: EnvironmentSpec):
        """Save environment specification to file."""
        spec_file = self.reproducibility_dir / f"environment_{name}.json"
        with open(spec_file, 'w') as f:
            json.dump(spec.to_dict(), f, indent=2)
        logger.info(f"Environment specification saved: {spec_file}")
    
    def compute_dataset_checksum(
        self,
        dataset_path: Union[str, Path],
        dataset_name: str,
        checksum_type: str = "sha256",
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> DatasetChecksum:
        """
        Compute checksum for dataset validation.
        
        Args:
            dataset_path: Path to dataset file or directory
            dataset_name: Name of the dataset
            checksum_type: Type of checksum (sha256, md5)
            preprocessing_config: Configuration used for preprocessing
            
        Returns:
            Dataset checksum information
        """
        logger.info(f"Computing checksum for dataset: {dataset_name}")
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        try:
            if dataset_path.is_file():
                checksum_value, file_size = self._compute_file_checksum(dataset_path, checksum_type)
                file_path = str(dataset_path)
            else:
                # For directories, compute checksum of all files
                checksum_value, file_size = self._compute_directory_checksum(dataset_path, checksum_type)
                file_path = str(dataset_path)
            
            # Try to determine number of samples
            num_samples = self._estimate_dataset_size(dataset_path, dataset_name)
            
            checksum = DatasetChecksum(
                dataset_name=dataset_name,
                file_path=file_path,
                checksum_type=checksum_type,
                checksum_value=checksum_value,
                file_size_bytes=file_size,
                num_samples=num_samples,
                preprocessing_config=preprocessing_config
            )
            
            # Store and save
            self._dataset_checksums[dataset_name] = checksum
            self._save_dataset_checksum(dataset_name, checksum)
            
            logger.info(f"Dataset checksum computed: {dataset_name} -> {checksum_value[:16]}...")
            return checksum
            
        except Exception as e:
            logger.error(f"Failed to compute dataset checksum: {str(e)}")
            raise RuntimeError(f"Checksum computation failed: {str(e)}") from e
    
    def _compute_file_checksum(self, file_path: Path, checksum_type: str) -> tuple[str, int]:
        """Compute checksum for a single file."""
        hash_func = hashlib.new(checksum_type)
        file_size = file_path.stat().st_size
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest(), file_size
    
    def _compute_directory_checksum(self, dir_path: Path, checksum_type: str) -> tuple[str, int]:
        """Compute checksum for all files in a directory."""
        hash_func = hashlib.new(checksum_type)
        total_size = 0
        
        # Get all files sorted for consistent ordering
        all_files = sorted(dir_path.rglob('*'))
        
        for file_path in all_files:
            if file_path.is_file():
                # Include file path in hash for structure consistency
                hash_func.update(str(file_path.relative_to(dir_path)).encode())
                
                file_size = file_path.stat().st_size
                total_size += file_size
                
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_func.update(chunk)
        
        return hash_func.hexdigest(), total_size
    
    def _estimate_dataset_size(self, dataset_path: Path, dataset_name: str) -> Optional[int]:
        """Estimate number of samples in dataset."""
        try:
            # For common dataset formats
            if dataset_name.lower() in ['cifar10', 'cifar-10']:
                return 60000  # 50k train + 10k test
            elif dataset_name.lower() in ['cifar100', 'cifar-100']:
                return 60000  # 50k train + 10k test
            elif dataset_name.lower() in ['tinyimagenet', 'tiny-imagenet']:
                return 110000  # 100k train + 10k val
            
            # Try to count files if it's a directory
            if dataset_path.is_dir():
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                count = sum(1 for f in dataset_path.rglob('*') 
                           if f.is_file() and f.suffix.lower() in image_extensions)
                return count if count > 0 else None
            
        except Exception as e:
            logger.warning(f"Could not estimate dataset size: {str(e)}")
        
        return None
    
    def _save_dataset_checksum(self, name: str, checksum: DatasetChecksum):
        """Save dataset checksum to file."""
        checksum_file = self.reproducibility_dir / f"dataset_{name}_checksum.json"
        with open(checksum_file, 'w') as f:
            json.dump(checksum.__dict__, f, indent=2)
        logger.info(f"Dataset checksum saved: {checksum_file}")
    
    def validate_dataset(self, dataset_name: str, dataset_path: Union[str, Path]) -> bool:
        """
        Validate dataset against stored checksum.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating dataset: {dataset_name}")
        
        if dataset_name not in self._dataset_checksums:
            # Try to load from file
            checksum_file = self.reproducibility_dir / f"dataset_{dataset_name}_checksum.json"
            if checksum_file.exists():
                with open(checksum_file, 'r') as f:
                    data = json.load(f)
                    self._dataset_checksums[dataset_name] = DatasetChecksum(**data)
            else:
                logger.error(f"No checksum found for dataset: {dataset_name}")
                return False
        
        stored_checksum = self._dataset_checksums[dataset_name]
        
        try:
            # Compute current checksum
            current_checksum = self.compute_dataset_checksum(
                dataset_path, 
                f"{dataset_name}_validation",
                stored_checksum.checksum_type
            )
            
            # Compare checksums
            if current_checksum.checksum_value == stored_checksum.checksum_value:
                logger.info(f"Dataset validation passed: {dataset_name}")
                return True
            else:
                logger.error(f"Dataset validation failed: {dataset_name}")
                logger.error(f"Expected: {stored_checksum.checksum_value}")
                logger.error(f"Got: {current_checksum.checksum_value}")
                return False
                
        except Exception as e:
            logger.error(f"Dataset validation error: {str(e)}")
            return False
    
    def create_reproduction_test(
        self,
        test_name: str,
        test_type: str,
        expected_results: Dict[str, Any],
        test_command: Optional[str] = None,
        test_script_path: Optional[str] = None,
        tolerance: Optional[Dict[str, float]] = None,
        dependencies: Optional[List[str]] = None
    ) -> ReproductionTest:
        """
        Create a reproduction test configuration.
        
        Args:
            test_name: Name of the test
            test_type: Type of test (unit, integration, end_to_end)
            expected_results: Expected results for validation
            test_command: Command to run the test
            test_script_path: Path to test script
            tolerance: Tolerance for numerical comparisons
            dependencies: List of required dependencies
            
        Returns:
            Reproduction test configuration
        """
        logger.info(f"Creating reproduction test: {test_name}")
        
        test = ReproductionTest(
            test_name=test_name,
            test_type=test_type,
            expected_results=expected_results,
            tolerance=tolerance or {"default": 1e-5},
            test_command=test_command,
            test_script_path=test_script_path,
            dependencies=dependencies or []
        )
        
        # Store and save
        self._reproduction_tests[test_name] = test
        self._save_reproduction_test(test_name, test)
        
        logger.info(f"Reproduction test created: {test_name}")
        return test
    
    def _save_reproduction_test(self, name: str, test: ReproductionTest):
        """Save reproduction test to file."""
        test_file = self.reproducibility_dir / f"test_{name}.json"
        with open(test_file, 'w') as f:
            json.dump(test.__dict__, f, indent=2)
        logger.info(f"Reproduction test saved: {test_file}")
    
    def run_reproduction_test(self, test_name: str) -> Dict[str, Any]:
        """
        Run a reproduction test and validate results.
        
        Args:
            test_name: Name of the test to run
            
        Returns:
            Test results and validation status
        """
        logger.info(f"Running reproduction test: {test_name}")
        
        if test_name not in self._reproduction_tests:
            # Try to load from file
            test_file = self.reproducibility_dir / f"test_{test_name}.json"
            if test_file.exists():
                with open(test_file, 'r') as f:
                    data = json.load(f)
                    self._reproduction_tests[test_name] = ReproductionTest(**data)
            else:
                raise ValueError(f"Reproduction test not found: {test_name}")
        
        test = self._reproduction_tests[test_name]
        
        try:
            start_time = time.time()
            
            # Run the test
            if test.test_command:
                result = self._run_test_command(test.test_command)
            elif test.test_script_path:
                result = self._run_test_script(test.test_script_path)
            else:
                raise ValueError(f"No test command or script specified for {test_name}")
            
            end_time = time.time()
            test_duration = end_time - start_time
            
            # Validate results
            validation_results = self._validate_test_results(
                result, test.expected_results, test.tolerance
            )
            
            test_results = {
                "test_name": test_name,
                "test_type": test.test_type,
                "duration_seconds": test_duration,
                "raw_results": result,
                "validation_passed": validation_results["passed"],
                "validation_details": validation_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
            
            # Save results
            self._save_test_results(test_name, test_results)
            
            if validation_results["passed"]:
                logger.info(f"Reproduction test passed: {test_name}")
            else:
                logger.error(f"Reproduction test failed: {test_name}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Reproduction test error: {str(e)}")
            return {
                "test_name": test_name,
                "error": str(e),
                "validation_passed": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
    
    def _run_test_command(self, command: str) -> Dict[str, Any]:
        """Run test command and capture results."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=self.project_root
            )
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": "Test command timed out",
                "success": False
            }
    
    def _run_test_script(self, script_path: str) -> Dict[str, Any]:
        """Run test script and capture results."""
        script_path = Path(script_path)
        if not script_path.is_absolute():
            script_path = self.project_root / script_path
        
        if not script_path.exists():
            return {
                "error": f"Test script not found: {script_path}",
                "success": False
            }
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=self.project_root
            )
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": "Test script timed out",
                "success": False
            }
    
    def _validate_test_results(
        self,
        actual_results: Dict[str, Any],
        expected_results: Dict[str, Any],
        tolerance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate test results against expected values."""
        validation = {
            "passed": True,
            "details": {},
            "errors": []
        }
        
        default_tolerance = tolerance.get("default", 1e-5)
        
        for key, expected_value in expected_results.items():
            if key not in actual_results:
                validation["passed"] = False
                validation["errors"].append(f"Missing result key: {key}")
                continue
            
            actual_value = actual_results[key]
            key_tolerance = tolerance.get(key, default_tolerance)
            
            # Validate based on type
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                diff = abs(actual_value - expected_value)
                passed = diff <= key_tolerance
                validation["details"][key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "difference": diff,
                    "tolerance": key_tolerance,
                    "passed": passed
                }
                if not passed:
                    validation["passed"] = False
                    validation["errors"].append(
                        f"Numerical validation failed for {key}: "
                        f"expected {expected_value}, got {actual_value}, "
                        f"difference {diff} > tolerance {key_tolerance}"
                    )
            
            elif expected_value == actual_value:
                validation["details"][key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "passed": True
                }
            
            else:
                validation["passed"] = False
                validation["details"][key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "passed": False
                }
                validation["errors"].append(
                    f"Value mismatch for {key}: expected {expected_value}, got {actual_value}"
                )
        
        return validation
    
    def _save_test_results(self, test_name: str, results: Dict[str, Any]):
        """Save test results to file."""
        results_file = self.reproducibility_dir / f"results_{test_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved: {results_file}")
    
    def generate_reproducibility_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive reproducibility report.
        
        Returns:
            Reproducibility report with all specifications and test results
        """
        logger.info("Generating reproducibility report")
        
        report = {
            "report_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "project_root": str(self.project_root),
            "environment_specifications": {},
            "dataset_checksums": {},
            "reproduction_tests": {},
            "summary": {
                "total_environments": len(self._environment_specs),
                "total_datasets": len(self._dataset_checksums),
                "total_tests": len(self._reproduction_tests),
                "reproducibility_score": 0.0
            }
        }
        
        # Add environment specifications
        for name, spec in self._environment_specs.items():
            report["environment_specifications"][name] = spec.to_dict()
        
        # Add dataset checksums
        for name, checksum in self._dataset_checksums.items():
            report["dataset_checksums"][name] = checksum.__dict__
        
        # Add reproduction tests
        for name, test in self._reproduction_tests.items():
            report["reproduction_tests"][name] = test.__dict__
        
        # Calculate reproducibility score (simple metric)
        total_components = len(self._environment_specs) + len(self._dataset_checksums) + len(self._reproduction_tests)
        if total_components > 0:
            # This is a simplified score - in practice, you might weight different components
            report["summary"]["reproducibility_score"] = min(total_components / 10.0, 1.0)
        
        # Save report
        report_file = self.reproducibility_dir / "reproducibility_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reproducibility report generated: {report_file}")
        return report
    
    def create_setup_instructions(self) -> str:
        """
        Create detailed setup instructions for reproduction.
        
        Returns:
            Markdown-formatted setup instructions
        """
        logger.info("Creating setup instructions")
        
        instructions = []
        instructions.append("# Reproducibility Setup Instructions")
        instructions.append("")
        instructions.append("This document provides step-by-step instructions to reproduce the research environment and results.")
        instructions.append("")
        
        # Environment setup
        instructions.append("## Environment Setup")
        instructions.append("")
        
        if self._environment_specs:
            env_spec = list(self._environment_specs.values())[0]  # Use first available
            
            instructions.append(f"### Python Version")
            instructions.append(f"- Required Python version: {env_spec.python_version.split()[0]}")
            instructions.append("")
            
            instructions.append("### Platform Requirements")
            for key, value in env_spec.platform_info.items():
                instructions.append(f"- {key}: {value}")
            instructions.append("")
            
            instructions.append("### Hardware Requirements")
            for key, value in env_spec.hardware_info.items():
                instructions.append(f"- {key}: {value}")
            instructions.append("")
            
            if env_spec.conda_environment:
                instructions.append("### Conda Environment")
                instructions.append("```bash")
                instructions.append("# Create conda environment from specification")
                instructions.append("conda env create -f environment.yml")
                instructions.append("conda activate peft-vision-transformer")
                instructions.append("```")
                instructions.append("")
            
            instructions.append("### Package Installation")
            instructions.append("```bash")
            instructions.append("# Install required packages")
            instructions.append("pip install -r requirements.txt")
            instructions.append("")
            instructions.append("# Or install specific versions:")
            for package, version in env_spec.package_versions.items():
                if version not in ['not_installed', 'unknown'] and not version.startswith('error'):
                    instructions.append(f"pip install {package}=={version}")
            instructions.append("```")
            instructions.append("")
        
        # Dataset setup
        if self._dataset_checksums:
            instructions.append("## Dataset Setup")
            instructions.append("")
            
            for name, checksum in self._dataset_checksums.items():
                instructions.append(f"### {name}")
                instructions.append(f"- File path: `{checksum.file_path}`")
                instructions.append(f"- Expected checksum ({checksum.checksum_type}): `{checksum.checksum_value}`")
                instructions.append(f"- File size: {checksum.file_size_bytes:,} bytes")
                if checksum.num_samples:
                    instructions.append(f"- Number of samples: {checksum.num_samples:,}")
                instructions.append("")
                
                instructions.append("```bash")
                instructions.append(f"# Validate {name} dataset")
                instructions.append(f"python -c \"")
                instructions.append(f"from src.utils.reproducibility import ReproducibilityManager")
                instructions.append(f"manager = ReproducibilityManager()")
                instructions.append(f"result = manager.validate_dataset('{name}', '{checksum.file_path}')")
                instructions.append(f"print('Validation passed:' if result else 'Validation failed')\"")
                instructions.append("```")
                instructions.append("")
        
        # Reproduction tests
        if self._reproduction_tests:
            instructions.append("## Running Reproduction Tests")
            instructions.append("")
            
            for name, test in self._reproduction_tests.items():
                instructions.append(f"### {test.test_name} ({test.test_type})")
                
                if test.dependencies:
                    instructions.append("Dependencies:")
                    for dep in test.dependencies:
                        instructions.append(f"- {dep}")
                    instructions.append("")
                
                if test.test_command:
                    instructions.append("```bash")
                    instructions.append(test.test_command)
                    instructions.append("```")
                elif test.test_script_path:
                    instructions.append("```bash")
                    instructions.append(f"python {test.test_script_path}")
                    instructions.append("```")
                
                instructions.append("")
                instructions.append("Expected results:")
                for key, value in test.expected_results.items():
                    instructions.append(f"- {key}: {value}")
                instructions.append("")
        
        # Additional instructions
        instructions.append("## Validation")
        instructions.append("")
        instructions.append("To validate the complete setup:")
        instructions.append("")
        instructions.append("```bash")
        instructions.append("# Run all reproduction tests")
        instructions.append("python -c \"")
        instructions.append("from src.utils.reproducibility import ReproducibilityManager")
        instructions.append("manager = ReproducibilityManager()")
        instructions.append("report = manager.generate_reproducibility_report()")
        instructions.append("print(f'Reproducibility score: {report[\\\"summary\\\"][\\\"reproducibility_score\\\"]:.2f}')\"")
        instructions.append("```")
        instructions.append("")
        
        instructions.append("## Troubleshooting")
        instructions.append("")
        instructions.append("### Common Issues")
        instructions.append("1. **Package version conflicts**: Use the exact versions specified in requirements.txt")
        instructions.append("2. **CUDA compatibility**: Ensure CUDA version matches PyTorch requirements")
        instructions.append("3. **Memory issues**: Reduce batch size if running on limited hardware")
        instructions.append("4. **Dataset validation failures**: Re-download datasets and verify checksums")
        instructions.append("")
        
        instructions.append("### Support")
        instructions.append("For additional support, please:")
        instructions.append("1. Check the project README.md")
        instructions.append("2. Review the reproducibility report for detailed environment information")
        instructions.append("3. Open an issue on the project repository")
        instructions.append("")
        
        instructions_text = "\n".join(instructions)
        
        # Save instructions
        instructions_file = self.reproducibility_dir / "SETUP_INSTRUCTIONS.md"
        with open(instructions_file, 'w') as f:
            f.write(instructions_text)
        
        logger.info(f"Setup instructions created: {instructions_file}")
        return instructions_text
    
    def load_all_specifications(self):
        """Load all specifications from saved files."""
        logger.info("Loading all reproducibility specifications")
        
        # Load environment specifications
        for spec_file in self.reproducibility_dir.glob("environment_*.json"):
            try:
                with open(spec_file, 'r') as f:
                    data = json.load(f)
                    name = spec_file.stem.replace("environment_", "")
                    self._environment_specs[name] = EnvironmentSpec.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load environment spec {spec_file}: {str(e)}")
        
        # Load dataset checksums
        for checksum_file in self.reproducibility_dir.glob("dataset_*_checksum.json"):
            try:
                with open(checksum_file, 'r') as f:
                    data = json.load(f)
                    name = data["dataset_name"]
                    self._dataset_checksums[name] = DatasetChecksum(**data)
            except Exception as e:
                logger.warning(f"Failed to load dataset checksum {checksum_file}: {str(e)}")
        
        # Load reproduction tests
        for test_file in self.reproducibility_dir.glob("test_*.json"):
            try:
                with open(test_file, 'r') as f:
                    data = json.load(f)
                    name = data["test_name"]
                    self._reproduction_tests[name] = ReproductionTest(**data)
            except Exception as e:
                logger.warning(f"Failed to load reproduction test {test_file}: {str(e)}")
        
        logger.info(f"Loaded {len(self._environment_specs)} environment specs, "
                   f"{len(self._dataset_checksums)} dataset checksums, "
                   f"{len(self._reproduction_tests)} reproduction tests")
    
    def clear_all_data(self):
        """Clear all stored reproducibility data."""
        self._environment_specs.clear()
        self._dataset_checksums.clear()
        self._reproduction_tests.clear()
        logger.info("All reproducibility data cleared")