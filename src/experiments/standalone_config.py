"""
Standalone configuration system for PEFT experiments.
This module provides configuration classes without heavy dependencies.
"""

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator, Tuple
from itertools import product

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""
    
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: str = "FEATURE_EXTRACTION"
    inference_mode: bool = False
    modules_to_save: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate LoRA configuration."""
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("LoRA dropout must be between 0 and 1")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError("LoRA bias must be 'none', 'all', or 'lora_only'")
        
        # Set default target modules if not specified
        if self.target_modules is None:
            self.target_modules = ["qkv", "query", "key", "value", "proj", "fc1", "fc2"]


@dataclass 
class QuantizationConfig:
    """Configuration for model quantization."""
    
    bits: int = 8
    compute_dtype: str = "float16"
    quant_type: str = "nf4"
    double_quant: bool = False
    
    def __post_init__(self):
        """Validate quantization configuration."""
        if self.bits not in [4, 8, 16]:
            raise ValueError("Quantization bits must be 4, 8, or 16")
        if self.compute_dtype not in ["float16", "float32", "bfloat16"]:
            raise ValueError("Compute dtype must be float16, float32, or bfloat16")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip_norm: Optional[float] = 1.0
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    early_stopping_patience: Optional[int] = None
    output_dir: str = "outputs"
    logging_dir: Optional[str] = None
    seed: int = 42
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        
        # Set default logging directory
        if self.logging_dir is None:
            self.logging_dir = str(Path(self.output_dir) / "logs")


@dataclass
class ModelConfig:
    """Configuration for model selection and loading."""
    
    name: str  # e.g., "deit_tiny_patch16_224", "vit_small_patch16_224"
    source: str = "timm"  # "timm" or "huggingface"
    pretrained: bool = True
    num_classes: int = 10  # Will be set based on dataset
    
    # Model-specific parameters
    image_size: int = 224
    patch_size: int = 16
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.source not in ["timm", "huggingface"]:
            raise ValueError("Model source must be 'timm' or 'huggingface'")
        if self.image_size <= 0:
            raise ValueError("Image size must be positive")
        if self.patch_size <= 0:
            raise ValueError("Patch size must be positive")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    name: str  # "cifar10", "cifar100", "tiny_imagenet"
    data_dir: Optional[str] = None
    
    # Data splits
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    
    # Preprocessing
    image_size: int = 224
    normalize: bool = True
    augmentation: bool = True
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """Validate dataset configuration."""
        if self.name not in ["cifar10", "cifar100", "tiny_imagenet"]:
            raise ValueError("Dataset must be one of: cifar10, cifar100, tiny_imagenet")
        if self.image_size <= 0:
            raise ValueError("Image size must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
    
    @property
    def num_classes(self) -> int:
        """Get number of classes for the dataset."""
        return {
            "cifar10": 10,
            "cifar100": 100,
            "tiny_imagenet": 200
        }[self.name]


@dataclass
class ExperimentConfig:
    """Complete configuration for a PEFT experiment."""
    
    # Experiment metadata
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Core configurations
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="deit_tiny_patch16_224"))
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="cifar10"))
    lora: Optional[LoRAConfig] = None
    quantization: Optional[QuantizationConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment settings
    seed: int = 42
    output_dir: str = "experiments/outputs"
    
    # Advanced PEFT methods
    use_adalora: bool = False
    use_qa_lora: bool = False
    
    # Resource constraints
    max_memory_gb: Optional[float] = None
    max_training_time_hours: Optional[float] = None
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        # Ensure model and dataset image sizes match
        if self.model.image_size != self.dataset.image_size:
            logger.warning(f"Model image size ({self.model.image_size}) != "
                          f"dataset image size ({self.dataset.image_size}). "
                          f"Using dataset image size.")
            self.model.image_size = self.dataset.image_size
        
        # Set model num_classes based on dataset
        self.model.num_classes = self.dataset.num_classes
        
        # Set training batch size from dataset if not explicitly set
        if hasattr(self.training, 'batch_size') and self.training.batch_size != self.dataset.batch_size:
            logger.info(f"Using dataset batch size ({self.dataset.batch_size}) for training")
            self.training.batch_size = self.dataset.batch_size
        
        # Validate PEFT method combinations
        if self.use_qa_lora and self.quantization is None:
            raise ValueError("QA-LoRA requires quantization configuration")
        
        if self.use_adalora and self.lora is None:
            raise ValueError("AdaLoRA requires LoRA configuration")
        
        # Set default LoRA config if using PEFT methods but none specified
        if (self.use_adalora or self.use_qa_lora) and self.lora is None:
            self.lora = LoRAConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        # Extract nested configurations
        model_config = ModelConfig(**config_dict.get("model", {}))
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        
        # Handle LoRA config
        lora_config = None
        if "lora" in config_dict and config_dict["lora"] is not None:
            lora_config = LoRAConfig(**config_dict["lora"])
        
        # Handle quantization config
        quantization_config = None
        if "quantization" in config_dict and config_dict["quantization"] is not None:
            quantization_config = QuantizationConfig(**config_dict["quantization"])
        
        # Handle training config
        training_config = TrainingConfig(**config_dict.get("training", {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["model", "dataset", "lora", "quantization", "training"]}
        
        return cls(
            model=model_config,
            dataset=dataset_config,
            lora=lora_config,
            quantization=quantization_config,
            training=training_config,
            **main_config
        )
    
    def save_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML not available - cannot save YAML files")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML not available - cannot load YAML files")
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {path}")
        return cls.from_dict(config_dict)
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID based on configuration."""
        components = [
            self.model.name,
            self.dataset.name,
            f"seed{self.seed}"
        ]
        
        if self.lora:
            components.append(f"lora_r{self.lora.rank}_a{self.lora.alpha}")
        
        if self.quantization:
            components.append(f"q{self.quantization.bits}bit")
        
        if self.use_adalora:
            components.append("adalora")
        
        if self.use_qa_lora:
            components.append("qalora")
        
        return "_".join(components)


class ExperimentMatrix:
    """Generator for systematic experiment configurations."""
    
    def __init__(self, base_config: ExperimentConfig):
        """
        Initialize experiment matrix with base configuration.
        
        Args:
            base_config: Base configuration to vary
        """
        self.base_config = base_config
        self.variations: Dict[str, List[Any]] = {}
    
    def add_model_variation(self, models: List[str]):
        """Add model name variations."""
        self.variations["model.name"] = models
        return self
    
    def add_dataset_variation(self, datasets: List[str]):
        """Add dataset variations."""
        self.variations["dataset.name"] = datasets
        return self
    
    def add_lora_rank_variation(self, ranks: List[int]):
        """Add LoRA rank variations."""
        self.variations["lora.rank"] = ranks
        return self
    
    def add_lora_alpha_variation(self, alphas: List[float]):
        """Add LoRA alpha variations."""
        self.variations["lora.alpha"] = alphas
        return self
    
    def add_quantization_variation(self, bit_widths: List[int]):
        """Add quantization bit width variations."""
        self.variations["quantization.bits"] = bit_widths
        return self
    
    def add_seed_variation(self, seeds: List[int]):
        """Add random seed variations."""
        self.variations["seed"] = seeds
        return self
    
    def add_batch_size_variation(self, batch_sizes: List[int]):
        """Add batch size variations."""
        self.variations["dataset.batch_size"] = batch_sizes
        return self
    
    def add_learning_rate_variation(self, learning_rates: List[float]):
        """Add learning rate variations."""
        self.variations["training.learning_rate"] = learning_rates
        return self
    
    def add_method_variation(self, methods: List[str]):
        """
        Add PEFT method variations.
        
        Args:
            methods: List of methods like ["lora", "adalora", "qa_lora"]
        """
        method_configs = []
        for method in methods:
            if method == "lora":
                method_configs.append({"use_adalora": False, "use_qa_lora": False})
            elif method == "adalora":
                method_configs.append({"use_adalora": True, "use_qa_lora": False})
            elif method == "qa_lora":
                method_configs.append({"use_adalora": False, "use_qa_lora": True})
            else:
                raise ValueError(f"Unknown method: {method}")
        
        self.variations["_method"] = method_configs
        return self
    
    def generate_configs(self) -> Iterator[ExperimentConfig]:
        """
        Generate all experiment configurations from the matrix.
        
        Yields:
            ExperimentConfig objects for each combination
        """
        if not self.variations:
            yield self.base_config
            return
        
        # Get all parameter names and their values
        param_names = list(self.variations.keys())
        param_values = list(self.variations.values())
        
        # Generate all combinations
        for combination in product(*param_values):
            # Create a copy of the base config
            config_dict = self.base_config.to_dict()
            
            # Apply variations
            for param_name, value in zip(param_names, combination):
                if param_name == "_method":
                    # Special handling for method variations
                    config_dict.update(value)
                else:
                    # Handle nested parameter names like "lora.rank"
                    self._set_nested_value(config_dict, param_name, value)
            
            # Create new config
            try:
                yield ExperimentConfig.from_dict(config_dict)
            except Exception as e:
                logger.warning(f"Skipping invalid configuration: {e}")
                continue
    
    def _set_nested_value(self, config_dict: Dict[str, Any], param_path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = param_path.split('.')
        current = config_dict
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def count_experiments(self) -> int:
        """Count total number of experiments in the matrix."""
        if not self.variations:
            return 1
        
        count = 1
        for values in self.variations.values():
            count *= len(values)
        
        return count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the experiment matrix."""
        return {
            "total_experiments": self.count_experiments(),
            "variations": {
                param: len(values) for param, values in self.variations.items()
            },
            "base_config": self.base_config.name
        }


class ConfigValidator:
    """Validator for experiment configurations."""
    
    @staticmethod
    def validate_config(config: ExperimentConfig) -> Tuple[bool, List[str]]:
        """
        Validate experiment configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Validate model configuration
            model_errors = ConfigValidator._validate_model_config(config.model)
            errors.extend(model_errors)
            
            # Validate dataset configuration
            dataset_errors = ConfigValidator._validate_dataset_config(config.dataset)
            errors.extend(dataset_errors)
            
            # Validate LoRA configuration if present
            if config.lora:
                lora_errors = ConfigValidator._validate_lora_config(config.lora)
                errors.extend(lora_errors)
            
            # Validate quantization configuration if present
            if config.quantization:
                quant_errors = ConfigValidator._validate_quantization_config(config.quantization)
                errors.extend(quant_errors)
            
            # Validate training configuration
            training_errors = ConfigValidator._validate_training_config(config.training)
            errors.extend(training_errors)
            
            # Validate method combinations
            method_errors = ConfigValidator._validate_method_combinations(config)
            errors.extend(method_errors)
            
            # Validate resource constraints
            resource_errors = ConfigValidator._validate_resource_constraints(config)
            errors.extend(resource_errors)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_model_config(model: ModelConfig) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        # Check supported models
        supported_models = [
            "deit_tiny_patch16_224",
            "deit_small_patch16_224", 
            "vit_small_patch16_224",
            "vit_base_patch16_224"
        ]
        
        if model.name not in supported_models:
            errors.append(f"Unsupported model: {model.name}. "
                         f"Supported models: {supported_models}")
        
        # Validate image and patch sizes
        if model.image_size % model.patch_size != 0:
            errors.append(f"Image size ({model.image_size}) must be divisible by "
                         f"patch size ({model.patch_size})")
        
        return errors
    
    @staticmethod
    def _validate_dataset_config(dataset: DatasetConfig) -> List[str]:
        """Validate dataset configuration."""
        errors = []
        
        # Check supported datasets
        supported_datasets = ["cifar10", "cifar100", "tiny_imagenet"]
        if dataset.name not in supported_datasets:
            errors.append(f"Unsupported dataset: {dataset.name}. "
                         f"Supported datasets: {supported_datasets}")
        
        # Validate batch size for memory constraints
        if dataset.batch_size > 128:
            errors.append(f"Batch size ({dataset.batch_size}) may be too large for M2 hardware")
        
        return errors
    
    @staticmethod
    def _validate_lora_config(lora: LoRAConfig) -> List[str]:
        """Validate LoRA configuration."""
        errors = []
        
        # Check rank constraints
        if lora.rank > 64:
            errors.append(f"LoRA rank ({lora.rank}) is very high and may not be efficient")
        
        # Check alpha/rank ratio
        if lora.alpha / lora.rank > 4:
            errors.append(f"LoRA alpha/rank ratio ({lora.alpha/lora.rank:.1f}) is high")
        
        return errors
    
    @staticmethod
    def _validate_quantization_config(quantization: QuantizationConfig) -> List[str]:
        """Validate quantization configuration."""
        errors = []
        
        # Check supported bit widths
        if quantization.bits not in [4, 8]:
            errors.append(f"Unsupported quantization bits: {quantization.bits}. "
                         f"Supported: 4, 8")
        
        return errors
    
    @staticmethod
    def _validate_training_config(training: TrainingConfig) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        # Check learning rate
        if training.learning_rate > 1e-2:
            errors.append(f"Learning rate ({training.learning_rate}) may be too high")
        
        # Check epoch count
        if training.num_epochs > 100:
            errors.append(f"Number of epochs ({training.num_epochs}) may be excessive")
        
        return errors
    
    @staticmethod
    def _validate_method_combinations(config: ExperimentConfig) -> List[str]:
        """Validate PEFT method combinations."""
        errors = []
        
        # Check QA-LoRA requirements
        if config.use_qa_lora and not config.quantization:
            errors.append("QA-LoRA requires quantization configuration")
        
        # Check AdaLoRA requirements
        if config.use_adalora and not config.lora:
            errors.append("AdaLoRA requires LoRA configuration")
        
        # Check conflicting methods
        if config.use_adalora and config.use_qa_lora:
            errors.append("AdaLoRA and QA-LoRA cannot be used simultaneously")
        
        return errors
    
    @staticmethod
    def _validate_resource_constraints(config: ExperimentConfig) -> List[str]:
        """Validate resource constraints for M2 hardware."""
        errors = []
        
        # Estimate memory usage
        estimated_memory = ConfigValidator._estimate_memory_usage(config)
        
        # Check against M2 MacBook constraints (assume 32GB available for experiments)
        max_memory_gb = config.max_memory_gb or 32.0
        
        if estimated_memory > max_memory_gb:
            errors.append(f"Estimated memory usage ({estimated_memory:.1f}GB) "
                         f"exceeds limit ({max_memory_gb}GB)")
        
        return errors
    
    @staticmethod
    def _estimate_memory_usage(config: ExperimentConfig) -> float:
        """Estimate memory usage in GB for the configuration."""
        # Base model memory (rough estimates)
        model_memory = {
            "deit_tiny_patch16_224": 0.5,
            "deit_small_patch16_224": 2.0,
            "vit_small_patch16_224": 2.0,
            "vit_base_patch16_224": 8.0
        }.get(config.model.name, 2.0)
        
        # Quantization reduces memory
        if config.quantization:
            if config.quantization.bits == 8:
                model_memory *= 0.5
            elif config.quantization.bits == 4:
                model_memory *= 0.25
        
        # LoRA adds minimal memory
        if config.lora:
            lora_memory = config.lora.rank * 0.001  # Very rough estimate
            model_memory += lora_memory
        
        # Batch size affects memory
        batch_memory = config.dataset.batch_size * 0.01  # Rough estimate
        
        # Training overhead
        training_overhead = model_memory * 2  # Gradients, optimizer states
        
        total_memory = model_memory + batch_memory + training_overhead
        
        return total_memory


def create_default_experiment_matrix() -> ExperimentMatrix:
    """Create a default experiment matrix for systematic evaluation."""
    # Base configuration
    base_config = ExperimentConfig(
        name="peft_vision_transformer_study",
        description="Systematic evaluation of PEFT methods on Vision Transformers",
        tags=["peft", "vision_transformer", "lora", "quantization"],
        model=ModelConfig(name="deit_tiny_patch16_224"),
        dataset=DatasetConfig(name="cifar10"),
        lora=LoRAConfig(rank=8, alpha=16.0),
        training=TrainingConfig(
            learning_rate=1e-4,
            batch_size=32,
            num_epochs=10,
            save_steps=500,
            eval_steps=100
        )
    )
    
    # Create matrix with variations
    matrix = ExperimentMatrix(base_config)
    
    # Add systematic variations
    matrix.add_model_variation([
        "deit_tiny_patch16_224",
        "deit_small_patch16_224"
    ])
    
    matrix.add_dataset_variation([
        "cifar10",
        "cifar100"
    ])
    
    matrix.add_lora_rank_variation([2, 4, 8, 16, 32])
    
    matrix.add_quantization_variation([8, 4])  # Will create configs with and without quantization
    
    matrix.add_seed_variation([42, 123, 456])  # Multiple seeds for statistical significance
    
    matrix.add_method_variation(["lora", "adalora"])
    
    return matrix