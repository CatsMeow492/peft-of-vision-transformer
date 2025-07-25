"""
LoRA configuration and adapter integration for Parameter-Efficient Fine-Tuning.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model, TaskType
else:
    try:
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        nn = None
        LoraConfig = None
        get_peft_model = None
        TaskType = None

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""
    
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: str = "FEATURE_EXTRACTION"  # Will be converted to TaskType enum
    inference_mode: bool = False
    modules_to_save: Optional[List[str]] = None
    
    # Advanced LoRA parameters
    use_rslora: bool = False  # Rank-Stabilized LoRA
    use_dora: bool = False    # Weight-Decomposed Low-Rank Adaptation
    lora_alpha_pattern: Optional[Dict[str, float]] = None
    lora_dropout_pattern: Optional[Dict[str, float]] = None
    
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
            self.target_modules = self._get_default_target_modules()
    
    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules for Vision Transformers."""
        return [
            "qkv",           # timm models - combined query, key, value
            "query",         # HuggingFace models - separate layers
            "key",
            "value", 
            "proj",          # projection layers
            "fc1",           # MLP layers
            "fc2"
        ]
    
    def to_peft_config(self) -> "LoraConfig":
        """
        Convert to HuggingFace PEFT LoraConfig.
        
        Returns:
            PEFT LoraConfig object
            
        Raises:
            RuntimeError: If PEFT library is not available
        """
        if LoraConfig is None or TaskType is None:
            raise RuntimeError("PEFT library not available")
        
        # Convert task type string to enum
        task_type_enum = getattr(TaskType, self.task_type, TaskType.FEATURE_EXTRACTION)
        
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=task_type_enum,
            inference_mode=self.inference_mode,
            modules_to_save=self.modules_to_save,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
            lora_alpha_pattern=self.lora_alpha_pattern,
            lora_dropout_pattern=self.lora_dropout_pattern
        )
    
    def get_trainable_params_ratio(self, total_params: int) -> float:
        """
        Estimate the ratio of trainable parameters with LoRA.
        
        Args:
            total_params: Total number of parameters in the base model
            
        Returns:
            Estimated ratio of trainable parameters
        """
        # Rough estimation: LoRA adds 2 * rank * (input_dim + output_dim) parameters
        # For Vision Transformers, typical attention dimension is 768-1024
        # This is a conservative estimate
        estimated_lora_params = len(self.target_modules) * 2 * self.rank * 768
        return min(estimated_lora_params / total_params, 1.0) if total_params > 0 else 0.0


class LoRAAdapter:
    """Manager for applying LoRA adapters to models."""
    
    def __init__(self):
        """Initialize LoRA adapter manager."""
        self._applied_models: Dict[str, Any] = {}
    
    def apply_lora(self, model, config: LoRAConfig, model_name: str = "model"):
        """
        Apply LoRA adapter to a model.
        
        Args:
            model: Base model to adapt
            config: LoRA configuration
            model_name: Name for tracking the adapted model
            
        Returns:
            Model with LoRA adapters applied
            
        Raises:
            RuntimeError: If PEFT library is not available or application fails
        """
        if get_peft_model is None:
            raise RuntimeError("PEFT library not available for LoRA adaptation")
        
        try:
            # Detect and validate target modules
            detected_modules = self._detect_target_modules(model, config.target_modules)
            if not detected_modules:
                logger.warning("No target modules found for LoRA adaptation")
                detected_modules = config.target_modules  # Use original list as fallback
            
            # Update config with detected modules
            updated_config = LoRAConfig(
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                target_modules=detected_modules,
                bias=config.bias,
                task_type=config.task_type,
                inference_mode=config.inference_mode,
                modules_to_save=config.modules_to_save,
                use_rslora=config.use_rslora,
                use_dora=config.use_dora,
                lora_alpha_pattern=config.lora_alpha_pattern,
                lora_dropout_pattern=config.lora_dropout_pattern
            )
            
            # Convert to PEFT config and apply
            peft_config = updated_config.to_peft_config()
            peft_model = get_peft_model(model, peft_config)
            
            # Cache the adapted model
            self._applied_models[model_name] = {
                "model": peft_model,
                "config": updated_config,
                "base_model": model
            }
            
            logger.info(f"Successfully applied LoRA to {model_name}")
            logger.info(f"LoRA config: rank={config.rank}, alpha={config.alpha}, "
                       f"dropout={config.dropout}, targets={len(detected_modules)}")
            
            return peft_model
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA to {model_name}: {str(e)}")
            raise RuntimeError(f"LoRA application failed: {str(e)}") from e
    
    def _detect_target_modules(self, model, target_patterns: List[str]) -> List[str]:
        """
        Detect actual target modules in the model based on patterns.
        
        Args:
            model: Model to analyze
            target_patterns: List of module name patterns to match
            
        Returns:
            List of actual module names that match the patterns
        """
        detected_modules = []
        
        for name, module in model.named_modules():
            # Check if module name contains any of the target patterns
            for pattern in target_patterns:
                if pattern in name and hasattr(module, 'weight'):
                    # Verify it's a linear layer (most common LoRA target)
                    if nn is not None and isinstance(module, nn.Linear):
                        detected_modules.append(name)
                        break
                    # For cases where nn is not available, check for weight attribute
                    elif hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                        if len(module.weight.shape) == 2:  # Linear layer has 2D weight
                            detected_modules.append(name)
                            break
        
        return sorted(list(set(detected_modules)))
    
    def validate_adapter(self, model, config: LoRAConfig) -> Dict[str, Any]:
        """
        Validate LoRA adapter application and return statistics.
        
        Args:
            model: Model with LoRA adapters
            config: LoRA configuration used
            
        Returns:
            Dictionary with validation statistics
        """
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Calculate ratios
            trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
            
            # Count LoRA modules
            lora_modules = 0
            for name, module in model.named_modules():
                if "lora" in name.lower():
                    lora_modules += 1
            
            validation_stats = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_ratio": trainable_ratio,
                "lora_modules_count": lora_modules,
                "target_modules": config.target_modules,
                "lora_rank": config.rank,
                "lora_alpha": config.alpha,
                "validation_passed": True
            }
            
            # Validation checks
            if trainable_ratio > 0.5:  # More than 50% trainable is unusual for LoRA
                logger.warning(f"High trainable ratio: {trainable_ratio:.2%}")
                validation_stats["warnings"] = ["High trainable parameter ratio"]
            
            if lora_modules == 0:
                logger.error("No LoRA modules found in adapted model")
                validation_stats["validation_passed"] = False
                validation_stats["errors"] = ["No LoRA modules detected"]
            
            logger.info(f"LoRA validation: {trainable_params:,}/{total_params:,} "
                       f"trainable ({trainable_ratio:.2%}), {lora_modules} LoRA modules")
            
            return validation_stats
            
        except Exception as e:
            logger.error(f"LoRA validation failed: {str(e)}")
            return {
                "validation_passed": False,
                "errors": [str(e)]
            }
    
    def get_parameter_ratio(self, model) -> float:
        """
        Get the ratio of trainable to total parameters.
        
        Args:
            model: Model to analyze
            
        Returns:
            Ratio of trainable parameters
        """
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return trainable_params / total_params if total_params > 0 else 0.0
        except Exception:
            return 0.0
    
    def list_applied_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all models with applied LoRA adapters.
        
        Returns:
            Dictionary of model information
        """
        return {
            name: {
                "config": info["config"],
                "parameter_ratio": self.get_parameter_ratio(info["model"])
            }
            for name, info in self._applied_models.items()
        }
    
    def clear_cache(self):
        """Clear the cache of applied models."""
        self._applied_models.clear()
        logger.info("LoRA adapter cache cleared")