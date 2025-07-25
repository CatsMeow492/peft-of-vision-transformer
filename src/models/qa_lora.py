"""
QA-LoRA (Quantization-Aware LoRA) implementation for Vision Transformers.

This module implements quantization-aware training for LoRA adapters, extending
QLoRA techniques from NLP to the vision domain with proper gradient scaling
and group-wise quantization operators.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    from peft import PeftModel
else:
    try:
        import torch
        import torch.nn as nn
        from torch.optim import Optimizer
        from peft import PeftModel
    except ImportError:
        torch = None
        nn = None
        Optimizer = None
        PeftModel = None

logger = logging.getLogger(__name__)


@dataclass
class QALoRAConfig:
    """Configuration for QA-LoRA quantization-aware training."""
    
    # Quantization settings
    quantization_bits: int = 4              # 4-bit or 8-bit quantization
    quantization_type: str = "nf4"          # "nf4", "fp4", "int4", "int8"
    double_quantization: bool = True        # Use double quantization
    compute_dtype: str = "float16"          # Compute dtype for quantized operations
    
    # LoRA settings
    lora_rank: int = 8                      # LoRA rank
    lora_alpha: float = 16.0                # LoRA scaling parameter
    lora_dropout: float = 0.1               # LoRA dropout
    
    # QA-LoRA specific settings
    gradient_scaling_factor: float = 1.0    # Gradient scaling for quantized training
    quantization_schedule: str = "constant" # "constant", "linear", "cosine"
    warmup_steps: int = 100                 # Steps before full quantization
    
    # Group-wise quantization
    use_group_quantization: bool = True     # Enable group-wise quantization
    quantization_group_size: int = 64       # Group size for quantization
    
    # Adaptation balance
    quantization_weight: float = 1.0        # Weight for quantization loss
    adaptation_weight: float = 1.0          # Weight for adaptation loss
    balance_schedule: str = "constant"      # "constant", "adaptive"
    
    # Training stability
    gradient_clipping: float = 1.0          # Gradient clipping for stability
    use_stable_embedding: bool = True       # Use stable embedding quantization
    freeze_quantization_after: Optional[int] = None  # Freeze quantization after N steps
    
    def __post_init__(self):
        """Validate QA-LoRA configuration."""
        if self.quantization_bits not in [4, 8]:
            raise ValueError("Quantization bits must be 4 or 8")
        if self.quantization_type not in ["nf4", "fp4", "int4", "int8"]:
            raise ValueError("Invalid quantization type")
        if self.lora_rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("LoRA dropout must be between 0 and 1")
        if self.gradient_scaling_factor <= 0:
            raise ValueError("Gradient scaling factor must be positive")
        if self.quantization_group_size <= 0:
            raise ValueError("Quantization group size must be positive")


@dataclass
class QuantizationState:
    """State tracking for quantization during training."""
    
    step: int = 0
    current_bits: float = 32.0              # Current effective bits
    quantization_ratio: float = 0.0         # Current quantization ratio
    gradient_scale: float = 1.0             # Current gradient scaling
    
    # Statistics
    quantization_error: float = 0.0         # Quantization error
    gradient_norm_before: float = 0.0       # Gradient norm before scaling
    gradient_norm_after: float = 0.0        # Gradient norm after scaling
    
    # Group statistics
    group_quantization_errors: Dict[str, float] = field(default_factory=dict)
    layer_quantization_ratios: Dict[str, float] = field(default_factory=dict)


class GroupWiseQuantizer:
    """Group-wise quantization operator for LoRA matrices."""
    
    def __init__(self, config: QALoRAConfig):
        """
        Initialize group-wise quantizer.
        
        Args:
            config: QA-LoRA configuration
        """
        self.config = config
        self.group_size = config.quantization_group_size
        self.bits = config.quantization_bits
        self.quant_type = config.quantization_type
        
        # Quantization parameters
        if self.bits == 4:
            if self.quant_type == "nf4":
                # NormalFloat4 quantization levels
                self.quant_levels = self._get_nf4_levels()
            elif self.quant_type == "fp4":
                # Float4 quantization levels
                self.quant_levels = self._get_fp4_levels()
            else:  # int4
                self.quant_levels = torch.linspace(-8, 7, 16)
        else:  # 8-bit
            if torch is not None:
                self.quant_levels = torch.linspace(-128, 127, 256)
            else:
                # Create linear levels without torch
                self.quant_levels = [i for i in range(-128, 128)]
    
    def _get_nf4_levels(self):
        """Get NormalFloat4 quantization levels."""
        # NF4 quantization levels optimized for normal distribution
        levels = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ]
        
        if torch is not None:
            return torch.tensor(levels)
        else:
            return levels
    
    def _get_fp4_levels(self):
        """Get Float4 quantization levels."""
        # FP4 quantization levels
        levels = [
            -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0,
            -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0
        ]
        
        if torch is not None:
            return torch.tensor(levels)
        else:
            return levels
    
    def quantize_tensor(self, tensor: "torch.Tensor") -> Tuple["torch.Tensor", Dict[str, Any]]:
        """
        Quantize tensor using group-wise quantization.
        
        Args:
            tensor: Input tensor to quantize
            
        Returns:
            Tuple of (quantized_tensor, quantization_info)
        """
        if torch is None:
            raise RuntimeError("PyTorch not available")
        
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # Pad tensor to be divisible by group size
        padding = (self.group_size - len(tensor_flat) % self.group_size) % self.group_size
        if padding > 0:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(padding, device=tensor.device)])
        
        # Reshape into groups
        groups = tensor_flat.view(-1, self.group_size)
        quantized_groups = []
        group_scales = []
        group_errors = []
        
        # Quantize each group
        for group in groups:
            quantized_group, scale, error = self._quantize_group(group)
            quantized_groups.append(quantized_group)
            group_scales.append(scale)
            group_errors.append(error)
        
        # Reconstruct tensor
        quantized_flat = torch.cat(quantized_groups)
        if padding > 0:
            quantized_flat = quantized_flat[:-padding]
        
        quantized_tensor = quantized_flat.view(original_shape)
        
        quantization_info = {
            "scales": torch.stack(group_scales),
            "group_errors": group_errors,
            "avg_error": sum(group_errors) / len(group_errors),
            "compression_ratio": 32.0 / self.bits
        }
        
        return quantized_tensor, quantization_info
    
    def _quantize_group(self, group: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", float]:
        """Quantize a single group."""
        # Calculate scale factor
        group_max = torch.max(torch.abs(group))
        if group_max == 0:
            return group, torch.tensor(1.0, device=group.device), 0.0
        
        scale = group_max / torch.max(torch.abs(self.quant_levels.to(group.device)))
        
        # Quantize
        normalized = group / scale
        quantized_indices = torch.searchsorted(
            self.quant_levels.to(group.device), normalized
        )
        quantized_indices = torch.clamp(quantized_indices, 0, len(self.quant_levels) - 1)
        
        quantized_values = self.quant_levels.to(group.device)[quantized_indices]
        quantized_group = quantized_values * scale
        
        # Calculate quantization error
        error = torch.mean((group - quantized_group) ** 2).item()
        
        return quantized_group, scale, error
    
    def dequantize_tensor(self, quantized_tensor: "torch.Tensor", 
                         quantization_info: Dict[str, Any]) -> "torch.Tensor":
        """
        Dequantize tensor (for validation purposes).
        
        Args:
            quantized_tensor: Quantized tensor
            quantization_info: Quantization information
            
        Returns:
            Dequantized tensor
        """
        # In practice, this would reconstruct the original precision
        # For now, return the quantized tensor as it's already dequantized
        return quantized_tensor


class QALoRATrainer:
    """Quantization-Aware LoRA trainer for Vision Transformers."""
    
    def __init__(self, config: QALoRAConfig):
        """
        Initialize QA-LoRA trainer.
        
        Args:
            config: QA-LoRA configuration
        """
        self.config = config
        self.quantizer = GroupWiseQuantizer(config)
        self.quantization_state = QuantizationState()
        
        # Training state
        self.step_count = 0
        self.quantization_frozen = False
        
        # LoRA layers tracking
        self.lora_layers: Dict[str, Any] = {}
        
        # Statistics tracking
        self.quantization_history: List[QuantizationState] = []
        self.layer_statistics: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"QA-LoRA trainer initialized with {config.quantization_bits}-bit quantization")
    
    def setup_model(self, model: "PeftModel") -> "PeftModel":
        """
        Set up model for QA-LoRA training.
        
        Args:
            model: PEFT model with LoRA adapters
            
        Returns:
            Model configured for QA-LoRA training
        """
        if torch is None or PeftModel is None:
            raise RuntimeError("PyTorch and PEFT not available")
        
        # Find LoRA layers
        self.lora_layers = {}
        for name, module in model.named_modules():
            if "lora" in name.lower() and hasattr(module, 'weight'):
                self.lora_layers[name] = module
        
        if not self.lora_layers:
            raise ValueError("No LoRA layers found in model")
        
        logger.info(f"Found {len(self.lora_layers)} LoRA layers for QA-LoRA training")
        
        # Apply initial quantization if needed
        if self.config.warmup_steps == 0:
            self._apply_quantization_to_model(model)
        
        return model
    
    def training_step(self, model: "PeftModel", optimizer: "Optimizer", 
                     loss: "torch.Tensor", step: int) -> Dict[str, Any]:
        """
        Perform QA-LoRA training step with quantization-aware updates.
        
        Args:
            model: PEFT model
            optimizer: Optimizer
            loss: Training loss
            step: Current training step
            
        Returns:
            Dictionary with training statistics
        """
        if torch is None:
            raise RuntimeError("PyTorch not available")
        
        self.step_count = step
        
        # Update quantization schedule
        self._update_quantization_schedule(step)
        
        # Apply quantization to LoRA layers if not frozen
        if not self.quantization_frozen:
            quantization_stats = self._apply_quantization_to_model(model)
        else:
            quantization_stats = {}
        
        # Compute gradients
        loss.backward()
        
        # Apply gradient scaling for quantized training
        self._apply_gradient_scaling(model)
        
        # Gradient clipping for stability
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config.gradient_clipping
            )
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update statistics
        self._update_statistics(model, quantization_stats)
        
        # Check if quantization should be frozen
        if (self.config.freeze_quantization_after is not None and 
            step >= self.config.freeze_quantization_after):
            self.quantization_frozen = True
        
        return {
            "quantization_ratio": self.quantization_state.quantization_ratio,
            "gradient_scale": self.quantization_state.gradient_scale,
            "quantization_error": self.quantization_state.quantization_error,
            "effective_bits": self.quantization_state.current_bits
        }
    
    def _update_quantization_schedule(self, step: int) -> None:
        """Update quantization parameters based on schedule."""
        if step < self.config.warmup_steps:
            # Gradual quantization during warmup
            progress = step / self.config.warmup_steps
            self.quantization_state.quantization_ratio = progress
        else:
            # Full quantization after warmup
            if self.config.quantization_schedule == "constant":
                self.quantization_state.quantization_ratio = 1.0
            elif self.config.quantization_schedule == "linear":
                # Linear increase in quantization
                max_steps = self.config.warmup_steps * 2
                progress = min(1.0, (step - self.config.warmup_steps) / max_steps)
                self.quantization_state.quantization_ratio = progress
            elif self.config.quantization_schedule == "cosine":
                # Cosine schedule for quantization
                max_steps = self.config.warmup_steps * 2
                progress = min(1.0, (step - self.config.warmup_steps) / max_steps)
                self.quantization_state.quantization_ratio = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        
        # Update effective bits
        full_bits = 32.0
        target_bits = float(self.config.quantization_bits)
        self.quantization_state.current_bits = (
            full_bits - (full_bits - target_bits) * self.quantization_state.quantization_ratio
        )
        
        # Update gradient scaling
        self.quantization_state.gradient_scale = (
            self.config.gradient_scaling_factor * 
            (1.0 + self.quantization_state.quantization_ratio)
        )
    
    def _apply_quantization_to_model(self, model: "PeftModel") -> Dict[str, Any]:
        """Apply quantization to LoRA layers in the model."""
        quantization_stats = {}
        total_error = 0.0
        
        for layer_name, module in self.lora_layers.items():
            if hasattr(module, 'weight') and module.weight.requires_grad:
                # Apply group-wise quantization
                original_weight = module.weight.data.clone()
                quantized_weight, quant_info = self.quantizer.quantize_tensor(original_weight)
                
                # Apply quantization with ratio
                mixed_weight = (
                    self.quantization_state.quantization_ratio * quantized_weight +
                    (1 - self.quantization_state.quantization_ratio) * original_weight
                )
                
                module.weight.data = mixed_weight
                
                # Track statistics
                quantization_stats[layer_name] = {
                    "quantization_error": quant_info["avg_error"],
                    "compression_ratio": quant_info["compression_ratio"]
                }
                
                total_error += quant_info["avg_error"]
        
        self.quantization_state.quantization_error = total_error / len(self.lora_layers)
        return quantization_stats
    
    def _apply_gradient_scaling(self, model: "PeftModel") -> None:
        """Apply gradient scaling for quantized training."""
        scale_factor = self.quantization_state.gradient_scale
        
        # Calculate gradient norms before scaling
        grad_norm_before = 0.0
        for name, module in self.lora_layers.items():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad_norm_before += torch.norm(module.weight.grad).item() ** 2
        
        self.quantization_state.gradient_norm_before = math.sqrt(grad_norm_before)
        
        # Apply scaling to LoRA layer gradients
        for name, module in self.lora_layers.items():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                module.weight.grad *= scale_factor
        
        # Calculate gradient norms after scaling
        grad_norm_after = 0.0
        for name, module in self.lora_layers.items():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad_norm_after += torch.norm(module.weight.grad).item() ** 2
        
        self.quantization_state.gradient_norm_after = math.sqrt(grad_norm_after)
    
    def _update_statistics(self, model: "PeftModel", quantization_stats: Dict[str, Any]) -> None:
        """Update training statistics."""
        # Update quantization state
        self.quantization_state.step = self.step_count
        
        # Store layer statistics
        for layer_name, stats in quantization_stats.items():
            if layer_name not in self.layer_statistics:
                self.layer_statistics[layer_name] = {
                    "avg_quantization_error": 0.0,
                    "avg_compression_ratio": 0.0,
                    "update_count": 0
                }
            
            layer_stats = self.layer_statistics[layer_name]
            layer_stats["update_count"] += 1
            
            # Running average
            alpha = 1.0 / layer_stats["update_count"]
            layer_stats["avg_quantization_error"] = (
                (1 - alpha) * layer_stats["avg_quantization_error"] +
                alpha * stats["quantization_error"]
            )
            layer_stats["avg_compression_ratio"] = (
                (1 - alpha) * layer_stats["avg_compression_ratio"] +
                alpha * stats["compression_ratio"]
            )
        
        # Store history
        if self.config.quantization_schedule != "constant" or len(self.quantization_history) < 100:
            self.quantization_history.append(QuantizationState(
                step=self.quantization_state.step,
                current_bits=self.quantization_state.current_bits,
                quantization_ratio=self.quantization_state.quantization_ratio,
                gradient_scale=self.quantization_state.gradient_scale,
                quantization_error=self.quantization_state.quantization_error,
                gradient_norm_before=self.quantization_state.gradient_norm_before,
                gradient_norm_after=self.quantization_state.gradient_norm_after
            ))
    
    def validate_quantization_adaptation_balance(self, model: "PeftModel") -> Dict[str, float]:
        """
        Validate the balance between quantization and adaptation.
        
        Args:
            model: PEFT model to validate
            
        Returns:
            Dictionary with validation metrics
        """
        if torch is None:
            raise RuntimeError("PyTorch not available")
        
        validation_metrics = {}
        
        # Calculate quantization impact
        total_quantization_error = 0.0
        total_parameters = 0
        
        for layer_name, module in self.lora_layers.items():
            if hasattr(module, 'weight'):
                # Simulate full quantization
                original_weight = module.weight.data.clone()
                quantized_weight, quant_info = self.quantizer.quantize_tensor(original_weight)
                
                # Calculate error
                error = torch.mean((original_weight - quantized_weight) ** 2).item()
                total_quantization_error += error * module.weight.numel()
                total_parameters += module.weight.numel()
        
        avg_quantization_error = total_quantization_error / total_parameters if total_parameters > 0 else 0.0
        
        # Calculate adaptation effectiveness
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        adaptation_ratio = trainable_params / total_params if total_params > 0 else 0.0
        
        # Balance metrics
        validation_metrics = {
            "avg_quantization_error": avg_quantization_error,
            "adaptation_ratio": adaptation_ratio,
            "balance_score": adaptation_ratio / (1.0 + avg_quantization_error),
            "effective_compression": 32.0 / self.quantization_state.current_bits,
            "gradient_stability": (
                self.quantization_state.gradient_norm_after / 
                max(self.quantization_state.gradient_norm_before, 1e-8)
            )
        }
        
        return validation_metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            "config": {
                "quantization_bits": self.config.quantization_bits,
                "quantization_type": self.config.quantization_type,
                "lora_rank": self.config.lora_rank,
                "gradient_scaling_factor": self.config.gradient_scaling_factor,
                "use_group_quantization": self.config.use_group_quantization
            },
            "current_state": {
                "step": self.quantization_state.step,
                "current_bits": self.quantization_state.current_bits,
                "quantization_ratio": self.quantization_state.quantization_ratio,
                "gradient_scale": self.quantization_state.gradient_scale,
                "quantization_error": self.quantization_state.quantization_error,
                "quantization_frozen": self.quantization_frozen
            },
            "layer_statistics": self.layer_statistics.copy(),
            "num_lora_layers": len(self.lora_layers),
            "training_history_length": len(self.quantization_history)
        }
    
    def export_quantization_data(self) -> Dict[str, Any]:
        """Export quantization training data for analysis."""
        return {
            "config": self.config,
            "quantization_history": [
                {
                    "step": state.step,
                    "current_bits": state.current_bits,
                    "quantization_ratio": state.quantization_ratio,
                    "gradient_scale": state.gradient_scale,
                    "quantization_error": state.quantization_error,
                    "gradient_norm_before": state.gradient_norm_before,
                    "gradient_norm_after": state.gradient_norm_after
                }
                for state in self.quantization_history
            ],
            "layer_statistics": self.layer_statistics.copy(),
            "final_state": {
                "quantization_frozen": self.quantization_frozen,
                "final_bits": self.quantization_state.current_bits,
                "final_error": self.quantization_state.quantization_error
            }
        }
    
    def visualize_quantization_progress(self, save_path: Optional[str] = None) -> Optional[Any]:
        """
        Visualize quantization training progress.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure if available, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not self.quantization_history:
                logger.warning("No quantization history available")
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract data
            steps = [state.step for state in self.quantization_history]
            bits = [state.current_bits for state in self.quantization_history]
            ratios = [state.quantization_ratio for state in self.quantization_history]
            errors = [state.quantization_error for state in self.quantization_history]
            grad_scales = [state.gradient_scale for state in self.quantization_history]
            
            # Plot effective bits over time
            ax1.plot(steps, bits, 'b-', linewidth=2)
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Effective Bits")
            ax1.set_title("Quantization Bits Evolution")
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=self.config.quantization_bits, color='r', linestyle='--', 
                       label=f'Target ({self.config.quantization_bits}-bit)')
            ax1.legend()
            
            # Plot quantization ratio
            ax2.plot(steps, ratios, 'g-', linewidth=2)
            ax2.set_xlabel("Training Step")
            ax2.set_ylabel("Quantization Ratio")
            ax2.set_title("Quantization Ratio Schedule")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            # Plot quantization error
            ax3.plot(steps, errors, 'r-', linewidth=2)
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Quantization Error")
            ax3.set_title("Quantization Error Over Time")
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Plot gradient scaling
            ax4.plot(steps, grad_scales, 'm-', linewidth=2)
            ax4.set_xlabel("Training Step")
            ax4.set_ylabel("Gradient Scale Factor")
            ax4.set_title("Gradient Scaling Evolution")
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Quantization progress plot saved to {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return None
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            return None
    
    def reset(self) -> None:
        """Reset trainer state."""
        self.step_count = 0
        self.quantization_frozen = False
        self.quantization_state = QuantizationState()
        self.quantization_history.clear()
        self.layer_statistics.clear()
        self.lora_layers.clear()
        
        logger.info("QA-LoRA trainer reset")


class QALoRAIntegratedTrainer:
    """Integrated trainer combining QA-LoRA with standard PEFT training."""
    
    def __init__(self, qa_lora_config: QALoRAConfig, base_trainer):
        """
        Initialize integrated QA-LoRA trainer.
        
        Args:
            qa_lora_config: QA-LoRA configuration
            base_trainer: Base PEFT trainer to extend
        """
        self.qa_lora_trainer = QALoRATrainer(qa_lora_config)
        self.base_trainer = base_trainer
        self.config = qa_lora_config
        
        logger.info("QA-LoRA integrated trainer initialized")
    
    def setup_model(self, model):
        """Set up model for integrated QA-LoRA training."""
        # Set up QA-LoRA
        model = self.qa_lora_trainer.setup_model(model)
        
        # Apply any base trainer setup
        if hasattr(self.base_trainer, 'setup_model'):
            model = self.base_trainer.setup_model(model)
        
        return model
    
    def training_step(self, model, optimizer, loss, step):
        """Perform integrated training step."""
        # QA-LoRA training step
        qa_stats = self.qa_lora_trainer.training_step(model, optimizer, loss, step)
        
        # Base trainer step (if applicable)
        base_stats = {}
        if hasattr(self.base_trainer, 'training_step'):
            base_stats = self.base_trainer.training_step(model, optimizer, loss, step)
        
        # Combine statistics
        combined_stats = {**base_stats, **qa_stats}
        combined_stats["qa_lora_active"] = True
        
        return combined_stats
    
    def validate(self, model):
        """Perform validation with QA-LoRA metrics."""
        # QA-LoRA validation
        qa_validation = self.qa_lora_trainer.validate_quantization_adaptation_balance(model)
        
        # Base validation (if applicable)
        base_validation = {}
        if hasattr(self.base_trainer, 'validate'):
            base_validation = self.base_trainer.validate(model)
        
        return {**base_validation, **qa_validation}
    
    def get_training_summary(self):
        """Get comprehensive training summary."""
        qa_summary = self.qa_lora_trainer.get_training_summary()
        
        base_summary = {}
        if hasattr(self.base_trainer, 'get_training_summary'):
            base_summary = self.base_trainer.get_training_summary()
        
        return {
            "qa_lora": qa_summary,
            "base_trainer": base_summary,
            "integration_active": True
        }