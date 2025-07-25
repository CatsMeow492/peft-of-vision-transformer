"""
Data models for model information and configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn
else:
    try:
        import torch.nn as nn
    except ImportError:
        nn = None


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    
    name: str
    total_params: int
    trainable_params: int
    trainable_ratio: float
    model_size_mb: float
    architecture: str
    input_size: tuple
    num_classes: int
    attention_layers: List[str]
    
    def __post_init__(self):
        """Validate model info after initialization."""
        if self.total_params <= 0:
            raise ValueError("Total parameters must be positive")
        if not 0 <= self.trainable_ratio <= 1:
            raise ValueError("Trainable ratio must be between 0 and 1")
        if self.model_size_mb <= 0:
            raise ValueError("Model size must be positive")


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    bits: int = 8
    compute_dtype: str = "float16"  # Will be converted to torch.dtype
    quant_type: str = "nf4"
    double_quant: bool = True
    
    def __post_init__(self):
        """Validate quantization configuration."""
        if self.bits not in [4, 8]:
            raise ValueError("Only 4-bit and 8-bit quantization supported")
        if self.compute_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError("Unsupported compute dtype")
        if self.quant_type not in ["fp4", "nf4"]:
            raise ValueError("Quantization type must be 'fp4' or 'nf4'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "bits": self.bits,
            "compute_dtype": self.compute_dtype,
            "quant_type": self.quant_type,
            "double_quant": self.double_quant
        }


def count_parameters(model) -> tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def find_attention_layers(model, model_name: str) -> List[str]:
    """
    Find attention layers in a Vision Transformer model.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for architecture-specific handling
        
    Returns:
        List of attention layer names suitable for LoRA targeting
    """
    attention_layers = []
    
    # Common ViT attention layer patterns
    attention_patterns = [
        "attn.qkv",  # timm models
        "attention.query",  # HuggingFace models
        "attention.key",
        "attention.value",
        "self_attention.query",
        "self_attention.key", 
        "self_attention.value"
    ]
    
    for name, module in model.named_modules():
        for pattern in attention_patterns:
            if pattern in name and hasattr(module, 'weight'):
                attention_layers.append(name)
                break
    
    # If no specific patterns found, look for linear layers in attention blocks
    if not attention_layers and nn is not None:
        for name, module in model.named_modules():
            if ("attn" in name.lower() or "attention" in name.lower()) and isinstance(module, nn.Linear):
                attention_layers.append(name)
    
    return sorted(list(set(attention_layers)))