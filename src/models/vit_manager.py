"""
Vision Transformer model management for loading and configuration.
"""

import logging
from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
    import timm
else:
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
        import timm
    except ImportError:
        torch = None
        nn = None
        AutoModel = None
        AutoConfig = None
        BitsAndBytesConfig = None
        timm = None

from .model_info import ModelInfo, QuantizationConfig, count_parameters, get_model_size_mb, find_attention_layers

logger = logging.getLogger(__name__)


class ViTModelManager:
    """
    Manager class for loading and configuring Vision Transformer models.
    
    Supports both timm and HuggingFace model sources with quantization options.
    """
    
    # Supported model configurations
    SUPPORTED_MODELS = {
        "deit_tiny_patch16_224": {
            "source": "timm",
            "params": "5M",
            "input_size": (224, 224),
            "description": "DeiT-tiny with 16x16 patches"
        },
        "deit_small_patch16_224": {
            "source": "timm", 
            "params": "22M",
            "input_size": (224, 224),
            "description": "DeiT-small with 16x16 patches"
        },
        "vit_small_patch16_224": {
            "source": "timm",
            "params": "20M", 
            "input_size": (224, 224),
            "description": "ViT-small with 16x16 patches"
        },
        "google/vit-base-patch16-224": {
            "source": "huggingface",
            "params": "86M",
            "input_size": (224, 224),
            "description": "ViT-base from HuggingFace"
        }
    }
    
    def __init__(self):
        """Initialize the ViT model manager."""
        self._loaded_models: Dict[str, object] = {}
        
    def load_model(
        self,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        quantization_config: Optional[QuantizationConfig] = None
    ):
        """
        Load a Vision Transformer model with optional quantization.
        
        Args:
            model_name: Name of the model to load
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
            quantization_config: Optional quantization configuration
            
        Returns:
            Loaded PyTorch model
            
        Raises:
            ValueError: If model name is not supported
            RuntimeError: If model loading fails
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        model_config = self.SUPPORTED_MODELS[model_name]
        cache_key = f"{model_name}_{num_classes}_{pretrained}_{quantization_config}"
        
        # Return cached model if available
        if cache_key in self._loaded_models:
            logger.info(f"Returning cached model: {model_name}")
            return self._loaded_models[cache_key]
        
        try:
            if model_config["source"] == "timm":
                model = self._load_timm_model(model_name, num_classes, pretrained, quantization_config)
            else:  # huggingface
                model = self._load_huggingface_model(model_name, num_classes, pretrained, quantization_config)
            
            # Cache the loaded model
            self._loaded_models[cache_key] = model
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def _load_timm_model(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        quantization_config: Optional[QuantizationConfig]
    ):
        """Load model from timm library."""
        logger.info(f"Loading timm model: {model_name}")
        
        # Load model from timm
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Apply quantization if specified
        if quantization_config is not None:
            model = self._apply_quantization(model, quantization_config)
        
        return model
    
    def _load_huggingface_model(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        quantization_config: Optional[QuantizationConfig]
    ):
        """Load model from HuggingFace transformers."""
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        # Prepare quantization config for HuggingFace
        bnb_config = None
        if quantization_config is not None:
            bnb_config = self._create_bnb_config(quantization_config)
        
        # Load model configuration
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        
        # Load model with optional quantization
        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if quantization_config else None
        )
        
        return model
    
    def _apply_quantization(self, model, config: QuantizationConfig):
        """Apply quantization to a timm model."""
        try:
            import bitsandbytes as bnb
            
            # Convert linear layers to quantized versions
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if config.bits == 8:
                        quantized_layer = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False
                        )
                    else:  # 4-bit
                        quantized_layer = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            compute_dtype=getattr(torch, config.compute_dtype),
                            quant_type=config.quant_type
                        )
                    
                    # Copy weights and bias
                    with torch.no_grad():
                        quantized_layer.weight.data = module.weight.data
                        if module.bias is not None:
                            quantized_layer.bias.data = module.bias.data
                    
                    # Replace the module
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent_module = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent_module, child_name, quantized_layer)
            
            logger.info(f"Applied {config.bits}-bit quantization to model")
            return model
            
        except ImportError:
            raise RuntimeError("bitsandbytes not available for quantization")
    
    def _create_bnb_config(self, config: QuantizationConfig) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig for HuggingFace models."""
        return BitsAndBytesConfig(
            load_in_4bit=(config.bits == 4),
            load_in_8bit=(config.bits == 8),
            bnb_4bit_compute_dtype=getattr(torch, config.compute_dtype),
            bnb_4bit_quant_type=config.quant_type,
            bnb_4bit_use_double_quant=config.double_quant
        )
    
    def get_model_info(self, model, model_name: str) -> ModelInfo:
        """
        Get comprehensive information about a loaded model.
        
        Args:
            model: Loaded PyTorch model
            model_name: Name of the model
            
        Returns:
            ModelInfo object with model statistics
        """
        total_params, trainable_params = count_parameters(model)
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
        model_size_mb = get_model_size_mb(model)
        attention_layers = find_attention_layers(model, model_name)
        
        # Get model configuration
        model_config = self.SUPPORTED_MODELS.get(model_name, {})
        input_size = model_config.get("input_size", (224, 224))
        
        # Determine number of classes
        num_classes = 1000  # default
        if hasattr(model, 'head') and hasattr(model.head, 'out_features'):
            num_classes = model.head.out_features
        elif hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
            num_classes = model.classifier.out_features
        elif hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        
        return ModelInfo(
            name=model_name,
            total_params=total_params,
            trainable_params=trainable_params,
            trainable_ratio=trainable_ratio,
            model_size_mb=model_size_mb,
            architecture=model.__class__.__name__,
            input_size=input_size,
            num_classes=num_classes,
            attention_layers=attention_layers
        )
    
    def validate_model(self, model, model_name: str) -> bool:
        """
        Validate that a model is properly loaded and configured.
        
        Args:
            model: Model to validate
            model_name: Name of the model
            
        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Check if model is in evaluation mode initially
            model.eval()
            
            # Get model info to validate structure
            model_info = self.get_model_info(model, model_name)
            
            # Basic validation checks
            if model_info.total_params <= 0:
                logger.error("Model has no parameters")
                return False
            
            if not model_info.attention_layers:
                logger.warning("No attention layers found - may affect LoRA application")
            
            # Test forward pass with dummy input
            model_config = self.SUPPORTED_MODELS.get(model_name, {})
            input_size = model_config.get("input_size", (224, 224))
            
            dummy_input = torch.randn(1, 3, *input_size)
            
            with torch.no_grad():
                try:
                    output = model(dummy_input)
                    if output is None:
                        logger.error("Model forward pass returned None")
                        return False
                except Exception as e:
                    logger.error(f"Model forward pass failed: {str(e)}")
                    return False
            
            logger.info(f"Model validation passed for {model_name}")
            logger.info(f"Model info: {model_info.total_params:,} params, "
                       f"{model_info.model_size_mb:.2f} MB, "
                       f"{len(model_info.attention_layers)} attention layers")
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def list_supported_models(self) -> Dict[str, Dict]:
        """
        Get information about all supported models.
        
        Returns:
            Dictionary of supported models and their configurations
        """
        return self.SUPPORTED_MODELS.copy()
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        self._loaded_models.clear()
        logger.info("Model cache cleared")