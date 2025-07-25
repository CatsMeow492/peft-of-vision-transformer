"""
Quantization management for memory-efficient model loading using bitsandbytes.
"""

import logging
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

try:
    import psutil
except ImportError:
    psutil = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
else:
    try:
        import torch
        import torch.nn as nn
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb
    except ImportError:
        torch = None
        nn = None
        BitsAndBytesConfig = None
        bnb = None

from .model_info import QuantizationConfig

logger = logging.getLogger(__name__)


class QuantizationManager:
    """Manager for model quantization using bitsandbytes."""
    
    def __init__(self):
        """Initialize quantization manager."""
        self._quantized_models: Dict[str, Dict[str, Any]] = {}
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if bnb is None:
            logger.warning("bitsandbytes not available - quantization features disabled")
        if torch is None:
            logger.warning("PyTorch not available - quantization features disabled")
    
    def create_bnb_config(self, config: QuantizationConfig) -> "BitsAndBytesConfig":
        """
        Create BitsAndBytesConfig for HuggingFace transformers.
        
        Args:
            config: Quantization configuration
            
        Returns:
            BitsAndBytesConfig object
            
        Raises:
            RuntimeError: If required libraries are not available
        """
        if BitsAndBytesConfig is None:
            raise RuntimeError("transformers library not available")
        
        if torch is None:
            raise RuntimeError("PyTorch not available")
        
        # Convert string dtype to torch dtype
        compute_dtype = getattr(torch, config.compute_dtype, torch.float16)
        
        if config.bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=config.quant_type,
                bnb_4bit_use_double_quant=config.double_quant
            )
        elif config.bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype
            )
        else:
            raise ValueError(f"Unsupported quantization bits: {config.bits}")
    
    def quantize_model(self, model, config: QuantizationConfig, model_name: str = "model"):
        """
        Apply quantization to a model using bitsandbytes.
        
        Args:
            model: Model to quantize
            config: Quantization configuration
            model_name: Name for tracking the quantized model
            
        Returns:
            Quantized model
            
        Raises:
            RuntimeError: If quantization fails or dependencies are missing
        """
        if bnb is None:
            raise RuntimeError("bitsandbytes library not available")
        
        if torch is None or nn is None:
            raise RuntimeError("PyTorch not available")
        
        try:
            logger.info(f"Applying {config.bits}-bit quantization to {model_name}")
            
            # Measure memory before quantization
            memory_before = self.measure_memory_usage(model)
            
            # Apply quantization to linear layers
            quantized_model = self._quantize_linear_layers(model, config)
            
            # Measure memory after quantization
            memory_after = self.measure_memory_usage(quantized_model)
            memory_reduction = ((memory_before - memory_after) / memory_before) * 100
            
            # Store quantization info
            self._quantized_models[model_name] = {
                "model": quantized_model,
                "config": config,
                "original_model": model,
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_reduction_percent": memory_reduction
            }
            
            logger.info(f"Quantization complete: {memory_before:.1f}MB → {memory_after:.1f}MB "
                       f"({memory_reduction:.1f}% reduction)")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed for {model_name}: {str(e)}")
            raise RuntimeError(f"Quantization failed: {str(e)}") from e
    
    def _quantize_linear_layers(self, model, config: QuantizationConfig):
        """
        Apply quantization to linear layers in the model.
        
        Args:
            model: Model to quantize
            config: Quantization configuration
            
        Returns:
            Model with quantized linear layers
        """
        quantized_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    # Create quantized layer based on configuration
                    if config.bits == 8:
                        quantized_layer = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False,
                            threshold=6.0
                        )
                    elif config.bits == 4:
                        # Convert compute dtype
                        compute_dtype = getattr(torch, config.compute_dtype, torch.float16)
                        
                        quantized_layer = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            compute_dtype=compute_dtype,
                            quant_type=config.quant_type,
                            use_double_quant=config.double_quant
                        )
                    else:
                        continue  # Skip unsupported bit widths
                    
                    # Copy weights and bias
                    with torch.no_grad():
                        quantized_layer.weight.data = module.weight.data.clone()
                        if module.bias is not None:
                            quantized_layer.bias.data = module.bias.data.clone()
                    
                    # Replace the module
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    
                    if parent_name:
                        parent_module = model.get_submodule(parent_name)
                    else:
                        parent_module = model
                    
                    setattr(parent_module, child_name, quantized_layer)
                    quantized_layers += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to quantize layer {name}: {str(e)}")
                    continue
        
        logger.info(f"Quantized {quantized_layers} linear layers")
        return model
    
    def measure_memory_usage(self, model) -> float:
        """
        Measure memory usage of a model in MB.
        
        Args:
            model: Model to measure
            
        Returns:
            Memory usage in MB
        """
        if model is None:
            return 0.0
        
        try:
            # Method 1: Calculate parameter memory
            param_memory = 0
            buffer_memory = 0
            
            for param in model.parameters():
                param_memory += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_memory += buffer.nelement() * buffer.element_size()
            
            total_memory_bytes = param_memory + buffer_memory
            total_memory_mb = total_memory_bytes / (1024 * 1024)
            
            return total_memory_mb
            
        except Exception as e:
            logger.warning(f"Failed to measure model memory: {str(e)}")
            return 0.0
    
    def measure_system_memory(self) -> Dict[str, float]:
        """
        Measure system memory usage.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        if psutil is None:
            logger.warning("psutil not available - cannot measure system memory")
            return {
                "total_mb": 0.0,
                "available_mb": 0.0,
                "used_mb": 0.0,
                "percent_used": 0.0
            }
        
        try:
            memory = psutil.virtual_memory()
            
            return {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "percent_used": memory.percent
            }
        except Exception as e:
            logger.warning(f"Failed to measure system memory: {str(e)}")
            return {
                "total_mb": 0.0,
                "available_mb": 0.0,
                "used_mb": 0.0,
                "percent_used": 0.0
            }
    
    def verify_quantization(self, original_model, quantized_model, config: QuantizationConfig) -> Dict[str, Any]:
        """
        Verify that quantization was applied correctly.
        
        Args:
            original_model: Original model before quantization
            quantized_model: Model after quantization
            config: Quantization configuration used
            
        Returns:
            Dictionary with verification results
        """
        try:
            verification_results = {
                "quantization_applied": False,
                "quantized_layers": 0,
                "total_layers": 0,
                "memory_reduction_mb": 0.0,
                "memory_reduction_percent": 0.0,
                "errors": []
            }
            
            # Count layers
            original_linear_layers = sum(1 for _, m in original_model.named_modules() if isinstance(m, nn.Linear))
            
            # Count quantized layers
            quantized_layers = 0
            for name, module in quantized_model.named_modules():
                if bnb is not None:
                    if isinstance(module, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)):
                        quantized_layers += 1
            
            verification_results["total_layers"] = original_linear_layers
            verification_results["quantized_layers"] = quantized_layers
            verification_results["quantization_applied"] = quantized_layers > 0
            
            # Measure memory difference
            original_memory = self.measure_memory_usage(original_model)
            quantized_memory = self.measure_memory_usage(quantized_model)
            
            memory_reduction_mb = original_memory - quantized_memory
            memory_reduction_percent = (memory_reduction_mb / original_memory * 100) if original_memory > 0 else 0
            
            verification_results["memory_reduction_mb"] = memory_reduction_mb
            verification_results["memory_reduction_percent"] = memory_reduction_percent
            
            # Validation checks
            if quantized_layers == 0:
                verification_results["errors"].append("No layers were quantized")
            
            if memory_reduction_percent < 10:  # Expect at least 10% reduction
                verification_results["errors"].append(f"Low memory reduction: {memory_reduction_percent:.1f}%")
            
            logger.info(f"Quantization verification: {quantized_layers}/{original_linear_layers} layers quantized, "
                       f"{memory_reduction_percent:.1f}% memory reduction")
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Quantization verification failed: {str(e)}")
            return {
                "quantization_applied": False,
                "errors": [str(e)]
            }
    
    def get_quantization_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a quantized model.
        
        Args:
            model_name: Name of the quantized model
            
        Returns:
            Dictionary with quantization information or None if not found
        """
        return self._quantized_models.get(model_name)
    
    def list_quantized_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all quantized models.
        
        Returns:
            Dictionary of quantized model information
        """
        return {
            name: {
                "config": info["config"],
                "memory_before_mb": info["memory_before_mb"],
                "memory_after_mb": info["memory_after_mb"],
                "memory_reduction_percent": info["memory_reduction_percent"]
            }
            for name, info in self._quantized_models.items()
        }
    
    def estimate_quantization_savings(self, model, config: QuantizationConfig) -> Dict[str, float]:
        """
        Estimate memory savings from quantization without applying it.
        
        Args:
            model: Model to analyze
            config: Quantization configuration
            
        Returns:
            Dictionary with estimated savings
        """
        try:
            current_memory = self.measure_memory_usage(model)
            
            # Count linear layers
            linear_layers = 0
            if nn is not None:
                linear_layers = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
            else:
                # Fallback: count modules with 'linear' in name
                linear_layers = sum(1 for name, _ in model.named_modules() if 'linear' in name.lower())
            
            # Estimate reduction based on bit width
            if config.bits == 8:
                # 8-bit typically reduces memory by ~50%
                estimated_reduction = 0.5
            elif config.bits == 4:
                # 4-bit typically reduces memory by ~75%
                estimated_reduction = 0.75
            else:
                estimated_reduction = 0.0
            
            estimated_memory_after = current_memory * (1 - estimated_reduction)
            estimated_savings_mb = current_memory - estimated_memory_after
            estimated_savings_percent = estimated_reduction * 100
            
            return {
                "current_memory_mb": current_memory,
                "estimated_memory_after_mb": estimated_memory_after,
                "estimated_savings_mb": estimated_savings_mb,
                "estimated_savings_percent": estimated_savings_percent,
                "linear_layers_count": linear_layers
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate quantization savings: {str(e)}")
            return {
                "current_memory_mb": 0.0,
                "estimated_memory_after_mb": 0.0,
                "estimated_savings_mb": 0.0,
                "estimated_savings_percent": 0.0,
                "linear_layers_count": 0
            }
    
    def clear_cache(self):
        """Clear the cache of quantized models."""
        self._quantized_models.clear()
        logger.info("Quantization manager cache cleared")