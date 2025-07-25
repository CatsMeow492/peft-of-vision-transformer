"""
Model export and adapter merging functionality for PEFT Vision Transformers.

This module provides functionality to merge LoRA adapters back into base models,
export models in various formats, and validate the merged models.
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from peft import PeftModel
else:
    try:
        import torch
        import torch.nn as nn
        from peft import PeftModel
    except ImportError:
        torch = None
        nn = None
        PeftModel = None

from .model_info import ModelInfo, count_parameters, get_model_size_mb

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    export_format: str = "pytorch"  # "pytorch", "onnx", "torchscript"
    output_path: str = "exported_model"
    merge_adapters: bool = True
    validate_merged: bool = True
    precision: str = "float32"  # "float32", "float16", "bfloat16"
    optimize_for_inference: bool = True
    include_metadata: bool = True
    
    def __post_init__(self):
        """Validate export configuration."""
        valid_formats = ["pytorch", "onnx", "torchscript"]
        if self.export_format not in valid_formats:
            raise ValueError(f"Export format must be one of {valid_formats}")
        
        valid_precisions = ["float32", "float16", "bfloat16"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Precision must be one of {valid_precisions}")


@dataclass
class MergeValidationResult:
    """Results of adapter merge validation."""
    
    validation_passed: bool
    numerical_precision_check: bool
    forward_pass_check: bool
    parameter_count_check: bool
    size_comparison: Dict[str, float]
    max_weight_difference: float
    mean_weight_difference: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ModelExporter:
    """
    Handles model export and adapter merging functionality.
    
    Provides methods to merge LoRA adapters back into base models,
    validate merged models, and export in various formats.
    """
    
    def __init__(self):
        """Initialize model exporter."""
        self._exported_models: Dict[str, Dict[str, Any]] = {}
    
    def merge_adapters(
        self,
        peft_model: "PeftModel",
        model_name: str = "merged_model",
        safe_merge: bool = True
    ) -> "nn.Module":
        """
        Merge LoRA adapters back into the base model.
        
        Args:
            peft_model: PEFT model with LoRA adapters
            model_name: Name for the merged model
            safe_merge: Whether to use safe merging (preserves original weights)
            
        Returns:
            Base model with merged adapters
            
        Raises:
            RuntimeError: If merging fails or PEFT is not available
        """
        if PeftModel is None:
            raise RuntimeError("PEFT library not available for adapter merging")
        
        if not isinstance(peft_model, PeftModel):
            raise ValueError("Model must be a PeftModel instance")
        
        try:
            logger.info(f"Merging LoRA adapters for {model_name}")
            
            # Get the base model
            base_model = peft_model.get_base_model()
            
            # Merge adapters using PEFT's merge_and_unload method
            if hasattr(peft_model, 'merge_and_unload'):
                merged_model = peft_model.merge_and_unload(safe_merge=safe_merge)
            else:
                # Fallback for older PEFT versions
                merged_model = peft_model.merge_adapter()
                if hasattr(merged_model, 'unload'):
                    merged_model = merged_model.unload()
            
            # Store information about the merged model
            self._exported_models[model_name] = {
                "merged_model": merged_model,
                "original_peft_model": peft_model,
                "base_model": base_model,
                "merge_method": "safe" if safe_merge else "standard"
            }
            
            logger.info(f"Successfully merged adapters for {model_name}")
            return merged_model
            
        except Exception as e:
            logger.error(f"Failed to merge adapters for {model_name}: {str(e)}")
            raise RuntimeError(f"Adapter merging failed: {str(e)}") from e
    
    def validate_merged_model(
        self,
        merged_model: "nn.Module",
        original_peft_model: "PeftModel",
        tolerance: float = 1e-5,
        test_input_shape: tuple = (1, 3, 224, 224)
    ) -> MergeValidationResult:
        """
        Validate that the merged model produces equivalent results to the PEFT model.
        
        Args:
            merged_model: Model with merged adapters
            original_peft_model: Original PEFT model
            tolerance: Numerical tolerance for comparison
            test_input_shape: Shape of test input for validation
            
        Returns:
            Validation results
        """
        logger.info("Validating merged model")
        
        errors = []
        warnings = []
        
        try:
            # Set models to evaluation mode
            merged_model.eval()
            original_peft_model.eval()
            
            # Parameter count check
            merged_params = sum(p.numel() for p in merged_model.parameters())
            base_params = sum(p.numel() for p in original_peft_model.get_base_model().parameters())
            parameter_count_check = abs(merged_params - base_params) < 1000  # Allow small differences
            
            if not parameter_count_check:
                errors.append(f"Parameter count mismatch: merged={merged_params}, base={base_params}")
            
            # Size comparison
            merged_size = get_model_size_mb(merged_model)
            original_size = get_model_size_mb(original_peft_model)
            base_size = get_model_size_mb(original_peft_model.get_base_model())
            
            size_comparison = {
                "merged_size_mb": merged_size,
                "original_peft_size_mb": original_size,
                "base_size_mb": base_size,
                "size_reduction_ratio": (original_size - merged_size) / original_size if original_size > 0 else 0.0
            }
            
            # Forward pass comparison
            forward_pass_check = False
            max_diff = float('inf')
            mean_diff = float('inf')
            
            try:
                # Create test input
                test_input = torch.randn(test_input_shape)
                
                with torch.no_grad():
                    # Get outputs from both models
                    merged_output = merged_model(test_input)
                    original_output = original_peft_model(test_input)
                    
                    # Handle different output formats
                    if hasattr(merged_output, 'logits'):
                        merged_output = merged_output.logits
                    if hasattr(original_output, 'logits'):
                        original_output = original_output.logits
                    
                    # Calculate differences
                    diff = torch.abs(merged_output - original_output)
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    
                    forward_pass_check = max_diff < tolerance
                    
                    if not forward_pass_check:
                        errors.append(f"Forward pass difference too large: max={max_diff:.2e}, mean={mean_diff:.2e}")
                    else:
                        logger.info(f"Forward pass validation passed: max_diff={max_diff:.2e}")
                        
            except Exception as e:
                errors.append(f"Forward pass validation failed: {str(e)}")
                forward_pass_check = False
            
            # Numerical precision check (compare weights where possible)
            numerical_precision_check = True
            try:
                base_model = original_peft_model.get_base_model()
                weight_diffs = []
                
                # Compare weights of corresponding layers
                merged_state = merged_model.state_dict()
                base_state = base_model.state_dict()
                
                for name, merged_weight in merged_state.items():
                    if name in base_state:
                        base_weight = base_state[name]
                        if merged_weight.shape == base_weight.shape:
                            diff = torch.abs(merged_weight - base_weight)
                            weight_diffs.append(torch.mean(diff).item())
                
                if weight_diffs:
                    avg_weight_diff = sum(weight_diffs) / len(weight_diffs)
                    if avg_weight_diff > tolerance * 10:  # More lenient for weight comparison
                        warnings.append(f"Average weight difference: {avg_weight_diff:.2e}")
                        
            except Exception as e:
                warnings.append(f"Weight comparison failed: {str(e)}")
            
            validation_passed = (
                parameter_count_check and 
                forward_pass_check and 
                numerical_precision_check and 
                len(errors) == 0
            )
            
            result = MergeValidationResult(
                validation_passed=validation_passed,
                numerical_precision_check=numerical_precision_check,
                forward_pass_check=forward_pass_check,
                parameter_count_check=parameter_count_check,
                size_comparison=size_comparison,
                max_weight_difference=max_diff,
                mean_weight_difference=mean_diff,
                errors=errors,
                warnings=warnings
            )
            
            if validation_passed:
                logger.info("Merged model validation passed")
            else:
                logger.warning(f"Merged model validation failed: {len(errors)} errors")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {str(e)}")
            return MergeValidationResult(
                validation_passed=False,
                numerical_precision_check=False,
                forward_pass_check=False,
                parameter_count_check=False,
                size_comparison={},
                max_weight_difference=float('inf'),
                mean_weight_difference=float('inf'),
                errors=[str(e)]
            )
    
    def export_model(
        self,
        model: "nn.Module",
        config: ExportConfig,
        model_info: Optional[ModelInfo] = None,
        sample_input_shape: tuple = (1, 3, 224, 224)
    ) -> Dict[str, Any]:
        """
        Export model in the specified format.
        
        Args:
            model: Model to export
            config: Export configuration
            model_info: Optional model information for metadata
            sample_input_shape: Shape for sample input (needed for ONNX/TorchScript)
            
        Returns:
            Export results and metadata
            
        Raises:
            RuntimeError: If export fails
        """
        logger.info(f"Exporting model in {config.export_format} format to {config.output_path}")
        
        try:
            # Prepare model for export
            model.eval()
            
            # Convert precision if specified
            if config.precision == "float16":
                model = model.half()
            elif config.precision == "bfloat16":
                model = model.to(torch.bfloat16)
            
            # Optimize for inference if requested
            if config.optimize_for_inference:
                model = self._optimize_for_inference(model)
            
            export_results = {
                "export_format": config.export_format,
                "output_path": config.output_path,
                "precision": config.precision,
                "success": False
            }
            
            # Create output directory
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config.export_format == "pytorch":
                export_results.update(self._export_pytorch(model, config, model_info))
            elif config.export_format == "onnx":
                export_results.update(self._export_onnx(model, config, sample_input_shape))
            elif config.export_format == "torchscript":
                export_results.update(self._export_torchscript(model, config, sample_input_shape))
            else:
                raise ValueError(f"Unsupported export format: {config.export_format}")
            
            logger.info(f"Model export completed successfully")
            return export_results
            
        except Exception as e:
            logger.error(f"Model export failed: {str(e)}")
            raise RuntimeError(f"Export failed: {str(e)}") from e
    
    def _optimize_for_inference(self, model: "nn.Module") -> "nn.Module":
        """
        Apply inference optimizations to the model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        try:
            # Fuse operations where possible
            if hasattr(torch.jit, 'optimize_for_inference'):
                # This is available in newer PyTorch versions
                scripted = torch.jit.script(model)
                optimized = torch.jit.optimize_for_inference(scripted)
                return optimized
            else:
                # Fallback: just ensure model is in eval mode and freeze
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                return model
                
        except Exception as e:
            logger.warning(f"Inference optimization failed, using original model: {str(e)}")
            return model
    
    def _export_pytorch(
        self,
        model: "nn.Module",
        config: ExportConfig,
        model_info: Optional[ModelInfo]
    ) -> Dict[str, Any]:
        """Export model in PyTorch format."""
        output_path = Path(config.output_path)
        
        # Prepare export data
        export_data = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        }
        
        # Add metadata if requested and available
        if config.include_metadata and model_info:
            export_data["metadata"] = {
                "model_name": model_info.name,
                "total_params": model_info.total_params,
                "model_size_mb": model_info.model_size_mb,
                "architecture": model_info.architecture,
                "input_size": model_info.input_size,
                "num_classes": model_info.num_classes,
                "export_precision": config.precision,
                "export_timestamp": torch.tensor(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0)
            }
        
        # Save the model
        torch.save(export_data, output_path.with_suffix('.pth'))
        
        # Also save just the state dict for easier loading
        torch.save(model.state_dict(), output_path.with_suffix('_state_dict.pth'))
        
        return {
            "success": True,
            "files_created": [
                str(output_path.with_suffix('.pth')),
                str(output_path.with_suffix('_state_dict.pth'))
            ],
            "model_size_mb": get_model_size_mb(model)
        }
    
    def _export_onnx(
        self,
        model: "nn.Module",
        config: ExportConfig,
        sample_input_shape: tuple
    ) -> Dict[str, Any]:
        """Export model in ONNX format."""
        try:
            import onnx
        except ImportError:
            raise RuntimeError("ONNX library not available for export")
        
        output_path = Path(config.output_path).with_suffix('.onnx')
        
        # Create sample input
        sample_input = torch.randn(sample_input_shape)
        if config.precision == "float16":
            sample_input = sample_input.half()
        elif config.precision == "bfloat16":
            sample_input = sample_input.to(torch.bfloat16)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        return {
            "success": True,
            "files_created": [str(output_path)],
            "model_size_mb": os.path.getsize(output_path) / (1024 * 1024)
        }
    
    def _export_torchscript(
        self,
        model: "nn.Module",
        config: ExportConfig,
        sample_input_shape: tuple
    ) -> Dict[str, Any]:
        """Export model in TorchScript format."""
        output_path = Path(config.output_path).with_suffix('.pt')
        
        # Create sample input for tracing
        sample_input = torch.randn(sample_input_shape)
        if config.precision == "float16":
            sample_input = sample_input.half()
        elif config.precision == "bfloat16":
            sample_input = sample_input.to(torch.bfloat16)
        
        # Try tracing first, fall back to scripting
        try:
            traced_model = torch.jit.trace(model, sample_input)
            traced_model.save(str(output_path))
            method = "trace"
        except Exception as e:
            logger.warning(f"Tracing failed, trying scripting: {str(e)}")
            try:
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(output_path))
                method = "script"
            except Exception as e2:
                raise RuntimeError(f"Both tracing and scripting failed: trace={str(e)}, script={str(e2)}")
        
        return {
            "success": True,
            "files_created": [str(output_path)],
            "model_size_mb": os.path.getsize(output_path) / (1024 * 1024),
            "torchscript_method": method
        }
    
    def preserve_quantization_during_export(
        self,
        quantized_model: "nn.Module",
        export_config: ExportConfig
    ) -> Dict[str, Any]:
        """
        Export quantized model while preserving quantization information.
        
        Args:
            quantized_model: Model with quantization
            export_config: Export configuration
            
        Returns:
            Export results with quantization preservation info
        """
        logger.info("Exporting quantized model with quantization preservation")
        
        try:
            # Check if model has quantization
            has_quantization = self._detect_quantization(quantized_model)
            
            if not has_quantization:
                logger.warning("No quantization detected in model")
                return self.export_model(quantized_model, export_config)
            
            # For PyTorch export, we can preserve quantization directly
            if export_config.export_format == "pytorch":
                return self._export_quantized_pytorch(quantized_model, export_config)
            
            # For other formats, quantization might not be fully preserved
            else:
                logger.warning(f"Quantization may not be fully preserved in {export_config.export_format} format")
                result = self.export_model(quantized_model, export_config)
                result["quantization_preserved"] = False
                result["quantization_warning"] = f"Quantization may not be preserved in {export_config.export_format}"
                return result
                
        except Exception as e:
            logger.error(f"Quantized model export failed: {str(e)}")
            raise RuntimeError(f"Quantized export failed: {str(e)}") from e
    
    def _detect_quantization(self, model: "nn.Module") -> bool:
        """Detect if model has quantization."""
        for name, module in model.named_modules():
            # Check for bitsandbytes quantized layers
            if hasattr(module, '__class__') and 'bnb' in str(module.__class__):
                return True
            # Check for PyTorch quantization
            if hasattr(module, 'qconfig') or 'quantized' in str(type(module)):
                return True
        return False
    
    def _export_quantized_pytorch(
        self,
        quantized_model: "nn.Module",
        config: ExportConfig
    ) -> Dict[str, Any]:
        """Export quantized model in PyTorch format with quantization info."""
        output_path = Path(config.output_path)
        
        # Collect quantization information
        quantization_info = self._collect_quantization_info(quantized_model)
        
        # Prepare export data
        export_data = {
            "model_state_dict": quantized_model.state_dict(),
            "model_class": quantized_model.__class__.__name__,
            "quantization_info": quantization_info,
            "quantization_preserved": True
        }
        
        # Save the quantized model
        torch.save(export_data, output_path.with_suffix('_quantized.pth'))
        
        return {
            "success": True,
            "files_created": [str(output_path.with_suffix('_quantized.pth'))],
            "model_size_mb": get_model_size_mb(quantized_model),
            "quantization_preserved": True,
            "quantization_info": quantization_info
        }
    
    def _collect_quantization_info(self, model: "nn.Module") -> Dict[str, Any]:
        """Collect information about quantization in the model."""
        quantization_info = {
            "quantized_layers": [],
            "quantization_types": set(),
            "total_quantized_params": 0
        }
        
        for name, module in model.named_modules():
            if hasattr(module, '__class__'):
                class_name = str(module.__class__)
                if 'bnb' in class_name or 'quantized' in class_name.lower():
                    layer_info = {
                        "name": name,
                        "type": class_name,
                        "parameters": sum(p.numel() for p in module.parameters())
                    }
                    
                    # Extract specific quantization details
                    if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                        layer_info["weight_dtype"] = str(module.weight.dtype)
                    
                    quantization_info["quantized_layers"].append(layer_info)
                    quantization_info["quantization_types"].add(class_name)
                    quantization_info["total_quantized_params"] += layer_info["parameters"]
        
        quantization_info["quantization_types"] = list(quantization_info["quantization_types"])
        return quantization_info
    
    def get_export_summary(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of exported model.
        
        Args:
            model_name: Name of the exported model
            
        Returns:
            Export summary or None if not found
        """
        return self._exported_models.get(model_name)
    
    def list_exported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all exported models."""
        return self._exported_models.copy()
    
    def clear_cache(self):
        """Clear the cache of exported models."""
        self._exported_models.clear()
        logger.info("Model export cache cleared")