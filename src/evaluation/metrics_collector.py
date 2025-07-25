"""
Comprehensive metrics collection for PEFT Vision Transformer evaluation.
"""

import logging
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
else:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        torch = None
        nn = None
        DataLoader = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Accuracy metrics
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    
    # Loss metrics
    average_loss: float = 0.0
    
    # Per-class metrics
    per_class_accuracy: Optional[List[float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Confidence and uncertainty
    prediction_confidence: Optional[List[float]] = None
    entropy: Optional[float] = None
    
    # Additional metrics
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    # Metadata
    num_samples: int = 0
    num_classes: int = 0
    evaluation_time: float = 0.0


@dataclass
class ModelMetrics:
    """Container for model-specific metrics."""
    
    # Parameter counts
    total_parameters: int = 0
    trainable_parameters: int = 0
    frozen_parameters: int = 0
    
    # Parameter ratios
    trainable_ratio: float = 0.0
    
    # Model size
    model_size_mb: float = 0.0
    
    # PEFT-specific metrics
    lora_parameters: int = 0
    lora_modules_count: int = 0
    lora_rank: Optional[int] = None
    lora_alpha: Optional[float] = None
    
    # Quantization metrics
    quantized_layers: int = 0
    quantization_bits: Optional[int] = None
    memory_reduction_percent: float = 0.0
    
    # Architecture info
    model_name: str = ""
    architecture: str = ""
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 0


@dataclass
class ResourceMetrics:
    """Container for resource usage metrics."""
    
    # Memory metrics (MB)
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    
    # System memory
    system_memory_total_mb: float = 0.0
    system_memory_used_mb: float = 0.0
    system_memory_percent: float = 0.0
    
    # Timing metrics (seconds)
    training_time: float = 0.0
    inference_time: float = 0.0
    data_loading_time: float = 0.0
    
    # Throughput metrics
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    
    # Hardware info
    device_type: str = "cpu"
    device_name: str = ""
    num_gpus: int = 0


class MetricsCollector:
    """
    Comprehensive metrics collector for PEFT Vision Transformer evaluation.
    
    Collects accuracy, loss, memory usage, timing, and model-specific metrics.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize metrics collector.
        
        Args:
            device: Device to monitor (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._memory_history: List[float] = []
        self._timing_history: List[float] = []
        
        logger.info(f"MetricsCollector initialized for device: {self.device}")
    
    def evaluate_model(
        self,
        model: "nn.Module",
        dataloader: "DataLoader",
        compute_detailed_metrics: bool = True,
        compute_per_class: bool = False
    ) -> EvaluationMetrics:
        """
        Evaluate model on a dataset and collect comprehensive metrics.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            compute_detailed_metrics: Whether to compute detailed metrics (F1, precision, etc.)
            compute_per_class: Whether to compute per-class accuracy
            
        Returns:
            EvaluationMetrics with all collected metrics
        """
        if torch is None or nn is None:
            raise RuntimeError("PyTorch not available")
        
        logger.info("Starting model evaluation")
        start_time = time.time()
        
        model.eval()
        
        # Initialize tracking variables
        total_loss = 0.0
        total_samples = 0
        correct_top1 = 0
        correct_top5 = 0
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        # Memory tracking
        self._reset_memory_tracking()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_start_time = time.time()
                
                # Move batch to device
                if isinstance(batch, dict):
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                else:
                    pixel_values, labels = batch
                    pixel_values = pixel_values.to(self.device)
                    labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(pixel_values)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()
                
                # Compute accuracy
                batch_size = labels.size(0)
                total_samples += batch_size
                
                # Top-1 accuracy
                _, pred_top1 = logits.topk(1, dim=1)
                correct_top1 += pred_top1.eq(labels.view(-1, 1)).sum().item()
                
                # Top-5 accuracy (if applicable)
                if logits.size(1) >= 5:
                    _, pred_top5 = logits.topk(5, dim=1)
                    correct_top5 += pred_top5.eq(labels.view(-1, 1)).sum().item()
                
                # Store predictions and labels for detailed metrics
                if compute_detailed_metrics or compute_per_class:
                    predictions = torch.argmax(logits, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Store confidence scores
                    probabilities = torch.softmax(logits, dim=1)
                    max_probs, _ = torch.max(probabilities, dim=1)
                    all_confidences.extend(max_probs.cpu().numpy())
                
                # Track memory usage
                self._track_memory_usage()
                
                # Log progress
                if batch_idx % 50 == 0:
                    batch_time = time.time() - batch_start_time
                    logger.debug(f"Batch {batch_idx}/{len(dataloader)} processed in {batch_time:.3f}s")
        
        evaluation_time = time.time() - start_time
        
        # Calculate basic metrics
        avg_loss = total_loss / len(dataloader)
        top1_acc = correct_top1 / total_samples
        top5_acc = correct_top5 / total_samples if logits.size(1) >= 5 else 0.0
        
        # Create evaluation metrics
        metrics = EvaluationMetrics(
            top1_accuracy=top1_acc,
            top5_accuracy=top5_acc,
            average_loss=avg_loss,
            num_samples=total_samples,
            num_classes=logits.size(1),
            evaluation_time=evaluation_time
        )
        
        # Compute detailed metrics if requested
        if compute_detailed_metrics and all_predictions:
            detailed_metrics = self._compute_detailed_metrics(all_predictions, all_labels)
            metrics.f1_score = detailed_metrics.get("f1_score")
            metrics.precision = detailed_metrics.get("precision")
            metrics.recall = detailed_metrics.get("recall")
            metrics.entropy = detailed_metrics.get("entropy")
        
        # Compute per-class accuracy if requested
        if compute_per_class and all_predictions:
            per_class_acc, confusion_mat = self._compute_per_class_metrics(
                all_predictions, all_labels, metrics.num_classes
            )
            metrics.per_class_accuracy = per_class_acc
            metrics.confusion_matrix = confusion_mat
        
        # Store confidence scores
        if all_confidences:
            metrics.prediction_confidence = all_confidences
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        logger.info(f"Top-1 Accuracy: {top1_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}, Loss: {avg_loss:.4f}")
        
        return metrics
    
    def collect_model_metrics(self, model: "nn.Module", model_name: str = "") -> ModelMetrics:
        """
        Collect comprehensive model metrics.
        
        Args:
            model: Model to analyze
            model_name: Name of the model
            
        Returns:
            ModelMetrics with model information
        """
        logger.info(f"Collecting model metrics for {model_name or 'unnamed model'}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
        
        # Calculate model size
        model_size_mb = self._calculate_model_size(model)
        
        # Detect PEFT-specific metrics
        lora_metrics = self._detect_lora_metrics(model)
        quantization_metrics = self._detect_quantization_metrics(model)
        
        # Get architecture info
        architecture_info = self._get_architecture_info(model)
        
        metrics = ModelMetrics(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            frozen_parameters=frozen_params,
            trainable_ratio=trainable_ratio,
            model_size_mb=model_size_mb,
            lora_parameters=lora_metrics.get("lora_parameters", 0),
            lora_modules_count=lora_metrics.get("lora_modules_count", 0),
            lora_rank=lora_metrics.get("lora_rank"),
            lora_alpha=lora_metrics.get("lora_alpha"),
            quantized_layers=quantization_metrics.get("quantized_layers", 0),
            quantization_bits=quantization_metrics.get("quantization_bits"),
            memory_reduction_percent=quantization_metrics.get("memory_reduction_percent", 0.0),
            model_name=model_name,
            architecture=architecture_info.get("architecture", model.__class__.__name__),
            input_size=architecture_info.get("input_size", (224, 224)),
            num_classes=architecture_info.get("num_classes", 0)
        )
        
        logger.info(f"Model metrics: {trainable_params:,}/{total_params:,} trainable "
                   f"({trainable_ratio:.2%}), {model_size_mb:.2f}MB")
        
        return metrics
    
    def collect_resource_metrics(
        self,
        training_time: float = 0.0,
        inference_time: float = 0.0,
        num_samples: int = 0
    ) -> ResourceMetrics:
        """
        Collect resource usage metrics.
        
        Args:
            training_time: Total training time in seconds
            inference_time: Total inference time in seconds
            num_samples: Number of samples processed
            
        Returns:
            ResourceMetrics with resource usage information
        """
        logger.info("Collecting resource usage metrics")
        
        # Memory metrics
        memory_stats = self._get_memory_stats()
        
        # System memory
        system_memory = self._get_system_memory_stats()
        
        # Calculate throughput
        samples_per_second = num_samples / inference_time if inference_time > 0 else 0.0
        
        # Hardware info
        hardware_info = self._get_hardware_info()
        
        metrics = ResourceMetrics(
            peak_memory_mb=max(self._memory_history) if self._memory_history else 0.0,
            average_memory_mb=sum(self._memory_history) / len(self._memory_history) if self._memory_history else 0.0,
            memory_allocated_mb=memory_stats.get("allocated_mb", 0.0),
            memory_reserved_mb=memory_stats.get("reserved_mb", 0.0),
            system_memory_total_mb=system_memory.get("total_mb", 0.0),
            system_memory_used_mb=system_memory.get("used_mb", 0.0),
            system_memory_percent=system_memory.get("percent", 0.0),
            training_time=training_time,
            inference_time=inference_time,
            samples_per_second=samples_per_second,
            device_type=hardware_info.get("device_type", "cpu"),
            device_name=hardware_info.get("device_name", ""),
            num_gpus=hardware_info.get("num_gpus", 0)
        )
        
        logger.info(f"Resource metrics: Peak memory {metrics.peak_memory_mb:.1f}MB, "
                   f"Throughput {samples_per_second:.1f} samples/s")
        
        return metrics
    
    def _compute_detailed_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Compute detailed classification metrics."""
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            import numpy as np
            
            # Convert to numpy arrays
            pred_array = np.array(predictions)
            label_array = np.array(labels)
            
            # Compute metrics
            f1 = f1_score(label_array, pred_array, average='weighted', zero_division=0)
            precision = precision_score(label_array, pred_array, average='weighted', zero_division=0)
            recall = recall_score(label_array, pred_array, average='weighted', zero_division=0)
            
            # Compute entropy (uncertainty measure)
            unique_labels, counts = np.unique(pred_array, return_counts=True)
            probabilities = counts / len(pred_array)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return {
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "entropy": float(entropy)
            }
            
        except ImportError:
            logger.warning("scikit-learn not available, skipping detailed metrics")
            return {}
        except Exception as e:
            logger.warning(f"Failed to compute detailed metrics: {str(e)}")
            return {}
    
    def _compute_per_class_metrics(
        self, 
        predictions: List[int], 
        labels: List[int], 
        num_classes: int
    ) -> Tuple[List[float], List[List[int]]]:
        """Compute per-class accuracy and confusion matrix."""
        try:
            import numpy as np
            
            pred_array = np.array(predictions)
            label_array = np.array(labels)
            
            # Per-class accuracy
            per_class_acc = []
            for class_idx in range(num_classes):
                class_mask = (label_array == class_idx)
                if class_mask.sum() > 0:
                    class_correct = (pred_array[class_mask] == class_idx).sum()
                    class_acc = class_correct / class_mask.sum()
                    per_class_acc.append(float(class_acc))
                else:
                    per_class_acc.append(0.0)
            
            # Confusion matrix
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
            for true_label, pred_label in zip(label_array, pred_array):
                if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
                    confusion_matrix[true_label][pred_label] += 1
            
            return per_class_acc, confusion_matrix.tolist()
            
        except Exception as e:
            logger.warning(f"Failed to compute per-class metrics: {str(e)}")
            return [], []
    
    def _calculate_model_size(self, model: "nn.Module") -> float:
        """Calculate model size in MB."""
        try:
            total_size = 0
            
            # Parameters
            for param in model.parameters():
                total_size += param.nelement() * param.element_size()
            
            # Buffers
            for buffer in model.buffers():
                total_size += buffer.nelement() * buffer.element_size()
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {str(e)}")
            return 0.0
    
    def _detect_lora_metrics(self, model: "nn.Module") -> Dict[str, Any]:
        """Detect LoRA-specific metrics."""
        lora_params = 0
        lora_modules = 0
        lora_rank = None
        lora_alpha = None
        
        try:
            for name, module in model.named_modules():
                if "lora" in name.lower():
                    lora_modules += 1
                    
                    # Count LoRA parameters
                    for param in module.parameters():
                        lora_params += param.numel()
                    
                    # Try to extract LoRA configuration
                    if hasattr(module, 'r'):
                        lora_rank = module.r
                    if hasattr(module, 'lora_alpha'):
                        lora_alpha = module.lora_alpha
            
            return {
                "lora_parameters": lora_params,
                "lora_modules_count": lora_modules,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha
            }
            
        except Exception as e:
            logger.warning(f"Failed to detect LoRA metrics: {str(e)}")
            return {}
    
    def _detect_quantization_metrics(self, model: "nn.Module") -> Dict[str, Any]:
        """Detect quantization-specific metrics."""
        quantized_layers = 0
        quantization_bits = None
        
        try:
            for name, module in model.named_modules():
                module_type = type(module).__name__
                
                if "8bit" in module_type.lower():
                    quantized_layers += 1
                    quantization_bits = 8
                elif "4bit" in module_type.lower():
                    quantized_layers += 1
                    quantization_bits = 4
            
            return {
                "quantized_layers": quantized_layers,
                "quantization_bits": quantization_bits,
                "memory_reduction_percent": 0.0  # Would need before/after comparison
            }
            
        except Exception as e:
            logger.warning(f"Failed to detect quantization metrics: {str(e)}")
            return {}
    
    def _get_architecture_info(self, model: "nn.Module") -> Dict[str, Any]:
        """Get model architecture information."""
        try:
            info = {
                "architecture": model.__class__.__name__,
                "input_size": (224, 224),  # Default
                "num_classes": 0
            }
            
            # Try to detect number of classes
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and ("head" in name or "classifier" in name):
                    info["num_classes"] = module.out_features
                    break
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get architecture info: {str(e)}")
            return {"architecture": "Unknown", "input_size": (224, 224), "num_classes": 0}
    
    def _reset_memory_tracking(self):
        """Reset memory tracking history."""
        self._memory_history.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def _track_memory_usage(self):
        """Track current memory usage."""
        try:
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                self._memory_history.append(memory_mb)
        except Exception:
            pass  # Ignore memory tracking errors
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        try:
            if torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024)
                }
            else:
                return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
        except Exception:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    
    def _get_system_memory_stats(self) -> Dict[str, float]:
        """Get system memory statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "percent": memory.percent
            }
        except Exception:
            return {"total_mb": 0.0, "used_mb": 0.0, "available_mb": 0.0, "percent": 0.0}
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        try:
            info = {
                "device_type": self.device,
                "device_name": "",
                "num_gpus": 0
            }
            
            if torch.cuda.is_available():
                info["num_gpus"] = torch.cuda.device_count()
                if info["num_gpus"] > 0:
                    info["device_name"] = torch.cuda.get_device_name(0)
            
            return info
            
        except Exception:
            return {"device_type": "cpu", "device_name": "", "num_gpus": 0}