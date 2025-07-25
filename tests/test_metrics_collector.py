"""
Tests for MetricsCollector class.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch

from src.evaluation.metrics_collector import (
    MetricsCollector, EvaluationMetrics, ModelMetrics, ResourceMetrics
)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, pixel_values):
        batch_size = pixel_values.size(0)
        features = torch.randn(batch_size, 768)
        return self.classifier(features)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel(num_classes=10)


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    pixel_values = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 10, (16,))
    
    dataset = TensorDataset(pixel_values, labels)
    
    def collate_fn(batch):
        pixel_values, labels = zip(*batch)
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.stack(labels)
        }
    
    return DataLoader(dataset, batch_size=4, collate_fn=collate_fn)


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values of EvaluationMetrics."""
        metrics = EvaluationMetrics()
        
        assert metrics.top1_accuracy == 0.0
        assert metrics.top5_accuracy == 0.0
        assert metrics.average_loss == 0.0
        assert metrics.per_class_accuracy is None
        assert metrics.confusion_matrix is None
        assert metrics.num_samples == 0
        assert metrics.num_classes == 0
    
    def test_custom_values(self):
        """Test EvaluationMetrics with custom values."""
        metrics = EvaluationMetrics(
            top1_accuracy=0.85,
            top5_accuracy=0.95,
            average_loss=0.3,
            num_samples=1000,
            num_classes=10,
            evaluation_time=120.5
        )
        
        assert metrics.top1_accuracy == 0.85
        assert metrics.top5_accuracy == 0.95
        assert metrics.average_loss == 0.3
        assert metrics.num_samples == 1000
        assert metrics.num_classes == 10
        assert metrics.evaluation_time == 120.5


class TestModelMetrics:
    """Test ModelMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values of ModelMetrics."""
        metrics = ModelMetrics()
        
        assert metrics.total_parameters == 0
        assert metrics.trainable_parameters == 0
        assert metrics.frozen_parameters == 0
        assert metrics.trainable_ratio == 0.0
        assert metrics.model_size_mb == 0.0
        assert metrics.lora_parameters == 0
        assert metrics.model_name == ""
    
    def test_parameter_calculations(self):
        """Test parameter-related calculations."""
        metrics = ModelMetrics(
            total_parameters=1000000,
            trainable_parameters=50000,
            frozen_parameters=950000
        )
        
        # Calculate trainable ratio
        metrics.trainable_ratio = metrics.trainable_parameters / metrics.total_parameters
        
        assert metrics.trainable_ratio == 0.05
        assert metrics.total_parameters == metrics.trainable_parameters + metrics.frozen_parameters


class TestResourceMetrics:
    """Test ResourceMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values of ResourceMetrics."""
        metrics = ResourceMetrics()
        
        assert metrics.peak_memory_mb == 0.0
        assert metrics.average_memory_mb == 0.0
        assert metrics.training_time == 0.0
        assert metrics.inference_time == 0.0
        assert metrics.samples_per_second == 0.0
        assert metrics.device_type == "cpu"
        assert metrics.num_gpus == 0
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = ResourceMetrics(
            inference_time=10.0,
            samples_per_second=100.0
        )
        
        # Verify that 100 samples/second for 10 seconds = 1000 samples
        total_samples = metrics.samples_per_second * metrics.inference_time
        assert total_samples == 1000.0


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        
        assert collector.device in ["cpu", "cuda"]
        assert isinstance(collector._memory_history, list)
        assert len(collector._memory_history) == 0
    
    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        collector = MetricsCollector(device="cpu")
        
        assert collector.device == "cpu"
    
    def test_model_metrics_collection(self, mock_model):
        """Test model metrics collection."""
        collector = MetricsCollector()
        
        metrics = collector.collect_model_metrics(mock_model, "test_model")
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.model_name == "test_model"
        assert metrics.total_parameters > 0
        assert metrics.trainable_parameters > 0
        assert metrics.model_size_mb > 0
        assert metrics.architecture == "MockModel"
        
        # Check that trainable ratio is calculated correctly
        expected_ratio = metrics.trainable_parameters / metrics.total_parameters
        assert abs(metrics.trainable_ratio - expected_ratio) < 1e-6
    
    def test_model_evaluation(self, mock_model, mock_dataloader):
        """Test model evaluation functionality."""
        collector = MetricsCollector()
        
        # Set model to evaluation mode
        mock_model.eval()
        
        metrics = collector.evaluate_model(
            model=mock_model,
            dataloader=mock_dataloader,
            compute_detailed_metrics=False,
            compute_per_class=False
        )
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.num_samples > 0
        assert metrics.num_classes == 10
        assert 0 <= metrics.top1_accuracy <= 1
        assert metrics.average_loss >= 0
        assert metrics.evaluation_time > 0
    
    def test_model_evaluation_with_detailed_metrics(self, mock_model, mock_dataloader):
        """Test model evaluation with detailed metrics."""
        collector = MetricsCollector()
        
        with patch('sklearn.metrics.f1_score', return_value=0.8), \
             patch('sklearn.metrics.precision_score', return_value=0.85), \
             patch('sklearn.metrics.recall_score', return_value=0.75):
            
            metrics = collector.evaluate_model(
                model=mock_model,
                dataloader=mock_dataloader,
                compute_detailed_metrics=True,
                compute_per_class=False
            )
            
            assert metrics.f1_score == 0.8
            assert metrics.precision == 0.85
            assert metrics.recall == 0.75
    
    def test_model_evaluation_with_per_class_metrics(self, mock_model, mock_dataloader):
        """Test model evaluation with per-class metrics."""
        collector = MetricsCollector()
        
        metrics = collector.evaluate_model(
            model=mock_model,
            dataloader=mock_dataloader,
            compute_detailed_metrics=False,
            compute_per_class=True
        )
        
        assert metrics.per_class_accuracy is not None
        assert len(metrics.per_class_accuracy) == 10  # num_classes
        assert metrics.confusion_matrix is not None
        assert len(metrics.confusion_matrix) == 10
        assert len(metrics.confusion_matrix[0]) == 10
    
    def test_resource_metrics_collection(self):
        """Test resource metrics collection."""
        collector = MetricsCollector()
        
        # Add some fake memory history
        collector._memory_history = [100.0, 150.0, 200.0, 120.0]
        
        metrics = collector.collect_resource_metrics(
            training_time=300.0,
            inference_time=60.0,
            num_samples=1000
        )
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.training_time == 300.0
        assert metrics.inference_time == 60.0
        assert metrics.peak_memory_mb == 200.0  # max of memory history
        assert metrics.average_memory_mb == 142.5  # average of memory history
        assert abs(metrics.samples_per_second - (1000/60.0)) < 1e-6
    
    def test_memory_tracking(self):
        """Test memory tracking functionality."""
        collector = MetricsCollector()
        
        # Test reset
        collector._memory_history = [100.0, 200.0]
        collector._reset_memory_tracking()
        assert len(collector._memory_history) == 0
        
        # Test tracking (this might not add anything on CPU)
        collector._track_memory_usage()
        # Memory history might still be empty on CPU, which is fine
        assert isinstance(collector._memory_history, list)
    
    def test_model_size_calculation(self, mock_model):
        """Test model size calculation."""
        collector = MetricsCollector()
        
        size_mb = collector._calculate_model_size(mock_model)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
    
    def test_lora_detection(self):
        """Test LoRA metrics detection."""
        collector = MetricsCollector()
        
        # Create a mock model with LoRA-like modules
        mock_model = Mock()
        mock_model.named_modules.return_value = [
            ("layer.lora_A", Mock()),
            ("layer.lora_B", Mock()),
            ("layer.normal", Mock())
        ]
        
        # Mock parameters for LoRA modules
        lora_param = Mock()
        lora_param.numel.return_value = 1000
        
        normal_param = Mock()
        normal_param.numel.return_value = 500
        
        mock_model.named_modules.return_value[0][1].parameters.return_value = [lora_param]
        mock_model.named_modules.return_value[1][1].parameters.return_value = [lora_param]
        mock_model.named_modules.return_value[2][1].parameters.return_value = [normal_param]
        
        lora_metrics = collector._detect_lora_metrics(mock_model)
        
        assert lora_metrics["lora_modules_count"] == 2
        assert lora_metrics["lora_parameters"] == 2000  # 2 * 1000
    
    def test_quantization_detection(self):
        """Test quantization metrics detection."""
        collector = MetricsCollector()
        
        # Create a mock model with quantized modules
        mock_model = Mock()
        
        # Mock modules with quantization-like names
        mock_8bit_module = Mock()
        mock_8bit_module.__class__.__name__ = "Linear8bitLt"
        
        mock_4bit_module = Mock()
        mock_4bit_module.__class__.__name__ = "Linear4bit"
        
        mock_normal_module = Mock()
        mock_normal_module.__class__.__name__ = "Linear"
        
        mock_model.named_modules.return_value = [
            ("layer1", mock_8bit_module),
            ("layer2", mock_4bit_module),
            ("layer3", mock_normal_module)
        ]
        
        quant_metrics = collector._detect_quantization_metrics(mock_model)
        
        assert quant_metrics["quantized_layers"] == 2
        assert quant_metrics["quantization_bits"] == 4  # Last detected
    
    @patch('psutil.virtual_memory')
    def test_system_memory_stats(self, mock_memory):
        """Test system memory statistics collection."""
        # Mock psutil memory info
        mock_memory_info = Mock()
        mock_memory_info.total = 16 * 1024**3  # 16 GB
        mock_memory_info.used = 8 * 1024**3   # 8 GB
        mock_memory_info.available = 8 * 1024**3  # 8 GB
        mock_memory_info.percent = 50.0
        
        mock_memory.return_value = mock_memory_info
        
        collector = MetricsCollector()
        stats = collector._get_system_memory_stats()
        
        assert stats["total_mb"] == 16 * 1024  # 16 GB in MB
        assert stats["used_mb"] == 8 * 1024    # 8 GB in MB
        assert stats["percent"] == 50.0
    
    def test_hardware_info(self):
        """Test hardware information collection."""
        collector = MetricsCollector()
        
        info = collector._get_hardware_info()
        
        assert "device_type" in info
        assert "device_name" in info
        assert "num_gpus" in info
        assert isinstance(info["num_gpus"], int)
        assert info["num_gpus"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])