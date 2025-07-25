"""
Tests for PEFTTrainer class.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch

from src.training.peft_trainer import PEFTTrainer, TrainingConfig, TrainingMetrics, TrainingResults


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, pixel_values):
        # Simulate ViT-like behavior
        batch_size = pixel_values.size(0)
        features = torch.randn(batch_size, 768)
        return self.classifier(features)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel(num_classes=10)


@pytest.fixture
def training_config():
    """Create a basic training configuration."""
    return TrainingConfig(
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=2,
        gradient_accumulation_steps=1,
        use_mixed_precision=False,  # Disable for testing
        save_steps=100,
        eval_steps=50,
        logging_steps=10,
        output_dir="test_outputs"
    )


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    # Create dummy data
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


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.num_epochs == 10
        assert config.optimizer == "adamw"
        assert config.use_mixed_precision is True
        assert config.gradient_accumulation_steps == 1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid learning rate
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            TrainingConfig(learning_rate=-1.0)
        
        # Test invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            TrainingConfig(batch_size=0)
        
        # Test invalid epochs
        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            TrainingConfig(num_epochs=0)
        
        # Test invalid gradient accumulation
        with pytest.raises(ValueError, match="Gradient accumulation steps must be positive"):
            TrainingConfig(gradient_accumulation_steps=0)
    
    def test_logging_dir_default(self):
        """Test that logging directory defaults correctly."""
        config = TrainingConfig(output_dir="custom_output")
        assert config.logging_dir == "custom_output/logs"


class TestTrainingMetrics:
    """Test TrainingMetrics class."""
    
    def test_metrics_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5,
            train_accuracy=0.8,
            eval_loss=0.6,
            eval_accuracy=0.75,
            learning_rate=1e-4
        )
        
        assert metrics.epoch == 1
        assert metrics.step == 100
        assert metrics.train_loss == 0.5
        assert metrics.train_accuracy == 0.8
        assert metrics.eval_loss == 0.6
        assert metrics.eval_accuracy == 0.75
        assert metrics.learning_rate == 1e-4
    
    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        metrics = TrainingMetrics(
            epoch=0,
            step=0,
            train_loss=1.0
        )
        
        assert metrics.train_accuracy is None
        assert metrics.eval_loss is None
        assert metrics.eval_accuracy is None
        assert metrics.learning_rate == 0.0


class TestPEFTTrainer:
    """Test PEFTTrainer class."""
    
    def test_trainer_initialization(self, mock_model, training_config, mock_dataloader):
        """Test trainer initialization."""
        trainer = PEFTTrainer(
            model=mock_model,
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        assert trainer.model is mock_model
        assert trainer.config is training_config
        assert trainer.train_dataloader is mock_dataloader
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.best_eval_accuracy is None
    
    def test_parameter_counting(self, mock_model, training_config, mock_dataloader):
        """Test parameter counting functionality."""
        trainer = PEFTTrainer(
            model=mock_model,
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        param_counts = trainer._count_parameters()
        
        assert "total" in param_counts
        assert "trainable" in param_counts
        assert "frozen" in param_counts
        assert param_counts["total"] > 0
        assert param_counts["trainable"] > 0
        assert param_counts["total"] == param_counts["trainable"] + param_counts["frozen"]
    
    def test_optimizer_creation(self, mock_model, training_config, mock_dataloader):
        """Test optimizer creation."""
        trainer = PEFTTrainer(
            model=mock_model,
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        # Test AdamW optimizer (default)
        assert trainer.optimizer is not None
        assert trainer.optimizer.__class__.__name__ == "AdamW"
        
        # Test SGD optimizer
        config_sgd = TrainingConfig(optimizer="sgd")
        trainer_sgd = PEFTTrainer(
            model=mock_model,
            config=config_sgd,
            train_dataloader=mock_dataloader
        )
        assert trainer_sgd.optimizer.__class__.__name__ == "SGD"
    
    def test_scheduler_creation(self, mock_model, training_config, mock_dataloader):
        """Test scheduler creation."""
        # Test cosine scheduler
        config_cosine = TrainingConfig(scheduler="cosine")
        trainer = PEFTTrainer(
            model=mock_model,
            config=config_cosine,
            train_dataloader=mock_dataloader
        )
        assert trainer.scheduler is not None
        assert "CosineAnnealingLR" in trainer.scheduler.__class__.__name__
        
        # Test constant scheduler (None)
        config_constant = TrainingConfig(scheduler="constant")
        trainer_constant = PEFTTrainer(
            model=mock_model,
            config=config_constant,
            train_dataloader=mock_dataloader
        )
        assert trainer_constant.scheduler is None
    
    def test_batch_device_movement(self, mock_model, training_config, mock_dataloader):
        """Test moving batch to device."""
        trainer = PEFTTrainer(
            model=mock_model,
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        # Create test batch
        batch = {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "labels": torch.randint(0, 10, (2,))
        }
        
        moved_batch = trainer._move_batch_to_device(batch)
        
        assert moved_batch["pixel_values"].device == trainer.device
        assert moved_batch["labels"].device == trainer.device
    
    def test_loss_computation(self, mock_model, training_config, mock_dataloader):
        """Test loss computation."""
        trainer = PEFTTrainer(
            model=mock_model,
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        # Create test batch
        batch = {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "labels": torch.randint(0, 10, (2,))
        }
        
        loss, logits = trainer._compute_loss(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert logits is not None
        assert logits.shape[0] == 2  # batch size
        assert logits.shape[1] == 10  # num classes
    
    def test_memory_usage_tracking(self, mock_model, training_config, mock_dataloader):
        """Test memory usage tracking."""
        trainer = PEFTTrainer(
            model=mock_model,
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        memory_stats = trainer._get_memory_usage()
        
        assert "allocated_mb" in memory_stats
        assert "reserved_mb" in memory_stats
        assert "max_allocated_mb" in memory_stats
        assert all(isinstance(v, float) for v in memory_stats.values())
    
    def test_early_stopping_logic(self, mock_model, training_config, mock_dataloader):
        """Test early stopping logic."""
        config_with_early_stop = TrainingConfig(
            early_stopping_patience=2,
            early_stopping_threshold=0.01
        )
        
        trainer = PEFTTrainer(
            model=mock_model,
            config=config_with_early_stop,
            train_dataloader=mock_dataloader
        )
        
        # Test no early stopping without eval metrics
        assert not trainer._should_early_stop(None)
        
        # Test no early stopping on first evaluation
        eval_metrics = {"accuracy": 0.8}
        assert not trainer._should_early_stop(eval_metrics)
        assert trainer.best_eval_accuracy == 0.8
        
        # Test no early stopping with improvement
        eval_metrics = {"accuracy": 0.85}
        assert not trainer._should_early_stop(eval_metrics)
        assert trainer.best_eval_accuracy == 0.85
        
        # Test early stopping counter increment
        eval_metrics = {"accuracy": 0.84}  # Small decrease
        assert not trainer._should_early_stop(eval_metrics)
        assert trainer.early_stopping_counter == 1
        
        # Test early stopping trigger
        eval_metrics = {"accuracy": 0.83}  # Another small decrease
        assert trainer._should_early_stop(eval_metrics)
        assert trainer.early_stopping_counter == 2
    
    @patch('torch.save')
    def test_checkpoint_saving(self, mock_save, mock_model, training_config, mock_dataloader, tmp_path):
        """Test checkpoint saving functionality."""
        config = TrainingConfig(output_dir=str(tmp_path))
        trainer = PEFTTrainer(
            model=mock_model,
            config=config,
            train_dataloader=mock_dataloader
        )
        
        # Test checkpoint saving
        trainer._save_checkpoint(epoch=1)
        
        # Verify torch.save was called
        assert mock_save.called
        
        # Check that checkpoint directory would be created
        expected_checkpoint_dir = tmp_path / "checkpoint-epoch-1"
        # Note: We can't check if directory exists because we mocked torch.save
    
    def test_training_summary(self, mock_model, training_config, mock_dataloader):
        """Test training summary generation."""
        trainer = PEFTTrainer(
            model=mock_model,
            config=training_config,
            train_dataloader=mock_dataloader
        )
        
        # Test summary before training
        summary = trainer.get_training_summary()
        assert summary["status"] == "not_started"
        
        # Add some fake training history
        trainer.training_history = [
            TrainingMetrics(epoch=0, step=10, train_loss=1.0),
            TrainingMetrics(epoch=1, step=20, train_loss=0.8)
        ]
        trainer.current_epoch = 1
        trainer.global_step = 20
        trainer.best_eval_accuracy = 0.75
        
        summary = trainer.get_training_summary()
        assert summary["status"] == "in_progress"
        assert summary["current_epoch"] == 1
        assert summary["global_step"] == 20
        assert summary["best_eval_accuracy"] == 0.75
        assert summary["training_history_length"] == 2
        assert "parameter_counts" in summary
        assert "memory_usage" in summary
    
    def test_invalid_optimizer(self, mock_model, mock_dataloader):
        """Test handling of invalid optimizer."""
        config = TrainingConfig(optimizer="invalid_optimizer")
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            PEFTTrainer(
                model=mock_model,
                config=config,
                train_dataloader=mock_dataloader
            )
    
    def test_invalid_scheduler(self, mock_model, mock_dataloader):
        """Test handling of invalid scheduler."""
        config = TrainingConfig(scheduler="invalid_scheduler")
        
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            PEFTTrainer(
                model=mock_model,
                config=config,
                train_dataloader=mock_dataloader
            )


class TestTrainingResults:
    """Test TrainingResults class."""
    
    def test_results_creation(self):
        """Test creating training results."""
        results = TrainingResults(
            final_train_loss=0.3,
            final_eval_loss=0.4,
            final_train_accuracy=0.9,
            final_eval_accuracy=0.85,
            best_eval_accuracy=0.87,
            total_epochs=10,
            total_steps=1000,
            total_training_time=3600.0,
            converged=True,
            early_stopped=False
        )
        
        assert results.final_train_loss == 0.3
        assert results.final_eval_loss == 0.4
        assert results.final_train_accuracy == 0.9
        assert results.final_eval_accuracy == 0.85
        assert results.best_eval_accuracy == 0.87
        assert results.total_epochs == 10
        assert results.total_steps == 1000
        assert results.total_training_time == 3600.0
        assert results.converged is True
        assert results.early_stopped is False
        assert len(results.training_history) == 0  # Default empty list


if __name__ == "__main__":
    pytest.main([__file__])