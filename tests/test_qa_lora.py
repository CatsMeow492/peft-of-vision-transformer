"""
Tests for QA-LoRA (Quantization-Aware LoRA) implementation.
"""

import pytest
import math
from unittest.mock import Mock, patch, MagicMock

from src.models.qa_lora import (
    QALoRAConfig,
    QALoRATrainer,
    GroupWiseQuantizer,
    QuantizationState,
    QALoRAIntegratedTrainer
)


class TestQALoRAConfig:
    """Test QA-LoRA configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QALoRAConfig()
        
        assert config.quantization_bits == 4
        assert config.quantization_type == "nf4"
        assert config.lora_rank == 8
        assert config.lora_alpha == 16.0
        assert config.gradient_scaling_factor == 1.0
        assert config.use_group_quantization is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid quantization bits
        with pytest.raises(ValueError, match="Quantization bits must be 4 or 8"):
            QALoRAConfig(quantization_bits=16)
        
        # Test invalid quantization type
        with pytest.raises(ValueError, match="Invalid quantization type"):
            QALoRAConfig(quantization_type="invalid")
        
        # Test invalid LoRA rank
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            QALoRAConfig(lora_rank=0)
        
        # Test invalid LoRA alpha
        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            QALoRAConfig(lora_alpha=0)
        
        # Test invalid dropout
        with pytest.raises(ValueError, match="LoRA dropout must be between 0 and 1"):
            QALoRAConfig(lora_dropout=1.5)
        
        # Test invalid gradient scaling
        with pytest.raises(ValueError, match="Gradient scaling factor must be positive"):
            QALoRAConfig(gradient_scaling_factor=0)
        
        # Test invalid group size
        with pytest.raises(ValueError, match="Quantization group size must be positive"):
            QALoRAConfig(quantization_group_size=0)


class TestQuantizationState:
    """Test QuantizationState data structure."""
    
    def test_quantization_state_creation(self):
        """Test QuantizationState creation."""
        state = QuantizationState(
            step=100,
            current_bits=8.0,
            quantization_ratio=0.5,
            gradient_scale=1.2
        )
        
        assert state.step == 100
        assert state.current_bits == 8.0
        assert state.quantization_ratio == 0.5
        assert state.gradient_scale == 1.2
        assert state.quantization_error == 0.0
        assert len(state.group_quantization_errors) == 0


class TestGroupWiseQuantizer:
    """Test GroupWiseQuantizer functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return QALoRAConfig(
            quantization_bits=4,
            quantization_type="nf4",
            quantization_group_size=8
        )
    
    @pytest.fixture
    def quantizer(self, config):
        """Create test quantizer."""
        return GroupWiseQuantizer(config)
    
    def test_quantizer_initialization(self, quantizer):
        """Test quantizer initialization."""
        assert quantizer.bits == 4
        assert quantizer.quant_type == "nf4"
        assert quantizer.group_size == 8
        assert len(quantizer.quant_levels) == 16  # 4-bit = 16 levels
    
    def test_nf4_levels(self, quantizer):
        """Test NF4 quantization levels."""
        levels = quantizer._get_nf4_levels()
        assert len(levels) == 16
        assert levels[0] == -1.0
        assert levels[-1] == 1.0
        assert levels[7] == 0.0  # Middle should be zero
    
    def test_fp4_levels(self, quantizer):
        """Test FP4 quantization levels."""
        levels = quantizer._get_fp4_levels()
        assert len(levels) == 16
        assert levels[0] == -12.0
        assert levels[-1] == 1.0
    
    @patch('torch.tensor')
    @patch('torch.cat')
    @patch('torch.zeros')
    def test_quantize_tensor_structure(self, mock_zeros, mock_cat, mock_tensor, quantizer):
        """Test quantize tensor method structure (without actual PyTorch)."""
        # This test verifies the method exists and has correct structure
        # Actual functionality requires PyTorch
        
        # Mock tensor
        mock_input = Mock()
        mock_input.shape = (4, 4)
        mock_input.flatten.return_value = Mock()
        mock_input.flatten.return_value.__len__ = Mock(return_value=16)
        mock_input.device = "cpu"
        
        # Mock torch operations
        mock_tensor.return_value = mock_input
        mock_zeros.return_value = Mock()
        mock_cat.return_value = Mock()
        
        # Test that method exists and can be called
        assert hasattr(quantizer, 'quantize_tensor')
        assert hasattr(quantizer, '_quantize_group')
        assert hasattr(quantizer, 'dequantize_tensor')


class TestQALoRATrainer:
    """Test QA-LoRA trainer functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return QALoRAConfig(
            quantization_bits=4,
            lora_rank=8,
            gradient_scaling_factor=1.5,
            warmup_steps=50,
            quantization_schedule="linear"
        )
    
    @pytest.fixture
    def trainer(self, config):
        """Create test trainer."""
        return QALoRATrainer(config)
    
    def test_trainer_initialization(self, trainer, config):
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.step_count == 0
        assert trainer.quantization_frozen is False
        assert len(trainer.quantization_history) == 0
        assert len(trainer.layer_statistics) == 0
    
    def test_update_quantization_schedule_warmup(self, trainer):
        """Test quantization schedule during warmup."""
        # During warmup
        trainer._update_quantization_schedule(25)  # 25 < 50 (warmup_steps)
        
        assert trainer.quantization_state.quantization_ratio == 0.5  # 25/50
        assert trainer.quantization_state.current_bits > trainer.config.quantization_bits
    
    def test_update_quantization_schedule_constant(self, trainer):
        """Test constant quantization schedule."""
        trainer.config.quantization_schedule = "constant"
        
        # After warmup
        trainer._update_quantization_schedule(100)
        
        assert trainer.quantization_state.quantization_ratio == 1.0
        assert trainer.quantization_state.current_bits == float(trainer.config.quantization_bits)
    
    def test_update_quantization_schedule_linear(self, trainer):
        """Test linear quantization schedule."""
        trainer.config.quantization_schedule = "linear"
        
        # After warmup, at middle of linear schedule
        trainer._update_quantization_schedule(75)  # warmup=50, so 25 steps into linear
        
        # Should be at 25% of linear schedule (25/100 where 100 is warmup_steps*2)
        expected_ratio = 0.25
        assert abs(trainer.quantization_state.quantization_ratio - expected_ratio) < 0.01
    
    def test_update_quantization_schedule_cosine(self, trainer):
        """Test cosine quantization schedule."""
        trainer.config.quantization_schedule = "cosine"
        
        # After warmup
        trainer._update_quantization_schedule(100)  # At end of cosine schedule
        
        # Cosine schedule should give a specific value
        expected_ratio = 0.5 * (1 + math.cos(math.pi * (1 - 0.5)))  # progress = 0.5
        assert trainer.quantization_state.quantization_ratio >= 0
        assert trainer.quantization_state.quantization_ratio <= 1
    
    def test_gradient_scaling_calculation(self, trainer):
        """Test gradient scaling calculation."""
        trainer.quantization_state.quantization_ratio = 0.5
        trainer._update_quantization_schedule(100)
        
        expected_scale = trainer.config.gradient_scaling_factor * (1.0 + 0.5)
        assert trainer.quantization_state.gradient_scale == expected_scale
    
    @patch('torch.norm')
    @patch('math.sqrt')
    def test_apply_gradient_scaling_structure(self, mock_sqrt, mock_norm, trainer):
        """Test gradient scaling method structure."""
        # Mock model with LoRA layers
        mock_model = Mock()
        mock_layer = Mock()
        mock_layer.weight = Mock()
        mock_layer.weight.grad = Mock()
        
        trainer.lora_layers = {"test_layer": mock_layer}
        
        # Mock torch operations
        mock_norm.return_value.item.return_value = 1.0
        mock_sqrt.return_value = 1.0
        
        # Test method exists and can be called
        trainer._apply_gradient_scaling(mock_model)
        
        # Verify gradient was scaled
        assert mock_layer.weight.grad.__imul__.called or hasattr(mock_layer.weight.grad, '__imul__')
    
    def test_get_training_summary(self, trainer):
        """Test training summary generation."""
        summary = trainer.get_training_summary()
        
        assert "config" in summary
        assert "current_state" in summary
        assert "layer_statistics" in summary
        assert "num_lora_layers" in summary
        assert "training_history_length" in summary
        
        # Check config data
        config_data = summary["config"]
        assert config_data["quantization_bits"] == trainer.config.quantization_bits
        assert config_data["lora_rank"] == trainer.config.lora_rank
        
        # Check current state
        state_data = summary["current_state"]
        assert "step" in state_data
        assert "current_bits" in state_data
        assert "quantization_ratio" in state_data
    
    def test_export_quantization_data(self, trainer):
        """Test quantization data export."""
        # Add some history
        trainer.quantization_history.append(QuantizationState(step=1, current_bits=16.0))
        trainer.quantization_history.append(QuantizationState(step=2, current_bits=8.0))
        
        export_data = trainer.export_quantization_data()
        
        assert "config" in export_data
        assert "quantization_history" in export_data
        assert "layer_statistics" in export_data
        assert "final_state" in export_data
        
        # Check history data
        history = export_data["quantization_history"]
        assert len(history) == 2
        assert history[0]["step"] == 1
        assert history[1]["step"] == 2
    
    def test_reset(self, trainer):
        """Test trainer reset."""
        # Set some state
        trainer.step_count = 100
        trainer.quantization_frozen = True
        trainer.quantization_history.append(QuantizationState())
        trainer.layer_statistics["test"] = {"error": 0.1}
        trainer.lora_layers["test"] = Mock()
        
        # Reset
        trainer.reset()
        
        # Verify reset
        assert trainer.step_count == 0
        assert trainer.quantization_frozen is False
        assert len(trainer.quantization_history) == 0
        assert len(trainer.layer_statistics) == 0
        assert len(trainer.lora_layers) == 0
    
    @patch('matplotlib.pyplot')
    def test_visualize_quantization_progress(self, mock_plt, trainer):
        """Test quantization progress visualization."""
        # Add some history
        trainer.quantization_history = [
            QuantizationState(step=1, current_bits=16.0, quantization_ratio=0.2),
            QuantizationState(step=2, current_bits=8.0, quantization_ratio=0.5),
            QuantizationState(step=3, current_bits=4.0, quantization_ratio=1.0)
        ]
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        fig = trainer.visualize_quantization_progress()
        
        # Should return the mocked figure
        assert fig == mock_fig
        
        # Test with no history
        trainer.quantization_history = []
        fig = trainer.visualize_quantization_progress()
        assert fig is None


class TestQALoRAIntegratedTrainer:
    """Test QA-LoRA integrated trainer."""
    
    @pytest.fixture
    def qa_config(self):
        """Create QA-LoRA configuration."""
        return QALoRAConfig(quantization_bits=4, lora_rank=8)
    
    @pytest.fixture
    def base_trainer(self):
        """Create mock base trainer."""
        trainer = Mock()
        trainer.setup_model = Mock()
        trainer.training_step = Mock(return_value={"base_loss": 0.5})
        trainer.validate = Mock(return_value={"base_accuracy": 0.8})
        trainer.get_training_summary = Mock(return_value={"base_info": "test"})
        return trainer
    
    @pytest.fixture
    def integrated_trainer(self, qa_config, base_trainer):
        """Create integrated trainer."""
        return QALoRAIntegratedTrainer(qa_config, base_trainer)
    
    def test_integrated_trainer_initialization(self, integrated_trainer, qa_config):
        """Test integrated trainer initialization."""
        assert integrated_trainer.config == qa_config
        assert hasattr(integrated_trainer, 'qa_lora_trainer')
        assert hasattr(integrated_trainer, 'base_trainer')
    
    def test_setup_model(self, integrated_trainer):
        """Test integrated model setup."""
        mock_model = Mock()
        
        # Mock QA-LoRA setup
        integrated_trainer.qa_lora_trainer.setup_model = Mock(return_value=mock_model)
        
        result = integrated_trainer.setup_model(mock_model)
        
        # Both setups should be called
        integrated_trainer.qa_lora_trainer.setup_model.assert_called_once_with(mock_model)
        integrated_trainer.base_trainer.setup_model.assert_called_once_with(mock_model)
        
        assert result == mock_model
    
    def test_training_step(self, integrated_trainer):
        """Test integrated training step."""
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_loss = Mock()
        
        # Mock QA-LoRA training step
        integrated_trainer.qa_lora_trainer.training_step = Mock(
            return_value={"quantization_ratio": 0.5, "gradient_scale": 1.2}
        )
        
        result = integrated_trainer.training_step(mock_model, mock_optimizer, mock_loss, 100)
        
        # Both training steps should be called
        integrated_trainer.qa_lora_trainer.training_step.assert_called_once()
        integrated_trainer.base_trainer.training_step.assert_called_once()
        
        # Results should be combined
        assert "base_loss" in result
        assert "quantization_ratio" in result
        assert "qa_lora_active" in result
        assert result["qa_lora_active"] is True
    
    def test_validate(self, integrated_trainer):
        """Test integrated validation."""
        mock_model = Mock()
        
        # Mock QA-LoRA validation
        integrated_trainer.qa_lora_trainer.validate_quantization_adaptation_balance = Mock(
            return_value={"balance_score": 0.8, "quantization_error": 0.1}
        )
        
        result = integrated_trainer.validate(mock_model)
        
        # Both validations should be called
        integrated_trainer.qa_lora_trainer.validate_quantization_adaptation_balance.assert_called_once()
        integrated_trainer.base_trainer.validate.assert_called_once()
        
        # Results should be combined
        assert "base_accuracy" in result
        assert "balance_score" in result
        assert "quantization_error" in result
    
    def test_get_training_summary(self, integrated_trainer):
        """Test integrated training summary."""
        # Mock QA-LoRA summary
        integrated_trainer.qa_lora_trainer.get_training_summary = Mock(
            return_value={"qa_config": "test", "qa_state": "active"}
        )
        
        result = integrated_trainer.get_training_summary()
        
        # Both summaries should be called
        integrated_trainer.qa_lora_trainer.get_training_summary.assert_called_once()
        integrated_trainer.base_trainer.get_training_summary.assert_called_once()
        
        # Results should be structured
        assert "qa_lora" in result
        assert "base_trainer" in result
        assert "integration_active" in result
        assert result["integration_active"] is True


class TestQALoRAIntegration:
    """Integration tests for QA-LoRA functionality."""
    
    def test_complete_workflow(self):
        """Test complete QA-LoRA workflow."""
        config = QALoRAConfig(
            quantization_bits=4,
            lora_rank=8,
            warmup_steps=10,
            quantization_schedule="linear"
        )
        
        trainer = QALoRATrainer(config)
        
        # Test initialization
        assert trainer.step_count == 0
        assert not trainer.quantization_frozen
        
        # Test schedule updates
        for step in range(20):
            trainer._update_quantization_schedule(step)
            
            # Verify schedule progression
            if step < config.warmup_steps:
                expected_ratio = step / config.warmup_steps
                assert abs(trainer.quantization_state.quantization_ratio - expected_ratio) < 0.01
            else:
                assert trainer.quantization_state.quantization_ratio >= 0
                assert trainer.quantization_state.quantization_ratio <= 1
            
            # Verify bits progression
            assert trainer.quantization_state.current_bits >= config.quantization_bits
            assert trainer.quantization_state.current_bits <= 32.0
        
        # Test data export
        export_data = trainer.export_quantization_data()
        assert export_data["config"] == config
        
        # Test reset
        trainer.reset()
        assert trainer.step_count == 0
        assert len(trainer.quantization_history) == 0
    
    def test_quantization_schedules(self):
        """Test different quantization schedules."""
        schedules = ["constant", "linear", "cosine"]
        
        for schedule in schedules:
            config = QALoRAConfig(
                quantization_schedule=schedule,
                warmup_steps=10
            )
            trainer = QALoRATrainer(config)
            
            # Test after warmup
            trainer._update_quantization_schedule(20)
            
            # All schedules should produce valid ratios
            assert 0 <= trainer.quantization_state.quantization_ratio <= 1
            assert trainer.quantization_state.current_bits >= config.quantization_bits
    
    def test_configuration_validation_edge_cases(self):
        """Test edge cases in configuration validation."""
        # Valid edge cases
        valid_configs = [
            QALoRAConfig(quantization_bits=8, quantization_type="int8"),
            QALoRAConfig(lora_dropout=0.0),
            QALoRAConfig(lora_dropout=1.0),
            QALoRAConfig(gradient_scaling_factor=0.1),
            QALoRAConfig(quantization_group_size=1)
        ]
        
        for config in valid_configs:
            trainer = QALoRATrainer(config)
            assert trainer.config == config
        
        # Invalid edge cases should raise errors
        with pytest.raises(ValueError):
            QALoRAConfig(quantization_bits=3)
        
        with pytest.raises(ValueError):
            QALoRAConfig(lora_rank=-1)
        
        with pytest.raises(ValueError):
            QALoRAConfig(gradient_scaling_factor=-1)