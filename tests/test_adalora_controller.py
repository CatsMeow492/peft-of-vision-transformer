"""
Tests for AdaLoRA controller implementation.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from src.models.adalora_controller import (
    AdaLoRAConfig, 
    AdaLoRAController, 
    LayerImportance
)


class TestAdaLoRAConfig:
    """Test AdaLoRA configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaLoRAConfig()
        
        assert config.total_rank_budget == 64
        assert config.min_rank == 1
        assert config.max_rank == 16
        assert config.importance_metric == "magnitude"
        assert config.allocation_strategy == "proportional"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid total budget
        with pytest.raises(ValueError, match="Total rank budget must be positive"):
            AdaLoRAConfig(total_rank_budget=0)
        
        # Test invalid min rank
        with pytest.raises(ValueError, match="Minimum rank must be positive"):
            AdaLoRAConfig(min_rank=0)
        
        # Test invalid max rank
        with pytest.raises(ValueError, match="Maximum rank must be >= minimum rank"):
            AdaLoRAConfig(min_rank=10, max_rank=5)
        
        # Test invalid reallocation ratio
        with pytest.raises(ValueError, match="Reallocation ratio must be between 0 and 1"):
            AdaLoRAConfig(reallocation_ratio=1.5)
        
        # Test invalid importance metric
        with pytest.raises(ValueError, match="Invalid importance metric"):
            AdaLoRAConfig(importance_metric="invalid")
        
        # Test invalid allocation strategy
        with pytest.raises(ValueError, match="Invalid allocation strategy"):
            AdaLoRAConfig(allocation_strategy="invalid")


class TestLayerImportance:
    """Test LayerImportance data structure."""
    
    def test_layer_importance_creation(self):
        """Test LayerImportance creation."""
        layer_info = LayerImportance(
            layer_name="test_layer",
            current_rank=8,
            importance_score=0.5,
            gradient_magnitude=0.1,
            weight_magnitude=0.2,
            svd_entropy=1.5
        )
        
        assert layer_info.layer_name == "test_layer"
        assert layer_info.current_rank == 8
        assert layer_info.importance_score == 0.5
        assert layer_info.update_count == 0
        assert layer_info.last_rank_change == 0
        assert len(layer_info.importance_history) == 0
        assert len(layer_info.rank_history) == 0


class TestAdaLoRAController:
    """Test AdaLoRA controller functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AdaLoRAConfig(
            total_rank_budget=32,
            min_rank=2,
            max_rank=8,
            update_frequency=10,
            warmup_steps=20
        )
    
    @pytest.fixture
    def controller(self, config):
        """Create test controller."""
        return AdaLoRAController(config)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock PEFT model."""
        model = Mock()
        
        # Create mock LoRA layers
        lora_layer1 = Mock()
        lora_layer1.weight = torch.randn(64, 32, requires_grad=True)
        lora_layer1.weight.grad = torch.randn(64, 32)
        
        lora_layer2 = Mock()
        lora_layer2.weight = torch.randn(32, 16, requires_grad=True)
        lora_layer2.weight.grad = torch.randn(32, 16)
        
        # Mock named_modules to return LoRA layers
        model.named_modules.return_value = [
            ("base_model.encoder.layer.0.attention.self.query.lora_A", lora_layer1),
            ("base_model.encoder.layer.0.attention.self.value.lora_B", lora_layer2),
            ("base_model.encoder.layer.1.output", Mock())  # Non-LoRA layer
        ]
        
        return model
    
    def test_controller_initialization(self, config):
        """Test controller initialization."""
        controller = AdaLoRAController(config)
        
        assert controller.config == config
        assert controller.step_count == 0
        assert controller.last_update_step == 0
        assert len(controller.layer_importance) == 0
        assert len(controller.lora_layers) == 0
        assert controller.total_allocated_rank == 0
    
    def test_initialize_from_model(self, controller, mock_model):
        """Test initialization from PEFT model."""
        controller.initialize_from_model(mock_model)
        
        # Should find 2 LoRA layers
        assert len(controller.lora_layers) == 2
        assert len(controller.layer_importance) == 2
        assert len(controller.current_budget_allocation) == 2
        
        # Check initial rank allocation
        expected_rank = controller.config.total_rank_budget // 2  # 32 // 2 = 16, but max is 8
        expected_rank = min(expected_rank, controller.config.max_rank)  # 8
        
        for layer_info in controller.layer_importance.values():
            assert layer_info.current_rank == expected_rank
            assert layer_info.importance_score == 1.0
        
        assert controller.total_allocated_rank <= controller.config.total_rank_budget
    
    def test_initialize_from_model_no_lora_layers(self, controller):
        """Test initialization with no LoRA layers."""
        mock_model = Mock()
        mock_model.named_modules.return_value = [
            ("base_model.encoder.layer.0.attention", Mock()),
            ("base_model.encoder.layer.1.output", Mock())
        ]
        
        with pytest.raises(ValueError, match="No LoRA layers found in model"):
            controller.initialize_from_model(mock_model)
    
    def test_update_importance_scores(self, controller, mock_model):
        """Test importance score calculation."""
        controller.initialize_from_model(mock_model)
        
        # Update importance scores
        importance_scores = controller.update_importance_scores(mock_model, step=1)
        
        assert len(importance_scores) == 2
        assert all(score >= 0 for score in importance_scores.values())
        
        # Check that layer information was updated
        for layer_info in controller.layer_importance.values():
            assert layer_info.update_count == 1
            assert layer_info.gradient_magnitude >= 0
            assert layer_info.weight_magnitude >= 0
            assert layer_info.svd_entropy >= 0
    
    def test_should_update_ranks(self, controller):
        """Test rank update timing logic."""
        # Before warmup
        assert not controller.should_update_ranks(10)
        
        # After warmup but before update frequency
        assert not controller.should_update_ranks(25)
        
        # After warmup and update frequency
        controller.last_update_step = 0
        assert controller.should_update_ranks(30)
        
        # Too soon after last update
        controller.last_update_step = 25
        assert not controller.should_update_ranks(30)
    
    def test_proportional_allocation(self, controller, mock_model):
        """Test proportional rank allocation."""
        controller.initialize_from_model(mock_model)
        
        # Create mock importance scores
        importance_scores = {
            list(controller.layer_importance.keys())[0]: 0.8,
            list(controller.layer_importance.keys())[1]: 0.2
        }
        
        new_allocation = controller._proportional_allocation(importance_scores)
        
        # Check that allocation respects budget
        total_allocated = sum(new_allocation.values())
        assert total_allocated <= controller.config.total_rank_budget
        
        # Check that higher importance gets more rank
        layer_names = list(importance_scores.keys())
        assert new_allocation[layer_names[0]] >= new_allocation[layer_names[1]]
    
    def test_threshold_allocation(self, controller, mock_model):
        """Test threshold-based rank allocation."""
        controller.initialize_from_model(mock_model)
        
        importance_scores = {
            list(controller.layer_importance.keys())[0]: 0.8,
            list(controller.layer_importance.keys())[1]: 0.2
        }
        
        new_allocation = controller._threshold_allocation(importance_scores)
        
        # Check budget constraint
        total_allocated = sum(new_allocation.values())
        assert total_allocated <= controller.config.total_rank_budget
        
        # Check min/max constraints
        for rank in new_allocation.values():
            assert controller.config.min_rank <= rank <= controller.config.max_rank
    
    def test_top_k_allocation(self, controller, mock_model):
        """Test top-k rank allocation."""
        controller.initialize_from_model(mock_model)
        
        importance_scores = {
            list(controller.layer_importance.keys())[0]: 0.8,
            list(controller.layer_importance.keys())[1]: 0.2
        }
        
        new_allocation = controller._top_k_allocation(importance_scores)
        
        # Check budget constraint
        total_allocated = sum(new_allocation.values())
        assert total_allocated <= controller.config.total_rank_budget
    
    def test_validate_allocation(self, controller, mock_model):
        """Test allocation validation."""
        controller.initialize_from_model(mock_model)
        
        # Create invalid allocation (exceeds max rank)
        layer_names = list(controller.layer_importance.keys())
        invalid_allocation = {
            layer_names[0]: 20,  # Exceeds max_rank (8)
            layer_names[1]: 0    # Below min_rank (2)
        }
        
        validated = controller._validate_allocation(invalid_allocation)
        
        # Check constraints are enforced
        assert validated[layer_names[0]] <= controller.config.max_rank
        assert validated[layer_names[1]] >= controller.config.min_rank
        
        # Check budget constraint
        total_allocated = sum(validated.values())
        assert total_allocated <= controller.config.total_rank_budget
    
    def test_allocation_changed_significantly(self, controller, mock_model):
        """Test significant change detection."""
        controller.initialize_from_model(mock_model)
        
        layer_names = list(controller.layer_importance.keys())
        
        # Small change
        small_change = controller.current_budget_allocation.copy()
        small_change[layer_names[0]] += 1
        assert not controller._allocation_changed_significantly(small_change)
        
        # Large change
        large_change = controller.current_budget_allocation.copy()
        large_change[layer_names[0]] = controller.config.max_rank
        large_change[layer_names[1]] = controller.config.min_rank
        # This might or might not be significant depending on the threshold
        # Just check it doesn't crash
        controller._allocation_changed_significantly(large_change)
    
    def test_reallocate_ranks(self, controller, mock_model):
        """Test complete rank reallocation process."""
        controller.initialize_from_model(mock_model)
        
        importance_scores = {
            list(controller.layer_importance.keys())[0]: 0.8,
            list(controller.layer_importance.keys())[1]: 0.2
        }
        
        # Should not update during warmup
        new_allocation = controller.reallocate_ranks(importance_scores, step=10)
        assert new_allocation == controller.current_budget_allocation
        
        # Should update after warmup and frequency
        new_allocation = controller.reallocate_ranks(importance_scores, step=50)
        
        # Check that allocation is valid
        total_allocated = sum(new_allocation.values())
        assert total_allocated <= controller.config.total_rank_budget
    
    def test_compute_gradient_magnitude(self, controller):
        """Test gradient magnitude computation."""
        # Module with gradient
        module = Mock()
        module.weight = torch.randn(10, 5, requires_grad=True)
        module.weight.grad = torch.randn(10, 5)
        
        grad_mag = controller._compute_gradient_magnitude(module)
        assert grad_mag >= 0
        
        # Module without gradient
        module.weight.grad = None
        grad_mag = controller._compute_gradient_magnitude(module)
        assert grad_mag == 0.0
        
        # Module without weight
        module_no_weight = Mock()
        del module_no_weight.weight
        grad_mag = controller._compute_gradient_magnitude(module_no_weight)
        assert grad_mag == 0.0
    
    def test_compute_weight_magnitude(self, controller):
        """Test weight magnitude computation."""
        # Module with weight
        module = Mock()
        module.weight = torch.randn(10, 5)
        
        weight_mag = controller._compute_weight_magnitude(module)
        assert weight_mag >= 0
        
        # Module without weight
        module_no_weight = Mock()
        del module_no_weight.weight
        weight_mag = controller._compute_weight_magnitude(module_no_weight)
        assert weight_mag == 0.0
    
    def test_compute_svd_entropy(self, controller):
        """Test SVD entropy computation."""
        # Module with weight
        module = Mock()
        module.weight = torch.randn(10, 5)
        
        entropy = controller._compute_svd_entropy(module)
        assert entropy >= 0
        
        # Module without weight
        module_no_weight = Mock()
        del module_no_weight.weight
        entropy = controller._compute_svd_entropy(module_no_weight)
        assert entropy == 0.0
    
    def test_get_layer_importance_summary(self, controller, mock_model):
        """Test layer importance summary."""
        controller.initialize_from_model(mock_model)
        controller.update_importance_scores(mock_model, step=1)
        
        summary = controller.get_layer_importance_summary()
        
        assert len(summary) == 2
        for layer_name, info in summary.items():
            assert "current_rank" in info
            assert "importance_score" in info
            assert "gradient_magnitude" in info
            assert "weight_magnitude" in info
            assert "svd_entropy" in info
            assert "update_count" in info
    
    def test_get_budget_utilization(self, controller, mock_model):
        """Test budget utilization statistics."""
        controller.initialize_from_model(mock_model)
        
        utilization = controller.get_budget_utilization()
        
        assert "total_budget" in utilization
        assert "allocated_budget" in utilization
        assert "utilization_ratio" in utilization
        assert "allocation_per_layer" in utilization
        assert "num_reallocations" in utilization
        
        assert utilization["total_budget"] == controller.config.total_rank_budget
        assert 0 <= utilization["utilization_ratio"] <= 1
    
    def test_export_importance_data(self, controller, mock_model):
        """Test importance data export."""
        controller.initialize_from_model(mock_model)
        controller.update_importance_scores(mock_model, step=1)
        
        export_data = controller.export_importance_data()
        
        assert "config" in export_data
        assert "layer_importance" in export_data
        assert "reallocation_history" in export_data
        assert "importance_statistics" in export_data
        assert "budget_utilization" in export_data
        
        # Check config data
        config_data = export_data["config"]
        assert config_data["total_rank_budget"] == controller.config.total_rank_budget
        assert config_data["importance_metric"] == controller.config.importance_metric
    
    def test_reset(self, controller, mock_model):
        """Test controller reset."""
        controller.initialize_from_model(mock_model)
        controller.update_importance_scores(mock_model, step=1)
        
        # Verify state before reset
        assert len(controller.layer_importance) > 0
        assert len(controller.lora_layers) > 0
        assert controller.total_allocated_rank > 0
        
        # Reset
        controller.reset()
        
        # Verify state after reset
        assert controller.step_count == 0
        assert controller.last_update_step == 0
        assert len(controller.layer_importance) == 0
        assert len(controller.lora_layers) == 0
        assert controller.total_allocated_rank == 0
        assert len(controller.reallocation_history) == 0
    
    @patch('matplotlib.pyplot')
    def test_visualize_importance_evolution(self, mock_plt, controller, mock_model):
        """Test importance evolution visualization."""
        # Enable history tracking
        controller.config.track_importance_history = True
        controller.initialize_from_model(mock_model)
        
        # Add some history
        for step in range(5):
            controller.update_importance_scores(mock_model, step=step)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_plt.subplots.return_value = (mock_fig, (Mock(), Mock()))
        
        fig = controller.visualize_importance_evolution()
        
        # Should return the mocked figure
        assert fig == mock_fig
        
        # Test with history tracking disabled
        controller.config.track_importance_history = False
        fig = controller.visualize_importance_evolution()
        assert fig is None
    
    def test_calculate_allocation_entropy(self, controller):
        """Test allocation entropy calculation."""
        # Uniform allocation
        uniform_allocation = {"layer1": 4, "layer2": 4, "layer3": 4}
        entropy = controller._calculate_allocation_entropy(uniform_allocation)
        assert entropy > 0
        
        # Non-uniform allocation
        skewed_allocation = {"layer1": 10, "layer2": 1, "layer3": 1}
        skewed_entropy = controller._calculate_allocation_entropy(skewed_allocation)
        assert skewed_entropy < entropy  # Less entropy for skewed distribution
        
        # Empty allocation
        empty_allocation = {"layer1": 0, "layer2": 0}
        empty_entropy = controller._calculate_allocation_entropy(empty_allocation)
        assert empty_entropy == 0.0


class TestAdaLoRAIntegration:
    """Integration tests for AdaLoRA with PEFT models."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        return model
    
    def test_integration_workflow(self):
        """Test complete AdaLoRA workflow."""
        config = AdaLoRAConfig(
            total_rank_budget=16,
            min_rank=1,
            max_rank=8,
            update_frequency=5,
            warmup_steps=10
        )
        
        controller = AdaLoRAController(config)
        
        # Create mock model with LoRA layers
        mock_model = Mock()
        lora_layer = Mock()
        lora_layer.weight = torch.randn(10, 5, requires_grad=True)
        lora_layer.weight.grad = torch.randn(10, 5)
        
        mock_model.named_modules.return_value = [
            ("lora_layer", lora_layer)
        ]
        
        # Initialize controller
        controller.initialize_from_model(mock_model)
        assert len(controller.layer_importance) == 1
        
        # Simulate training steps
        for step in range(20):
            importance_scores = controller.update_importance_scores(mock_model, step)
            new_allocation = controller.reallocate_ranks(importance_scores, step)
            
            # Verify allocation is always valid
            total_allocated = sum(new_allocation.values())
            assert total_allocated <= config.total_rank_budget
            
            for rank in new_allocation.values():
                assert config.min_rank <= rank <= config.max_rank
        
        # Check that we have tracking data
        summary = controller.get_layer_importance_summary()
        assert len(summary) == 1
        
        utilization = controller.get_budget_utilization()
        assert utilization["total_budget"] == config.total_rank_budget