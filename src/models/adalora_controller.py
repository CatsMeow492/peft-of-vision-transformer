"""
AdaLoRA (Adaptive LoRA) implementation for dynamic rank allocation in Vision Transformers.

This module implements adaptive rank allocation based on importance scoring using
SVD-based analysis of weight updates, following the AdaLoRA methodology adapted
for Vision Transformers.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

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

logger = logging.getLogger(__name__)


@dataclass
class AdaLoRAConfig:
    """Configuration for AdaLoRA adaptive rank allocation."""
    
    # Budget management
    total_rank_budget: int = 64  # Total rank budget across all layers
    min_rank: int = 1           # Minimum rank per layer
    max_rank: int = 16          # Maximum rank per layer
    
    # Importance scoring
    importance_metric: str = "magnitude"  # "magnitude", "fisher", "gradient_norm"
    update_frequency: int = 100          # Steps between rank updates
    sensitivity_threshold: float = 0.01   # Threshold for rank changes
    
    # Rank allocation strategy
    allocation_strategy: str = "proportional"  # "proportional", "threshold", "top_k"
    reallocation_ratio: float = 0.1           # Fraction of budget to reallocate
    
    # Stability and convergence
    warmup_steps: int = 500              # Steps before first rank update
    cooldown_steps: int = 100            # Steps to wait after rank change
    max_rank_changes: int = 5            # Maximum rank changes per layer
    
    # Visualization and logging
    track_importance_history: bool = True
    save_importance_plots: bool = False
    
    def __post_init__(self):
        """Validate AdaLoRA configuration."""
        if self.total_rank_budget <= 0:
            raise ValueError("Total rank budget must be positive")
        if self.min_rank <= 0:
            raise ValueError("Minimum rank must be positive")
        if self.max_rank < self.min_rank:
            raise ValueError("Maximum rank must be >= minimum rank")
        if not 0 < self.reallocation_ratio <= 1:
            raise ValueError("Reallocation ratio must be between 0 and 1")
        if self.importance_metric not in ["magnitude", "fisher", "gradient_norm"]:
            raise ValueError("Invalid importance metric")
        if self.allocation_strategy not in ["proportional", "threshold", "top_k"]:
            raise ValueError("Invalid allocation strategy")


@dataclass
class LayerImportance:
    """Container for layer importance information."""
    
    layer_name: str
    current_rank: int
    importance_score: float
    gradient_magnitude: float
    weight_magnitude: float
    svd_entropy: float
    update_count: int = 0
    last_rank_change: int = 0
    
    # Historical tracking
    importance_history: List[float] = field(default_factory=list)
    rank_history: List[int] = field(default_factory=list)


class AdaLoRAController:
    """
    Controller for adaptive LoRA rank allocation in Vision Transformers.
    
    Implements importance-based rank reallocation using SVD analysis of weight
    updates to dynamically adjust LoRA ranks during training.
    """
    
    def __init__(self, config: AdaLoRAConfig):
        """
        Initialize AdaLoRA controller.
        
        Args:
            config: AdaLoRA configuration
        """
        self.config = config
        self.step_count = 0
        self.last_update_step = 0
        
        # Layer tracking
        self.layer_importance: Dict[str, LayerImportance] = {}
        self.lora_layers: Dict[str, nn.Module] = {}
        self.previous_weights: Dict[str, torch.Tensor] = {}
        
        # Budget management
        self.current_budget_allocation: Dict[str, int] = {}
        self.total_allocated_rank = 0
        
        # Statistics
        self.reallocation_history: List[Dict[str, Any]] = []
        self.importance_statistics: Dict[str, List[float]] = {}
        
        logger.info(f"AdaLoRA controller initialized with budget: {config.total_rank_budget}")
    
    def initialize_from_model(self, model: "PeftModel") -> None:
        """
        Initialize controller from a PEFT model with LoRA adapters.
        
        Args:
            model: PEFT model with LoRA adapters applied
        """
        if torch is None:
            raise RuntimeError("PyTorch not available for model initialization")
        self.lora_layers.clear()
        self.layer_importance.clear()
        self.current_budget_allocation.clear()
        
        # Find all LoRA layers
        lora_layer_names = []
        for name, module in model.named_modules():
            if "lora" in name.lower() and hasattr(module, 'weight'):
                self.lora_layers[name] = module
                lora_layer_names.append(name)
        
        if not self.lora_layers:
            raise ValueError("No LoRA layers found in model")
        
        # Initialize uniform rank allocation
        num_layers = len(self.lora_layers)
        initial_rank = max(self.config.min_rank, 
                          self.config.total_rank_budget // num_layers)
        
        # Ensure we don't exceed budget
        if initial_rank * num_layers > self.config.total_rank_budget:
            initial_rank = self.config.total_rank_budget // num_layers
        
        # Initialize layer importance tracking
        for layer_name in lora_layer_names:
            self.layer_importance[layer_name] = LayerImportance(
                layer_name=layer_name,
                current_rank=initial_rank,
                importance_score=1.0,  # Start with uniform importance
                gradient_magnitude=0.0,
                weight_magnitude=0.0,
                svd_entropy=0.0
            )
            self.current_budget_allocation[layer_name] = initial_rank
        
        self.total_allocated_rank = sum(self.current_budget_allocation.values())
        
        # Store initial weights for tracking updates
        self._store_current_weights()
        
        logger.info(f"Initialized AdaLoRA for {num_layers} layers with rank {initial_rank}")
        logger.info(f"Total allocated rank: {self.total_allocated_rank}/{self.config.total_rank_budget}")
    
    def update_importance_scores(self, model: "PeftModel", step: int) -> Dict[str, float]:
        """
        Update importance scores for all LoRA layers.
        
        Args:
            model: PEFT model to analyze
            step: Current training step
            
        Returns:
            Dictionary of layer names to importance scores
        """
        self.step_count = step
        importance_scores = {}
        
        for layer_name, layer_info in self.layer_importance.items():
            if layer_name not in self.lora_layers:
                continue
            
            module = self.lora_layers[layer_name]
            
            # Compute different importance metrics
            gradient_magnitude = self._compute_gradient_magnitude(module)
            weight_magnitude = self._compute_weight_magnitude(module)
            svd_entropy = self._compute_svd_entropy(module)
            
            # Combine metrics based on configuration
            if self.config.importance_metric == "magnitude":
                importance = weight_magnitude
            elif self.config.importance_metric == "gradient_norm":
                importance = gradient_magnitude
            elif self.config.importance_metric == "fisher":
                # Simplified Fisher information approximation
                importance = gradient_magnitude * weight_magnitude
            else:
                importance = weight_magnitude  # Default fallback
            
            # Update layer information
            layer_info.importance_score = importance
            layer_info.gradient_magnitude = gradient_magnitude
            layer_info.weight_magnitude = weight_magnitude
            layer_info.svd_entropy = svd_entropy
            layer_info.update_count += 1
            
            # Track history if enabled
            if self.config.track_importance_history:
                layer_info.importance_history.append(importance)
                layer_info.rank_history.append(layer_info.current_rank)
            
            importance_scores[layer_name] = importance
        
        # Update statistics
        for layer_name, score in importance_scores.items():
            if layer_name not in self.importance_statistics:
                self.importance_statistics[layer_name] = []
            self.importance_statistics[layer_name].append(score)
        
        return importance_scores
    
    def should_update_ranks(self, step: int) -> bool:
        """
        Determine if ranks should be updated at the current step.
        
        Args:
            step: Current training step
            
        Returns:
            True if ranks should be updated
        """
        # Check warmup period
        if step < self.config.warmup_steps:
            return False
        
        # Check update frequency
        if step - self.last_update_step < self.config.update_frequency:
            return False
        
        # Check cooldown period
        if step - self.last_update_step < self.config.cooldown_steps:
            return False
        
        return True
    
    def reallocate_ranks(self, importance_scores: Dict[str, float], step: int) -> Dict[str, int]:
        """
        Reallocate ranks based on importance scores.
        
        Args:
            importance_scores: Dictionary of layer importance scores
            step: Current training step
            
        Returns:
            Dictionary of new rank allocations
        """
        if not self.should_update_ranks(step):
            return self.current_budget_allocation.copy()
        
        # Calculate new rank allocation
        new_allocation = self._calculate_new_allocation(importance_scores)
        
        # Validate and adjust allocation
        new_allocation = self._validate_allocation(new_allocation)
        
        # Check if changes are significant enough
        if not self._allocation_changed_significantly(new_allocation):
            return self.current_budget_allocation.copy()
        
        # Apply rank changes
        rank_changes = self._apply_rank_changes(new_allocation, step)
        
        # Update tracking
        self.last_update_step = step
        self._record_reallocation(importance_scores, new_allocation, step)
        
        logger.info(f"Rank reallocation at step {step}: {rank_changes} changes")
        
        return new_allocation
    
    def _calculate_new_allocation(self, importance_scores: Dict[str, float]) -> Dict[str, int]:
        """Calculate new rank allocation based on importance scores."""
        if self.config.allocation_strategy == "proportional":
            return self._proportional_allocation(importance_scores)
        elif self.config.allocation_strategy == "threshold":
            return self._threshold_allocation(importance_scores)
        elif self.config.allocation_strategy == "top_k":
            return self._top_k_allocation(importance_scores)
        else:
            raise ValueError(f"Unknown allocation strategy: {self.config.allocation_strategy}")
    
    def _proportional_allocation(self, importance_scores: Dict[str, float]) -> Dict[str, int]:
        """Allocate ranks proportionally to importance scores."""
        # Normalize importance scores
        total_importance = sum(importance_scores.values())
        if total_importance == 0:
            return self.current_budget_allocation.copy()
        
        new_allocation = {}
        remaining_budget = self.config.total_rank_budget
        
        # Calculate proportional allocation
        for layer_name, importance in importance_scores.items():
            proportion = importance / total_importance
            target_rank = int(proportion * self.config.total_rank_budget)
            
            # Apply constraints
            target_rank = max(self.config.min_rank, 
                            min(self.config.max_rank, target_rank))
            
            new_allocation[layer_name] = target_rank
            remaining_budget -= target_rank
        
        # Distribute remaining budget to highest importance layers
        if remaining_budget > 0:
            sorted_layers = sorted(importance_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            for layer_name, _ in sorted_layers:
                if remaining_budget <= 0:
                    break
                
                current_rank = new_allocation[layer_name]
                if current_rank < self.config.max_rank:
                    additional = min(remaining_budget, 
                                   self.config.max_rank - current_rank)
                    new_allocation[layer_name] += additional
                    remaining_budget -= additional
        
        return new_allocation
    
    def _threshold_allocation(self, importance_scores: Dict[str, float]) -> Dict[str, int]:
        """Allocate ranks based on importance threshold."""
        # Calculate threshold as percentile of importance scores
        scores = list(importance_scores.values())
        threshold = torch.quantile(torch.tensor(scores), 0.5).item()
        
        new_allocation = {}
        for layer_name, importance in importance_scores.items():
            if importance > threshold:
                new_allocation[layer_name] = self.config.max_rank
            else:
                new_allocation[layer_name] = self.config.min_rank
        
        # Adjust to fit budget
        return self._adjust_to_budget(new_allocation)
    
    def _top_k_allocation(self, importance_scores: Dict[str, float]) -> Dict[str, int]:
        """Allocate high ranks to top-k most important layers."""
        k = len(importance_scores) // 2  # Top half gets high rank
        
        sorted_layers = sorted(importance_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        new_allocation = {}
        for i, (layer_name, _) in enumerate(sorted_layers):
            if i < k:
                new_allocation[layer_name] = self.config.max_rank
            else:
                new_allocation[layer_name] = self.config.min_rank
        
        return self._adjust_to_budget(new_allocation)
    
    def _adjust_to_budget(self, allocation: Dict[str, int]) -> Dict[str, int]:
        """Adjust allocation to fit within budget constraints."""
        total_allocated = sum(allocation.values())
        
        if total_allocated <= self.config.total_rank_budget:
            return allocation
        
        # Scale down proportionally
        scale_factor = self.config.total_rank_budget / total_allocated
        
        adjusted_allocation = {}
        for layer_name, rank in allocation.items():
            adjusted_rank = max(self.config.min_rank, 
                              int(rank * scale_factor))
            adjusted_allocation[layer_name] = adjusted_rank
        
        return adjusted_allocation
    
    def _validate_allocation(self, allocation: Dict[str, int]) -> Dict[str, int]:
        """Validate and fix allocation constraints."""
        validated_allocation = {}
        
        for layer_name, rank in allocation.items():
            # Apply min/max constraints
            validated_rank = max(self.config.min_rank, 
                               min(self.config.max_rank, rank))
            validated_allocation[layer_name] = validated_rank
        
        # Ensure budget constraint
        total_allocated = sum(validated_allocation.values())
        if total_allocated > self.config.total_rank_budget:
            validated_allocation = self._adjust_to_budget(validated_allocation)
        
        return validated_allocation
    
    def _allocation_changed_significantly(self, new_allocation: Dict[str, int]) -> bool:
        """Check if allocation changed significantly."""
        total_change = 0
        for layer_name, new_rank in new_allocation.items():
            old_rank = self.current_budget_allocation.get(layer_name, 0)
            total_change += abs(new_rank - old_rank)
        
        # Consider significant if change is above threshold
        change_ratio = total_change / self.config.total_rank_budget
        return change_ratio > self.config.sensitivity_threshold
    
    def _apply_rank_changes(self, new_allocation: Dict[str, int], step: int) -> int:
        """Apply rank changes and update tracking."""
        changes_count = 0
        
        for layer_name, new_rank in new_allocation.items():
            old_rank = self.current_budget_allocation.get(layer_name, 0)
            
            if new_rank != old_rank:
                # Check if layer has exceeded max changes
                layer_info = self.layer_importance[layer_name]
                if (layer_info.last_rank_change > 0 and 
                    step - layer_info.last_rank_change < self.config.cooldown_steps):
                    continue  # Skip this change due to cooldown
                
                # Update allocation
                self.current_budget_allocation[layer_name] = new_rank
                layer_info.current_rank = new_rank
                layer_info.last_rank_change = step
                changes_count += 1
        
        # Update total allocated rank
        self.total_allocated_rank = sum(self.current_budget_allocation.values())
        
        return changes_count
    
    def _record_reallocation(self, importance_scores: Dict[str, float], 
                           new_allocation: Dict[str, int], step: int) -> None:
        """Record reallocation event for analysis."""
        reallocation_record = {
            "step": step,
            "importance_scores": importance_scores.copy(),
            "old_allocation": self.current_budget_allocation.copy(),
            "new_allocation": new_allocation.copy(),
            "total_budget_used": sum(new_allocation.values()),
            "allocation_entropy": self._calculate_allocation_entropy(new_allocation)
        }
        
        self.reallocation_history.append(reallocation_record)
    
    def _calculate_allocation_entropy(self, allocation: Dict[str, int]) -> float:
        """Calculate entropy of rank allocation distribution."""
        ranks = list(allocation.values())
        total_rank = sum(ranks)
        
        if total_rank == 0:
            return 0.0
        
        # Calculate normalized probabilities
        probabilities = [r / total_rank for r in ranks]
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _compute_gradient_magnitude(self, module) -> float:
        """Compute gradient magnitude for a module."""
        if not hasattr(module, 'weight') or module.weight.grad is None:
            return 0.0
        
        grad_norm = torch.norm(module.weight.grad).item()
        return grad_norm
    
    def _compute_weight_magnitude(self, module) -> float:
        """Compute weight magnitude for a module."""
        if not hasattr(module, 'weight'):
            return 0.0
        
        weight_norm = torch.norm(module.weight).item()
        return weight_norm
    
    def _compute_svd_entropy(self, module) -> float:
        """Compute SVD-based entropy of weight matrix."""
        if not hasattr(module, 'weight'):
            return 0.0
        
        try:
            # Perform SVD
            U, S, V = torch.svd(module.weight.detach())
            
            # Normalize singular values
            S_normalized = S / torch.sum(S)
            
            # Calculate entropy
            entropy = 0.0
            for s in S_normalized:
                if s > 1e-8:  # Avoid log(0)
                    entropy -= s * torch.log(s)
            
            return entropy.item()
            
        except Exception:
            return 0.0
    
    def _store_current_weights(self) -> None:
        """Store current weights for tracking updates."""
        for layer_name, module in self.lora_layers.items():
            if hasattr(module, 'weight'):
                self.previous_weights[layer_name] = module.weight.detach().clone()
    
    def get_layer_importance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of layer importance information."""
        summary = {}
        
        for layer_name, layer_info in self.layer_importance.items():
            summary[layer_name] = {
                "current_rank": layer_info.current_rank,
                "importance_score": layer_info.importance_score,
                "gradient_magnitude": layer_info.gradient_magnitude,
                "weight_magnitude": layer_info.weight_magnitude,
                "svd_entropy": layer_info.svd_entropy,
                "update_count": layer_info.update_count,
                "last_rank_change": layer_info.last_rank_change
            }
            
            # Add historical statistics if available
            if layer_info.importance_history:
                summary[layer_name]["avg_importance"] = sum(layer_info.importance_history) / len(layer_info.importance_history)
                summary[layer_name]["importance_std"] = torch.std(torch.tensor(layer_info.importance_history)).item()
        
        return summary
    
    def get_budget_utilization(self) -> Dict[str, Any]:
        """Get budget utilization statistics."""
        return {
            "total_budget": self.config.total_rank_budget,
            "allocated_budget": self.total_allocated_rank,
            "utilization_ratio": self.total_allocated_rank / self.config.total_rank_budget,
            "allocation_per_layer": self.current_budget_allocation.copy(),
            "num_reallocations": len(self.reallocation_history),
            "last_reallocation_step": self.last_update_step
        }
    
    def visualize_importance_evolution(self, save_path: Optional[str] = None) -> Optional[Any]:
        """
        Visualize the evolution of layer importance over time.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure if available, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not self.config.track_importance_history:
                logger.warning("Importance history tracking is disabled")
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot importance evolution
            for layer_name, layer_info in self.layer_importance.items():
                if layer_info.importance_history:
                    steps = range(len(layer_info.importance_history))
                    ax1.plot(steps, layer_info.importance_history, 
                            label=layer_name, alpha=0.7)
            
            ax1.set_xlabel("Update Step")
            ax1.set_ylabel("Importance Score")
            ax1.set_title("Layer Importance Evolution")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot rank evolution
            for layer_name, layer_info in self.layer_importance.items():
                if layer_info.rank_history:
                    steps = range(len(layer_info.rank_history))
                    ax2.plot(steps, layer_info.rank_history, 
                            label=layer_name, marker='o', markersize=3)
            
            ax2.set_xlabel("Update Step")
            ax2.set_ylabel("Allocated Rank")
            ax2.set_title("Rank Allocation Evolution")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Importance evolution plot saved to {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return None
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            return None
    
    def export_importance_data(self) -> Dict[str, Any]:
        """Export importance tracking data for analysis."""
        export_data = {
            "config": {
                "total_rank_budget": self.config.total_rank_budget,
                "min_rank": self.config.min_rank,
                "max_rank": self.config.max_rank,
                "importance_metric": self.config.importance_metric,
                "allocation_strategy": self.config.allocation_strategy
            },
            "layer_importance": {},
            "reallocation_history": self.reallocation_history.copy(),
            "importance_statistics": self.importance_statistics.copy(),
            "budget_utilization": self.get_budget_utilization()
        }
        
        # Export layer-wise data
        for layer_name, layer_info in self.layer_importance.items():
            export_data["layer_importance"][layer_name] = {
                "current_rank": layer_info.current_rank,
                "importance_score": layer_info.importance_score,
                "gradient_magnitude": layer_info.gradient_magnitude,
                "weight_magnitude": layer_info.weight_magnitude,
                "svd_entropy": layer_info.svd_entropy,
                "update_count": layer_info.update_count,
                "importance_history": layer_info.importance_history.copy(),
                "rank_history": layer_info.rank_history.copy()
            }
        
        return export_data
    
    def reset(self) -> None:
        """Reset controller state."""
        self.step_count = 0
        self.last_update_step = 0
        self.layer_importance.clear()
        self.lora_layers.clear()
        self.previous_weights.clear()
        self.current_budget_allocation.clear()
        self.total_allocated_rank = 0
        self.reallocation_history.clear()
        self.importance_statistics.clear()
        
        logger.info("AdaLoRA controller reset")