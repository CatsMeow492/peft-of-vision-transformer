#!/usr/bin/env python3
"""
Example demonstrating AdaLoRA (Adaptive LoRA) usage with Vision Transformers.

This example shows how to:
1. Set up a ViT model with LoRA adapters
2. Initialize AdaLoRA controller for adaptive rank allocation
3. Integrate AdaLoRA with the training pipeline
4. Monitor importance scores and rank changes during training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import our PEFT components
from models import (
    ViTModelManager, 
    LoRAConfig, 
    LoRAAdapter,
    AdaLoRAConfig,
    AdaLoRAController
)
from training import PEFTTrainer, TrainingConfig


def create_dummy_dataset(num_samples=100, num_classes=10):
    """Create a dummy dataset for demonstration."""
    # Create random image data (batch_size, channels, height, width)
    pixel_values = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(pixel_values, labels)
    return dataset


def create_dataloader_with_dict_format(dataset, batch_size=8):
    """Create dataloader that returns dict format expected by trainer."""
    def collate_fn(batch):
        pixel_values, labels = zip(*batch)
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.stack(labels)
        }
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


class AdaLoRATrainer(PEFTTrainer):
    """Extended PEFT trainer with AdaLoRA support."""
    
    def __init__(self, adalora_controller: AdaLoRAController, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adalora_controller = adalora_controller
        
        # Initialize AdaLoRA from the model
        self.adalora_controller.initialize_from_model(self.model)
        
        print(f"AdaLoRA initialized with {len(self.adalora_controller.layer_importance)} layers")
        print(f"Initial budget allocation: {self.adalora_controller.get_budget_utilization()}")
    
    def _train_epoch(self):
        """Override train epoch to include AdaLoRA updates."""
        # Call parent training method
        epoch_results = super()._train_epoch()
        
        # Update importance scores and potentially reallocate ranks
        importance_scores = self.adalora_controller.update_importance_scores(
            self.model, self.global_step
        )
        
        # Attempt rank reallocation
        new_allocation = self.adalora_controller.reallocate_ranks(
            importance_scores, self.global_step
        )
        
        # Log AdaLoRA statistics
        if self.global_step % 50 == 0:  # Log every 50 steps
            self._log_adalora_stats(importance_scores, new_allocation)
        
        return epoch_results
    
    def _log_adalora_stats(self, importance_scores, allocation):
        """Log AdaLoRA statistics."""
        print(f"\n--- AdaLoRA Stats (Step {self.global_step}) ---")
        
        # Log importance scores
        print("Layer Importance Scores:")
        for layer_name, score in importance_scores.items():
            current_rank = allocation.get(layer_name, 0)
            print(f"  {layer_name}: {score:.4f} (rank: {current_rank})")
        
        # Log budget utilization
        budget_stats = self.adalora_controller.get_budget_utilization()
        print(f"Budget Utilization: {budget_stats['allocated_budget']}/{budget_stats['total_budget']} "
              f"({budget_stats['utilization_ratio']:.2%})")
        
        # Log recent reallocations
        if budget_stats['num_reallocations'] > 0:
            print(f"Total Reallocations: {budget_stats['num_reallocations']}")
        
        print("--- End AdaLoRA Stats ---\n")


def main():
    """Main demonstration function."""
    print("AdaLoRA (Adaptive LoRA) Example for Vision Transformers")
    print("=" * 60)
    
    # Check if we have the required dependencies
    try:
        import timm
        print("✓ timm available")
    except ImportError:
        print("✗ timm not available - using mock model")
        timm = None
    
    try:
        from peft import get_peft_model
        print("✓ PEFT available")
    except ImportError:
        print("✗ PEFT not available - demonstration will be limited")
        get_peft_model = None
    
    # 1. Create a simple model for demonstration
    print("\n1. Setting up model...")
    
    if timm is not None:
        try:
            # Try to load a real ViT model
            model_manager = ViTModelManager()
            model = model_manager.load_model("deit_tiny_patch16_224", num_classes=10)
            print("✓ Loaded DeiT-tiny model")
        except Exception as e:
            print(f"✗ Failed to load real model: {e}")
            print("Using simple mock model instead")
            model = create_mock_vit_model()
    else:
        model = create_mock_vit_model()
    
    # 2. Configure LoRA
    print("\n2. Configuring LoRA...")
    lora_config = LoRAConfig(
        rank=8,  # This will be the initial rank, AdaLoRA will adjust it
        alpha=16.0,
        dropout=0.1,
        target_modules=["qkv", "proj"]  # Target attention layers
    )
    
    # Apply LoRA adapters
    if get_peft_model is not None:
        try:
            lora_adapter = LoRAAdapter()
            peft_model = lora_adapter.apply_lora(model, lora_config)
            print("✓ LoRA adapters applied")
        except Exception as e:
            print(f"✗ Failed to apply LoRA: {e}")
            print("Using original model")
            peft_model = model
    else:
        peft_model = model
    
    # 3. Configure AdaLoRA
    print("\n3. Configuring AdaLoRA...")
    adalora_config = AdaLoRAConfig(
        total_rank_budget=32,      # Total rank budget across all layers
        min_rank=2,                # Minimum rank per layer
        max_rank=12,               # Maximum rank per layer
        importance_metric="magnitude",  # Use weight magnitude for importance
        allocation_strategy="proportional",  # Proportional allocation
        update_frequency=20,       # Update ranks every 20 steps
        warmup_steps=50,          # Wait 50 steps before first update
        track_importance_history=True  # Track history for analysis
    )
    
    adalora_controller = AdaLoRAController(adalora_config)
    print("✓ AdaLoRA controller created")
    
    # 4. Create training data
    print("\n4. Creating training data...")
    train_dataset = create_dummy_dataset(num_samples=200, num_classes=10)
    eval_dataset = create_dummy_dataset(num_samples=50, num_classes=10)
    
    train_dataloader = create_dataloader_with_dict_format(train_dataset, batch_size=8)
    eval_dataloader = create_dataloader_with_dict_format(eval_dataset, batch_size=8)
    print("✓ Training data created")
    
    # 5. Configure training
    print("\n5. Configuring training...")
    training_config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=3,  # Short training for demonstration
        warmup_steps=10,
        save_steps=100,
        eval_steps=25,
        logging_steps=10,
        use_mixed_precision=False,  # Disable for compatibility
        output_dir="outputs/adalora_example"
    )
    
    # 6. Create AdaLoRA trainer
    print("\n6. Creating AdaLoRA trainer...")
    try:
        trainer = AdaLoRATrainer(
            adalora_controller=adalora_controller,
            model=peft_model,
            config=training_config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader
        )
        print("✓ AdaLoRA trainer created")
    except Exception as e:
        print(f"✗ Failed to create trainer: {e}")
        print("This is expected if PEFT dependencies are not available")
        return
    
    # 7. Run training
    print("\n7. Starting training with AdaLoRA...")
    print("Watch for AdaLoRA statistics during training!")
    
    try:
        results = trainer.train()
        print("\n✓ Training completed successfully!")
        
        # 8. Analyze AdaLoRA results
        print("\n8. AdaLoRA Analysis:")
        print("-" * 40)
        
        # Get final importance summary
        importance_summary = adalora_controller.get_layer_importance_summary()
        print("Final Layer Importance:")
        for layer_name, info in importance_summary.items():
            print(f"  {layer_name}:")
            print(f"    Current Rank: {info['current_rank']}")
            print(f"    Importance Score: {info['importance_score']:.4f}")
            print(f"    Updates: {info['update_count']}")
        
        # Get budget utilization
        budget_stats = adalora_controller.get_budget_utilization()
        print(f"\nFinal Budget Utilization:")
        print(f"  Total Budget: {budget_stats['total_budget']}")
        print(f"  Allocated: {budget_stats['allocated_budget']}")
        print(f"  Utilization: {budget_stats['utilization_ratio']:.2%}")
        print(f"  Reallocations: {budget_stats['num_reallocations']}")
        
        # Export data for further analysis
        export_data = adalora_controller.export_importance_data()
        print(f"\nExported AdaLoRA data with {len(export_data['reallocation_history'])} reallocation events")
        
        # Try to create visualization
        print("\n9. Creating visualization...")
        try:
            fig = adalora_controller.visualize_importance_evolution(
                save_path="outputs/adalora_example/importance_evolution.png"
            )
            if fig is not None:
                print("✓ Importance evolution plot saved")
            else:
                print("✗ Visualization not available (matplotlib required)")
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
        
        print("\n" + "=" * 60)
        print("AdaLoRA Example Completed Successfully!")
        print("Key Benefits Demonstrated:")
        print("- Adaptive rank allocation based on layer importance")
        print("- Dynamic budget management during training")
        print("- Comprehensive tracking and analysis")
        print("- Integration with existing PEFT training pipeline")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print("This might be due to missing dependencies or hardware limitations")


def create_mock_vit_model():
    """Create a simple mock ViT-like model for demonstration."""
    class MockViTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(3 * 16 * 16, 192)  # Patch embedding
            
            # Mock attention layers that AdaLoRA can target
            self.attention_qkv = nn.Linear(192, 192 * 3)  # Combined Q, K, V
            self.attention_proj = nn.Linear(192, 192)      # Output projection
            
            # Mock MLP layers
            self.mlp_fc1 = nn.Linear(192, 768)
            self.mlp_fc2 = nn.Linear(768, 192)
            
            # Classification head
            self.head = nn.Linear(192, 10)
        
        def forward(self, x):
            # Simple forward pass for demonstration
            batch_size = x.shape[0]
            
            # Flatten patches (simplified)
            x = x.view(batch_size, -1)
            if x.shape[1] != 3 * 16 * 16:
                # Adapt to expected input size
                x = torch.randn(batch_size, 3 * 16 * 16, device=x.device)
            
            # Patch embedding
            x = self.patch_embed(x)
            
            # Mock attention
            qkv = self.attention_qkv(x)
            x = self.attention_proj(qkv[:, :192])  # Use only Q part
            
            # Mock MLP
            x = self.mlp_fc2(torch.relu(self.mlp_fc1(x)))
            
            # Classification
            x = self.head(x)
            
            return x
    
    return MockViTModel()


if __name__ == "__main__":
    main()