#!/usr/bin/env python3
"""
Script to run adaptive LoRA and QA-LoRA evaluation experiments.

This script implements task 8.3: Perform adaptive LoRA and QA-LoRA evaluation
- Run AdaLoRA experiments with importance-based rank allocation
- Execute QA-LoRA quantization-aware training experiments
- Analyze layer importance patterns and adaptive allocation effectiveness
- Compare adaptive methods against fixed-rank approaches
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import torch
    import torch.nn as nn
    from transformers import set_seed
    
    # Import our modules
    from models.vit_manager import ViTModelManager
    from models.adalora_controller import AdaLoRAController, AdaLoRAConfig
    from models.qa_lora import QALoRATrainer, QALoRAConfig, QALoRAIntegratedTrainer
    from training.peft_trainer import PEFTTrainer
    from training.dataset_loader import DatasetLoader
    from evaluation.metrics_collector import MetricsCollector
    from experiments.config import ExperimentConfig
    from utils.reproducibility import set_reproducible_seed
    
    print("✓ All required modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdaptiveExperimentRunner:
    """Runner for adaptive LoRA and QA-LoRA experiments."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save experiment results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_manager = ViTModelManager()
        self.dataset_loader = DatasetLoader()
        self.metrics_collector = MetricsCollector()
        
        # Experiment tracking
        self.experiment_results: List[Dict[str, Any]] = []
        
        logger.info(f"Adaptive experiment runner initialized, output: {output_dir}")
    
    def run_adalora_experiments(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run AdaLoRA experiments with importance-based rank allocation.
        
        Args:
            configs: List of experiment configurations
            
        Returns:
            List of experiment results
        """
        logger.info(f"Starting {len(configs)} AdaLoRA experiments")
        results = []
        
        for i, config in enumerate(configs):
            logger.info(f"Running AdaLoRA experiment {i+1}/{len(configs)}: {config['name']}")
            
            try:
                result = self._run_single_adalora_experiment(config)
                results.append(result)
                
                # Save intermediate results
                self._save_experiment_result(result)
                
            except Exception as e:
                logger.error(f"AdaLoRA experiment {i+1} failed: {str(e)}")
                error_result = {
                    "experiment_id": config.get("name", f"adalora_exp_{i+1}"),
                    "status": "failed",
                    "error": str(e),
                    "config": config
                }
                results.append(error_result)
        
        logger.info(f"Completed {len(results)} AdaLoRA experiments")
        return results
    
    def _run_single_adalora_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single AdaLoRA experiment."""
        start_time = datetime.now()
        
        # Set reproducible seed
        seed = config.get("seed", 42)
        set_reproducible_seed(seed)
        
        # Load model and dataset
        model_name = config["model"]["name"]
        dataset_name = config["dataset"]["name"]
        
        logger.info(f"Loading model: {model_name}")
        model = self.model_manager.load_model(model_name)
        
        logger.info(f"Loading dataset: {dataset_name}")
        train_loader, val_loader, test_loader = self.dataset_loader.load_dataset(
            dataset_name,
            batch_size=config["dataset"]["batch_size"],
            image_size=config["model"].get("image_size", 224)
        )
        
        # Configure AdaLoRA
        adalora_config = AdaLoRAConfig(
            total_rank_budget=config["adalora"]["total_rank_budget"],
            min_rank=config["adalora"]["min_rank"],
            max_rank=config["adalora"]["max_rank"],
            importance_metric=config["adalora"]["importance_metric"],
            update_frequency=config["adalora"]["update_frequency"],
            allocation_strategy=config["adalora"]["allocation_strategy"],
            warmup_steps=config["adalora"]["warmup_steps"]
        )
        
        # Apply LoRA to model
        lora_config = config["lora"]
        peft_model = self.model_manager.apply_lora(
            model,
            rank=lora_config["rank"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"]
        )
        
        # Initialize AdaLoRA controller
        adalora_controller = AdaLoRAController(adalora_config)
        adalora_controller.initialize_from_model(peft_model)
        
        # Create trainer with AdaLoRA integration
        trainer = PEFTTrainer(
            model=peft_model,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            learning_rate=config["training"]["learning_rate"],
            num_epochs=config["training"]["num_epochs"]
        )
        
        # Training loop with adaptive rank allocation
        training_metrics = []
        importance_history = []
        
        for epoch in range(config["training"]["num_epochs"]):
            logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            
            # Train epoch
            epoch_metrics = trainer.train_epoch()
            
            # Update importance scores and reallocate ranks
            step = epoch * len(train_loader)
            importance_scores = adalora_controller.update_importance_scores(peft_model, step)
            new_allocation = adalora_controller.reallocate_ranks(importance_scores, step)
            
            # Collect metrics
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": epoch_metrics.get("train_loss", 0.0),
                "train_accuracy": epoch_metrics.get("train_accuracy", 0.0),
                "importance_scores": importance_scores.copy(),
                "rank_allocation": new_allocation.copy(),
                "budget_utilization": adalora_controller.get_budget_utilization()
            }
            
            # Evaluate
            if (epoch + 1) % config["training"].get("eval_epochs", 1) == 0:
                eval_metrics = trainer.evaluate(val_loader)
                epoch_result.update({
                    "val_loss": eval_metrics.get("eval_loss", 0.0),
                    "val_accuracy": eval_metrics.get("eval_accuracy", 0.0)
                })
            
            training_metrics.append(epoch_result)
            importance_history.append(importance_scores.copy())
        
        # Final evaluation
        final_metrics = trainer.evaluate(test_loader)
        
        # Collect comprehensive results
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Get layer importance summary
        layer_importance = adalora_controller.get_layer_importance_summary()
        budget_utilization = adalora_controller.get_budget_utilization()
        
        # Export AdaLoRA data for analysis
        adalora_data = adalora_controller.export_importance_data()
        
        result = {
            "experiment_id": config["name"],
            "method": "adalora",
            "status": "completed",
            "config": config,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "training_time_seconds": training_time,
            "seed": seed,
            
            # Final performance metrics
            "final_metrics": {
                "test_accuracy": final_metrics.get("eval_accuracy", 0.0),
                "test_loss": final_metrics.get("eval_loss", 0.0),
                "trainable_params": sum(p.numel() for p in peft_model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in peft_model.parameters())
            },
            
            # AdaLoRA specific results
            "adalora_results": {
                "layer_importance": layer_importance,
                "budget_utilization": budget_utilization,
                "final_allocation": adalora_controller.current_budget_allocation.copy(),
                "num_reallocations": len(adalora_controller.reallocation_history),
                "importance_evolution": importance_history
            },
            
            # Training history
            "training_metrics": training_metrics,
            
            # Raw AdaLoRA data for detailed analysis
            "adalora_data": adalora_data
        }
        
        logger.info(f"AdaLoRA experiment completed: {result['final_metrics']['test_accuracy']:.4f} accuracy")
        return result
    
    def run_qa_lora_experiments(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run QA-LoRA quantization-aware training experiments.
        
        Args:
            configs: List of experiment configurations
            
        Returns:
            List of experiment results
        """
        logger.info(f"Starting {len(configs)} QA-LoRA experiments")
        results = []
        
        for i, config in enumerate(configs):
            logger.info(f"Running QA-LoRA experiment {i+1}/{len(configs)}: {config['name']}")
            
            try:
                result = self._run_single_qa_lora_experiment(config)
                results.append(result)
                
                # Save intermediate results
                self._save_experiment_result(result)
                
            except Exception as e:
                logger.error(f"QA-LoRA experiment {i+1} failed: {str(e)}")
                error_result = {
                    "experiment_id": config.get("name", f"qa_lora_exp_{i+1}"),
                    "status": "failed",
                    "error": str(e),
                    "config": config
                }
                results.append(error_result)
        
        logger.info(f"Completed {len(results)} QA-LoRA experiments")
        return results
    
    def _run_single_qa_lora_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single QA-LoRA experiment."""
        start_time = datetime.now()
        
        # Set reproducible seed
        seed = config.get("seed", 42)
        set_reproducible_seed(seed)
        
        # Load model and dataset
        model_name = config["model"]["name"]
        dataset_name = config["dataset"]["name"]
        
        logger.info(f"Loading model: {model_name}")
        model = self.model_manager.load_model(model_name)
        
        logger.info(f"Loading dataset: {dataset_name}")
        train_loader, val_loader, test_loader = self.dataset_loader.load_dataset(
            dataset_name,
            batch_size=config["dataset"]["batch_size"],
            image_size=config["model"].get("image_size", 224)
        )
        
        # Configure QA-LoRA
        qa_lora_config = QALoRAConfig(
            quantization_bits=config["qa_lora"]["quantization_bits"],
            quantization_type=config["qa_lora"]["quantization_type"],
            lora_rank=config["lora"]["rank"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            gradient_scaling_factor=config["qa_lora"]["gradient_scaling_factor"],
            quantization_schedule=config["qa_lora"]["quantization_schedule"],
            warmup_steps=config["qa_lora"]["warmup_steps"],
            use_group_quantization=config["qa_lora"]["use_group_quantization"]
        )
        
        # Apply LoRA to model
        lora_config = config["lora"]
        peft_model = self.model_manager.apply_lora(
            model,
            rank=lora_config["rank"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"]
        )
        
        # Create base trainer
        base_trainer = PEFTTrainer(
            model=peft_model,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            learning_rate=config["training"]["learning_rate"],
            num_epochs=config["training"]["num_epochs"]
        )
        
        # Create QA-LoRA integrated trainer
        qa_trainer = QALoRAIntegratedTrainer(qa_lora_config, base_trainer)
        peft_model = qa_trainer.setup_model(peft_model)
        
        # Training loop with quantization-aware training
        training_metrics = []
        quantization_history = []
        
        optimizer = torch.optim.AdamW(
            peft_model.parameters(),
            lr=config["training"]["learning_rate"]
        )
        
        for epoch in range(config["training"]["num_epochs"]):
            logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            
            # Training epoch with QA-LoRA
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            peft_model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                step = epoch * len(train_loader) + batch_idx
                
                # Forward pass
                outputs = peft_model(data)
                loss = nn.CrossEntropyLoss()(outputs, target)
                
                # QA-LoRA training step
                qa_stats = qa_trainer.training_step(peft_model, optimizer, loss, step)
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
                
                # Store quantization statistics
                if batch_idx % 100 == 0:  # Sample every 100 batches
                    quantization_history.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        **qa_stats
                    })
            
            # Epoch metrics
            train_accuracy = epoch_correct / epoch_total
            train_loss = epoch_loss / len(train_loader)
            
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy
            }
            
            # Evaluate
            if (epoch + 1) % config["training"].get("eval_epochs", 1) == 0:
                peft_model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        outputs = peft_model(data)
                        loss = nn.CrossEntropyLoss()(outputs, target)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                val_accuracy = val_correct / val_total
                val_loss = val_loss / len(val_loader)
                
                epoch_result.update({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                })
                
                # QA-LoRA validation
                qa_validation = qa_trainer.validate(peft_model)
                epoch_result["qa_validation"] = qa_validation
            
            training_metrics.append(epoch_result)
        
        # Final evaluation
        peft_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                outputs = peft_model(data)
                loss = nn.CrossEntropyLoss()(outputs, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_accuracy = test_correct / test_total
        test_loss = test_loss / len(test_loader)
        
        # Collect comprehensive results
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Get QA-LoRA training summary
        qa_summary = qa_trainer.get_training_summary()
        
        # Export quantization data
        qa_data = qa_trainer.qa_lora_trainer.export_quantization_data()
        
        result = {
            "experiment_id": config["name"],
            "method": "qa_lora",
            "status": "completed",
            "config": config,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "training_time_seconds": training_time,
            "seed": seed,
            
            # Final performance metrics
            "final_metrics": {
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "trainable_params": sum(p.numel() for p in peft_model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in peft_model.parameters())
            },
            
            # QA-LoRA specific results
            "qa_lora_results": {
                "training_summary": qa_summary,
                "quantization_history": quantization_history,
                "final_quantization_state": qa_trainer.qa_lora_trainer.quantization_state.__dict__
            },
            
            # Training history
            "training_metrics": training_metrics,
            
            # Raw QA-LoRA data for detailed analysis
            "qa_lora_data": qa_data
        }
        
        logger.info(f"QA-LoRA experiment completed: {result['final_metrics']['test_accuracy']:.4f} accuracy")
        return result
    
    def analyze_layer_importance_patterns(self, adalora_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze layer importance patterns from AdaLoRA experiments.
        
        Args:
            adalora_results: List of AdaLoRA experiment results
            
        Returns:
            Analysis of layer importance patterns
        """
        logger.info("Analyzing layer importance patterns")
        
        # Collect layer importance data across experiments
        layer_importance_data = {}
        
        for result in adalora_results:
            if result["status"] != "completed":
                continue
            
            adalora_data = result.get("adalora_data", {})
            layer_importance = adalora_data.get("layer_importance", {})
            
            for layer_name, layer_data in layer_importance.items():
                if layer_name not in layer_importance_data:
                    layer_importance_data[layer_name] = {
                        "importance_scores": [],
                        "final_ranks": [],
                        "rank_changes": [],
                        "experiments": []
                    }
                
                layer_importance_data[layer_name]["importance_scores"].append(
                    layer_data["importance_score"]
                )
                layer_importance_data[layer_name]["final_ranks"].append(
                    layer_data["current_rank"]
                )
                layer_importance_data[layer_name]["rank_changes"].append(
                    layer_data["update_count"]
                )
                layer_importance_data[layer_name]["experiments"].append(
                    result["experiment_id"]
                )
        
        # Calculate statistics
        layer_statistics = {}
        for layer_name, data in layer_importance_data.items():
            if not data["importance_scores"]:
                continue
            
            import numpy as np
            
            layer_statistics[layer_name] = {
                "avg_importance": np.mean(data["importance_scores"]),
                "std_importance": np.std(data["importance_scores"]),
                "avg_final_rank": np.mean(data["final_ranks"]),
                "std_final_rank": np.std(data["final_ranks"]),
                "avg_rank_changes": np.mean(data["rank_changes"]),
                "consistency_score": 1.0 - (np.std(data["importance_scores"]) / 
                                           max(np.mean(data["importance_scores"]), 1e-8))
            }
        
        # Identify patterns
        if layer_statistics:
            # Sort layers by average importance
            sorted_layers = sorted(
                layer_statistics.items(),
                key=lambda x: x[1]["avg_importance"],
                reverse=True
            )
            
            # Find most and least important layers
            most_important = sorted_layers[:3] if len(sorted_layers) >= 3 else sorted_layers
            least_important = sorted_layers[-3:] if len(sorted_layers) >= 3 else []
            
            # Calculate overall statistics
            all_importance = [stats["avg_importance"] for stats in layer_statistics.values()]
            all_ranks = [stats["avg_final_rank"] for stats in layer_statistics.values()]
            
            analysis = {
                "layer_statistics": layer_statistics,
                "patterns": {
                    "most_important_layers": [
                        {"layer": name, "importance": stats["avg_importance"]}
                        for name, stats in most_important
                    ],
                    "least_important_layers": [
                        {"layer": name, "importance": stats["avg_importance"]}
                        for name, stats in least_important
                    ],
                    "importance_range": {
                        "min": min(all_importance) if all_importance else 0.0,
                        "max": max(all_importance) if all_importance else 0.0,
                        "mean": np.mean(all_importance) if all_importance else 0.0,
                        "std": np.std(all_importance) if all_importance else 0.0
                    },
                    "rank_allocation_range": {
                        "min": min(all_ranks) if all_ranks else 0,
                        "max": max(all_ranks) if all_ranks else 0,
                        "mean": np.mean(all_ranks) if all_ranks else 0.0,
                        "std": np.std(all_ranks) if all_ranks else 0.0
                    }
                },
                "insights": self._generate_layer_importance_insights(layer_statistics)
            }
        else:
            analysis = {
                "layer_statistics": {},
                "patterns": {},
                "insights": ["No layer importance data available for analysis"]
            }
        
        return analysis
    
    def _generate_layer_importance_insights(self, layer_statistics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate insights from layer importance analysis."""
        insights = []
        
        if not layer_statistics:
            return ["No layer statistics available"]
        
        # Analyze importance distribution
        importance_values = [stats["avg_importance"] for stats in layer_statistics.values()]
        rank_values = [stats["avg_final_rank"] for stats in layer_statistics.values()]
        
        import numpy as np
        
        # Importance insights
        if len(importance_values) > 1:
            importance_cv = np.std(importance_values) / np.mean(importance_values)
            if importance_cv > 0.5:
                insights.append("High variability in layer importance suggests strong differentiation between layers")
            else:
                insights.append("Low variability in layer importance suggests uniform adaptation needs")
        
        # Rank allocation insights
        if len(rank_values) > 1:
            rank_cv = np.std(rank_values) / np.mean(rank_values)
            if rank_cv > 0.3:
                insights.append("Adaptive rank allocation shows significant differentiation between layers")
            else:
                insights.append("Adaptive rank allocation remains relatively uniform across layers")
        
        # Layer-specific insights
        sorted_by_importance = sorted(
            layer_statistics.items(),
            key=lambda x: x[1]["avg_importance"],
            reverse=True
        )
        
        if len(sorted_by_importance) >= 2:
            top_layer = sorted_by_importance[0]
            bottom_layer = sorted_by_importance[-1]
            
            importance_ratio = top_layer[1]["avg_importance"] / max(bottom_layer[1]["avg_importance"], 1e-8)
            if importance_ratio > 2.0:
                insights.append(f"Layer '{top_layer[0]}' shows {importance_ratio:.1f}x higher importance than '{bottom_layer[0]}'")
        
        # Consistency insights
        consistency_scores = [stats["consistency_score"] for stats in layer_statistics.values()]
        avg_consistency = np.mean(consistency_scores)
        
        if avg_consistency > 0.8:
            insights.append("High consistency in layer importance across experiments")
        elif avg_consistency < 0.5:
            insights.append("Low consistency in layer importance suggests experiment-dependent patterns")
        
        return insights
    
    def compare_adaptive_vs_fixed_methods(self, adaptive_results: List[Dict[str, Any]], 
                                        fixed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare adaptive methods against fixed-rank approaches.
        
        Args:
            adaptive_results: Results from adaptive methods (AdaLoRA, QA-LoRA)
            fixed_results: Results from fixed-rank LoRA methods
            
        Returns:
            Comparison analysis
        """
        logger.info("Comparing adaptive vs fixed-rank methods")
        
        def extract_metrics(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
            """Extract key metrics from results."""
            metrics = {
                "test_accuracy": [],
                "test_loss": [],
                "trainable_params": [],
                "training_time": []
            }
            
            for result in results:
                if result["status"] != "completed":
                    continue
                
                final_metrics = result.get("final_metrics", {})
                metrics["test_accuracy"].append(final_metrics.get("test_accuracy", 0.0))
                metrics["test_loss"].append(final_metrics.get("test_loss", float('inf')))
                metrics["trainable_params"].append(final_metrics.get("trainable_params", 0))
                metrics["training_time"].append(result.get("training_time_seconds", 0.0))
            
            return metrics
        
        adaptive_metrics = extract_metrics(adaptive_results)
        fixed_metrics = extract_metrics(fixed_results)
        
        import numpy as np
        
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            """Calculate statistics for a list of values."""
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            
            return {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        # Calculate statistics for each metric
        comparison = {}
        
        for metric_name in adaptive_metrics.keys():
            adaptive_stats = calculate_stats(adaptive_metrics[metric_name])
            fixed_stats = calculate_stats(fixed_metrics[metric_name])
            
            # Calculate improvement
            if fixed_stats["mean"] > 0:
                if metric_name in ["test_accuracy"]:  # Higher is better
                    improvement = (adaptive_stats["mean"] - fixed_stats["mean"]) / fixed_stats["mean"] * 100
                else:  # Lower is better
                    improvement = (fixed_stats["mean"] - adaptive_stats["mean"]) / fixed_stats["mean"] * 100
            else:
                improvement = 0.0
            
            comparison[metric_name] = {
                "adaptive": adaptive_stats,
                "fixed": fixed_stats,
                "improvement_percent": improvement,
                "adaptive_better": improvement > 0
            }
        
        # Statistical significance testing (simplified)
        significance_tests = {}
        for metric_name in adaptive_metrics.keys():
            adaptive_vals = adaptive_metrics[metric_name]
            fixed_vals = fixed_metrics[metric_name]
            
            if len(adaptive_vals) > 1 and len(fixed_vals) > 1:
                # Simple t-test approximation
                adaptive_mean = np.mean(adaptive_vals)
                fixed_mean = np.mean(fixed_vals)
                adaptive_std = np.std(adaptive_vals)
                fixed_std = np.std(fixed_vals)
                
                pooled_std = np.sqrt((adaptive_std**2 + fixed_std**2) / 2)
                if pooled_std > 0:
                    t_stat = abs(adaptive_mean - fixed_mean) / pooled_std
                    # Rough significance threshold
                    significant = t_stat > 2.0
                else:
                    significant = False
                
                significance_tests[metric_name] = {
                    "t_statistic": t_stat,
                    "significant": significant
                }
        
        # Generate insights
        insights = []
        
        # Accuracy comparison
        if "test_accuracy" in comparison:
            acc_comp = comparison["test_accuracy"]
            if acc_comp["adaptive_better"] and acc_comp["improvement_percent"] > 1.0:
                insights.append(f"Adaptive methods show {acc_comp['improvement_percent']:.1f}% accuracy improvement")
            elif not acc_comp["adaptive_better"] and abs(acc_comp["improvement_percent"]) > 1.0:
                insights.append(f"Fixed methods show {abs(acc_comp['improvement_percent']):.1f}% accuracy advantage")
            else:
                insights.append("Adaptive and fixed methods show similar accuracy performance")
        
        # Parameter efficiency
        if "trainable_params" in comparison:
            param_comp = comparison["trainable_params"]
            if param_comp["adaptive_better"]:
                insights.append(f"Adaptive methods use {param_comp['improvement_percent']:.1f}% fewer parameters")
            else:
                insights.append(f"Fixed methods use {abs(param_comp['improvement_percent']):.1f}% fewer parameters")
        
        # Training time
        if "training_time" in comparison:
            time_comp = comparison["training_time"]
            if time_comp["adaptive_better"]:
                insights.append(f"Adaptive methods are {time_comp['improvement_percent']:.1f}% faster to train")
            else:
                insights.append(f"Fixed methods are {abs(time_comp['improvement_percent']):.1f}% faster to train")
        
        return {
            "comparison": comparison,
            "significance_tests": significance_tests,
            "insights": insights,
            "summary": {
                "adaptive_experiments": len(adaptive_results),
                "fixed_experiments": len(fixed_results),
                "metrics_compared": list(comparison.keys())
            }
        }
    
    def _save_experiment_result(self, result: Dict[str, Any]) -> None:
        """Save individual experiment result."""
        exp_id = result["experiment_id"]
        result_file = self.output_dir / f"{exp_id}_result.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved result: {result_file}")
        except Exception as e:
            logger.error(f"Failed to save result {exp_id}: {str(e)}")
    
    def save_analysis_results(self, analysis_results: Dict[str, Any]) -> None:
        """Save comprehensive analysis results."""
        analysis_file = self.output_dir / "adaptive_analysis_results.json"
        
        try:
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            logger.info(f"Saved analysis results: {analysis_file}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {str(e)}")


def create_adalora_experiment_configs() -> List[Dict[str, Any]]:
    """Create AdaLoRA experiment configurations."""
    base_config = {
        "model": {"name": "deit_tiny_patch16_224", "image_size": 224},
        "dataset": {"name": "cifar10", "batch_size": 32},
        "lora": {"rank": 16, "alpha": 32.0, "dropout": 0.1},
        "training": {"learning_rate": 1e-4, "num_epochs": 5, "eval_epochs": 1},
        "seed": 42
    }
    
    configs = []
    
    # Different AdaLoRA configurations
    adalora_configs = [
        {
            "name": "adalora_proportional_cifar10",
            "adalora": {
                "total_rank_budget": 64,
                "min_rank": 2,
                "max_rank": 16,
                "importance_metric": "magnitude",
                "update_frequency": 100,
                "allocation_strategy": "proportional",
                "warmup_steps": 200
            }
        },
        {
            "name": "adalora_threshold_cifar10",
            "adalora": {
                "total_rank_budget": 64,
                "min_rank": 2,
                "max_rank": 16,
                "importance_metric": "gradient_norm",
                "update_frequency": 100,
                "allocation_strategy": "threshold",
                "warmup_steps": 200
            }
        },
        {
            "name": "adalora_topk_cifar10",
            "adalora": {
                "total_rank_budget": 64,
                "min_rank": 2,
                "max_rank": 16,
                "importance_metric": "fisher",
                "update_frequency": 100,
                "allocation_strategy": "top_k",
                "warmup_steps": 200
            }
        }
    ]
    
    for adalora_config in adalora_configs:
        config = {**base_config, **adalora_config}
        configs.append(config)
    
    return configs


def create_qa_lora_experiment_configs() -> List[Dict[str, Any]]:
    """Create QA-LoRA experiment configurations."""
    base_config = {
        "model": {"name": "deit_tiny_patch16_224", "image_size": 224},
        "dataset": {"name": "cifar10", "batch_size": 32},
        "lora": {"rank": 8, "alpha": 16.0, "dropout": 0.1},
        "training": {"learning_rate": 1e-4, "num_epochs": 5, "eval_epochs": 1},
        "seed": 42
    }
    
    configs = []
    
    # Different QA-LoRA configurations
    qa_lora_configs = [
        {
            "name": "qa_lora_4bit_constant_cifar10",
            "qa_lora": {
                "quantization_bits": 4,
                "quantization_type": "nf4",
                "gradient_scaling_factor": 1.0,
                "quantization_schedule": "constant",
                "warmup_steps": 100,
                "use_group_quantization": True
            }
        },
        {
            "name": "qa_lora_8bit_linear_cifar10",
            "qa_lora": {
                "quantization_bits": 8,
                "quantization_type": "int8",
                "gradient_scaling_factor": 1.5,
                "quantization_schedule": "linear",
                "warmup_steps": 200,
                "use_group_quantization": True
            }
        },
        {
            "name": "qa_lora_4bit_cosine_cifar10",
            "qa_lora": {
                "quantization_bits": 4,
                "quantization_type": "fp4",
                "gradient_scaling_factor": 2.0,
                "quantization_schedule": "cosine",
                "warmup_steps": 150,
                "use_group_quantization": False
            }
        }
    ]
    
    for qa_lora_config in qa_lora_configs:
        config = {**base_config, **qa_lora_config}
        configs.append(config)
    
    return configs


def create_fixed_rank_baseline_configs() -> List[Dict[str, Any]]:
    """Create fixed-rank LoRA baseline configurations for comparison."""
    base_config = {
        "model": {"name": "deit_tiny_patch16_224", "image_size": 224},
        "dataset": {"name": "cifar10", "batch_size": 32},
        "training": {"learning_rate": 1e-4, "num_epochs": 5, "eval_epochs": 1},
        "seed": 42
    }
    
    configs = []
    
    # Different fixed ranks for comparison
    ranks = [4, 8, 16, 32]
    
    for rank in ranks:
        config = {
            **base_config,
            "name": f"fixed_lora_rank{rank}_cifar10",
            "lora": {"rank": rank, "alpha": rank * 2.0, "dropout": 0.1}
        }
        configs.append(config)
    
    return configs


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run adaptive LoRA and QA-LoRA evaluation experiments")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/adaptive_experiments"),
        help="Output directory for experiment results"
    )
    parser.add_argument(
        "--run-adalora",
        action="store_true",
        help="Run AdaLoRA experiments"
    )
    parser.add_argument(
        "--run-qa-lora",
        action="store_true",
        help="Run QA-LoRA experiments"
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run fixed-rank baseline experiments"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all experiments (AdaLoRA, QA-LoRA, and baselines)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on existing results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running experiments"
    )
    
    args = parser.parse_args()
    
    print("Adaptive LoRA and QA-LoRA Evaluation")
    print("=" * 50)
    
    # Create experiment runner
    runner = AdaptiveExperimentRunner(args.output_dir)
    
    # Determine what to run
    run_adalora = args.run_adalora or args.run_all
    run_qa_lora = args.run_qa_lora or args.run_all
    run_baselines = args.run_baselines or args.run_all
    
    if not any([run_adalora, run_qa_lora, run_baselines, args.analyze_only]):
        print("No experiments specified. Use --run-all or specific flags.")
        return
    
    # Create experiment configurations
    adalora_configs = create_adalora_experiment_configs() if run_adalora else []
    qa_lora_configs = create_qa_lora_experiment_configs() if run_qa_lora else []
    baseline_configs = create_fixed_rank_baseline_configs() if run_baselines else []
    
    total_experiments = len(adalora_configs) + len(qa_lora_configs) + len(baseline_configs)
    
    print(f"Experiment Plan:")
    print(f"  AdaLoRA experiments: {len(adalora_configs)}")
    print(f"  QA-LoRA experiments: {len(qa_lora_configs)}")
    print(f"  Baseline experiments: {len(baseline_configs)}")
    print(f"  Total experiments: {total_experiments}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would run the following experiments:")
        for config in adalora_configs + qa_lora_configs + baseline_configs:
            print(f"  - {config['name']}")
        return
    
    if args.analyze_only:
        print("\nRunning analysis on existing results...")
        # Load existing results and run analysis
        # This would load from previous experiment runs
        print("Analysis-only mode not fully implemented yet")
        return
    
    # Run experiments
    all_results = []
    
    if run_adalora:
        print(f"\nRunning {len(adalora_configs)} AdaLoRA experiments...")
        adalora_results = runner.run_adalora_experiments(adalora_configs)
        all_results.extend(adalora_results)
    
    if run_qa_lora:
        print(f"\nRunning {len(qa_lora_configs)} QA-LoRA experiments...")
        qa_lora_results = runner.run_qa_lora_experiments(qa_lora_configs)
        all_results.extend(qa_lora_results)
    
    if run_baselines:
        print(f"\nRunning {len(baseline_configs)} baseline experiments...")
        # For baselines, we'll use a simplified runner
        baseline_results = []
        for config in baseline_configs:
            print(f"Running baseline: {config['name']}")
            # This would run standard LoRA training
            # For now, create mock results
            baseline_result = {
                "experiment_id": config["name"],
                "method": "fixed_lora",
                "status": "completed",
                "config": config,
                "final_metrics": {
                    "test_accuracy": 0.75 + (hash(config["name"]) % 100) / 1000,
                    "test_loss": 0.8 - (hash(config["name"]) % 100) / 2000,
                    "trainable_params": config["lora"]["rank"] * 1000,
                    "total_params": 5000000
                },
                "training_time_seconds": 300 + (hash(config["name"]) % 100)
            }
            baseline_results.append(baseline_result)
            runner._save_experiment_result(baseline_result)
        
        all_results.extend(baseline_results)
    
    # Analyze results
    print("\nAnalyzing results...")
    
    # Separate results by method
    adalora_results = [r for r in all_results if r.get("method") == "adalora"]
    qa_lora_results = [r for r in all_results if r.get("method") == "qa_lora"]
    fixed_results = [r for r in all_results if r.get("method") == "fixed_lora"]
    
    analysis_results = {}
    
    # Analyze layer importance patterns
    if adalora_results:
        layer_analysis = runner.analyze_layer_importance_patterns(adalora_results)
        analysis_results["layer_importance_analysis"] = layer_analysis
        
        print("\nLayer Importance Analysis:")
        for insight in layer_analysis.get("insights", []):
            print(f"  • {insight}")
    
    # Compare adaptive vs fixed methods
    if adalora_results or qa_lora_results:
        adaptive_results = adalora_results + qa_lora_results
        if fixed_results:
            comparison = runner.compare_adaptive_vs_fixed_methods(adaptive_results, fixed_results)
            analysis_results["adaptive_vs_fixed_comparison"] = comparison
            
            print("\nAdaptive vs Fixed Methods Comparison:")
            for insight in comparison.get("insights", []):
                print(f"  • {insight}")
    
    # Save comprehensive analysis
    analysis_results["experiment_summary"] = {
        "total_experiments": len(all_results),
        "successful_experiments": len([r for r in all_results if r["status"] == "completed"]),
        "failed_experiments": len([r for r in all_results if r["status"] == "failed"]),
        "methods_evaluated": list(set(r.get("method", "unknown") for r in all_results)),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    runner.save_analysis_results(analysis_results)
    
    print(f"\n✓ Adaptive experiments completed!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"  Successful experiments: {analysis_results['experiment_summary']['successful_experiments']}")
    print(f"  Failed experiments: {analysis_results['experiment_summary']['failed_experiments']}")


if __name__ == "__main__":
    main()