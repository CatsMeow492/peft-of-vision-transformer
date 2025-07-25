"""
Publication-quality figure generation for PEFT Vision Transformer research.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass

from .plot_styles import PlotStyleManager
from .pareto_analysis import ParetoAnalyzer, ParetoPoint

# Handle optional imports
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

try:
    from ..evaluation.metrics_collector import EvaluationMetrics, ModelMetrics, ResourceMetrics
    from ..evaluation.statistical_analyzer import StatisticalAnalyzer, ConfidenceInterval
    from ..experiments.results import ExperimentResult
    EVALUATION_AVAILABLE = True
except ImportError:
    # Create dummy classes for type hints
    class EvaluationMetrics:
        pass
    class ModelMetrics:
        pass
    class ResourceMetrics:
        pass
    class StatisticalAnalyzer:
        pass
    class ConfidenceInterval:
        pass
    class ExperimentResult:
        pass
    EVALUATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    
    output_dir: Path
    formats: List[str] = None
    dpi: int = 300
    figsize: Tuple[float, float] = (8, 6)
    style: str = "publication"
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ['pdf', 'png']
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


class FigureGenerator:
    """
    Publication-quality figure generator for PEFT Vision Transformer research.
    
    Generates all required figures including:
    - Accuracy vs efficiency trade-offs (Pareto frontiers)
    - Layer importance heatmaps for AdaLoRA analysis
    - Convergence curves with statistical confidence bands
    - Method comparison charts with error bars
    """
    
    def __init__(self, config: FigureConfig):
        """
        Initialize figure generator.
        
        Args:
            config: Figure generation configuration
        """
        self.config = config
        self.style_manager = PlotStyleManager(style=config.style)
        self.pareto_analyzer = ParetoAnalyzer()
        
        if EVALUATION_AVAILABLE:
            self.stats_analyzer = StatisticalAnalyzer()
        else:
            self.stats_analyzer = None
            logger.warning("Statistical analyzer not available")
        
        logger.info(f"FigureGenerator initialized with output_dir: {config.output_dir}")
    
    def generate_pareto_frontier_plot(
        self,
        results: List[ExperimentResult],
        efficiency_metric: str = "trainable_parameters",
        accuracy_metric: str = "final_accuracy",
        filename: str = "pareto_frontier"
    ) -> Path:
        """
        Generate Pareto frontier plot showing accuracy vs efficiency trade-offs.
        
        Args:
            results: List of experiment results
            efficiency_metric: Metric to use for efficiency axis
            accuracy_metric: Metric to use for accuracy axis
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Generating Pareto frontier plot with {len(results)} results")
        
        # Extract data points
        pareto_points = []
        for result in results:
            if not result.is_successful:
                continue
            
            accuracy = result.metrics.get(accuracy_metric, 0)
            efficiency = result.metrics.get(efficiency_metric, 0)
            
            if accuracy > 0 and efficiency > 0:
                point = ParetoPoint(
                    method_name=self._get_method_name(result),
                    accuracy=accuracy,
                    efficiency_metric=efficiency,
                    additional_metrics=result.metrics
                )
                pareto_points.append(point)
        
        if not pareto_points:
            logger.error("No valid data points for Pareto frontier")
            return None
        
        # Compute Pareto frontier
        frontier_points = self.pareto_analyzer.compute_pareto_frontier(pareto_points)
        dominated_points = self.pareto_analyzer.compute_dominated_points(pareto_points)
        
        # Create figure
        fig, ax = self.style_manager.setup_figure(figsize=self.config.figsize)
        
        # Plot dominated points
        if dominated_points:
            dom_x = [p.efficiency_metric for p in dominated_points]
            dom_y = [p.accuracy for p in dominated_points]
            ax.scatter(
                dom_x, dom_y,
                alpha=0.6, s=60,
                color=self.style_manager.COLORS['gray'],
                label='Dominated methods',
                zorder=1
            )
        
        # Plot Pareto frontier points
        if frontier_points:
            front_x = [p.efficiency_metric for p in frontier_points]
            front_y = [p.accuracy for p in frontier_points]
            
            # Color by method type
            for point in frontier_points:
                color = self.style_manager.get_method_color(point.method_name)
                ax.scatter(
                    point.efficiency_metric, point.accuracy,
                    color=color, s=100, alpha=0.8,
                    edgecolors='black', linewidth=1,
                    zorder=3
                )
            
            # Connect frontier points
            sorted_frontier = sorted(frontier_points, key=lambda p: p.efficiency_metric)
            front_x_sorted = [p.efficiency_metric for p in sorted_frontier]
            front_y_sorted = [p.accuracy for p in sorted_frontier]
            
            ax.plot(
                front_x_sorted, front_y_sorted,
                color=self.style_manager.COLORS['primary'],
                linewidth=2, alpha=0.7,
                linestyle='--',
                label='Pareto frontier',
                zorder=2
            )
        
        # Annotate points with method names
        for point in frontier_points:
            ax.annotate(
                self._format_method_name(point.method_name),
                (point.efficiency_metric, point.accuracy),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        # Formatting
        self.style_manager.format_axes_labels(
            ax,
            xlabel=self._format_metric_name(efficiency_metric),
            ylabel=self._format_metric_name(accuracy_metric),
            title="Accuracy vs Efficiency Trade-offs"
        )
        
        # Set axis limits with padding
        if pareto_points:
            all_x = [p.efficiency_metric for p in pareto_points]
            all_y = [p.accuracy for p in pareto_points]
            
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            
            ax.set_xlim(min(all_x) - 0.05 * x_range, max(all_x) + 0.05 * x_range)
            ax.set_ylim(min(all_y) - 0.05 * y_range, max(all_y) + 0.05 * y_range)
        
        # Legend
        self.style_manager.create_legend(ax, location='lower right')
        
        # Save figure
        output_path = self.config.output_dir / filename
        self.style_manager.save_figure(fig, str(output_path), self.config.formats, self.config.dpi)
        
        plt.close(fig)
        
        logger.info(f"Pareto frontier plot saved to {output_path}")
        return output_path
    
    def generate_layer_importance_heatmap(
        self,
        importance_data: Dict[str, Dict[str, float]],
        filename: str = "layer_importance_heatmap"
    ) -> Path:
        """
        Generate heatmap showing layer importance scores for AdaLoRA analysis.
        
        Args:
            importance_data: Nested dict {experiment_id: {layer_name: importance_score}}
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        logger.info("Generating layer importance heatmap")
        
        if not importance_data:
            logger.error("No importance data provided")
            return None
        
        # Prepare data matrix
        all_layers = set()
        for exp_data in importance_data.values():
            all_layers.update(exp_data.keys())
        
        layer_names = sorted(list(all_layers))
        experiment_names = list(importance_data.keys())
        
        # Create matrix
        matrix = np.zeros((len(experiment_names), len(layer_names)))
        
        for i, exp_name in enumerate(experiment_names):
            for j, layer_name in enumerate(layer_names):
                matrix[i, j] = importance_data[exp_name].get(layer_name, 0)
        
        # Create figure
        fig, ax = self.style_manager.setup_figure(
            figsize=(max(8, len(layer_names) * 0.5), max(6, len(experiment_names) * 0.4))
        )
        
        # Create heatmap
        if SEABORN_AVAILABLE:
            sns.heatmap(
                matrix,
                xticklabels=layer_names,
                yticklabels=experiment_names,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'Importance Score'},
                ax=ax
            )
        else:
            # Fallback matplotlib heatmap
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Importance Score')
            
            # Set ticks and labels
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.set_yticks(range(len(experiment_names)))
            ax.set_yticklabels(experiment_names)
            
            # Add text annotations
            for i in range(len(experiment_names)):
                for j in range(len(layer_names)):
                    ax.text(j, i, f'{matrix[i, j]:.3f}',
                           ha='center', va='center', fontsize=8)
        
        # Formatting
        ax.set_title("Layer Importance Scores (AdaLoRA)", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Transformer Layers", fontsize=12)
        ax.set_ylabel("Experiments", fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.config.output_dir / filename
        self.style_manager.save_figure(fig, str(output_path), self.config.formats, self.config.dpi)
        
        plt.close(fig)
        
        logger.info(f"Layer importance heatmap saved to {output_path}")
        return output_path
    
    def generate_convergence_curves(
        self,
        training_histories: Dict[str, List[Dict[str, float]]],
        metric: str = "accuracy",
        filename: str = "convergence_curves"
    ) -> Path:
        """
        Generate convergence curves with statistical confidence bands.
        
        Args:
            training_histories: Dict mapping method names to training history lists
            metric: Metric to plot (e.g., 'accuracy', 'loss')
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Generating convergence curves for {len(training_histories)} methods")
        
        if not training_histories:
            logger.error("No training histories provided")
            return None
        
        # Create figure
        fig, ax = self.style_manager.setup_figure(figsize=self.config.figsize)
        
        for method_name, histories in training_histories.items():
            if not histories:
                continue
            
            # Extract metric values across epochs
            max_epochs = max(len(history) for history in histories)
            epoch_data = {epoch: [] for epoch in range(max_epochs)}
            
            for history in histories:
                for epoch, epoch_metrics in enumerate(history):
                    if metric in epoch_metrics:
                        epoch_data[epoch].append(epoch_metrics[metric])
            
            # Compute statistics for each epoch
            epochs = []
            means = []
            lower_bounds = []
            upper_bounds = []
            
            for epoch in range(max_epochs):
                if epoch_data[epoch]:  # Has data for this epoch
                    values = epoch_data[epoch]
                    
                    if self.stats_analyzer and len(values) > 1:
                        ci = self.stats_analyzer.compute_confidence_interval(values)
                        mean = ci.mean
                        lower = ci.lower_bound
                        upper = ci.upper_bound
                    else:
                        mean = np.mean(values)
                        std = np.std(values) if len(values) > 1 else 0
                        lower = mean - std
                        upper = mean + std
                    
                    epochs.append(epoch)
                    means.append(mean)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
            
            if epochs:
                color = self.style_manager.get_method_color(method_name)
                
                # Plot mean line with confidence bands
                self.style_manager.add_confidence_bands(
                    ax, epochs, means, lower_bounds, upper_bounds,
                    color=color, alpha=0.2,
                    label=self._format_method_name(method_name)
                )
        
        # Formatting
        self.style_manager.format_axes_labels(
            ax,
            xlabel="Epoch",
            ylabel=self._format_metric_name(metric),
            title=f"Training Convergence ({metric.title()})"
        )
        
        # Legend
        self.style_manager.create_legend(ax, location='best')
        
        # Save figure
        output_path = self.config.output_dir / filename
        self.style_manager.save_figure(fig, str(output_path), self.config.formats, self.config.dpi)
        
        plt.close(fig)
        
        logger.info(f"Convergence curves saved to {output_path}")
        return output_path
    
    def generate_method_comparison_chart(
        self,
        results: List[ExperimentResult],
        metric: str = "final_accuracy",
        group_by: str = "method",
        filename: str = "method_comparison"
    ) -> Path:
        """
        Generate method comparison chart with error bars and significance indicators.
        
        Args:
            results: List of experiment results
            metric: Metric to compare
            group_by: How to group results ('method', 'dataset', 'model')
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Generating method comparison chart for {metric}")
        
        # Group results
        grouped_data = {}
        for result in results:
            if not result.is_successful or metric not in result.metrics:
                continue
            
            if group_by == "method":
                group_key = self._get_method_name(result)
            elif group_by == "dataset":
                group_key = result.config.dataset.name if result.config else "unknown"
            elif group_by == "model":
                group_key = result.config.model.name if result.config else "unknown"
            else:
                group_key = "all"
            
            if group_key not in grouped_data:
                grouped_data[group_key] = []
            
            grouped_data[group_key].append(result.metrics[metric])
        
        if not grouped_data:
            logger.error("No valid data for comparison chart")
            return None
        
        # Compute statistics
        group_names = list(grouped_data.keys())
        means = []
        errors = []
        
        for group_name in group_names:
            values = grouped_data[group_name]
            
            if self.stats_analyzer and len(values) > 1:
                ci = self.stats_analyzer.compute_confidence_interval(values)
                mean = ci.mean
                error = ci.margin_of_error
            else:
                mean = np.mean(values)
                error = np.std(values) if len(values) > 1 else 0
            
            means.append(mean)
            errors.append(error)
        
        # Create figure
        fig, ax = self.style_manager.setup_figure(
            figsize=(max(8, len(group_names) * 0.8), 6)
        )
        
        # Create bar chart
        x_positions = np.arange(len(group_names))
        colors = [self.style_manager.get_method_color(name) for name in group_names]
        
        bars = ax.bar(
            x_positions, means, yerr=errors,
            color=colors, alpha=0.8,
            capsize=5, error_kw={'linewidth': 1.5}
        )
        
        # Add value labels on bars
        for i, (bar, mean, error) in enumerate(zip(bars, means, errors)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height + error + 0.01,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=10
            )
        
        # Statistical significance testing
        if self.stats_analyzer and len(grouped_data) > 1:
            self._add_significance_indicators(ax, grouped_data, group_names, max(means) + max(errors))
        
        # Formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels([self._format_method_name(name) for name in group_names], 
                          rotation=45, ha='right')
        
        self.style_manager.format_axes_labels(
            ax,
            xlabel=group_by.title(),
            ylabel=self._format_metric_name(metric),
            title=f"{group_by.title()} Comparison ({metric.replace('_', ' ').title()})"
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.config.output_dir / filename
        self.style_manager.save_figure(fig, str(output_path), self.config.formats, self.config.dpi)
        
        plt.close(fig)
        
        logger.info(f"Method comparison chart saved to {output_path}")
        return output_path
    
    def generate_resource_usage_visualization(
        self,
        results: List[ExperimentResult],
        filename: str = "resource_usage"
    ) -> Path:
        """
        Generate resource usage visualization (memory, time, model size).
        
        Args:
            results: List of experiment results
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        logger.info("Generating resource usage visualization")
        
        # Extract resource data
        methods = []
        memory_usage = []
        training_times = []
        model_sizes = []
        
        for result in results:
            if not result.is_successful:
                continue
            
            method_name = self._get_method_name(result)
            methods.append(method_name)
            
            memory_usage.append(result.peak_memory_gb)
            training_times.append(result.duration_seconds / 60)  # Convert to minutes
            model_sizes.append(result.metrics.get('model_size_mb', 0))
        
        if not methods:
            logger.error("No valid data for resource usage visualization")
            return None
        
        # Create subplots
        fig, axes = self.style_manager.setup_figure(
            figsize=(12, 8),
            subplot_layout=(2, 2)
        )
        
        axes = axes.flatten()
        
        # Memory usage
        ax1 = axes[0]
        colors = [self.style_manager.get_method_color(method) for method in methods]
        bars1 = ax1.bar(range(len(methods)), memory_usage, color=colors, alpha=0.8)
        ax1.set_title("Peak Memory Usage")
        ax1.set_ylabel("Memory (GB)")
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels([self._format_method_name(m) for m in methods], rotation=45, ha='right')
        
        # Training time
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(methods)), training_times, color=colors, alpha=0.8)
        ax2.set_title("Training Time")
        ax2.set_ylabel("Time (minutes)")
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([self._format_method_name(m) for m in methods], rotation=45, ha='right')
        
        # Model size
        ax3 = axes[2]
        bars3 = ax3.bar(range(len(methods)), model_sizes, color=colors, alpha=0.8)
        ax3.set_title("Model Size")
        ax3.set_ylabel("Size (MB)")
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels([self._format_method_name(m) for m in methods], rotation=45, ha='right')
        
        # Combined efficiency plot (memory vs time)
        ax4 = axes[3]
        scatter = ax4.scatter(memory_usage, training_times, c=model_sizes, 
                            s=100, alpha=0.7, cmap='viridis')
        ax4.set_xlabel("Peak Memory (GB)")
        ax4.set_ylabel("Training Time (minutes)")
        ax4.set_title("Resource Efficiency")
        
        # Add colorbar for model size
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("Model Size (MB)")
        
        # Annotate points
        for i, method in enumerate(methods):
            ax4.annotate(
                self._format_method_name(method),
                (memory_usage[i], training_times[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.config.output_dir / filename
        self.style_manager.save_figure(fig, str(output_path), self.config.formats, self.config.dpi)
        
        plt.close(fig)
        
        logger.info(f"Resource usage visualization saved to {output_path}")
        return output_path
    
    def _get_method_name(self, result: ExperimentResult) -> str:
        """Extract method name from experiment result."""
        if not result.config:
            return "unknown"
        
        # Build method name from configuration
        method_parts = []
        
        # Base method
        if hasattr(result.config, 'peft') and result.config.peft:
            if result.config.peft.method == "lora":
                method_parts.append(f"lora_r{result.config.peft.rank}")
            elif result.config.peft.method == "adalora":
                method_parts.append("adalora")
            else:
                method_parts.append(result.config.peft.method)
        else:
            method_parts.append("full_finetune")
        
        # Quantization
        if hasattr(result.config, 'quantization') and result.config.quantization:
            if result.config.quantization.bits == 8:
                method_parts.append("8bit")
            elif result.config.quantization.bits == 4:
                method_parts.append("4bit")
        
        return "_".join(method_parts)
    
    def _format_method_name(self, method_name: str) -> str:
        """Format method name for display."""
        # Replace underscores and capitalize
        formatted = method_name.replace('_', ' ').title()
        
        # Special formatting for common terms
        formatted = formatted.replace('Lora', 'LoRA')
        formatted = formatted.replace('Adalora', 'AdaLoRA')
        formatted = formatted.replace('8bit', '8-bit')
        formatted = formatted.replace('4bit', '4-bit')
        formatted = formatted.replace('Finetune', 'Fine-tune')
        
        return formatted
    
    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display."""
        # Common metric name mappings
        metric_mappings = {
            'final_accuracy': 'Accuracy',
            'trainable_parameters': 'Trainable Parameters',
            'model_size_mb': 'Model Size (MB)',
            'peak_memory_gb': 'Peak Memory (GB)',
            'training_time': 'Training Time (s)',
            'final_loss': 'Loss'
        }
        
        return metric_mappings.get(metric_name, metric_name.replace('_', ' ').title())
    
    def _add_significance_indicators(
        self,
        ax: plt.Axes,
        grouped_data: Dict[str, List[float]],
        group_names: List[str],
        y_position: float
    ):
        """Add statistical significance indicators to comparison chart."""
        if not self.stats_analyzer:
            return
        
        try:
            # Perform multiple comparisons
            comparison_results = self.stats_analyzer.compare_multiple_methods(grouped_data)
            
            if "pairwise_comparisons" not in comparison_results:
                return
            
            # Add significance indicators
            y_offset = y_position * 0.05
            
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names[i+1:], i+1):
                    comparison_key = f"{group1}_vs_{group2}"
                    
                    if comparison_key in comparison_results["pairwise_comparisons"]:
                        test_result = comparison_results["pairwise_comparisons"][comparison_key]
                        
                        if test_result.is_significant:
                            # Draw significance line
                            x1, x2 = i, j
                            y = y_position + y_offset * (j - i)
                            
                            ax.plot([x1, x2], [y, y], 'k-', linewidth=1)
                            ax.plot([x1, x1], [y - y_offset*0.1, y + y_offset*0.1], 'k-', linewidth=1)
                            ax.plot([x2, x2], [y - y_offset*0.1, y + y_offset*0.1], 'k-', linewidth=1)
                            
                            # Add significance symbol
                            sig_symbol = self.style_manager.get_significance_symbol(test_result.p_value)
                            ax.text((x1 + x2) / 2, y + y_offset*0.2, sig_symbol,
                                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        except Exception as e:
            logger.warning(f"Failed to add significance indicators: {e}")
    
    def generate_ablation_study_chart(
        self,
        results: List[ExperimentResult],
        ablation_factor: str = "lora_rank",
        metric: str = "final_accuracy",
        filename: str = "ablation_study"
    ) -> Path:
        """
        Generate ablation study chart showing effect of different configurations.
        
        Args:
            results: List of experiment results
            ablation_factor: Factor being ablated (e.g., 'lora_rank', 'quantization')
            metric: Metric to analyze
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Generating ablation study chart for {ablation_factor}")
        
        # Group results by ablation factor
        factor_results = {}
        for result in results:
            if not result.is_successful or metric not in result.metrics:
                continue
            
            factor_value = self._extract_ablation_factor(result, ablation_factor)
            if factor_value is not None:
                if factor_value not in factor_results:
                    factor_results[factor_value] = []
                factor_results[factor_value].append(result.metrics[metric])
        
        if not factor_results:
            logger.error(f"No valid data for ablation factor: {ablation_factor}")
            return None
        
        # Sort factor values appropriately
        if ablation_factor == "lora_rank":
            sorted_factors = sorted(factor_results.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        else:
            sorted_factors = sorted(factor_results.keys())
        
        # Compute statistics
        means = []
        errors = []
        
        for factor_value in sorted_factors:
            values = factor_results[factor_value]
            
            if self.stats_analyzer and len(values) > 1:
                ci = self.stats_analyzer.compute_confidence_interval(values)
                mean = ci.mean
                error = ci.margin_of_error
            else:
                mean = np.mean(values)
                error = np.std(values) if len(values) > 1 else 0
            
            means.append(mean)
            errors.append(error)
        
        # Create figure
        fig, ax = self.style_manager.setup_figure(figsize=self.config.figsize)
        
        # Line plot with error bars
        x_positions = range(len(sorted_factors))
        ax.errorbar(
            x_positions, means, yerr=errors,
            marker='o', markersize=8, linewidth=2,
            capsize=5, capthick=1.5,
            color=self.style_manager.COLORS['primary']
        )
        
        # Formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(f) for f in sorted_factors])
        
        self.style_manager.format_axes_labels(
            ax,
            xlabel=ablation_factor.replace('_', ' ').title(),
            ylabel=self._format_metric_name(metric),
            title=f"Ablation Study: {ablation_factor.replace('_', ' ').title()}"
        )
        
        # Save figure
        output_path = self.config.output_dir / filename
        self.style_manager.save_figure(fig, str(output_path), self.config.formats, self.config.dpi)
        
        plt.close(fig)
        
        logger.info(f"Ablation study chart saved to {output_path}")
        return output_path
    
    def generate_efficiency_scatter_plot(
        self,
        results: List[ExperimentResult],
        x_metric: str = "trainable_parameters",
        y_metric: str = "final_accuracy",
        size_metric: str = "peak_memory_gb",
        filename: str = "efficiency_scatter"
    ) -> Path:
        """
        Generate scatter plot showing multi-dimensional efficiency analysis.
        
        Args:
            results: List of experiment results
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            size_metric: Metric for point size
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        logger.info("Generating efficiency scatter plot")
        
        # Extract data
        methods = []
        x_values = []
        y_values = []
        size_values = []
        
        for result in results:
            if not result.is_successful:
                continue
            
            if (x_metric in result.metrics and y_metric in result.metrics and 
                size_metric in result.metrics):
                
                methods.append(self._get_method_name(result))
                x_values.append(result.metrics[x_metric])
                y_values.append(result.metrics[y_metric])
                size_values.append(result.metrics[size_metric])
        
        if not methods:
            logger.error("No valid data for efficiency scatter plot")
            return None
        
        # Create figure
        fig, ax = self.style_manager.setup_figure(figsize=self.config.figsize)
        
        # Normalize sizes for visualization
        min_size, max_size = min(size_values), max(size_values)
        size_range = max_size - min_size if max_size > min_size else 1
        normalized_sizes = [50 + 200 * (s - min_size) / size_range for s in size_values]
        
        # Color by method type
        colors = [self.style_manager.get_method_color(method) for method in methods]
        
        # Create scatter plot
        scatter = ax.scatter(
            x_values, y_values, s=normalized_sizes, c=colors,
            alpha=0.7, edgecolors='black', linewidth=1
        )
        
        # Annotate points
        for i, method in enumerate(methods):
            ax.annotate(
                self._format_method_name(method),
                (x_values[i], y_values[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        # Formatting
        self.style_manager.format_axes_labels(
            ax,
            xlabel=self._format_metric_name(x_metric),
            ylabel=self._format_metric_name(y_metric),
            title="Multi-dimensional Efficiency Analysis"
        )
        
        # Add size legend
        size_legend_elements = []
        for size_val in [min_size, (min_size + max_size) / 2, max_size]:
            norm_size = 50 + 200 * (size_val - min_size) / size_range
            size_legend_elements.append(
                plt.scatter([], [], s=norm_size, c='gray', alpha=0.7,
                           label=f'{size_val:.1f}')
            )
        
        size_legend = ax.legend(
            handles=size_legend_elements,
            title=self._format_metric_name(size_metric),
            loc='upper left',
            bbox_to_anchor=(1.05, 1)
        )
        
        # Save figure
        output_path = self.config.output_dir / filename
        self.style_manager.save_figure(fig, str(output_path), self.config.formats, self.config.dpi)
        
        plt.close(fig)
        
        logger.info(f"Efficiency scatter plot saved to {output_path}")
        return output_path
    
    def _extract_ablation_factor(self, result: ExperimentResult, factor: str) -> Optional[Any]:
        """Extract ablation factor value from result."""
        if not result.config:
            return None
        
        if factor == "lora_rank":
            if hasattr(result.config, 'peft') and result.config.peft:
                return getattr(result.config.peft, 'rank', None)
        elif factor == "quantization":
            if hasattr(result.config, 'quantization') and result.config.quantization:
                return f"{result.config.quantization.bits}bit"
            else:
                return "none"
        elif factor == "model":
            return getattr(result.config.model, 'name', None)
        elif factor == "dataset":
            return getattr(result.config.dataset, 'name', None)
        
        return None
    
    def generate_all_figures(self, results: List[ExperimentResult]) -> Dict[str, Path]:
        """
        Generate all standard figures for the research.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        logger.info(f"Generating all figures from {len(results)} results")
        
        generated_figures = {}
        
        try:
            # Pareto frontier plot
            pareto_path = self.generate_pareto_frontier_plot(results)
            if pareto_path:
                generated_figures["pareto_frontier"] = pareto_path
            
            # Method comparison chart
            comparison_path = self.generate_method_comparison_chart(results)
            if comparison_path:
                generated_figures["method_comparison"] = comparison_path
            
            # Resource usage visualization
            resource_path = self.generate_resource_usage_visualization(results)
            if resource_path:
                generated_figures["resource_usage"] = resource_path
            
            # Ablation study chart
            ablation_path = self.generate_ablation_study_chart(results, "lora_rank")
            if ablation_path:
                generated_figures["ablation_study"] = ablation_path
            
            # Efficiency scatter plot
            scatter_path = self.generate_efficiency_scatter_plot(results)
            if scatter_path:
                generated_figures["efficiency_scatter"] = scatter_path
            
            # Convergence curves (if training history available)
            training_histories = {}
            for result in results:
                if result.training_history:
                    method_name = self._get_method_name(result)
                    if method_name not in training_histories:
                        training_histories[method_name] = []
                    training_histories[method_name].append(result.training_history)
            
            if training_histories:
                convergence_path = self.generate_convergence_curves(training_histories)
                if convergence_path:
                    generated_figures["convergence_curves"] = convergence_path
            
            logger.info(f"Generated {len(generated_figures)} figures")
            
        except Exception as e:
            logger.error(f"Error generating figures: {e}")
        
        return generated_figures