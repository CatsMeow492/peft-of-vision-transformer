"""
Enhanced statistical analysis for PEFT experiment results with multiple seeds and aggregation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import json

try:
    from .results import ExperimentResult, ResultsManager
    from ..evaluation.statistical_analyzer import StatisticalAnalyzer, ConfidenceInterval, SignificanceTest
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ValueError):
    # Create dummy classes for standalone usage
    class ExperimentResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ResultsManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class StatisticalAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
    
    class ConfidenceInterval:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SignificanceTest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AggregatedResults:
    """Container for aggregated experiment results across multiple seeds."""
    
    experiment_base_id: str  # ID without seed
    method_name: str
    model_name: str
    dataset_name: str
    
    # Aggregated metrics
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, ConfidenceInterval] = field(default_factory=dict)
    
    # Individual results
    individual_results: List[ExperimentResult] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    
    # Metadata
    num_seeds: int = 0
    success_rate: float = 0.0  # Fraction of successful runs
    
    def add_result(self, result: ExperimentResult, seed: int):
        """Add an individual experiment result."""
        self.individual_results.append(result)
        self.seeds.append(seed)
        self.num_seeds = len(self.individual_results)
        
        # Update success rate
        successful_results = [r for r in self.individual_results if r.is_successful]
        self.success_rate = len(successful_results) / self.num_seeds
    
    def compute_statistics(self, analyzer: StatisticalAnalyzer):
        """Compute aggregated statistics using the statistical analyzer."""
        if not self.individual_results:
            return
        
        # Get successful results only
        successful_results = [r for r in self.individual_results if r.is_successful]
        
        if not successful_results:
            logger.warning(f"No successful results for {self.experiment_base_id}")
            return
        
        # Aggregate metrics
        all_metrics = {}
        for result in successful_results:
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        # Compute statistics for each metric
        for metric_name, values in all_metrics.items():
            if len(values) > 1:
                # Compute mean and std
                self.mean_metrics[metric_name] = sum(values) / len(values)
                variance = sum((x - self.mean_metrics[metric_name]) ** 2 for x in values) / (len(values) - 1)
                self.std_metrics[metric_name] = variance ** 0.5
                
                # Compute confidence interval
                try:
                    ci = analyzer.compute_confidence_interval(values)
                    self.confidence_intervals[metric_name] = ci
                except Exception as e:
                    logger.warning(f"Failed to compute CI for {metric_name}: {e}")
            else:
                # Single value
                self.mean_metrics[metric_name] = values[0]
                self.std_metrics[metric_name] = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of aggregated results."""
        return {
            "experiment_base_id": self.experiment_base_id,
            "method_name": self.method_name,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "num_seeds": self.num_seeds,
            "success_rate": self.success_rate,
            "mean_metrics": self.mean_metrics,
            "std_metrics": self.std_metrics,
            "seeds": self.seeds
        }


@dataclass
class AblationStudyResult:
    """Container for ablation study analysis results."""
    
    study_name: str
    baseline_method: str
    component_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Overall analysis
    most_important_component: Optional[str] = None
    least_important_component: Optional[str] = None
    
    def add_component_effect(
        self, 
        component_name: str, 
        baseline_results: List[float],
        modified_results: List[float],
        analyzer: StatisticalAnalyzer
    ):
        """Add analysis of a component's effect."""
        try:
            # Perform statistical test
            significance_test = analyzer.paired_t_test(baseline_results, modified_results)
            
            # Calculate effect statistics
            baseline_mean = sum(baseline_results) / len(baseline_results)
            modified_mean = sum(modified_results) / len(modified_results)
            effect_magnitude = modified_mean - baseline_mean
            relative_effect = (effect_magnitude / baseline_mean) * 100 if baseline_mean != 0 else 0
            
            self.component_effects[component_name] = {
                "baseline_mean": baseline_mean,
                "modified_mean": modified_mean,
                "effect_magnitude": effect_magnitude,
                "relative_effect_percent": relative_effect,
                "significance_test": significance_test,
                "is_beneficial": effect_magnitude > 0,
                "is_significant": significance_test.is_significant
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze component {component_name}: {e}")
    
    def analyze_importance(self):
        """Analyze component importance based on effect magnitudes."""
        if not self.component_effects:
            return
        
        # Find most and least important components
        effects = [(name, data["effect_magnitude"]) for name, data in self.component_effects.items()]
        effects.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if effects:
            self.most_important_component = effects[0][0]
            self.least_important_component = effects[-1][0]


class ExperimentStatisticalAnalyzer:
    """
    Enhanced statistical analyzer for PEFT experiment results.
    
    Handles multiple random seeds, result aggregation, and comparative analysis
    for publication-quality research.
    """
    
    def __init__(self, results_manager: ResultsManager, confidence_level: float = 0.95):
        """
        Initialize experiment statistical analyzer.
        
        Args:
            results_manager: ResultsManager for accessing experiment results
            confidence_level: Default confidence level for statistical tests
        """
        self.results_manager = results_manager
        self.analyzer = StatisticalAnalyzer(confidence_level)
        self.confidence_level = confidence_level
        
        logger.info("ExperimentStatisticalAnalyzer initialized")
    
    def aggregate_results_by_method(
        self, 
        metric_name: str = "final_accuracy"
    ) -> Dict[str, AggregatedResults]:
        """
        Aggregate experiment results by method across multiple seeds.
        
        Args:
            metric_name: Primary metric to analyze
            
        Returns:
            Dictionary mapping method names to aggregated results
        """
        all_results = self.results_manager.list_results(status_filter="completed")
        
        # Group results by method configuration (excluding seed)
        method_groups = {}
        
        for result in all_results:
            if not result.config:
                continue
            
            # Create method identifier (without seed)
            method_id = self._get_method_identifier(result)
            
            if method_id not in method_groups:
                method_groups[method_id] = AggregatedResults(
                    experiment_base_id=method_id,
                    method_name=self._get_method_name(result),
                    model_name=result.config.model.name,
                    dataset_name=result.config.dataset.name
                )
            
            # Extract seed from config
            seed = getattr(result.config, 'seed', 42)
            method_groups[method_id].add_result(result, seed)
        
        # Compute statistics for each method
        for method_id, aggregated in method_groups.items():
            aggregated.compute_statistics(self.analyzer)
        
        return method_groups
    
    def _get_method_identifier(self, result: ExperimentResult) -> str:
        """Get method identifier excluding seed for grouping."""
        config = result.config
        
        components = [
            config.model.name,
            config.dataset.name
        ]
        
        if config.lora:
            components.append(f"lora_r{config.lora.rank}_a{config.lora.alpha}")
        
        if config.quantization:
            components.append(f"q{config.quantization.bits}bit")
        
        if config.use_adalora:
            components.append("adalora")
        
        if config.use_qa_lora:
            components.append("qalora")
        
        return "_".join(components)
    
    def _get_method_name(self, result: ExperimentResult) -> str:
        """Get human-readable method name."""
        config = result.config
        
        if config.use_adalora:
            method = "AdaLoRA"
        elif config.use_qa_lora:
            method = "QA-LoRA"
        elif config.lora:
            method = "LoRA"
        else:
            method = "Full Fine-tuning"
        
        if config.quantization:
            method += f" ({config.quantization.bits}-bit)"
        
        return method
    
    def compare_methods(
        self, 
        metric_name: str = "final_accuracy",
        min_seeds: int = 3
    ) -> Dict[str, Any]:
        """
        Compare different PEFT methods statistically.
        
        Args:
            metric_name: Metric to compare
            min_seeds: Minimum number of seeds required for comparison
            
        Returns:
            Dictionary with comparison results
        """
        aggregated_results = self.aggregate_results_by_method(metric_name)
        
        # Filter methods with sufficient seeds
        valid_methods = {
            name: agg for name, agg in aggregated_results.items()
            if agg.num_seeds >= min_seeds and agg.success_rate > 0.5
        }
        
        if len(valid_methods) < 2:
            logger.warning(f"Insufficient methods for comparison (need >= 2, got {len(valid_methods)})")
            return {"error": "Insufficient methods for comparison"}
        
        # Prepare data for comparison
        method_results = {}
        for method_name, agg_result in valid_methods.items():
            # Extract metric values from successful results
            values = []
            for result in agg_result.individual_results:
                if result.is_successful and metric_name in result.metrics:
                    values.append(result.metrics[metric_name])
            
            if values:
                method_results[agg_result.method_name] = values
        
        if len(method_results) < 2:
            return {"error": "Insufficient valid results for comparison"}
        
        # Perform statistical comparison
        try:
            comparison_results = self.analyzer.compare_multiple_methods(
                method_results, 
                self.confidence_level
            )
            
            # Add method metadata
            comparison_results["method_metadata"] = {
                name: agg.get_summary() for name, agg in valid_methods.items()
            }
            
            # Rank methods by performance
            method_means = {
                method: agg.mean_metrics.get(metric_name, 0)
                for method, agg in valid_methods.items()
            }
            
            ranked_methods = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
            comparison_results["method_ranking"] = ranked_methods
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Method comparison failed: {e}")
            return {"error": str(e)}
    
    def analyze_lora_rank_effects(
        self, 
        model_name: str,
        dataset_name: str,
        metric_name: str = "final_accuracy"
    ) -> Dict[str, Any]:
        """
        Analyze the effect of different LoRA ranks.
        
        Args:
            model_name: Model to analyze
            dataset_name: Dataset to analyze
            metric_name: Metric to analyze
            
        Returns:
            Analysis results
        """
        # Filter results for specific model/dataset with LoRA
        all_results = self.results_manager.list_results(
            status_filter="completed",
            model_filter=model_name,
            dataset_filter=dataset_name
        )
        
        lora_results = [r for r in all_results if r.config and r.config.lora and not r.config.use_adalora]
        
        if not lora_results:
            return {"error": "No LoRA results found for specified model/dataset"}
        
        # Group by LoRA rank
        rank_groups = {}
        for result in lora_results:
            rank = result.config.lora.rank
            if rank not in rank_groups:
                rank_groups[rank] = []
            
            if result.is_successful and metric_name in result.metrics:
                rank_groups[rank].append(result.metrics[metric_name])
        
        # Filter ranks with sufficient data
        valid_ranks = {rank: values for rank, values in rank_groups.items() if len(values) >= 2}
        
        if len(valid_ranks) < 2:
            return {"error": "Insufficient data for rank analysis"}
        
        # Compute statistics for each rank
        rank_statistics = {}
        for rank, values in valid_ranks.items():
            ci = self.analyzer.compute_confidence_interval(values)
            rank_statistics[rank] = {
                "mean": ci.mean,
                "confidence_interval": (ci.lower_bound, ci.upper_bound),
                "sample_size": ci.sample_size,
                "values": values
            }
        
        # Perform pairwise comparisons
        pairwise_comparisons = {}
        ranks = sorted(valid_ranks.keys())
        
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                rank1, rank2 = ranks[i], ranks[j]
                comparison_key = f"rank_{rank1}_vs_rank_{rank2}"
                
                try:
                    test_result = self.analyzer.independent_t_test(
                        valid_ranks[rank1],
                        valid_ranks[rank2]
                    )
                    pairwise_comparisons[comparison_key] = test_result
                except Exception as e:
                    logger.warning(f"Failed comparison {comparison_key}: {e}")
        
        # Find optimal rank
        best_rank = max(rank_statistics.keys(), key=lambda r: rank_statistics[r]["mean"])
        
        return {
            "model": model_name,
            "dataset": dataset_name,
            "metric": metric_name,
            "rank_statistics": rank_statistics,
            "pairwise_comparisons": pairwise_comparisons,
            "best_rank": best_rank,
            "best_performance": rank_statistics[best_rank]["mean"],
            "analysis_summary": f"Best LoRA rank: {best_rank} (mean {metric_name}: {rank_statistics[best_rank]['mean']:.4f})"
        }
    
    def perform_ablation_study(
        self,
        baseline_config_pattern: str,
        component_variations: Dict[str, str],
        metric_name: str = "final_accuracy"
    ) -> AblationStudyResult:
        """
        Perform ablation study to analyze component contributions.
        
        Args:
            baseline_config_pattern: Pattern to identify baseline experiments
            component_variations: Dict mapping component names to config patterns
            metric_name: Metric to analyze
            
        Returns:
            AblationStudyResult with analysis
        """
        ablation_result = AblationStudyResult(
            study_name=f"Ablation Study - {metric_name}",
            baseline_method=baseline_config_pattern
        )
        
        # Get baseline results
        all_results = self.results_manager.list_results(status_filter="completed")
        baseline_results = [
            r for r in all_results 
            if baseline_config_pattern in r.experiment_id and r.is_successful
        ]
        
        if not baseline_results:
            logger.error(f"No baseline results found for pattern: {baseline_config_pattern}")
            return ablation_result
        
        baseline_values = [r.metrics[metric_name] for r in baseline_results if metric_name in r.metrics]
        
        # Analyze each component
        for component_name, component_pattern in component_variations.items():
            component_results = [
                r for r in all_results
                if component_pattern in r.experiment_id and r.is_successful
            ]
            
            if not component_results:
                logger.warning(f"No results found for component: {component_name}")
                continue
            
            component_values = [r.metrics[metric_name] for r in component_results if metric_name in r.metrics]
            
            if len(component_values) >= 2 and len(baseline_values) >= 2:
                ablation_result.add_component_effect(
                    component_name,
                    baseline_values,
                    component_values,
                    self.analyzer
                )
        
        # Analyze component importance
        ablation_result.analyze_importance()
        
        return ablation_result
    
    def generate_publication_summary(
        self, 
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate publication-ready statistical summary.
        
        Args:
            output_path: Optional path to save the summary
            
        Returns:
            Formatted summary string
        """
        summary_lines = [
            "PEFT Vision Transformer - Statistical Analysis Summary",
            "=" * 60,
            ""
        ]
        
        # Overall experiment statistics
        all_results = self.results_manager.list_results()
        completed_results = [r for r in all_results if r.status == "completed"]
        
        summary_lines.extend([
            f"Total Experiments: {len(all_results)}",
            f"Completed Successfully: {len(completed_results)}",
            f"Success Rate: {len(completed_results)/len(all_results)*100:.1f}%",
            ""
        ])
        
        # Method comparison
        try:
            method_comparison = self.compare_methods("final_accuracy")
            
            if "error" not in method_comparison:
                summary_lines.extend([
                    "Method Performance Comparison (Final Accuracy):",
                    "-" * 50
                ])
                
                for i, (method, mean_acc) in enumerate(method_comparison["method_ranking"], 1):
                    summary_lines.append(f"{i}. {method}: {mean_acc:.4f}")
                
                summary_lines.append("")
                
                # Statistical significance
                if method_comparison.get("anova", {}).get("is_significant"):
                    summary_lines.append("✓ Significant differences found between methods (ANOVA)")
                else:
                    summary_lines.append("✗ No significant differences between methods (ANOVA)")
                
                summary_lines.append("")
        
        except Exception as e:
            logger.warning(f"Failed to generate method comparison: {e}")
        
        # LoRA rank analysis for each model/dataset combination
        try:
            # Get unique model/dataset combinations
            model_dataset_pairs = set()
            for result in completed_results:
                if result.config:
                    model_dataset_pairs.add((result.config.model.name, result.config.dataset.name))
            
            summary_lines.extend([
                "LoRA Rank Analysis:",
                "-" * 30
            ])
            
            for model, dataset in sorted(model_dataset_pairs):
                rank_analysis = self.analyze_lora_rank_effects(model, dataset)
                
                if "error" not in rank_analysis:
                    summary_lines.extend([
                        f"{model} on {dataset}:",
                        f"  {rank_analysis['analysis_summary']}",
                        ""
                    ])
        
        except Exception as e:
            logger.warning(f"Failed to generate LoRA rank analysis: {e}")
        
        # Resource usage statistics
        try:
            memory_usage = [r.peak_memory_gb for r in completed_results if r.peak_memory_gb > 0]
            training_times = [r.duration_seconds for r in completed_results if r.duration_seconds > 0]
            
            if memory_usage:
                avg_memory = sum(memory_usage) / len(memory_usage)
                max_memory = max(memory_usage)
                
                summary_lines.extend([
                    "Resource Usage Statistics:",
                    "-" * 30,
                    f"Average Peak Memory: {avg_memory:.1f}GB",
                    f"Maximum Peak Memory: {max_memory:.1f}GB",
                ])
            
            if training_times:
                avg_time = sum(training_times) / len(training_times)
                total_time = sum(training_times)
                
                summary_lines.extend([
                    f"Average Training Time: {avg_time:.1f}s",
                    f"Total Training Time: {total_time/3600:.1f}h",
                    ""
                ])
        
        except Exception as e:
            logger.warning(f"Failed to generate resource statistics: {e}")
        
        summary_lines.extend([
            "Statistical Analysis Notes:",
            "-" * 30,
            f"• Confidence Level: {self.confidence_level*100}%",
            "• Multiple comparisons corrected using appropriate statistical tests",
            "• Results aggregated across multiple random seeds for robustness",
            "• Only successful experiments included in analysis",
            ""
        ])
        
        summary_text = "\n".join(summary_lines)
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(summary_text)
            
            logger.info(f"Publication summary saved to {output_path}")
        
        return summary_text
    
    def export_results_for_plotting(
        self, 
        output_path: Union[str, Path],
        metric_name: str = "final_accuracy"
    ):
        """
        Export aggregated results in format suitable for plotting.
        
        Args:
            output_path: Path to save the results
            metric_name: Primary metric to export
        """
        aggregated_results = self.aggregate_results_by_method(metric_name)
        
        # Prepare data for export
        export_data = {
            "metadata": {
                "metric_name": metric_name,
                "confidence_level": self.confidence_level,
                "export_timestamp": str(datetime.now())
            },
            "methods": {}
        }
        
        for method_id, agg_result in aggregated_results.items():
            if agg_result.num_seeds >= 2:  # Only export methods with multiple seeds
                method_data = {
                    "method_name": agg_result.method_name,
                    "model": agg_result.model_name,
                    "dataset": agg_result.dataset_name,
                    "num_seeds": agg_result.num_seeds,
                    "success_rate": agg_result.success_rate,
                    "metrics": {}
                }
                
                # Export all metrics with statistics
                for metric, mean_val in agg_result.mean_metrics.items():
                    metric_data = {
                        "mean": mean_val,
                        "std": agg_result.std_metrics.get(metric, 0.0)
                    }
                    
                    # Add confidence interval if available
                    if metric in agg_result.confidence_intervals:
                        ci = agg_result.confidence_intervals[metric]
                        metric_data["confidence_interval"] = {
                            "lower": ci.lower_bound,
                            "upper": ci.upper_bound,
                            "margin_of_error": ci.margin_of_error
                        }
                    
                    # Add individual values
                    individual_values = []
                    for result in agg_result.individual_results:
                        if result.is_successful and metric in result.metrics:
                            individual_values.append(result.metrics[metric])
                    
                    metric_data["individual_values"] = individual_values
                    method_data["metrics"][metric] = metric_data
                
                export_data["methods"][method_id] = method_data
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Results exported for plotting to {output_path}")


# Utility functions for common statistical analysis patterns

def analyze_experiment_directory(
    results_dir: Union[str, Path],
    confidence_level: float = 0.95
) -> ExperimentStatisticalAnalyzer:
    """
    Create analyzer for experiment directory.
    
    Args:
        results_dir: Directory containing experiment results
        confidence_level: Confidence level for statistical tests
        
    Returns:
        Configured ExperimentStatisticalAnalyzer
    """
    results_manager = ResultsManager(results_dir, use_database=False)
    return ExperimentStatisticalAnalyzer(results_manager, confidence_level)


def quick_method_comparison(
    results_dir: Union[str, Path],
    metric_name: str = "final_accuracy"
) -> Dict[str, Any]:
    """
    Quick comparison of methods in results directory.
    
    Args:
        results_dir: Directory containing experiment results
        metric_name: Metric to compare
        
    Returns:
        Comparison results
    """
    analyzer = analyze_experiment_directory(results_dir)
    return analyzer.compare_methods(metric_name)


def generate_statistical_report(
    results_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate comprehensive statistical report for experiment directory.
    
    Args:
        results_dir: Directory containing experiment results
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    analyzer = analyze_experiment_directory(results_dir)
    return analyzer.generate_publication_summary(output_path)