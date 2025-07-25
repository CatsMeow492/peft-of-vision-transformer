"""
LaTeX table generation for PEFT Vision Transformer research results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

# Handle optional imports
try:
    from ..evaluation.statistical_analyzer import StatisticalAnalyzer, ConfidenceInterval
    from ..experiments.results import ExperimentResult
    EVALUATION_AVAILABLE = True
except ImportError:
    class StatisticalAnalyzer:
        pass
    class ConfidenceInterval:
        pass
    class ExperimentResult:
        pass
    EVALUATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TableConfig:
    """Configuration for table generation."""
    
    output_dir: Path
    precision: int = 3
    confidence_level: float = 0.95
    include_significance: bool = True
    table_style: str = "booktabs"  # booktabs, standard, minimal
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


class LaTeXTableGenerator:
    """
    Generator for publication-quality LaTeX tables.
    
    Creates comprehensive results tables with statistical analysis,
    method comparisons, and ablation study results.
    """
    
    def __init__(self, config: TableConfig):
        """
        Initialize table generator.
        
        Args:
            config: Table generation configuration
        """
        self.config = config
        
        if EVALUATION_AVAILABLE:
            self.stats_analyzer = StatisticalAnalyzer(
                default_confidence_level=config.confidence_level
            )
        else:
            self.stats_analyzer = None
            logger.warning("Statistical analyzer not available")
        
        logger.info(f"LaTeXTableGenerator initialized with output_dir: {config.output_dir}")
    
    def generate_main_results_table(
        self,
        results: List[ExperimentResult],
        metrics: List[str] = None,
        filename: str = "main_results"
    ) -> Path:
        """
        Generate main results table with all methods and key metrics.
        
        Args:
            results: List of experiment results
            metrics: List of metrics to include (default: key metrics)
            filename: Output filename (without extension)
            
        Returns:
            Path to saved LaTeX file
        """
        if metrics is None:
            metrics = [
                "final_accuracy", "trainable_parameters", "model_size_mb",
                "peak_memory_gb", "training_time"
            ]
        
        logger.info(f"Generating main results table with {len(results)} results")
        
        # Group results by method
        method_results = {}
        for result in results:
            if not result.is_successful:
                continue
            
            method_name = self._get_method_name(result)
            if method_name not in method_results:
                method_results[method_name] = []
            method_results[method_name].append(result)
        
        if not method_results:
            logger.error("No valid results for main table")
            return None
        
        # Compute statistics for each method and metric
        table_data = {}
        for method_name, method_results_list in method_results.items():
            table_data[method_name] = {}
            
            for metric in metrics:
                values = [r.metrics.get(metric, 0) for r in method_results_list 
                         if metric in r.metrics]
                
                if values:
                    if self.stats_analyzer and len(values) > 1:
                        ci = self.stats_analyzer.compute_confidence_interval(values)
                        table_data[method_name][metric] = {
                            "mean": ci.mean,
                            "ci_lower": ci.lower_bound,
                            "ci_upper": ci.upper_bound,
                            "n": ci.sample_size
                        }
                    else:
                        mean_val = np.mean(values)
                        table_data[method_name][metric] = {
                            "mean": mean_val,
                            "ci_lower": mean_val,
                            "ci_upper": mean_val,
                            "n": len(values)
                        }
                else:
                    table_data[method_name][metric] = {
                        "mean": 0, "ci_lower": 0, "ci_upper": 0, "n": 0
                    }
        
        # Generate LaTeX table
        latex_content = self._generate_latex_table(
            table_data, metrics, "Main Results",
            caption="Comprehensive comparison of PEFT methods on Vision Transformers. "
                   f"Results show mean ± {int(self.config.confidence_level*100)}% confidence interval.",
            label="tab:main_results"
        )
        
        # Save to file
        output_path = self.config.output_dir / f"{filename}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Main results table saved to {output_path}")
        return output_path
    
    def generate_ablation_study_table(
        self,
        results: List[ExperimentResult],
        ablation_factor: str = "lora_rank",
        filename: str = "ablation_study"
    ) -> Path:
        """
        Generate ablation study table showing effect of different configurations.
        
        Args:
            results: List of experiment results
            ablation_factor: Factor being ablated (e.g., 'lora_rank', 'quantization')
            filename: Output filename (without extension)
            
        Returns:
            Path to saved LaTeX file
        """
        logger.info(f"Generating ablation study table for {ablation_factor}")
        
        # Group results by ablation factor
        factor_results = {}
        for result in results:
            if not result.is_successful:
                continue
            
            factor_value = self._extract_ablation_factor(result, ablation_factor)
            if factor_value is not None:
                if factor_value not in factor_results:
                    factor_results[factor_value] = []
                factor_results[factor_value].append(result)
        
        if not factor_results:
            logger.error(f"No valid results for ablation factor: {ablation_factor}")
            return None
        
        # Key metrics for ablation study
        metrics = ["final_accuracy", "trainable_parameters", "training_time"]
        
        # Compute statistics
        table_data = {}
        for factor_value, factor_results_list in factor_results.items():
            table_data[str(factor_value)] = {}
            
            for metric in metrics:
                values = [r.metrics.get(metric, 0) for r in factor_results_list 
                         if metric in r.metrics]
                
                if values:
                    if self.stats_analyzer and len(values) > 1:
                        ci = self.stats_analyzer.compute_confidence_interval(values)
                        table_data[str(factor_value)][metric] = {
                            "mean": ci.mean,
                            "ci_lower": ci.lower_bound,
                            "ci_upper": ci.upper_bound,
                            "n": ci.sample_size
                        }
                    else:
                        mean_val = np.mean(values)
                        table_data[str(factor_value)][metric] = {
                            "mean": mean_val,
                            "ci_lower": mean_val,
                            "ci_upper": mean_val,
                            "n": len(values)
                        }
        
        # Generate LaTeX table
        latex_content = self._generate_latex_table(
            table_data, metrics, f"Ablation Study: {ablation_factor.replace('_', ' ').title()}",
            caption=f"Ablation study results showing the effect of {ablation_factor.replace('_', ' ')} "
                   f"on model performance. Results show mean ± {int(self.config.confidence_level*100)}% confidence interval.",
            label=f"tab:ablation_{ablation_factor}"
        )
        
        # Save to file
        output_path = self.config.output_dir / f"{filename}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Ablation study table saved to {output_path}")
        return output_path
    
    def generate_statistical_significance_table(
        self,
        results: List[ExperimentResult],
        metric: str = "final_accuracy",
        filename: str = "significance_tests"
    ) -> Path:
        """
        Generate table showing statistical significance between methods.
        
        Args:
            results: List of experiment results
            metric: Metric to test for significance
            filename: Output filename (without extension)
            
        Returns:
            Path to saved LaTeX file
        """
        if not self.stats_analyzer:
            logger.error("Statistical analyzer not available for significance testing")
            return None
        
        logger.info(f"Generating statistical significance table for {metric}")
        
        # Group results by method
        method_results = {}
        for result in results:
            if not result.is_successful or metric not in result.metrics:
                continue
            
            method_name = self._get_method_name(result)
            if method_name not in method_results:
                method_results[method_name] = []
            method_results[method_name].append(result.metrics[metric])
        
        if len(method_results) < 2:
            logger.error("Need at least 2 methods for significance testing")
            return None
        
        # Perform pairwise comparisons
        method_names = list(method_results.keys())
        n_methods = len(method_names)
        
        # Create significance matrix
        significance_matrix = {}
        p_value_matrix = {}
        
        for i, method1 in enumerate(method_names):
            significance_matrix[method1] = {}
            p_value_matrix[method1] = {}
            
            for j, method2 in enumerate(method_names):
                if i == j:
                    significance_matrix[method1][method2] = "-"
                    p_value_matrix[method1][method2] = 1.0
                elif i < j:
                    # Perform t-test
                    test_result = self.stats_analyzer.independent_t_test(
                        method_results[method1],
                        method_results[method2]
                    )
                    
                    sig_symbol = self._get_significance_symbol(test_result.p_value)
                    significance_matrix[method1][method2] = sig_symbol
                    significance_matrix[method2][method1] = sig_symbol
                    p_value_matrix[method1][method2] = test_result.p_value
                    p_value_matrix[method2][method1] = test_result.p_value
        
        # Generate LaTeX table
        latex_content = self._generate_significance_table(
            significance_matrix, p_value_matrix, method_names, metric
        )
        
        # Save to file
        output_path = self.config.output_dir / f"{filename}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Statistical significance table saved to {output_path}")
        return output_path
    
    def generate_resource_efficiency_table(
        self,
        results: List[ExperimentResult],
        filename: str = "resource_efficiency"
    ) -> Path:
        """
        Generate table focusing on resource efficiency metrics.
        
        Args:
            results: List of experiment results
            filename: Output filename (without extension)
            
        Returns:
            Path to saved LaTeX file
        """
        logger.info("Generating resource efficiency table")
        
        # Resource-focused metrics
        metrics = [
            "final_accuracy", "trainable_parameters", "model_size_mb",
            "peak_memory_gb", "training_time", "samples_per_second"
        ]
        
        # Group results by method
        method_results = {}
        for result in results:
            if not result.is_successful:
                continue
            
            method_name = self._get_method_name(result)
            if method_name not in method_results:
                method_results[method_name] = []
            method_results[method_name].append(result)
        
        # Compute efficiency ratios relative to full fine-tuning
        baseline_method = "full_finetune"
        baseline_data = method_results.get(baseline_method, [])
        
        if not baseline_data:
            logger.warning("No baseline (full fine-tuning) results found")
            baseline_metrics = None
        else:
            baseline_metrics = {}
            for metric in metrics:
                values = [r.metrics.get(metric, 0) for r in baseline_data if metric in r.metrics]
                if values:
                    baseline_metrics[metric] = np.mean(values)
        
        # Compute table data with efficiency ratios
        table_data = {}
        for method_name, method_results_list in method_results.items():
            table_data[method_name] = {}
            
            for metric in metrics:
                values = [r.metrics.get(metric, 0) for r in method_results_list 
                         if metric in r.metrics]
                
                if values:
                    mean_val = np.mean(values)
                    table_data[method_name][metric] = {"mean": mean_val}
                    
                    # Add efficiency ratio if baseline available
                    if baseline_metrics and metric in baseline_metrics and baseline_metrics[metric] > 0:
                        if metric in ["trainable_parameters", "model_size_mb", "peak_memory_gb", "training_time"]:
                            # Lower is better - compute reduction ratio
                            ratio = mean_val / baseline_metrics[metric]
                            table_data[method_name][f"{metric}_ratio"] = {"mean": ratio}
                        elif metric in ["final_accuracy", "samples_per_second"]:
                            # Higher is better - compute improvement ratio
                            ratio = mean_val / baseline_metrics[metric]
                            table_data[method_name][f"{metric}_ratio"] = {"mean": ratio}
        
        # Generate LaTeX table with efficiency focus
        extended_metrics = metrics + [f"{m}_ratio" for m in metrics if f"{m}_ratio" in next(iter(table_data.values()), {})]
        
        latex_content = self._generate_efficiency_table(table_data, extended_metrics)
        
        # Save to file
        output_path = self.config.output_dir / f"{filename}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Resource efficiency table saved to {output_path}")
        return output_path
    
    def _generate_latex_table(
        self,
        table_data: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str],
        title: str,
        caption: str,
        label: str
    ) -> str:
        """Generate LaTeX table content."""
        
        method_names = list(table_data.keys())
        n_cols = len(metrics) + 1  # +1 for method name column
        
        # Table header
        if self.config.table_style == "booktabs":
            latex_lines = [
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}",
                "\\toprule"
            ]
        else:
            latex_lines = [
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                f"\\begin{{tabular}}{{|l{'|c' * len(metrics)}|}}",
                "\\hline"
            ]
        
        # Column headers
        header_row = "Method"
        for metric in metrics:
            header_row += f" & {self._format_metric_name_latex(metric)}"
        header_row += " \\\\"
        latex_lines.append(header_row)
        
        if self.config.table_style == "booktabs":
            latex_lines.append("\\midrule")
        else:
            latex_lines.append("\\hline")
        
        # Data rows
        for method_name in sorted(method_names):
            row = self._format_method_name_latex(method_name)
            
            for metric in metrics:
                if metric in table_data[method_name]:
                    data = table_data[method_name][metric]
                    mean = data["mean"]
                    
                    if "ci_lower" in data and "ci_upper" in data and data["n"] > 1:
                        # Format with confidence interval
                        ci_lower = data["ci_lower"]
                        ci_upper = data["ci_upper"]
                        formatted_val = f"{mean:.{self.config.precision}f} ± {(ci_upper-ci_lower)/2:.{self.config.precision}f}"
                    else:
                        # Format without confidence interval
                        formatted_val = f"{mean:.{self.config.precision}f}"
                    
                    row += f" & {formatted_val}"
                else:
                    row += " & -"
            
            row += " \\\\"
            latex_lines.append(row)
        
        # Table footer
        if self.config.table_style == "booktabs":
            latex_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}"
            ])
        else:
            latex_lines.extend([
                "\\hline",
                "\\end{tabular}",
                "\\end{table}"
            ])
        
        return "\n".join(latex_lines)
    
    def _generate_significance_table(
        self,
        significance_matrix: Dict[str, Dict[str, str]],
        p_value_matrix: Dict[str, Dict[str, float]],
        method_names: List[str],
        metric: str
    ) -> str:
        """Generate LaTeX table for statistical significance."""
        
        n_methods = len(method_names)
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Statistical significance tests for {self._format_metric_name_latex(metric)}. "
            "Symbols: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant.}}",
            f"\\label{{tab:significance_{metric}}}",
            f"\\begin{{tabular}}{{l{'c' * n_methods}}}",
            "\\toprule"
        ]
        
        # Header row
        header_row = "Method"
        for method in method_names:
            header_row += f" & {self._format_method_name_latex(method)}"
        header_row += " \\\\"
        latex_lines.append(header_row)
        latex_lines.append("\\midrule")
        
        # Data rows
        for i, method1 in enumerate(method_names):
            row = self._format_method_name_latex(method1)
            
            for j, method2 in enumerate(method_names):
                if i == j:
                    row += " & -"
                else:
                    sig_symbol = significance_matrix[method1][method2]
                    row += f" & {sig_symbol}"
            
            row += " \\\\"
            latex_lines.append(row)
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def _generate_efficiency_table(
        self,
        table_data: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str]
    ) -> str:
        """Generate specialized efficiency table with ratios."""
        
        method_names = list(table_data.keys())
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Resource efficiency comparison. Ratios show improvement relative to full fine-tuning "
            "(lower is better for parameters/memory/time, higher is better for accuracy/throughput).}",
            "\\label{tab:resource_efficiency}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule"
        ]
        
        # Specialized header for efficiency table
        header_row = "Method & Accuracy & Params & Memory & Time & Acc. Ratio & Param. Ratio \\\\"
        latex_lines.append(header_row)
        latex_lines.append("\\midrule")
        
        # Data rows with efficiency focus
        for method_name in sorted(method_names):
            if method_name == "full_finetune":
                continue  # Skip baseline in efficiency table
            
            row = self._format_method_name_latex(method_name)
            
            # Core metrics
            for metric in ["final_accuracy", "trainable_parameters", "peak_memory_gb", "training_time"]:
                if metric in table_data[method_name]:
                    mean = table_data[method_name][metric]["mean"]
                    
                    if metric == "trainable_parameters":
                        # Format as scientific notation for large numbers
                        row += f" & {mean:.1e}"
                    elif metric == "training_time":
                        # Format time in minutes
                        row += f" & {mean/60:.1f}m"
                    else:
                        row += f" & {mean:.{self.config.precision}f}"
                else:
                    row += " & -"
            
            # Efficiency ratios
            for metric in ["final_accuracy_ratio", "trainable_parameters_ratio"]:
                if metric in table_data[method_name]:
                    ratio = table_data[method_name][metric]["mean"]
                    if metric == "trainable_parameters_ratio":
                        # Show as percentage reduction
                        reduction = (1 - ratio) * 100
                        row += f" & {reduction:.1f}\\%"
                    else:
                        row += f" & {ratio:.3f}"
                else:
                    row += " & -"
            
            row += " \\\\"
            latex_lines.append(row)
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def _get_method_name(self, result: ExperimentResult) -> str:
        """Extract method name from experiment result."""
        if not result.config:
            return "unknown"
        
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
    
    def _format_method_name_latex(self, method_name: str) -> str:
        """Format method name for LaTeX display."""
        # Replace underscores with spaces and format for LaTeX
        formatted = method_name.replace('_', ' ')
        
        # Special LaTeX formatting
        formatted = formatted.replace('lora', '\\textsc{LoRA}')
        formatted = formatted.replace('adalora', '\\textsc{AdaLoRA}')
        formatted = formatted.replace('8bit', '8-bit')
        formatted = formatted.replace('4bit', '4-bit')
        formatted = formatted.replace('finetune', 'Fine-tune')
        
        return formatted
    
    def _format_metric_name_latex(self, metric_name: str) -> str:
        """Format metric name for LaTeX display."""
        metric_mappings = {
            'final_accuracy': 'Accuracy',
            'trainable_parameters': 'Trainable Params',
            'model_size_mb': 'Model Size (MB)',
            'peak_memory_gb': 'Peak Memory (GB)',
            'training_time': 'Training Time (s)',
            'samples_per_second': 'Throughput (samples/s)',
            'final_loss': 'Loss'
        }
        
        formatted = metric_mappings.get(metric_name, metric_name.replace('_', ' ').title())
        
        # LaTeX formatting for units
        formatted = formatted.replace('(MB)', '(\\si{\\mega\\byte})')
        formatted = formatted.replace('(GB)', '(\\si{\\giga\\byte})')
        formatted = formatted.replace('(s)', '(\\si{\\second})')
        
        return formatted
    
    def _get_significance_symbol(self, p_value: float) -> str:
        """Convert p-value to significance symbol."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
    
    def generate_all_tables(self, results: List[ExperimentResult]) -> Dict[str, Path]:
        """
        Generate all standard tables for the research.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary mapping table names to file paths
        """
        logger.info(f"Generating all tables from {len(results)} results")
        
        generated_tables = {}
        
        try:
            # Main results table
            main_path = self.generate_main_results_table(results)
            if main_path:
                generated_tables["main_results"] = main_path
            
            # Ablation study tables
            ablation_path = self.generate_ablation_study_table(results, "lora_rank")
            if ablation_path:
                generated_tables["ablation_lora_rank"] = ablation_path
            
            # Resource efficiency table
            efficiency_path = self.generate_resource_efficiency_table(results)
            if efficiency_path:
                generated_tables["resource_efficiency"] = efficiency_path
            
            # Statistical significance table
            if self.stats_analyzer:
                significance_path = self.generate_statistical_significance_table(results)
                if significance_path:
                    generated_tables["statistical_significance"] = significance_path
            
            logger.info(f"Generated {len(generated_tables)} tables")
            
        except Exception as e:
            logger.error(f"Error generating tables: {e}")
        
        return generated_tables