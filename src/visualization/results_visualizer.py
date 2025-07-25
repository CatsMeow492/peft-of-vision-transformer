"""
Comprehensive results visualization manager for PEFT Vision Transformer research.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .figure_generator import FigureGenerator, FigureConfig
from .table_generator import LaTeXTableGenerator, TableConfig

# Handle optional imports
try:
    from ..experiments.results import ExperimentResult
    EXPERIMENTS_AVAILABLE = True
except ImportError:
    class ExperimentResult:
        pass
    EXPERIMENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for comprehensive visualization generation."""
    
    output_dir: Path
    figure_formats: List[str] = None
    table_precision: int = 3
    confidence_level: float = 0.95
    dpi: int = 300
    style: str = "publication"
    
    def __post_init__(self):
        if self.figure_formats is None:
            self.figure_formats = ['pdf', 'png']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)


class ResultsVisualizer:
    """
    Comprehensive visualization manager for PEFT research results.
    
    Coordinates figure generation and table creation to produce
    a complete set of publication-ready visualizations.
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize results visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        
        # Initialize figure generator
        figure_config = FigureConfig(
            output_dir=config.output_dir / "figures",
            formats=config.figure_formats,
            dpi=config.dpi,
            style=config.style
        )
        self.figure_generator = FigureGenerator(figure_config)
        
        # Initialize table generator
        table_config = TableConfig(
            output_dir=config.output_dir / "tables",
            precision=config.table_precision,
            confidence_level=config.confidence_level
        )
        self.table_generator = LaTeXTableGenerator(table_config)
        
        logger.info(f"ResultsVisualizer initialized with output_dir: {config.output_dir}")
    
    def generate_complete_analysis(
        self,
        results: List[ExperimentResult],
        include_figures: bool = True,
        include_tables: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete visualization analysis including all figures and tables.
        
        Args:
            results: List of experiment results
            include_figures: Whether to generate figures
            include_tables: Whether to generate tables
            
        Returns:
            Dictionary with paths to generated visualizations
        """
        logger.info(f"Generating complete analysis from {len(results)} results")
        
        analysis_results = {
            "figures": {},
            "tables": {},
            "summary": {}
        }
        
        if not results:
            logger.warning("No results provided for visualization")
            return analysis_results
        
        try:
            # Generate figures
            if include_figures:
                logger.info("Generating figures...")
                figures = self.figure_generator.generate_all_figures(results)
                analysis_results["figures"] = figures
                logger.info(f"Generated {len(figures)} figures")
            
            # Generate tables
            if include_tables:
                logger.info("Generating tables...")
                tables = self.table_generator.generate_all_tables(results)
                analysis_results["tables"] = tables
                logger.info(f"Generated {len(tables)} tables")
            
            # Generate summary statistics
            analysis_results["summary"] = self._generate_analysis_summary(results)
            
            # Create index file
            self._create_visualization_index(analysis_results)
            
            logger.info("Complete analysis generation finished")
            
        except Exception as e:
            logger.error(f"Error during complete analysis generation: {e}")
            raise
        
        return analysis_results
    
    def generate_paper_figures(
        self,
        results: List[ExperimentResult],
        figure_specs: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Generate specific figures for paper publication.
        
        Args:
            results: List of experiment results
            figure_specs: Specifications for each figure
            
        Returns:
            Dictionary mapping figure names to paths
        """
        if figure_specs is None:
            figure_specs = {
                "main_results": {
                    "type": "pareto_frontier",
                    "efficiency_metric": "trainable_parameters",
                    "accuracy_metric": "final_accuracy"
                },
                "convergence": {
                    "type": "convergence_curves",
                    "metric": "accuracy"
                },
                "ablation": {
                    "type": "ablation_study",
                    "factor": "lora_rank",
                    "metric": "final_accuracy"
                },
                "resource_analysis": {
                    "type": "resource_usage"
                }
            }
        
        logger.info("Generating paper-specific figures")
        
        paper_figures = {}
        
        for fig_name, spec in figure_specs.items():
            try:
                if spec["type"] == "pareto_frontier":
                    path = self.figure_generator.generate_pareto_frontier_plot(
                        results,
                        efficiency_metric=spec.get("efficiency_metric", "trainable_parameters"),
                        accuracy_metric=spec.get("accuracy_metric", "final_accuracy"),
                        filename=f"paper_{fig_name}"
                    )
                elif spec["type"] == "convergence_curves":
                    # Extract training histories
                    training_histories = self._extract_training_histories(results)
                    if training_histories:
                        path = self.figure_generator.generate_convergence_curves(
                            training_histories,
                            metric=spec.get("metric", "accuracy"),
                            filename=f"paper_{fig_name}"
                        )
                    else:
                        path = None
                elif spec["type"] == "ablation_study":
                    path = self.figure_generator.generate_ablation_study_chart(
                        results,
                        ablation_factor=spec.get("factor", "lora_rank"),
                        metric=spec.get("metric", "final_accuracy"),
                        filename=f"paper_{fig_name}"
                    )
                elif spec["type"] == "resource_usage":
                    path = self.figure_generator.generate_resource_usage_visualization(
                        results,
                        filename=f"paper_{fig_name}"
                    )
                else:
                    logger.warning(f"Unknown figure type: {spec['type']}")
                    path = None
                
                if path:
                    paper_figures[fig_name] = path
                    
            except Exception as e:
                logger.error(f"Failed to generate figure {fig_name}: {e}")
        
        logger.info(f"Generated {len(paper_figures)} paper figures")
        return paper_figures
    
    def generate_paper_tables(
        self,
        results: List[ExperimentResult],
        table_specs: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Generate specific tables for paper publication.
        
        Args:
            results: List of experiment results
            table_specs: Specifications for each table
            
        Returns:
            Dictionary mapping table names to paths
        """
        if table_specs is None:
            table_specs = {
                "main_results": {
                    "type": "main_results",
                    "metrics": ["final_accuracy", "trainable_parameters", "peak_memory_gb"]
                },
                "ablation_rank": {
                    "type": "ablation_study",
                    "factor": "lora_rank"
                },
                "efficiency": {
                    "type": "resource_efficiency"
                },
                "significance": {
                    "type": "statistical_significance",
                    "metric": "final_accuracy"
                }
            }
        
        logger.info("Generating paper-specific tables")
        
        paper_tables = {}
        
        for table_name, spec in table_specs.items():
            try:
                if spec["type"] == "main_results":
                    path = self.table_generator.generate_main_results_table(
                        results,
                        metrics=spec.get("metrics"),
                        filename=f"paper_{table_name}"
                    )
                elif spec["type"] == "ablation_study":
                    path = self.table_generator.generate_ablation_study_table(
                        results,
                        ablation_factor=spec.get("factor", "lora_rank"),
                        filename=f"paper_{table_name}"
                    )
                elif spec["type"] == "resource_efficiency":
                    path = self.table_generator.generate_resource_efficiency_table(
                        results,
                        filename=f"paper_{table_name}"
                    )
                elif spec["type"] == "statistical_significance":
                    path = self.table_generator.generate_statistical_significance_table(
                        results,
                        metric=spec.get("metric", "final_accuracy"),
                        filename=f"paper_{table_name}"
                    )
                else:
                    logger.warning(f"Unknown table type: {spec['type']}")
                    path = None
                
                if path:
                    paper_tables[table_name] = path
                    
            except Exception as e:
                logger.error(f"Failed to generate table {table_name}: {e}")
        
        logger.info(f"Generated {len(paper_tables)} paper tables")
        return paper_tables
    
    def generate_layer_importance_analysis(
        self,
        importance_data: Dict[str, Dict[str, float]],
        filename: str = "layer_importance_analysis"
    ) -> Optional[Path]:
        """
        Generate comprehensive layer importance analysis for AdaLoRA.
        
        Args:
            importance_data: Nested dict {experiment_id: {layer_name: importance_score}}
            filename: Output filename
            
        Returns:
            Path to generated heatmap
        """
        logger.info("Generating layer importance analysis")
        
        if not importance_data:
            logger.warning("No importance data provided")
            return None
        
        return self.figure_generator.generate_layer_importance_heatmap(
            importance_data, filename
        )
    
    def _extract_training_histories(
        self, 
        results: List[ExperimentResult]
    ) -> Dict[str, List[Dict[str, float]]]:
        """Extract training histories grouped by method."""
        training_histories = {}
        
        for result in results:
            if result.training_history:
                method_name = self.figure_generator._get_method_name(result)
                if method_name not in training_histories:
                    training_histories[method_name] = []
                training_histories[method_name].append(result.training_history)
        
        return training_histories
    
    def _generate_analysis_summary(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.is_successful]
        
        summary = {
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "methods_analyzed": len(set(
                self.figure_generator._get_method_name(r) for r in successful_results
            )),
            "datasets_used": len(set(
                r.config.dataset.name for r in successful_results 
                if r.config and hasattr(r.config, 'dataset')
            )),
            "models_used": len(set(
                r.config.model.name for r in successful_results 
                if r.config and hasattr(r.config, 'model')
            ))
        }
        
        # Performance statistics
        if successful_results:
            accuracies = [r.metrics.get("final_accuracy", 0) for r in successful_results]
            training_times = [r.duration_seconds for r in successful_results]
            memory_usage = [r.peak_memory_gb for r in successful_results]
            
            summary["performance"] = {
                "best_accuracy": max(accuracies) if accuracies else 0,
                "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                "avg_training_time": sum(training_times) / len(training_times) if training_times else 0,
                "avg_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0
            }
        
        return summary
    
    def _create_visualization_index(self, analysis_results: Dict[str, Any]):
        """Create an index file listing all generated visualizations."""
        index_path = self.config.output_dir / "visualization_index.md"
        
        with open(index_path, 'w') as f:
            f.write("# PEFT Vision Transformer - Visualization Index\n\n")
            
            # Summary
            if "summary" in analysis_results:
                summary = analysis_results["summary"]
                f.write("## Analysis Summary\n\n")
                f.write(f"- Total Experiments: {summary.get('total_experiments', 0)}\n")
                f.write(f"- Successful Experiments: {summary.get('successful_experiments', 0)}\n")
                f.write(f"- Success Rate: {summary.get('success_rate', 0):.1%}\n")
                f.write(f"- Methods Analyzed: {summary.get('methods_analyzed', 0)}\n")
                f.write(f"- Datasets Used: {summary.get('datasets_used', 0)}\n")
                f.write(f"- Models Used: {summary.get('models_used', 0)}\n\n")
                
                if "performance" in summary:
                    perf = summary["performance"]
                    f.write("### Performance Highlights\n\n")
                    f.write(f"- Best Accuracy: {perf.get('best_accuracy', 0):.3f}\n")
                    f.write(f"- Average Accuracy: {perf.get('avg_accuracy', 0):.3f}\n")
                    f.write(f"- Average Training Time: {perf.get('avg_training_time', 0):.1f}s\n")
                    f.write(f"- Average Memory Usage: {perf.get('avg_memory_usage', 0):.1f}GB\n\n")
            
            # Figures
            if "figures" in analysis_results and analysis_results["figures"]:
                f.write("## Generated Figures\n\n")
                for fig_name, fig_path in analysis_results["figures"].items():
                    f.write(f"- **{fig_name.replace('_', ' ').title()}**: `{fig_path.name}`\n")
                f.write("\n")
            
            # Tables
            if "tables" in analysis_results and analysis_results["tables"]:
                f.write("## Generated Tables\n\n")
                for table_name, table_path in analysis_results["tables"].items():
                    f.write(f"- **{table_name.replace('_', ' ').title()}**: `{table_path.name}`\n")
                f.write("\n")
            
            f.write("---\n")
            f.write(f"Generated by PEFT Vision Transformer Research Framework\n")
        
        logger.info(f"Visualization index created: {index_path}")
    
    def export_for_paper(
        self,
        results: List[ExperimentResult],
        paper_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Export publication-ready figures and tables for paper submission.
        
        Args:
            results: List of experiment results
            paper_dir: Directory for paper exports (default: output_dir/paper)
            
        Returns:
            Dictionary with exported file paths
        """
        if paper_dir is None:
            paper_dir = self.config.output_dir / "paper"
        
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting paper materials to {paper_dir}")
        
        exported_files = {}
        
        try:
            # Generate paper figures
            paper_figures = self.generate_paper_figures(results)
            
            # Copy figures to paper directory
            for fig_name, fig_path in paper_figures.items():
                # Copy PDF version for LaTeX
                pdf_source = fig_path.with_suffix('.pdf')
                if pdf_source.exists():
                    pdf_dest = paper_dir / f"fig_{fig_name}.pdf"
                    pdf_dest.write_bytes(pdf_source.read_bytes())
                    exported_files[f"figure_{fig_name}"] = pdf_dest
            
            # Generate paper tables
            paper_tables = self.generate_paper_tables(results)
            
            # Copy tables to paper directory
            for table_name, table_path in paper_tables.items():
                table_dest = paper_dir / f"table_{table_name}.tex"
                table_dest.write_text(table_path.read_text())
                exported_files[f"table_{table_name}"] = table_dest
            
            # Create paper README
            readme_path = paper_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write("# Paper Materials\n\n")
                f.write("This directory contains publication-ready figures and tables.\n\n")
                f.write("## Figures\n\n")
                for fig_name in paper_figures.keys():
                    f.write(f"- `fig_{fig_name}.pdf`: {fig_name.replace('_', ' ').title()}\n")
                f.write("\n## Tables\n\n")
                for table_name in paper_tables.keys():
                    f.write(f"- `table_{table_name}.tex`: {table_name.replace('_', ' ').title()}\n")
                f.write("\n## Usage\n\n")
                f.write("Include figures in LaTeX with:\n")
                f.write("```latex\n\\includegraphics{fig_main_results.pdf}\n```\n\n")
                f.write("Include tables in LaTeX with:\n")
                f.write("```latex\n\\input{table_main_results.tex}\n```\n")
            
            exported_files["readme"] = readme_path
            
            logger.info(f"Exported {len(exported_files)} paper materials")
            
        except Exception as e:
            logger.error(f"Error exporting paper materials: {e}")
            raise
        
        return exported_files