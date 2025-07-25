"""
Tests for visualization components.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from src.visualization import (
    FigureGenerator, PlotStyleManager, ParetoAnalyzer, 
    LaTeXTableGenerator, ResultsVisualizer
)
from src.visualization.figure_generator import FigureConfig
from src.visualization.table_generator import TableConfig
from src.visualization.results_visualizer import VisualizationConfig
from src.visualization.pareto_analysis import ParetoPoint


class TestPlotStyleManager:
    """Test plot style manager."""
    
    def test_initialization(self):
        """Test style manager initialization."""
        style_manager = PlotStyleManager()
        assert style_manager.style == "publication"
        assert len(style_manager.COLORS) > 0
        assert len(style_manager.METHOD_COLORS) > 0
    
    def test_method_color_mapping(self):
        """Test method color retrieval."""
        style_manager = PlotStyleManager()
        
        # Test known method
        color = style_manager.get_method_color("lora_r4")
        assert color.startswith("#")
        
        # Test unknown method (should return default)
        color = style_manager.get_method_color("unknown_method")
        assert color == style_manager.COLORS['primary']
    
    def test_dataset_marker_mapping(self):
        """Test dataset marker retrieval."""
        style_manager = PlotStyleManager()
        
        marker = style_manager.get_dataset_marker("cifar10")
        assert marker in ['o', 's', '^']
        
        # Test unknown dataset (should return default)
        marker = style_manager.get_dataset_marker("unknown_dataset")
        assert marker == 'o'
    
    def test_significance_symbol(self):
        """Test significance symbol conversion."""
        assert PlotStyleManager.get_significance_symbol(0.0001) == '***'
        assert PlotStyleManager.get_significance_symbol(0.005) == '**'
        assert PlotStyleManager.get_significance_symbol(0.03) == '*'
        assert PlotStyleManager.get_significance_symbol(0.1) == 'ns'


class TestParetoAnalyzer:
    """Test Pareto frontier analysis."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ParetoAnalyzer()
        assert analyzer.maximize_accuracy is True
        assert analyzer.minimize_efficiency is True
    
    def test_pareto_frontier_computation(self):
        """Test Pareto frontier computation."""
        analyzer = ParetoAnalyzer()
        
        # Create test points
        points = [
            ParetoPoint("method1", 0.9, 100, {}),  # High accuracy, high efficiency cost
            ParetoPoint("method2", 0.8, 50, {}),   # Medium accuracy, medium efficiency
            ParetoPoint("method3", 0.7, 25, {}),   # Low accuracy, low efficiency cost
            ParetoPoint("method4", 0.85, 75, {}),  # Dominated point
        ]
        
        frontier = analyzer.compute_pareto_frontier(points)
        
        # Should have 3 points on frontier (method4 is dominated)
        assert len(frontier) == 3
        frontier_methods = {p.method_name for p in frontier}
        assert "method4" not in frontier_methods
    
    def test_empty_points_list(self):
        """Test handling of empty points list."""
        analyzer = ParetoAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.compute_pareto_frontier([])
    
    def test_hypervolume_computation(self):
        """Test hypervolume computation."""
        analyzer = ParetoAnalyzer()
        
        points = [
            ParetoPoint("method1", 0.9, 100, {}),
            ParetoPoint("method2", 0.8, 50, {}),
            ParetoPoint("method3", 0.7, 25, {}),
        ]
        
        hypervolume = analyzer.compute_hypervolume(points)
        assert hypervolume > 0
    
    def test_knee_point_detection(self):
        """Test knee point detection."""
        analyzer = ParetoAnalyzer()
        
        points = [
            ParetoPoint("method1", 0.95, 1000, {}),
            ParetoPoint("method2", 0.90, 500, {}),   # Likely knee point
            ParetoPoint("method3", 0.85, 400, {}),
            ParetoPoint("method4", 0.70, 100, {}),
        ]
        
        knee_point = analyzer.find_knee_point(points)
        assert knee_point is not None
        # The knee point should be one of the Pareto optimal points
        frontier = analyzer.compute_pareto_frontier(points)
        assert knee_point in frontier


class TestFigureGenerator:
    """Test figure generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = FigureConfig(
            output_dir=self.temp_dir,
            formats=['png'],  # Use PNG for faster testing
            dpi=100  # Lower DPI for faster testing
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test figure generator initialization."""
        generator = FigureGenerator(self.config)
        assert generator.config == self.config
        assert generator.style_manager is not None
        assert generator.pareto_analyzer is not None
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_pareto_frontier_plot_generation(self, mock_close, mock_savefig):
        """Test Pareto frontier plot generation."""
        generator = FigureGenerator(self.config)
        
        # Create mock results
        mock_results = []
        for i in range(3):
            result = Mock()
            result.is_successful = True
            result.metrics = {
                "final_accuracy": 0.8 + i * 0.05,
                "trainable_parameters": 1000 - i * 200
            }
            result.config = Mock()
            result.config.peft = Mock()
            result.config.peft.method = "lora"
            result.config.peft.rank = 4 + i * 2
            result.config.quantization = None
            mock_results.append(result)
        
        # Generate plot
        output_path = generator.generate_pareto_frontier_plot(mock_results)
        
        # Verify plot was generated
        assert mock_savefig.called
        assert mock_close.called
        assert output_path is not None
    
    def test_empty_results_handling(self):
        """Test handling of empty results."""
        generator = FigureGenerator(self.config)
        
        # Should handle empty results gracefully
        output_path = generator.generate_pareto_frontier_plot([])
        assert output_path is None


class TestLaTeXTableGenerator:
    """Test LaTeX table generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = TableConfig(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test table generator initialization."""
        generator = LaTeXTableGenerator(self.config)
        assert generator.config == self.config
    
    def test_main_results_table_generation(self):
        """Test main results table generation."""
        generator = LaTeXTableGenerator(self.config)
        
        # Create mock results
        mock_results = []
        for i in range(2):
            result = Mock()
            result.is_successful = True
            result.metrics = {
                "final_accuracy": 0.8 + i * 0.05,
                "trainable_parameters": 1000 - i * 200,
                "model_size_mb": 100 + i * 10
            }
            result.config = Mock()
            result.config.peft = Mock()
            result.config.peft.method = "lora"
            result.config.peft.rank = 4
            result.config.quantization = None
            mock_results.append(result)
        
        # Generate table
        output_path = generator.generate_main_results_table(mock_results)
        
        # Verify table was generated
        assert output_path is not None
        assert output_path.exists()
        
        # Check content
        content = output_path.read_text()
        assert "\\begin{table}" in content
        assert "\\end{table}" in content
        assert "final_accuracy" in content or "Accuracy" in content
    
    def test_method_name_formatting(self):
        """Test method name formatting for LaTeX."""
        generator = LaTeXTableGenerator(self.config)
        
        formatted = generator._format_method_name_latex("lora_r4")
        assert "LoRA" in formatted or "lora" in formatted
        
        formatted = generator._format_method_name_latex("adalora")
        assert "AdaLoRA" in formatted or "adalora" in formatted


class TestResultsVisualizer:
    """Test comprehensive results visualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = VisualizationConfig(
            output_dir=self.temp_dir,
            figure_formats=['png'],
            dpi=100
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test visualizer initialization."""
        visualizer = ResultsVisualizer(self.config)
        assert visualizer.config == self.config
        assert visualizer.figure_generator is not None
        assert visualizer.table_generator is not None
    
    def test_directory_creation(self):
        """Test that required directories are created."""
        ResultsVisualizer(self.config)
        
        assert (self.temp_dir / "figures").exists()
        assert (self.temp_dir / "tables").exists()
    
    @patch('src.visualization.figure_generator.FigureGenerator.generate_all_figures')
    @patch('src.visualization.table_generator.LaTeXTableGenerator.generate_all_tables')
    def test_complete_analysis_generation(self, mock_tables, mock_figures):
        """Test complete analysis generation."""
        # Mock return values
        mock_figures.return_value = {"pareto": Path("test.png")}
        mock_tables.return_value = {"main": Path("test.tex")}
        
        visualizer = ResultsVisualizer(self.config)
        
        # Create mock results
        mock_results = [Mock()]
        mock_results[0].is_successful = True
        mock_results[0].metrics = {"final_accuracy": 0.85}
        mock_results[0].duration_seconds = 100
        mock_results[0].peak_memory_gb = 2.0
        mock_results[0].config = Mock()
        mock_results[0].config.dataset = Mock()
        mock_results[0].config.dataset.name = "cifar10"
        mock_results[0].config.model = Mock()
        mock_results[0].config.model.name = "deit_tiny"
        
        # Generate analysis
        results = visualizer.generate_complete_analysis(mock_results)
        
        # Verify structure
        assert "figures" in results
        assert "tables" in results
        assert "summary" in results
        
        # Verify methods were called
        mock_figures.assert_called_once()
        mock_tables.assert_called_once()
    
    def test_analysis_summary_generation(self):
        """Test analysis summary generation."""
        visualizer = ResultsVisualizer(self.config)
        
        # Create mock results
        mock_results = []
        for i in range(3):
            result = Mock()
            result.is_successful = i < 2  # 2 successful, 1 failed
            result.metrics = {"final_accuracy": 0.8 + i * 0.05}
            result.duration_seconds = 100 + i * 50
            result.peak_memory_gb = 2.0 + i * 0.5
            result.config = Mock()
            result.config.dataset = Mock()
            result.config.dataset.name = f"dataset_{i}"
            result.config.model = Mock()
            result.config.model.name = f"model_{i}"
            mock_results.append(result)
        
        summary = visualizer._generate_analysis_summary(mock_results)
        
        assert summary["total_experiments"] == 3
        assert summary["successful_experiments"] == 2
        assert summary["success_rate"] == 2/3
        assert "performance" in summary


if __name__ == "__main__":
    pytest.main([__file__])