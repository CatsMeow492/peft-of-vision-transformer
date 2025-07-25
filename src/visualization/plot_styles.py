"""
Publication-quality plot styling for PEFT Vision Transformer research.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import seaborn as sns

logger = logging.getLogger(__name__)


class PlotStyleManager:
    """
    Manager for consistent publication-quality plot styling.
    
    Provides standardized colors, fonts, and layout settings for all figures.
    """
    
    # Publication-ready color palettes (colorblind-friendly)
    COLORS = {
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e',    # Orange  
        'success': '#2ca02c',      # Green
        'danger': '#d62728',       # Red
        'warning': '#ff7f0e',      # Orange
        'info': '#17becf',         # Cyan
        'purple': '#9467bd',       # Purple
        'brown': '#8c564b',        # Brown
        'pink': '#e377c2',         # Pink
        'gray': '#7f7f7f',         # Gray
        'olive': '#bcbd22',        # Olive
        'cyan': '#17becf'          # Cyan
    }
    
    # Method-specific color mapping
    METHOD_COLORS = {
        'full_finetune': '#d62728',     # Red - baseline
        'lora_r2': '#1f77b4',           # Blue
        'lora_r4': '#ff7f0e',           # Orange
        'lora_r8': '#2ca02c',           # Green
        'lora_r16': '#9467bd',          # Purple
        'lora_r32': '#8c564b',          # Brown
        'adalora': '#e377c2',           # Pink
        'qa_lora_8bit': '#7f7f7f',      # Gray
        'qa_lora_4bit': '#bcbd22',      # Olive
        'quantized_8bit': '#17becf',    # Cyan
        'quantized_4bit': '#ffbb78'     # Light orange
    }
    
    # Dataset-specific styling
    DATASET_MARKERS = {
        'cifar10': 'o',      # Circle
        'cifar100': 's',     # Square
        'tiny_imagenet': '^' # Triangle
    }
    
    # Model-specific line styles
    MODEL_LINESTYLES = {
        'deit_tiny': '-',     # Solid
        'deit_small': '--',   # Dashed
        'vit_small': '-.'     # Dash-dot
    }
    
    def __init__(self, style: str = "publication"):
        """
        Initialize plot style manager.
        
        Args:
            style: Style preset ('publication', 'presentation', 'paper')
        """
        self.style = style
        self._setup_matplotlib_defaults()
        self._setup_seaborn_style()
        
        logger.info(f"PlotStyleManager initialized with style: {style}")
    
    def _setup_matplotlib_defaults(self):
        """Configure matplotlib defaults for publication quality."""
        # Font settings
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        rcParams['font.size'] = 12
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        rcParams['figure.titlesize'] = 16
        
        # Figure settings
        rcParams['figure.dpi'] = 300
        rcParams['savefig.dpi'] = 300
        rcParams['savefig.format'] = 'pdf'
        rcParams['savefig.bbox'] = 'tight'
        rcParams['savefig.pad_inches'] = 0.1
        
        # Line and marker settings
        rcParams['lines.linewidth'] = 1.5
        rcParams['lines.markersize'] = 6
        rcParams['axes.linewidth'] = 1.2
        rcParams['grid.linewidth'] = 0.8
        rcParams['xtick.major.width'] = 1.2
        rcParams['ytick.major.width'] = 1.2
        
        # Layout settings
        rcParams['figure.autolayout'] = True
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.3
        
        # Color settings
        rcParams['axes.prop_cycle'] = plt.cycler('color', list(self.COLORS.values()))
    
    def _setup_seaborn_style(self):
        """Configure seaborn styling."""
        try:
            if self.style == "publication":
                sns.set_style("whitegrid", {
                    "axes.spines.left": True,
                    "axes.spines.bottom": True,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "grid.color": "#b0b0b0",
                    "grid.alpha": 0.3
                })
            elif self.style == "presentation":
                sns.set_style("white")
            else:
                sns.set_style("whitegrid")
                
            # Set color palette
            sns.set_palette(list(self.COLORS.values()))
            
        except Exception as e:
            logger.warning(f"Failed to setup seaborn style: {e}")
    
    def get_method_color(self, method_name: str) -> str:
        """
        Get standardized color for a method.
        
        Args:
            method_name: Name of the method
            
        Returns:
            Hex color code
        """
        # Normalize method name
        method_key = method_name.lower().replace('-', '_').replace(' ', '_')
        
        return self.METHOD_COLORS.get(method_key, self.COLORS['primary'])
    
    def get_dataset_marker(self, dataset_name: str) -> str:
        """
        Get standardized marker for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Matplotlib marker style
        """
        dataset_key = dataset_name.lower().replace('-', '_')
        return self.DATASET_MARKERS.get(dataset_key, 'o')
    
    def get_model_linestyle(self, model_name: str) -> str:
        """
        Get standardized line style for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Matplotlib line style
        """
        model_key = model_name.lower().replace('-', '_')
        return self.MODEL_LINESTYLES.get(model_key, '-')
    
    def setup_figure(
        self,
        figsize: Tuple[float, float] = (8, 6),
        subplot_layout: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, Any]:
        """
        Create a properly styled figure.
        
        Args:
            figsize: Figure size in inches (width, height)
            subplot_layout: Subplot layout (rows, cols) if needed
            
        Returns:
            Tuple of (figure, axes)
        """
        if subplot_layout:
            fig, axes = plt.subplots(*subplot_layout, figsize=figsize)
        else:
            fig, axes = plt.subplots(figsize=figsize)
        
        # Apply consistent styling
        if hasattr(axes, '__iter__'):
            for ax in axes.flat:
                self._style_axes(ax)
        else:
            self._style_axes(axes)
        
        return fig, axes
    
    def _style_axes(self, ax: plt.Axes):
        """Apply consistent styling to axes."""
        # Grid styling
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Tick styling
        ax.tick_params(width=1.2, length=4)
    
    def add_confidence_bands(
        self,
        ax: plt.Axes,
        x_data: List[float],
        y_mean: List[float],
        y_lower: List[float],
        y_upper: List[float],
        color: Optional[str] = None,
        alpha: float = 0.2,
        label: Optional[str] = None
    ):
        """
        Add confidence bands to a plot.
        
        Args:
            ax: Matplotlib axes
            x_data: X-axis data
            y_mean: Mean values
            y_lower: Lower confidence bounds
            y_upper: Upper confidence bounds
            color: Fill color (auto-selected if None)
            alpha: Transparency level
            label: Legend label
        """
        if color is None:
            color = self.COLORS['primary']
        
        # Plot mean line
        line = ax.plot(x_data, y_mean, color=color, linewidth=2, label=label)[0]
        
        # Add confidence band
        ax.fill_between(
            x_data, y_lower, y_upper,
            color=color, alpha=alpha,
            linewidth=0
        )
        
        return line
    
    def add_significance_indicators(
        self,
        ax: plt.Axes,
        x_positions: List[float],
        y_position: float,
        significance_levels: List[str],
        height_offset: float = 0.02
    ):
        """
        Add significance indicators to a plot.
        
        Args:
            ax: Matplotlib axes
            x_positions: X positions for indicators
            y_position: Y position for indicators
            significance_levels: List of significance indicators ('*', '**', '***', 'ns')
            height_offset: Vertical offset for indicators
        """
        for x_pos, sig_level in zip(x_positions, significance_levels):
            if sig_level != 'ns':  # Don't show non-significant
                ax.text(
                    x_pos, y_position + height_offset,
                    sig_level,
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold'
                )
    
    def create_legend(
        self,
        ax: plt.Axes,
        location: str = 'best',
        frameon: bool = True,
        fancybox: bool = True,
        shadow: bool = False,
        ncol: int = 1
    ) -> plt.Legend:
        """
        Create a styled legend.
        
        Args:
            ax: Matplotlib axes
            location: Legend location
            frameon: Whether to draw frame
            fancybox: Whether to use rounded corners
            shadow: Whether to add shadow
            ncol: Number of columns
            
        Returns:
            Legend object
        """
        legend = ax.legend(
            loc=location,
            frameon=frameon,
            fancybox=fancybox,
            shadow=shadow,
            ncol=ncol,
            fontsize=10
        )
        
        if frameon:
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(0.8)
        
        return legend
    
    def format_axes_labels(
        self,
        ax: plt.Axes,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None
    ):
        """
        Format axes labels with consistent styling.
        
        Args:
            ax: Matplotlib axes
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title (optional)
        """
        ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    def save_figure(
        self,
        fig: plt.Figure,
        filepath: str,
        formats: List[str] = ['pdf', 'png'],
        dpi: int = 300,
        bbox_inches: str = 'tight',
        pad_inches: float = 0.1
    ):
        """
        Save figure in multiple formats with consistent settings.
        
        Args:
            fig: Matplotlib figure
            filepath: Base filepath (without extension)
            formats: List of formats to save
            dpi: Resolution for raster formats
            bbox_inches: Bounding box setting
            pad_inches: Padding around figure
        """
        for fmt in formats:
            output_path = f"{filepath}.{fmt}"
            
            try:
                fig.savefig(
                    output_path,
                    format=fmt,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    facecolor='white',
                    edgecolor='none'
                )
                logger.info(f"Figure saved: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save figure as {fmt}: {e}")
    
    def get_color_palette(self, n_colors: int) -> List[str]:
        """
        Get a color palette with specified number of colors.
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of hex color codes
        """
        if n_colors <= len(self.COLORS):
            return list(self.COLORS.values())[:n_colors]
        else:
            # Generate additional colors using seaborn
            try:
                palette = sns.color_palette("husl", n_colors)
                return [mpl.colors.rgb2hex(color) for color in palette]
            except:
                # Fallback: cycle through existing colors
                colors = list(self.COLORS.values())
                return [colors[i % len(colors)] for i in range(n_colors)]
    
    def apply_style_context(self):
        """
        Return a context manager for temporary style application.
        
        Returns:
            Context manager for matplotlib style
        """
        return plt.style.context({
            'font.family': 'serif',
            'font.size': 12,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'figure.dpi': 300
        })
    
    @staticmethod
    def get_significance_symbol(p_value: float) -> str:
        """
        Convert p-value to significance symbol.
        
        Args:
            p_value: Statistical p-value
            
        Returns:
            Significance symbol
        """
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
    
    @staticmethod
    def format_number(value: float, precision: int = 3) -> str:
        """
        Format number for publication display.
        
        Args:
            value: Number to format
            precision: Decimal precision
            
        Returns:
            Formatted string
        """
        if abs(value) >= 1000:
            return f"{value:.0f}"
        elif abs(value) >= 1:
            return f"{value:.{precision}f}"
        else:
            return f"{value:.{precision}f}"