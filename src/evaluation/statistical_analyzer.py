"""
Statistical analysis tools for PEFT Vision Transformer evaluation.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Container for confidence interval results."""
    
    mean: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    margin_of_error: float
    sample_size: int


@dataclass
class SignificanceTest:
    """Container for statistical significance test results."""
    
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    interpretation: str = ""


class StatisticalAnalyzer:
    """
    Statistical analysis tools for PEFT evaluation results.
    
    Provides confidence intervals, significance testing, and comparative analysis
    for publication-quality research results.
    """
    
    def __init__(self, default_confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            default_confidence_level: Default confidence level for tests (0.95 = 95%)
        """
        self.default_confidence_level = default_confidence_level
        logger.info(f"StatisticalAnalyzer initialized with {default_confidence_level*100}% confidence level")
    
    def compute_confidence_interval(
        self,
        values: List[float],
        confidence_level: Optional[float] = None
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for a list of values.
        
        Args:
            values: List of numerical values
            confidence_level: Confidence level (default: class default)
            
        Returns:
            ConfidenceInterval with statistics
            
        Raises:
            ValueError: If values list is empty or invalid
        """
        if not values:
            raise ValueError("Values list cannot be empty")
        
        confidence_level = confidence_level or self.default_confidence_level
        
        try:
            import numpy as np
            from scipy import stats
            
            values_array = np.array(values)
            n = len(values_array)
            mean = np.mean(values_array)
            std_err = stats.sem(values_array)  # Standard error of the mean
            
            # Calculate t-critical value for given confidence level
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            
            # Calculate margin of error and bounds
            margin_of_error = t_critical * std_err
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error
            
            return ConfidenceInterval(
                mean=float(mean),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                confidence_level=confidence_level,
                margin_of_error=float(margin_of_error),
                sample_size=n
            )
            
        except ImportError:
            # Fallback implementation without scipy
            logger.warning("scipy not available, using simplified confidence interval calculation")
            return self._compute_confidence_interval_fallback(values, confidence_level)
        except Exception as e:
            logger.error(f"Failed to compute confidence interval: {str(e)}")
            raise RuntimeError(f"Confidence interval calculation failed: {str(e)}") from e
    
    def _compute_confidence_interval_fallback(
        self, 
        values: List[float], 
        confidence_level: float
    ) -> ConfidenceInterval:
        """Fallback confidence interval calculation without scipy."""
        import math
        
        n = len(values)
        mean = sum(values) / n
        
        # Calculate sample standard deviation
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std_dev = math.sqrt(variance)
        std_err = std_dev / math.sqrt(n)
        
        # Use approximate t-critical values for common confidence levels
        t_critical_values = {
            0.90: 1.645,  # Approximate for large n
            0.95: 1.96,
            0.99: 2.576
        }
        
        t_critical = t_critical_values.get(confidence_level, 1.96)
        
        margin_of_error = t_critical * std_err
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
        return ConfidenceInterval(
            mean=mean,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            margin_of_error=margin_of_error,
            sample_size=n
        )
    
    def paired_t_test(
        self,
        group1: List[float],
        group2: List[float],
        confidence_level: Optional[float] = None
    ) -> SignificanceTest:
        """
        Perform paired t-test to compare two related groups.
        
        Args:
            group1: First group of values
            group2: Second group of values (must be same length as group1)
            confidence_level: Confidence level for test
            
        Returns:
            SignificanceTest with test results
            
        Raises:
            ValueError: If groups have different lengths or are empty
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have the same length for paired t-test")
        
        if not group1 or not group2:
            raise ValueError("Groups cannot be empty")
        
        confidence_level = confidence_level or self.default_confidence_level
        alpha = 1 - confidence_level
        
        try:
            from scipy import stats
            import numpy as np
            
            # Perform paired t-test
            statistic, p_value = stats.ttest_rel(group1, group2)
            
            # Calculate effect size (Cohen's d for paired samples)
            differences = np.array(group1) - np.array(group2)
            effect_size = np.mean(differences) / np.std(differences, ddof=1)
            
            is_significant = p_value < alpha
            
            # Interpretation
            if is_significant:
                interpretation = f"Significant difference (p={p_value:.4f} < {alpha})"
            else:
                interpretation = f"No significant difference (p={p_value:.4f} >= {alpha})"
            
            return SignificanceTest(
                test_name="Paired t-test",
                statistic=float(statistic),
                p_value=float(p_value),
                is_significant=is_significant,
                confidence_level=confidence_level,
                effect_size=float(effect_size),
                interpretation=interpretation
            )
            
        except ImportError:
            logger.warning("scipy not available, using simplified t-test")
            return self._paired_t_test_fallback(group1, group2, confidence_level)
        except Exception as e:
            logger.error(f"Paired t-test failed: {str(e)}")
            raise RuntimeError(f"Paired t-test failed: {str(e)}") from e
    
    def independent_t_test(
        self,
        group1: List[float],
        group2: List[float],
        confidence_level: Optional[float] = None,
        equal_variances: bool = True
    ) -> SignificanceTest:
        """
        Perform independent samples t-test.
        
        Args:
            group1: First group of values
            group2: Second group of values
            confidence_level: Confidence level for test
            equal_variances: Whether to assume equal variances (Welch's t-test if False)
            
        Returns:
            SignificanceTest with test results
        """
        if not group1 or not group2:
            raise ValueError("Groups cannot be empty")
        
        confidence_level = confidence_level or self.default_confidence_level
        alpha = 1 - confidence_level
        
        try:
            from scipy import stats
            import numpy as np
            
            # Perform independent t-test
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_variances)
            
            # Calculate effect size (Cohen's d)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            
            if equal_variances:
                # Pooled standard deviation
                n1, n2 = len(group1), len(group2)
                pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                effect_size = (mean1 - mean2) / pooled_std
            else:
                # Use average of standard deviations
                effect_size = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
            
            is_significant = p_value < alpha
            
            test_name = "Independent t-test" if equal_variances else "Welch's t-test"
            
            if is_significant:
                interpretation = f"Significant difference (p={p_value:.4f} < {alpha})"
            else:
                interpretation = f"No significant difference (p={p_value:.4f} >= {alpha})"
            
            return SignificanceTest(
                test_name=test_name,
                statistic=float(statistic),
                p_value=float(p_value),
                is_significant=is_significant,
                confidence_level=confidence_level,
                effect_size=float(effect_size),
                interpretation=interpretation
            )
            
        except ImportError:
            logger.warning("scipy not available, using simplified t-test")
            return self._independent_t_test_fallback(group1, group2, confidence_level)
        except Exception as e:
            logger.error(f"Independent t-test failed: {str(e)}")
            raise RuntimeError(f"Independent t-test failed: {str(e)}") from e
    
    def _paired_t_test_fallback(
        self, 
        group1: List[float], 
        group2: List[float], 
        confidence_level: float
    ) -> SignificanceTest:
        """Fallback paired t-test without scipy."""
        import math
        
        # Calculate differences
        differences = [x1 - x2 for x1, x2 in zip(group1, group2)]
        n = len(differences)
        
        # Calculate statistics
        mean_diff = sum(differences) / n
        var_diff = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
        std_err = math.sqrt(var_diff / n)
        
        # Calculate t-statistic
        t_statistic = mean_diff / std_err if std_err > 0 else 0
        
        # Approximate p-value (very rough approximation)
        # This is not accurate but provides a basic indication
        p_value = 2 * (1 - abs(t_statistic) / (abs(t_statistic) + math.sqrt(n - 1)))
        p_value = max(0.001, min(0.999, p_value))  # Clamp to reasonable range
        
        alpha = 1 - confidence_level
        is_significant = p_value < alpha
        
        return SignificanceTest(
            test_name="Paired t-test (approximate)",
            statistic=t_statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=confidence_level,
            interpretation=f"Approximate test: {'Significant' if is_significant else 'Not significant'}"
        )
    
    def _independent_t_test_fallback(
        self, 
        group1: List[float], 
        group2: List[float], 
        confidence_level: float
    ) -> SignificanceTest:
        """Fallback independent t-test without scipy."""
        import math
        
        n1, n2 = len(group1), len(group2)
        mean1 = sum(group1) / n1
        mean2 = sum(group2) / n2
        
        # Calculate variances
        var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
        
        # Pooled standard error
        pooled_se = math.sqrt(var1/n1 + var2/n2)
        
        # Calculate t-statistic
        t_statistic = (mean1 - mean2) / pooled_se if pooled_se > 0 else 0
        
        # Approximate degrees of freedom
        df = n1 + n2 - 2
        
        # Very rough p-value approximation
        p_value = 2 * (1 - abs(t_statistic) / (abs(t_statistic) + math.sqrt(df)))
        p_value = max(0.001, min(0.999, p_value))
        
        alpha = 1 - confidence_level
        is_significant = p_value < alpha
        
        return SignificanceTest(
            test_name="Independent t-test (approximate)",
            statistic=t_statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=confidence_level,
            interpretation=f"Approximate test: {'Significant' if is_significant else 'Not significant'}"
        )
    
    def compare_multiple_methods(
        self,
        results_dict: Dict[str, List[float]],
        confidence_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple methods using ANOVA and pairwise comparisons.
        
        Args:
            results_dict: Dictionary mapping method names to lists of results
            confidence_level: Confidence level for tests
            
        Returns:
            Dictionary with ANOVA results and pairwise comparisons
        """
        if len(results_dict) < 2:
            raise ValueError("Need at least 2 methods for comparison")
        
        confidence_level = confidence_level or self.default_confidence_level
        
        try:
            from scipy import stats
            import numpy as np
            
            # Prepare data for ANOVA
            groups = list(results_dict.values())
            
            # Perform one-way ANOVA
            f_statistic, p_value = stats.f_oneway(*groups)
            
            alpha = 1 - confidence_level
            anova_significant = p_value < alpha
            
            # Pairwise comparisons if ANOVA is significant
            pairwise_results = {}
            if anova_significant:
                method_names = list(results_dict.keys())
                for i in range(len(method_names)):
                    for j in range(i + 1, len(method_names)):
                        method1, method2 = method_names[i], method_names[j]
                        comparison_key = f"{method1}_vs_{method2}"
                        
                        pairwise_test = self.independent_t_test(
                            results_dict[method1],
                            results_dict[method2],
                            confidence_level
                        )
                        pairwise_results[comparison_key] = pairwise_test
            
            # Calculate descriptive statistics for each method
            descriptive_stats = {}
            for method_name, values in results_dict.items():
                ci = self.compute_confidence_interval(values, confidence_level)
                descriptive_stats[method_name] = {
                    "mean": ci.mean,
                    "confidence_interval": (ci.lower_bound, ci.upper_bound),
                    "sample_size": ci.sample_size
                }
            
            return {
                "anova": {
                    "f_statistic": float(f_statistic),
                    "p_value": float(p_value),
                    "is_significant": anova_significant,
                    "interpretation": f"{'Significant' if anova_significant else 'No significant'} difference between methods"
                },
                "pairwise_comparisons": pairwise_results,
                "descriptive_statistics": descriptive_stats,
                "num_methods": len(results_dict),
                "confidence_level": confidence_level
            }
            
        except ImportError:
            logger.warning("scipy not available, performing simplified comparisons")
            return self._compare_multiple_methods_fallback(results_dict, confidence_level)
        except Exception as e:
            logger.error(f"Multiple method comparison failed: {str(e)}")
            raise RuntimeError(f"Multiple method comparison failed: {str(e)}") from e
    
    def _compare_multiple_methods_fallback(
        self, 
        results_dict: Dict[str, List[float]], 
        confidence_level: float
    ) -> Dict[str, Any]:
        """Fallback multiple method comparison without scipy."""
        # Just do pairwise t-tests without ANOVA
        pairwise_results = {}
        method_names = list(results_dict.keys())
        
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                comparison_key = f"{method1}_vs_{method2}"
                
                pairwise_test = self.independent_t_test(
                    results_dict[method1],
                    results_dict[method2],
                    confidence_level
                )
                pairwise_results[comparison_key] = pairwise_test
        
        # Calculate descriptive statistics
        descriptive_stats = {}
        for method_name, values in results_dict.items():
            ci = self.compute_confidence_interval(values, confidence_level)
            descriptive_stats[method_name] = {
                "mean": ci.mean,
                "confidence_interval": (ci.lower_bound, ci.upper_bound),
                "sample_size": ci.sample_size
            }
        
        return {
            "anova": {
                "f_statistic": None,
                "p_value": None,
                "is_significant": None,
                "interpretation": "ANOVA not available (scipy required)"
            },
            "pairwise_comparisons": pairwise_results,
            "descriptive_statistics": descriptive_stats,
            "num_methods": len(results_dict),
            "confidence_level": confidence_level
        }
    
    def effect_size_interpretation(self, effect_size: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            effect_size: Cohen's d value
            
        Returns:
            String interpretation of effect size
        """
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Perform statistical power analysis.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size
            alpha: Type I error rate
            
        Returns:
            Dictionary with power analysis results
        """
        try:
            from scipy import stats
            import math
            
            # Calculate power for two-sample t-test
            # This is a simplified calculation
            delta = effect_size * math.sqrt(sample_size / 2)
            t_critical = stats.t.ppf(1 - alpha/2, df=2*sample_size-2)
            
            # Non-central t-distribution for power calculation
            power = 1 - stats.t.cdf(t_critical, df=2*sample_size-2, loc=delta)
            power += stats.t.cdf(-t_critical, df=2*sample_size-2, loc=delta)
            
            return {
                "power": float(power),
                "effect_size": effect_size,
                "sample_size": sample_size,
                "alpha": alpha,
                "interpretation": f"Power = {power:.3f} ({'adequate' if power >= 0.8 else 'inadequate'})"
            }
            
        except ImportError:
            logger.warning("scipy not available for power analysis")
            return {
                "power": None,
                "effect_size": effect_size,
                "sample_size": sample_size,
                "alpha": alpha,
                "interpretation": "Power analysis requires scipy"
            }
        except Exception as e:
            logger.error(f"Power analysis failed: {str(e)}")
            return {
                "power": None,
                "error": str(e)
            }
    
    def generate_summary_report(
        self,
        results_dict: Dict[str, List[float]],
        confidence_level: Optional[float] = None
    ) -> str:
        """
        Generate a comprehensive statistical summary report.
        
        Args:
            results_dict: Dictionary mapping method names to results
            confidence_level: Confidence level for analysis
            
        Returns:
            Formatted string report
        """
        confidence_level = confidence_level or self.default_confidence_level
        
        report_lines = [
            "Statistical Analysis Report",
            "=" * 50,
            f"Confidence Level: {confidence_level*100}%",
            f"Number of Methods: {len(results_dict)}",
            ""
        ]
        
        # Descriptive statistics
        report_lines.append("Descriptive Statistics:")
        report_lines.append("-" * 30)
        
        for method_name, values in results_dict.items():
            ci = self.compute_confidence_interval(values, confidence_level)
            report_lines.extend([
                f"{method_name}:",
                f"  Mean: {ci.mean:.4f}",
                f"  95% CI: [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]",
                f"  Sample Size: {ci.sample_size}",
                ""
            ])
        
        # Multiple comparisons if more than 2 methods
        if len(results_dict) > 1:
            comparison_results = self.compare_multiple_methods(results_dict, confidence_level)
            
            report_lines.extend([
                "Statistical Comparisons:",
                "-" * 30
            ])
            
            if comparison_results["anova"]["p_value"] is not None:
                anova = comparison_results["anova"]
                report_lines.extend([
                    f"ANOVA F-statistic: {anova['f_statistic']:.4f}",
                    f"ANOVA p-value: {anova['p_value']:.4f}",
                    f"Overall significance: {anova['interpretation']}",
                    ""
                ])
            
            # Pairwise comparisons
            if comparison_results["pairwise_comparisons"]:
                report_lines.append("Pairwise Comparisons:")
                for comparison_name, test_result in comparison_results["pairwise_comparisons"].items():
                    report_lines.extend([
                        f"{comparison_name}:",
                        f"  t-statistic: {test_result.statistic:.4f}",
                        f"  p-value: {test_result.p_value:.4f}",
                        f"  Effect size: {test_result.effect_size:.4f} ({self.effect_size_interpretation(test_result.effect_size)})" if test_result.effect_size is not None else "  Effect size: N/A",
                        f"  Result: {test_result.interpretation}",
                        ""
                    ])
        
        return "\n".join(report_lines)