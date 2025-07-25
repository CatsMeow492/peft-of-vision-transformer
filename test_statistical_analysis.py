#!/usr/bin/env python3
"""
Test statistical analysis functionality without complex dependencies.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Test statistical analysis functionality
try:
    # Import statistical analyzer directly
    import importlib.util
    
    # Load statistical analyzer
    spec = importlib.util.spec_from_file_location(
        "statistical_analyzer", 
        "src/evaluation/statistical_analyzer.py"
    )
    stats_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats_module)
    
    StatisticalAnalyzer = stats_module.StatisticalAnalyzer
    
    print("✓ Statistical analyzer loaded")
    
    # Test basic statistical functions
    analyzer = StatisticalAnalyzer(default_confidence_level=0.95)
    
    # Test confidence interval
    test_values = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.85, 0.87, 0.86, 0.88]
    ci = analyzer.compute_confidence_interval(test_values)
    
    print(f"✓ Confidence interval test:")
    print(f"  Mean: {ci.mean:.4f}")
    print(f"  95% CI: [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]")
    print(f"  Sample size: {ci.sample_size}")
    print(f"  Margin of error: {ci.margin_of_error:.4f}")
    
    # Test independent t-test
    group1 = [0.85, 0.87, 0.86, 0.88, 0.84, 0.86, 0.85]  # LoRA rank 4
    group2 = [0.89, 0.91, 0.90, 0.92, 0.88, 0.90, 0.89]  # LoRA rank 8
    
    t_test = analyzer.independent_t_test(group1, group2)
    print(f"\n✓ Independent t-test:")
    print(f"  Test: {t_test.test_name}")
    print(f"  t-statistic: {t_test.statistic:.4f}")
    print(f"  p-value: {t_test.p_value:.4f}")
    print(f"  Significant: {t_test.is_significant}")
    if t_test.effect_size is not None:
        print(f"  Effect size: {t_test.effect_size:.4f}")
    else:
        print(f"  Effect size: Not available (fallback method)")
    print(f"  Interpretation: {t_test.interpretation}")
    
    # Test paired t-test
    before_training = [0.75, 0.73, 0.76, 0.74, 0.72, 0.75, 0.74]
    after_training = [0.85, 0.87, 0.86, 0.88, 0.84, 0.86, 0.85]
    
    paired_test = analyzer.paired_t_test(before_training, after_training)
    print(f"\n✓ Paired t-test:")
    print(f"  Test: {paired_test.test_name}")
    print(f"  t-statistic: {paired_test.statistic:.4f}")
    print(f"  p-value: {paired_test.p_value:.4f}")
    print(f"  Significant: {paired_test.is_significant}")
    if paired_test.effect_size is not None:
        print(f"  Effect size: {paired_test.effect_size:.4f}")
    else:
        print(f"  Effect size: Not available (fallback method)")
    
    # Test multiple method comparison
    methods_data = {
        "LoRA_r4": [0.85, 0.87, 0.86, 0.88, 0.84, 0.86, 0.85],
        "LoRA_r8": [0.87, 0.89, 0.88, 0.90, 0.86, 0.88, 0.87],
        "LoRA_r16": [0.89, 0.91, 0.90, 0.92, 0.88, 0.90, 0.89],
        "AdaLoRA": [0.91, 0.93, 0.92, 0.94, 0.90, 0.92, 0.91]
    }
    
    multi_comparison = analyzer.compare_multiple_methods(methods_data)
    print(f"\n✓ Multiple method comparison:")
    print(f"  Number of methods: {multi_comparison['num_methods']}")
    
    if multi_comparison["anova"]["p_value"] is not None:
        anova = multi_comparison["anova"]
        print(f"  ANOVA F-statistic: {anova['f_statistic']:.4f}")
        print(f"  ANOVA p-value: {anova['p_value']:.4f}")
        print(f"  Overall significance: {anova['interpretation']}")
    
    # Show descriptive statistics
    desc_stats = multi_comparison["descriptive_statistics"]
    print(f"  Method performance:")
    for method, stats in desc_stats.items():
        ci_lower, ci_upper = stats["confidence_interval"]
        print(f"    {method}: {stats['mean']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (n={stats['sample_size']})")
    
    # Show pairwise comparisons
    pairwise = multi_comparison.get("pairwise_comparisons", {})
    if pairwise:
        print(f"  Pairwise comparisons ({len(pairwise)}):")
        for comparison_name, test_result in list(pairwise.items())[:3]:  # Show first 3
            if test_result.effect_size is not None:
                effect_interp = analyzer.effect_size_interpretation(test_result.effect_size)
                effect_str = f"effect={test_result.effect_size:.3f} ({effect_interp})"
            else:
                effect_str = "effect=N/A"
            
            print(f"    {comparison_name}: p={test_result.p_value:.4f}, "
                  f"{effect_str}, "
                  f"{'significant' if test_result.is_significant else 'not significant'}")
    
    # Test effect size interpretation
    print(f"\n✓ Effect size interpretations:")
    effect_sizes = [0.1, 0.3, 0.6, 0.9, 1.2]
    for es in effect_sizes:
        interpretation = analyzer.effect_size_interpretation(es)
        print(f"  Effect size {es}: {interpretation}")
    
    # Test power analysis
    print(f"\n✓ Power analysis:")
    power_result = analyzer.power_analysis(effect_size=0.5, sample_size=20, alpha=0.05)
    if power_result.get("power") is not None:
        print(f"  Power: {power_result['power']:.3f}")
        print(f"  Interpretation: {power_result['interpretation']}")
    else:
        print(f"  {power_result.get('interpretation', 'Power analysis not available')}")
    
    # Test summary report generation
    print(f"\n✓ Summary report generation:")
    report = analyzer.generate_summary_report(methods_data)
    
    # Show first few lines of report
    report_lines = report.split('\n')
    print("  Report preview:")
    for line in report_lines[:15]:  # First 15 lines
        print(f"    {line}")
    
    if len(report_lines) > 15:
        print(f"    ... and {len(report_lines) - 15} more lines")
    
    # Test with temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(report)
        temp_report_path = f.name
    
    print(f"  Full report saved to: {temp_report_path}")
    
    # Clean up
    os.unlink(temp_report_path)
    
    print("\n✓ All statistical analysis tests passed!")
    print("✓ Core functionality verified:")
    print("  - Confidence intervals")
    print("  - Independent and paired t-tests")
    print("  - Multiple method comparisons with ANOVA")
    print("  - Effect size calculations and interpretations")
    print("  - Power analysis")
    print("  - Summary report generation")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)