#!/usr/bin/env python3
"""
Script to analyze PEFT experiment results with statistical significance testing.
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from experiments.statistical_analysis import (
        ExperimentStatisticalAnalyzer,
        analyze_experiment_directory,
        quick_method_comparison,
        generate_statistical_report
    )
    from experiments.results import ResultsManager, ExperimentResult
    from evaluation.statistical_analyzer import StatisticalAnalyzer
    print("✓ Statistical analysis system loaded")
except ImportError as e:
    print(f"✗ Failed to import statistical analysis system: {e}")
    sys.exit(1)


def create_mock_results(output_dir: Path, num_methods: int = 3, num_seeds: int = 5):
    """Create mock experiment results for testing statistical analysis."""
    import random
    from datetime import datetime
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test methods
    methods = [
        {"name": "lora_r4", "base_acc": 0.85, "variance": 0.02},
        {"name": "lora_r8", "base_acc": 0.87, "variance": 0.015},
        {"name": "lora_r16", "base_acc": 0.89, "variance": 0.018},
        {"name": "adalora", "base_acc": 0.91, "variance": 0.012},
        {"name": "qa_lora", "base_acc": 0.88, "variance": 0.020}
    ]
    
    seeds = [42, 123, 456, 789, 999][:num_seeds]
    
    results_created = 0
    
    for method_idx in range(min(num_methods, len(methods))):
        method = methods[method_idx]
        
        for seed in seeds:
            # Generate realistic results with some variation
            random.seed(seed + method_idx * 1000)  # Deterministic but varied
            
            accuracy = method["base_acc"] + random.gauss(0, method["variance"])
            accuracy = max(0.0, min(1.0, accuracy))  # Clamp to valid range
            
            loss = 0.5 - (accuracy - 0.5) * 0.8 + random.gauss(0, 0.05)
            loss = max(0.01, loss)
            
            training_time = 120 + random.gauss(0, 20)
            memory_usage = 2.5 + random.gauss(0, 0.3)
            
            # Create experiment ID
            exp_id = f"deit_tiny_patch16_224_cifar10_seed{seed}_{method['name']}"
            
            # Create result
            result_data = {
                "experiment_id": exp_id,
                "status": "completed",
                "metrics": {
                    "final_accuracy": accuracy,
                    "final_loss": loss,
                    "training_time_seconds": training_time
                },
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "peak_memory_gb": memory_usage,
                "config": {
                    "name": f"test_{method['name']}_seed{seed}",
                    "model": {"name": "deit_tiny_patch16_224"},
                    "dataset": {"name": "cifar10"},
                    "seed": seed,
                    "lora": {"rank": int(method['name'].split('_r')[-1]) if '_r' in method['name'] else 8},
                    "use_adalora": "adalora" in method['name'],
                    "use_qa_lora": "qa_lora" in method['name']
                }
            }
            
            # Save result file
            result_file = output_dir / f"{exp_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            results_created += 1
    
    print(f"✓ Created {results_created} mock experiment results in {output_dir}")
    return results_created


def analyze_results_directory(results_dir: Path):
    """Analyze results in a directory."""
    print(f"\nAnalyzing results in: {results_dir}")
    print("-" * 50)
    
    # Create analyzer
    try:
        analyzer = analyze_experiment_directory(results_dir)
        print("✓ Statistical analyzer created")
    except Exception as e:
        print(f"✗ Failed to create analyzer: {e}")
        return
    
    # Get basic statistics
    try:
        stats = analyzer.results_manager.get_summary_statistics()
        print(f"✓ Total experiments: {stats['total_experiments']}")
        print(f"✓ Completed: {stats['completed']}")
        print(f"✓ Success rate: {stats['success_rate']:.1%}")
        
        if stats.get('performance'):
            perf = stats['performance']
            print(f"✓ Average accuracy: {perf.get('avg_accuracy', 0):.4f}")
            print(f"✓ Best accuracy: {perf.get('max_accuracy', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed to get basic statistics: {e}")
    
    # Aggregate results by method
    try:
        print("\nAggregating results by method...")
        aggregated = analyzer.aggregate_results_by_method("final_accuracy")
        
        print(f"✓ Found {len(aggregated)} unique method configurations")
        
        for method_id, agg_result in aggregated.items():
            print(f"  {agg_result.method_name}:")
            print(f"    Seeds: {agg_result.num_seeds}")
            print(f"    Success rate: {agg_result.success_rate:.1%}")
            
            if "final_accuracy" in agg_result.mean_metrics:
                mean_acc = agg_result.mean_metrics["final_accuracy"]
                std_acc = agg_result.std_metrics.get("final_accuracy", 0)
                print(f"    Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                
                if "final_accuracy" in agg_result.confidence_intervals:
                    ci = agg_result.confidence_intervals["final_accuracy"]
                    print(f"    95% CI: [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]")
    
    except Exception as e:
        print(f"✗ Failed to aggregate results: {e}")
    
    # Compare methods
    try:
        print("\nComparing methods statistically...")
        comparison = analyzer.compare_methods("final_accuracy", min_seeds=2)
        
        if "error" in comparison:
            print(f"✗ Method comparison failed: {comparison['error']}")
        else:
            print("✓ Method comparison completed")
            
            # Show rankings
            if "method_ranking" in comparison:
                print("  Method rankings by accuracy:")
                for i, (method, mean_acc) in enumerate(comparison["method_ranking"], 1):
                    print(f"    {i}. {method}: {mean_acc:.4f}")
            
            # Show ANOVA results
            if comparison.get("anova", {}).get("p_value") is not None:
                anova = comparison["anova"]
                print(f"  ANOVA: F={anova['f_statistic']:.3f}, p={anova['p_value']:.4f}")
                print(f"  Overall significance: {anova['interpretation']}")
            
            # Show significant pairwise comparisons
            pairwise = comparison.get("pairwise_comparisons", {})
            significant_pairs = [
                (name, test) for name, test in pairwise.items() 
                if test.is_significant
            ]
            
            if significant_pairs:
                print(f"  Significant pairwise differences ({len(significant_pairs)}):")
                for name, test in significant_pairs[:5]:  # Show first 5
                    print(f"    {name}: p={test.p_value:.4f}, effect={test.effect_size:.3f}")
            else:
                print("  No significant pairwise differences found")
    
    except Exception as e:
        print(f"✗ Failed to compare methods: {e}")
    
    # LoRA rank analysis
    try:
        print("\nAnalyzing LoRA rank effects...")
        rank_analysis = analyzer.analyze_lora_rank_effects(
            "deit_tiny_patch16_224", 
            "cifar10", 
            "final_accuracy"
        )
        
        if "error" in rank_analysis:
            print(f"✗ LoRA rank analysis failed: {rank_analysis['error']}")
        else:
            print("✓ LoRA rank analysis completed")
            print(f"  {rank_analysis['analysis_summary']}")
            
            # Show rank statistics
            rank_stats = rank_analysis.get("rank_statistics", {})
            if rank_stats:
                print("  Rank performance:")
                for rank in sorted(rank_stats.keys()):
                    stats = rank_stats[rank]
                    ci = stats["confidence_interval"]
                    print(f"    Rank {rank}: {stats['mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] (n={stats['sample_size']})")
    
    except Exception as e:
        print(f"✗ Failed LoRA rank analysis: {e}")


def generate_report(results_dir: Path, output_file: Path):
    """Generate comprehensive statistical report."""
    print(f"\nGenerating statistical report...")
    
    try:
        report = generate_statistical_report(results_dir, output_file)
        print(f"✓ Report generated and saved to {output_file}")
        
        # Show preview of report
        lines = report.split('\n')
        preview_lines = lines[:20]  # First 20 lines
        
        print("\nReport preview:")
        print("-" * 30)
        for line in preview_lines:
            print(line)
        
        if len(lines) > 20:
            print(f"... and {len(lines) - 20} more lines")
    
    except Exception as e:
        print(f"✗ Failed to generate report: {e}")


def test_statistical_functions():
    """Test basic statistical functions."""
    print("\nTesting statistical functions...")
    
    try:
        analyzer = StatisticalAnalyzer()
        
        # Test confidence interval
        test_values = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.85, 0.87]
        ci = analyzer.compute_confidence_interval(test_values)
        
        print(f"✓ Confidence interval test:")
        print(f"  Mean: {ci.mean:.4f}")
        print(f"  95% CI: [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]")
        print(f"  Sample size: {ci.sample_size}")
        
        # Test t-test
        group1 = [0.85, 0.87, 0.86, 0.88, 0.84]
        group2 = [0.89, 0.91, 0.90, 0.92, 0.88]
        
        t_test = analyzer.independent_t_test(group1, group2)
        print(f"✓ T-test result:")
        print(f"  t-statistic: {t_test.statistic:.4f}")
        print(f"  p-value: {t_test.p_value:.4f}")
        print(f"  Significant: {t_test.is_significant}")
        print(f"  Effect size: {t_test.effect_size:.4f}")
        
        # Test multiple comparisons
        methods_data = {
            "LoRA_r4": [0.85, 0.87, 0.86, 0.88, 0.84],
            "LoRA_r8": [0.87, 0.89, 0.88, 0.90, 0.86],
            "AdaLoRA": [0.89, 0.91, 0.90, 0.92, 0.88]
        }
        
        multi_comparison = analyzer.compare_multiple_methods(methods_data)
        print(f"✓ Multiple comparison test:")
        
        if multi_comparison["anova"]["p_value"] is not None:
            anova = multi_comparison["anova"]
            print(f"  ANOVA F-statistic: {anova['f_statistic']:.4f}")
            print(f"  ANOVA p-value: {anova['p_value']:.4f}")
        
        pairwise = multi_comparison.get("pairwise_comparisons", {})
        print(f"  Pairwise comparisons: {len(pairwise)}")
        
    except Exception as e:
        print(f"✗ Statistical function test failed: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze PEFT experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/test_outputs"),
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--create-mock-data",
        action="store_true",
        help="Create mock experiment data for testing"
    )
    parser.add_argument(
        "--num-methods",
        type=int,
        default=4,
        help="Number of methods for mock data"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds per method for mock data"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comprehensive statistical report"
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("experiments/statistical_report.txt"),
        help="Output file for statistical report"
    )
    parser.add_argument(
        "--test-functions",
        action="store_true",
        help="Test basic statistical functions"
    )
    
    args = parser.parse_args()
    
    print("PEFT Experiment Results Statistical Analysis")
    print("=" * 50)
    
    # Test statistical functions if requested
    if args.test_functions:
        test_statistical_functions()
    
    # Create mock data if requested
    if args.create_mock_data:
        print(f"\nCreating mock data...")
        mock_dir = args.results_dir / "mock_results"
        create_mock_results(mock_dir, args.num_methods, args.num_seeds)
        args.results_dir = mock_dir
    
    # Check if results directory exists
    if not args.results_dir.exists():
        print(f"✗ Results directory not found: {args.results_dir}")
        print("Use --create-mock-data to generate test data")
        return
    
    # Analyze results
    analyze_results_directory(args.results_dir)
    
    # Generate report if requested
    if args.generate_report:
        generate_report(args.results_dir, args.report_output)
    
    print(f"\n✓ Analysis complete!")
    print(f"Results directory: {args.results_dir}")
    
    if args.generate_report:
        print(f"Report saved to: {args.report_output}")


if __name__ == "__main__":
    main()