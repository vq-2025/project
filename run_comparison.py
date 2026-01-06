"""
Unified script to run benchmark and generate all comparisons
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Compare VQ-GAN, VQ-Diffusion, and Efficient-VQGAN performance'
    )
    parser.add_argument(
        '--skip-benchmark',
        action='store_true',
        help='Skip benchmarking and use existing results'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        default='results/benchmark_results.json',
        help='Path to benchmark results JSON file'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("VQ-GAN Model Comparison Suite")
    print("="*80 + "\n")

    # Run benchmark
    if not args.skip_benchmark:
        print("Step 1: Running benchmarks...")
        print("-" * 80)
        from benchmark import compare_models
        results = compare_models(save_results=True)
        print("\n✓ Benchmarking completed!\n")
    else:
        print("Step 1: Skipping benchmark (using existing results)")
        if not Path(args.results_path).exists():
            print(f"Error: Results file not found at {args.results_path}")
            print("Please run benchmark first or check the path.")
            sys.exit(1)

    # Generate visualizations
    print("Step 2: Generating visualizations...")
    print("-" * 80)
    from visualize_comparison import create_all_visualizations
    create_all_visualizations(args.results_path)
    print("\n✓ Visualizations completed!\n")

    # Generate report
    print("Step 3: Generating detailed report...")
    print("-" * 80)
    from generate_report import generate_report
    generate_report(args.results_path)
    print("\n✓ Report generated!\n")

    print("="*80)
    print("All tasks completed successfully!")
    print("\nGenerated files:")
    print("  - results/benchmark_results.json     (Raw benchmark data)")
    print("  - results/comparison_table.csv       (Comparison table)")
    print("  - results/*.png                      (Visualization plots)")
    print("  - results/comparison_report.md       (Detailed report)")
    print("  - results/comparison_report.html     (HTML report)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
