"""
Benchmark Script for Workforce Scheduling Algorithms

Systematically compares different scheduling algorithms across problem instances
of varying size and complexity.

Author: Ernest Owusu
"""

import time
import argparse
from typing import List, Dict
import pandas as pd
from data_generator import generate_scheduling_problem
from greedy_scheduler import GreedyScheduler
from genetic_scheduler import GeneticScheduler


def run_benchmark(
    problem_sizes: List[tuple] = None,
    num_trials: int = 5,
    output_file: str = "benchmark_results.csv"
) -> pd.DataFrame:
    """
    Run comprehensive benchmarks comparing algorithms.
    
    Args:
        problem_sizes: List of (workers, shifts, days) tuples
        num_trials: Number of trials per configuration
        output_file: Where to save results
        
    Returns:
        DataFrame with benchmark results
    """
    if problem_sizes is None:
        problem_sizes = [
            (10, 5, 7),    # Small
            (20, 10, 7),   # Medium
            (30, 15, 7),   # Large
            (50, 20, 14),  # Extra Large
        ]
    
    results = []
    
    print("=" * 70)
    print("WORKFORCE SCHEDULING ALGORITHM BENCHMARK")
    print("=" * 70)
    print(f"Problem sizes: {problem_sizes}")
    print(f"Trials per size: {num_trials}")
    print()
    
    for size_idx, (num_workers, num_shifts, num_days) in enumerate(problem_sizes):
        print(f"\nProblem Size {size_idx + 1}: {num_workers}W × {num_shifts}S × {num_days}D")
        print("-" * 70)
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end=" ")
            
            # Generate problem instance
            problem = generate_scheduling_problem(
                num_workers=num_workers,
                num_shifts=num_shifts,
                num_days=num_days,
                seed=42 + trial
            )
            
            # Test Greedy Algorithm (cost priority)
            greedy_cost = GreedyScheduler(priority_metric='cost')
            greedy_cost_sol = greedy_cost.solve(problem)
            
            results.append({
                'problem_size': f"{num_workers}W-{num_shifts}S-{num_days}D",
                'workers': num_workers,
                'shifts_per_day': num_shifts,
                'days': num_days,
                'trial': trial + 1,
                'algorithm': 'Greedy-Cost',
                'total_cost': greedy_cost_sol.total_cost,
                'coverage_rate': greedy_cost_sol.coverage_rate,
                'solve_time': greedy_cost_sol.solve_time,
                'feasible': greedy_cost_sol.feasible
            })
            
            # Test Greedy Algorithm (balanced priority)
            greedy_balanced = GreedyScheduler(priority_metric='balanced')
            greedy_balanced_sol = greedy_balanced.solve(problem)
            
            results.append({
                'problem_size': f"{num_workers}W-{num_shifts}S-{num_days}D",
                'workers': num_workers,
                'shifts_per_day': num_shifts,
                'days': num_days,
                'trial': trial + 1,
                'algorithm': 'Greedy-Balanced',
                'total_cost': greedy_balanced_sol.total_cost,
                'coverage_rate': greedy_balanced_sol.coverage_rate,
                'solve_time': greedy_balanced_sol.solve_time,
                'feasible': greedy_balanced_sol.feasible
            })
            
            # Test Genetic Algorithm (scaled parameters by problem size)
            pop_size = min(100, num_workers * 3)
            generations = min(50, 30 + num_workers // 2)
            
            ga = GeneticScheduler(
                population_size=pop_size,
                generations=generations,
                crossover_rate=0.8,
                mutation_rate=0.15
            )
            ga_sol = ga.solve(problem)
            
            results.append({
                'problem_size': f"{num_workers}W-{num_shifts}S-{num_days}D",
                'workers': num_workers,
                'shifts_per_day': num_shifts,
                'days': num_days,
                'trial': trial + 1,
                'algorithm': 'Genetic-Algorithm',
                'total_cost': ga_sol.total_cost,
                'coverage_rate': ga_sol.coverage_rate,
                'solve_time': ga_sol.solve_time,
                'feasible': ga_sol.feasible
            })
            
            print("Done")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to {output_file}")
    
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics from benchmark results."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Group by problem size and algorithm
    summary = df.groupby(['problem_size', 'algorithm']).agg({
        'total_cost': ['mean', 'std'],
        'coverage_rate': ['mean', 'std'],
        'solve_time': ['mean', 'std'],
        'feasible': 'mean'
    }).round(4)
    
    print("\nAverage Performance by Algorithm and Problem Size:")
    print(summary)
    
    # Compare algorithms
    print("\n" + "-" * 70)
    print("ALGORITHM COMPARISON (Averaged across all problem sizes)")
    print("-" * 70)
    
    algo_comparison = df.groupby('algorithm').agg({
        'total_cost': 'mean',
        'coverage_rate': 'mean',
        'solve_time': 'mean',
        'feasible': 'mean'
    }).round(4)
    
    print(algo_comparison)
    
    # Cost improvement analysis
    print("\n" + "-" * 70)
    print("COST IMPROVEMENT: Genetic Algorithm vs Greedy")
    print("-" * 70)
    
    for size in df['problem_size'].unique():
        size_data = df[df['problem_size'] == size]
        
        greedy_cost = size_data[size_data['algorithm'] == 'Greedy-Cost']['total_cost'].mean()
        ga_cost = size_data[size_data['algorithm'] == 'Genetic-Algorithm']['total_cost'].mean()
        
        improvement = ((greedy_cost - ga_cost) / greedy_cost) * 100
        
        print(f"{size}:")
        print(f"  Greedy Cost: ${greedy_cost:.2f}")
        print(f"  GA Cost: ${ga_cost:.2f}")
        print(f"  Improvement: {improvement:.2f}%")
        print()


def create_visualizations(df: pd.DataFrame):
    """Create benchmark visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Workforce Scheduling Algorithm Benchmarks', fontsize=16, fontweight='bold')
        
        # 1. Cost Comparison
        ax1 = axes[0, 0]
        cost_data = df.groupby(['problem_size', 'algorithm'])['total_cost'].mean().reset_index()
        for algo in df['algorithm'].unique():
            algo_data = cost_data[cost_data['algorithm'] == algo]
            ax1.plot(range(len(algo_data)), algo_data['total_cost'], marker='o', label=algo, linewidth=2)
        ax1.set_xlabel('Problem Size', fontsize=11)
        ax1.set_ylabel('Average Total Cost ($)', fontsize=11)
        ax1.set_title('Cost Comparison', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.set_xticks(range(len(df['problem_size'].unique())))
        ax1.set_xticklabels(df['problem_size'].unique(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Solve Time Comparison
        ax2 = axes[0, 1]
        time_data = df.groupby(['problem_size', 'algorithm'])['solve_time'].mean().reset_index()
        for algo in df['algorithm'].unique():
            algo_data = time_data[time_data['algorithm'] == algo]
            ax2.plot(range(len(algo_data)), algo_data['solve_time'], marker='s', label=algo, linewidth=2)
        ax2.set_xlabel('Problem Size', fontsize=11)
        ax2.set_ylabel('Average Solve Time (seconds)', fontsize=11)
        ax2.set_title('Computational Time Comparison', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.set_xticks(range(len(df['problem_size'].unique())))
        ax2.set_xticklabels(df['problem_size'].unique(), rotation=45, ha='right')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Coverage Rate Comparison
        ax3 = axes[1, 0]
        coverage_data = df.groupby(['problem_size', 'algorithm'])['coverage_rate'].mean().reset_index()
        coverage_data['coverage_pct'] = coverage_data['coverage_rate'] * 100
        for algo in df['algorithm'].unique():
            algo_data = coverage_data[coverage_data['algorithm'] == algo]
            ax3.plot(range(len(algo_data)), algo_data['coverage_pct'], marker='^', label=algo, linewidth=2)
        ax3.set_xlabel('Problem Size', fontsize=11)
        ax3.set_ylabel('Coverage Rate (%)', fontsize=11)
        ax3.set_title('Shift Coverage Comparison', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_xticks(range(len(df['problem_size'].unique())))
        ax3.set_xticklabels(df['problem_size'].unique(), rotation=45, ha='right')
        ax3.set_ylim([90, 101])
        ax3.grid(True, alpha=0.3)
        
        # 4. Cost-Time Tradeoff
        ax4 = axes[1, 1]
        tradeoff_data = df.groupby('algorithm').agg({
            'total_cost': 'mean',
            'solve_time': 'mean'
        }).reset_index()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for idx, row in tradeoff_data.iterrows():
            ax4.scatter(row['solve_time'], row['total_cost'], 
                       s=200, alpha=0.6, c=[colors[idx]], 
                       label=row['algorithm'], edgecolors='black', linewidths=2)
        
        ax4.set_xlabel('Average Solve Time (seconds)', fontsize=11)
        ax4.set_ylabel('Average Total Cost ($)', fontsize=11)
        ax4.set_title('Cost-Time Tradeoff', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to 'benchmark_results.png'")
        plt.show()
        
    except ImportError:
        print("\nNote: matplotlib/seaborn not installed. Skipping visualizations.")
        print("Install with: pip install matplotlib seaborn")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='Benchmark workforce scheduling algorithms')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (overrides default sizes)')
    parser.add_argument('--shifts', type=int, default=None, help='Number of shifts (overrides default sizes)')
    parser.add_argument('--days', type=int, default=None, help='Number of days (overrides default sizes)')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per size')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='Output CSV file')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Determine problem sizes
    if args.workers and args.shifts and args.days:
        problem_sizes = [(args.workers, args.shifts, args.days)]
    else:
        problem_sizes = None  # Use defaults
    
    # Run benchmarks
    df = run_benchmark(
        problem_sizes=problem_sizes,
        num_trials=args.trials,
        output_file=args.output
    )
    
    # Print statistics
    print_summary_statistics(df)
    
    # Create visualizations
    if not args.no_viz:
        create_visualizations(df)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
