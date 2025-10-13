"""
Run multiple Monte Carlo simulations with different configurations.

This script runs multiple simulations to understand:
- How algorithm performance varies with different judge distributions
- Impact of number of comparisons on ranking quality
- Robustness to unreliable judges

Usage:
    python run_batch_simulations.py --scenarios all
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
from analyze_simulation import SimulationRunner, app
from simulation_config import (
    DEFAULT_NUM_PROJECTS,
    DEFAULT_NUM_JUDGES,
    DEFAULT_NUM_COMPARISONS,
    DEFAULT_NUM_TRIALS,
    CONVERGENCE_COMPARISON_COUNTS,
    SCALE_CONFIGS
)

SCENARIOS = {
    'baseline': {
        'name': 'Baseline (Mostly Reliable Judges)',
        'judge_distribution': {
            'reliable': 0.7,
            'unreliable': 0.2,
            'biased': 0.05,
            'random': 0.05
        },
        'description': 'Standard scenario with majority reliable judges'
    },
    'high_quality': {
        'name': 'High Quality Judges',
        'judge_distribution': {
            'reliable': 0.9,
            'unreliable': 0.05,
            'biased': 0.03,
            'random': 0.02
        },
        'description': '90% reliable judges - best case scenario'
    },
    'mixed_quality': {
        'name': 'Mixed Quality Judges',
        'judge_distribution': {
            'reliable': 0.5,
            'unreliable': 0.3,
            'biased': 0.1,
            'random': 0.1
        },
        'description': 'Half reliable, half problematic judges'
    },
    'low_quality': {
        'name': 'Low Quality Judges',
        'judge_distribution': {
            'reliable': 0.3,
            'unreliable': 0.4,
            'biased': 0.2,
            'random': 0.1
        },
        'description': 'Majority unreliable judges - worst case'
    },
    'biased_judges': {
        'name': 'Heavily Biased Judges',
        'judge_distribution': {
            'reliable': 0.4,
            'unreliable': 0.1,
            'biased': 0.4,
            'random': 0.1
        },
        'description': 'Many judges with systematic biases'
    }
}


def run_convergence_study(output_dir: str, num_projects: int = DEFAULT_NUM_PROJECTS, num_judges: int = DEFAULT_NUM_JUDGES):
    """
    Study how ranking quality improves with more comparisons.
    """
    print("\n" + "="*70)
    print("CONVERGENCE STUDY: How does accuracy improve with more comparisons?")
    print("="*70)

    comparison_counts = CONVERGENCE_COMPARISON_COUNTS
    results = []

    for num_comparisons in comparison_counts:
        print(f"\n--- Running with {num_comparisons} comparisons ---")

        sim = SimulationRunner(
            num_projects=num_projects,
            num_judges=num_judges,
            judge_distribution=SCENARIOS['baseline']['judge_distribution']
        )

        sim.setup_database()
        sim.run_simulation(num_comparisons=num_comparisons)
        metrics = sim.analyze_results()

        results.append({
            'num_comparisons': num_comparisons,
            'spearman': metrics['spearman_correlation'],
            'kendall_tau': metrics['kendall_tau'],
            'top_5_accuracy': metrics['top_k_accuracy'].get(5, 0),
            'mean_rank_error': metrics['mean_rank_error']
        })

        print(f"  Spearman: {metrics['spearman_correlation']:.3f}")
        print(f"  Top-5 Accuracy: {metrics['top_k_accuracy'].get(5, 0):.1%}")

    # Save results
    with open(f'{output_dir}/convergence_study.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Convergence study results saved to {output_dir}/convergence_study.json")
    return results


def run_scenario_comparison(output_dir: str, scenarios_to_run: list,
                           num_projects: int = DEFAULT_NUM_PROJECTS,
                           num_judges: int = DEFAULT_NUM_JUDGES,
                           num_comparisons: int = DEFAULT_NUM_COMPARISONS):
    """
    Compare algorithm performance across different judge distributions.
    """
    print("\n" + "="*70)
    print("SCENARIO COMPARISON: Testing different judge distributions")
    print("="*70)

    results = {}

    for scenario_name in scenarios_to_run:
        if scenario_name not in SCENARIOS:
            print(f"Warning: Unknown scenario '{scenario_name}', skipping")
            continue

        scenario = SCENARIOS[scenario_name]
        print(f"\n--- {scenario['name']} ---")
        print(f"    {scenario['description']}")
        print(f"    Distribution: {scenario['judge_distribution']}")

        sim = SimulationRunner(
            num_projects=num_projects,
            num_judges=num_judges,
            judge_distribution=scenario['judge_distribution']
        )

        sim.setup_database()
        sim.run_simulation(num_comparisons=num_comparisons)
        metrics = sim.analyze_results()

        results[scenario_name] = {
            'scenario_info': scenario,
            'metrics': {
                'spearman': float(metrics['spearman_correlation']),
                'kendall_tau': float(metrics['kendall_tau']),
                'top_k_accuracy': {int(k): float(v) for k, v in metrics['top_k_accuracy'].items()},
                'mean_rank_error': float(metrics['mean_rank_error'])
            },
            'judge_analysis': metrics['judge_analysis']
        }

        print(f"    Results:")
        print(f"      Spearman:       {metrics['spearman_correlation']:.3f}")
        print(f"      Top-5 Accuracy: {metrics['top_k_accuracy'].get(5, 0):.1%}")
        print(f"      Mean Error:     {metrics['mean_rank_error']:.1f} positions")

    # Save results
    with open(f'{output_dir}/scenario_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Scenario comparison saved to {output_dir}/scenario_comparison.json")
    return results


def run_monte_carlo_trials(output_dir: str, num_trials: int = DEFAULT_NUM_TRIALS,
                           num_projects: int = DEFAULT_NUM_PROJECTS,
                           num_judges: int = DEFAULT_NUM_JUDGES,
                           num_comparisons: int = DEFAULT_NUM_COMPARISONS):
    """
    Run multiple trials to get statistical confidence in results.
    """
    print("\n" + "="*70)
    print(f"MONTE CARLO TRIALS: Running {num_trials} independent simulations")
    print("="*70)

    all_results = []

    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")

        sim = SimulationRunner(
            num_projects=num_projects,
            num_judges=num_judges,
            judge_distribution=SCENARIOS['baseline']['judge_distribution'],
            seed=trial  # Different seed for each trial
        )

        sim.setup_database()
        sim.run_simulation(num_comparisons=num_comparisons)
        metrics = sim.analyze_results()

        trial_result = {
            'trial': trial,
            'spearman': float(metrics['spearman_correlation']),
            'kendall_tau': float(metrics['kendall_tau']),
            'top_5_accuracy': float(metrics['top_k_accuracy'].get(5, 0)),
            'top_5_in_top_10': float(metrics['top_5_in_top_10']),
            'mean_rank_error': float(metrics['mean_rank_error'])
        }
        all_results.append(trial_result)

        print(f"  Spearman: {metrics['spearman_correlation']:.3f}")

    # Calculate summary statistics
    spearmans = [r['spearman'] for r in all_results]
    top5s = [r['top_5_accuracy'] for r in all_results]
    top5_in_top10s = [r['top_5_in_top_10'] for r in all_results]

    summary = {
        'num_trials': num_trials,
        'trials': all_results,
        'summary_statistics': {
            'spearman_mean': float(np.mean(spearmans)),
            'spearman_std': float(np.std(spearmans)),
            'spearman_min': float(np.min(spearmans)),
            'spearman_max': float(np.max(spearmans)),
            'top5_mean': float(np.mean(top5s)),
            'top5_std': float(np.std(top5s)),
            'top5_in_top10_mean': float(np.mean(top5_in_top10s)),
            'top5_in_top10_std': float(np.std(top5_in_top10s))
        }
    }

    # Save results
    with open(f'{output_dir}/monte_carlo_trials.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("MONTE CARLO SUMMARY")
    print("="*70)
    print(f"Spearman correlation: {summary['summary_statistics']['spearman_mean']:.3f} ± {summary['summary_statistics']['spearman_std']:.3f}")
    print(f"  Range: [{summary['summary_statistics']['spearman_min']:.3f}, {summary['summary_statistics']['spearman_max']:.3f}]")
    print(f"Top-5 accuracy:       {summary['summary_statistics']['top5_mean']:.1%} ± {summary['summary_statistics']['top5_std']:.1%}")
    print(f"Top-5 in Top-10:      {summary['summary_statistics']['top5_in_top10_mean']:.1%} ± {summary['summary_statistics']['top5_in_top10_std']:.1%}")

    print(f"\n✓ Monte Carlo trials saved to {output_dir}/monte_carlo_trials.json")
    return summary


def run_scale_study(output_dir: str, num_comparisons: int = DEFAULT_NUM_COMPARISONS):
    """
    Study how algorithm performs with different dataset sizes.
    """
    print("\n" + "="*70)
    print("SCALE STUDY: Testing with different numbers of projects and judges")
    print("="*70)

    configs = SCALE_CONFIGS

    results = []

    for config in configs:
        print(f"\n--- {config['name']}: {config['projects']} projects, {config['judges']} judges ---")

        sim = SimulationRunner(
            num_projects=config['projects'],
            num_judges=config['judges'],
            judge_distribution=SCENARIOS['baseline']['judge_distribution']
        )

        sim.setup_database()
        sim.run_simulation(num_comparisons=num_comparisons)
        metrics = sim.analyze_results()

        results.append({
            'config': config,
            'spearman': float(metrics['spearman_correlation']),
            'kendall_tau': float(metrics['kendall_tau']),
            'top_5_accuracy': float(metrics['top_k_accuracy'].get(5, 0)),
            'mean_rank_error': float(metrics['mean_rank_error'])
        })

        print(f"  Spearman: {metrics['spearman_correlation']:.3f}")
        print(f"  Top-5 Accuracy: {metrics['top_k_accuracy'].get(5, 0):.1%}")

    # Save results
    with open(f'{output_dir}/scale_study.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Scale study saved to {output_dir}/scale_study.json")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run batch Monte Carlo simulations')
    parser.add_argument('--study', type=str, default='all',
                       choices=['convergence', 'scenarios', 'monte_carlo', 'scale', 'all'],
                       help='Which study to run')
    parser.add_argument('--scenarios', nargs='+', default=list(SCENARIOS.keys()),
                       help='Which scenarios to compare (for scenario study)')
    parser.add_argument('--num-trials', type=int, default=DEFAULT_NUM_TRIALS,
                       help=f'Number of trials for Monte Carlo study (default: {DEFAULT_NUM_TRIALS})')
    parser.add_argument('--num-projects', type=int, default=DEFAULT_NUM_PROJECTS,
                       help=f'Number of projects (default: {DEFAULT_NUM_PROJECTS})')
    parser.add_argument('--num-judges', type=int, default=DEFAULT_NUM_JUDGES,
                       help=f'Number of judges (default: {DEFAULT_NUM_JUDGES})')
    parser.add_argument('--comparisons', type=int, default=DEFAULT_NUM_COMPARISONS,
                       help=f'Number of comparisons per simulation (default: {DEFAULT_NUM_COMPARISONS})')
    parser.add_argument('--output-dir', type=str, default='./simulation_results',
                       help='Output directory')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/batch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("GAVEL MONTE CARLO SIMULATION SUITE")
    print("="*70)
    print(f"Output directory: {output_dir}")

    # Run requested studies
    if args.study in ['convergence', 'all']:
        run_convergence_study(output_dir, args.num_projects, args.num_judges)

    if args.study in ['scenarios', 'all']:
        run_scenario_comparison(output_dir, args.scenarios,
                              args.num_projects, args.num_judges, args.comparisons)

    if args.study in ['monte_carlo', 'all']:
        run_monte_carlo_trials(output_dir, args.num_trials,
                              args.num_projects, args.num_judges, args.comparisons)

    if args.study in ['scale', 'all']:
        run_scale_study(output_dir, args.comparisons)

    print("\n" + "="*70)
    print("ALL STUDIES COMPLETE!")
    print("="*70)
    print(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
