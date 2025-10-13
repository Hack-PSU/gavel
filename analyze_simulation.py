"""
Analysis script for Gavel simulations using real database data.

This script:
1. Uses Gavel's actual database models and CrowdBT functions
2. Populates the database with synthetic projects with known ground truth
3. Simulates judges with different reliability profiles making comparisons
4. Analyzes how well the algorithm recovers true rankings

Usage:
    python analyze_simulation.py --num-projects 50 --num-judges 20 --comparisons 500
"""

import os
os.environ['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://localhost/gavel_simulation')
os.environ['IGNORE_CONFIG_FILE'] = 'true'
os.environ['SECRET_KEY'] = 'simulation-secret-key'
os.environ['ENABLE_PROJECT_SYNC'] = 'false'  # Disable project sync during simulation

import numpy as np
from numpy.random import normal, beta as beta_dist, choice, shuffle, random
import argparse
import json
from datetime import datetime
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

# Import Gavel components
from gavel import app
from gavel.models import db, Item, Annotator, Decision
import gavel.crowd_bt as crowd_bt
from simulation_config import (
    DEFAULT_NUM_PROJECTS,
    DEFAULT_NUM_JUDGES,
    DEFAULT_NUM_COMPARISONS,
    DEFAULT_JUDGE_DISTRIBUTION,
    JUDGE_TYPES
)


class SimulationRunner:
    """Runs Monte Carlo simulations using actual Gavel database."""

    def __init__(
        self,
        num_projects: int,
        num_judges: int,
        judge_distribution: dict = None,
        seed: int = None
    ):
        self.num_projects = num_projects
        self.num_judges = num_judges

        if seed is not None:
            np.random.seed(seed)

        # Default judge distribution
        if judge_distribution is None:
            judge_distribution = {
                'reliable': 0.6,
                'unreliable': 0.2,
                'biased': 0.1,
                'random': 0.1
            }
        self.judge_distribution = judge_distribution

        # Ground truth mapping: project_id -> true_quality
        self.ground_truth = {}
        # Judge profiles: annotator_id -> (type, bias, initial_alpha, initial_beta)
        self.judge_profiles = {}

    def setup_database(self):
        """Initialize database with projects and judges."""
        with app.app_context():
            # Clear existing data - must respect foreign key constraints
            # Delete in order: decisions, then clear ignore table, then annotators and items
            db.session.query(Decision).delete()

            # Clear the many-to-many ignore table
            db.session.execute(db.text('DELETE FROM ignore'))

            # Clear the many-to-many view table
            db.session.execute(db.text('DELETE FROM view'))

            # Now we can safely delete annotators and items
            db.session.query(Annotator).delete()
            db.session.query(Item).delete()
            db.session.commit()

            # Create projects with ground truth quality
            print(f"Creating {self.num_projects} projects...")
            for i in range(self.num_projects):
                true_quality = normal(0, 1)  # Sample from standard normal

                item = Item(
                    name=f"Project {i+1}",
                    location=f"Table {i+1}",
                    description=f"Simulated project with true quality: {true_quality:.2f}"
                )
                db.session.add(item)
                db.session.flush()

                self.ground_truth[item.id] = true_quality

            # Create judges with different reliability profiles
            print(f"Creating {self.num_judges} judges...")
            judge_types = []
            for judge_type, proportion in self.judge_distribution.items():
                count = int(self.num_judges * proportion)
                judge_types.extend([judge_type] * count)

            # Fill remaining slots
            while len(judge_types) < self.num_judges:
                judge_types.append('reliable')

            shuffle(judge_types)

            for i, judge_type in enumerate(judge_types):
                if judge_type == 'reliable':
                    alpha = np.random.uniform(8, 12)
                    beta = np.random.uniform(0.5, 1.5)
                    bias = 0.0
                elif judge_type == 'unreliable':
                    alpha = np.random.uniform(3, 6)
                    beta = np.random.uniform(2, 4)
                    bias = 0.0
                elif judge_type == 'biased':
                    alpha = np.random.uniform(8, 12)
                    beta = np.random.uniform(0.5, 1.5)
                    bias = np.random.uniform(-0.5, 0.5)
                elif judge_type == 'random':
                    alpha = np.random.uniform(1, 2)
                    beta = np.random.uniform(1, 2)
                    bias = 0.0
                else:
                    raise ValueError(f"Unknown judge type: {judge_type}")

                annotator = Annotator(
                    name=f"Judge {i+1} ({judge_type})",
                    email=f"judge{i+1}@simulation.local",
                    description=f"Simulated {judge_type} judge"
                )
                # Override default alpha/beta with our simulation values
                annotator.alpha = alpha
                annotator.beta = beta

                db.session.add(annotator)
                db.session.flush()

                self.judge_profiles[annotator.id] = {
                    'type': judge_type,
                    'bias': bias,
                    'initial_alpha': alpha,
                    'initial_beta': beta
                }

            db.session.commit()
            print("Database setup complete!")

    def simulate_judge_decision(self, judge_id: int, proj_a_id: int, proj_b_id: int) -> bool:
        """
        Simulate a judge's decision between two projects.

        Returns True if proj_a is preferred, False if proj_b is preferred.
        """
        profile = self.judge_profiles[judge_id]
        bias = profile['bias']

        true_quality_a = self.ground_truth[proj_a_id]
        true_quality_b = self.ground_truth[proj_b_id]

        # Calculate quality difference with bias
        quality_diff = (true_quality_a + bias) - (true_quality_b + bias)

        # Sample judge's accuracy from their current alpha/beta
        with app.app_context():
            annotator = Annotator.query.get(judge_id)
            judge_accuracy = beta_dist(annotator.alpha, annotator.beta)

        # Probability of correct decision
        true_prob = 1 / (1 + np.exp(-quality_diff))

        # Judge's perceived probability
        perceived_prob = judge_accuracy * true_prob + (1 - judge_accuracy) * 0.5

        # Make decision
        return random() < perceived_prob

    def run_comparison(self, judge_id: int, proj_a_id: int, proj_b_id: int):
        """Perform a single comparison and update Gavel's parameters."""
        with app.app_context():
            annotator = Annotator.query.get(judge_id)
            proj_a = Item.query.get(proj_a_id)
            proj_b = Item.query.get(proj_b_id)

            # Simulate judge's decision
            a_wins = self.simulate_judge_decision(judge_id, proj_a_id, proj_b_id)

            if a_wins:
                winner = proj_a
                loser = proj_b
            else:
                winner = proj_b
                loser = proj_a

            # Use Gavel's actual update function
            u_alpha, u_beta, u_winner_mu, u_winner_sigma_sq, u_loser_mu, u_loser_sigma_sq = crowd_bt.update(
                annotator.alpha,
                annotator.beta,
                winner.mu,
                winner.sigma_sq,
                loser.mu,
                loser.sigma_sq
            )

            # Update parameters (exactly as Gavel does)
            annotator.alpha = u_alpha
            annotator.beta = u_beta
            winner.mu = u_winner_mu
            winner.sigma_sq = u_winner_sigma_sq
            loser.mu = u_loser_mu
            loser.sigma_sq = u_loser_sigma_sq

            # Record decision
            decision = Decision(
                annotator=annotator,
                winner=winner,
                loser=loser
            )
            db.session.add(decision)
            db.session.commit()

    def run_simulation(self, num_comparisons: int):
        """Run the simulation with specified number of comparisons."""
        print(f"\nRunning simulation with {num_comparisons} comparisons...")

        with app.app_context():
            project_ids = [item.id for item in Item.query.all()]
            judge_ids = [ann.id for ann in Annotator.query.all()]

        for i in range(num_comparisons):
            # Randomly select judge
            judge_id = int(choice(judge_ids))

            # Randomly select two different projects
            proj_a_id, proj_b_id = np.random.choice(project_ids, size=2, replace=False)
            proj_a_id = int(proj_a_id)
            proj_b_id = int(proj_b_id)

            # Perform comparison
            self.run_comparison(judge_id, proj_a_id, proj_b_id)

            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{num_comparisons} comparisons")

        print("Simulation complete!")

    def analyze_results(self):
        """Analyze simulation results and compute metrics."""
        with app.app_context():
            items = Item.query.all()
            annotators = Annotator.query.all()

            # Get true rankings vs estimated rankings
            true_qualities = [self.ground_truth[item.id] for item in items]
            estimated_mus = [item.mu for item in items]

            # Sort by true quality and estimated mu
            true_ranking = sorted(items, key=lambda x: self.ground_truth[x.id], reverse=True)
            estimated_ranking = sorted(items, key=lambda x: x.mu, reverse=True)

            # Calculate correlations
            spearman_corr, _ = spearmanr(true_qualities, estimated_mus)
            kendall_corr, _ = kendalltau(true_qualities, estimated_mus)

            # Calculate top-k accuracy
            top_k_accuracy = {}
            for k in [5, 10, 20]:
                if k > len(items):
                    continue
                true_top_k = set(p.id for p in true_ranking[:k])
                estimated_top_k = set(p.id for p in estimated_ranking[:k])
                accuracy = len(true_top_k & estimated_top_k) / k
                top_k_accuracy[k] = accuracy

            # Calculate "top-5 in top-10" metric - what % of true top-5 are in estimated top-10
            top_5_in_top_10 = 0.0
            if len(items) >= 10:
                true_top_5 = set(p.id for p in true_ranking[:3])
                estimated_top_10 = set(p.id for p in estimated_ranking[:10])
                top_5_in_top_10 = len(true_top_5 & estimated_top_10) / 5.0

            # Calculate ranking error
            true_rank_map = {p.id: i for i, p in enumerate(true_ranking)}
            estimated_rank_map = {p.id: i for i, p in enumerate(estimated_ranking)}
            rank_errors = [abs(true_rank_map[p.id] - estimated_rank_map[p.id]) for p in items]
            mean_rank_error = np.mean(rank_errors)

            # Judge analysis
            judge_analysis = []
            for ann in annotators:
                profile = self.judge_profiles[ann.id]
                judge_analysis.append({
                    'id': ann.id,
                    'name': ann.name,
                    'type': profile['type'],
                    'initial_alpha': profile['initial_alpha'],
                    'final_alpha': ann.alpha,
                    'initial_beta': profile['initial_beta'],
                    'final_beta': ann.beta,
                    'alpha_change': ann.alpha - profile['initial_alpha'],
                    'beta_change': ann.beta - profile['initial_beta']
                })

            results = {
                'spearman_correlation': spearman_corr,
                'kendall_tau': kendall_corr,
                'top_k_accuracy': top_k_accuracy,
                'top_5_in_top_10': top_5_in_top_10,
                'mean_rank_error': mean_rank_error,
                'num_projects': len(items),
                'num_judges': len(annotators),
                'num_comparisons': Decision.query.count(),
                'true_qualities': true_qualities,
                'estimated_mus': estimated_mus,
                'judge_analysis': judge_analysis
            }

            return results

    def print_results(self, results):
        """Print analysis results."""
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        print(f"\nCorrelation Metrics:")
        print(f"  Spearman correlation: {results['spearman_correlation']:.3f}")
        print(f"  Kendall tau:          {results['kendall_tau']:.3f}")

        print(f"\nTop-K Accuracy:")
        for k, acc in sorted(results['top_k_accuracy'].items()):
            print(f"  Top-{k:2d}: {acc:6.1%}")

        print(f"\nTop-5 Projects Coverage:")
        print(f"  Top-5 in Top-10: {results['top_5_in_top_10']:6.1%}")

        print(f"\nRanking Quality:")
        print(f"  Mean rank error: {results['mean_rank_error']:.1f} positions")

        print(f"\nSimulation Stats:")
        print(f"  Projects:    {results['num_projects']}")
        print(f"  Judges:      {results['num_judges']}")
        print(f"  Comparisons: {results['num_comparisons']}")

        print(f"\nJudge Performance (Alpha/Beta Changes):")
        for judge in sorted(results['judge_analysis'], key=lambda x: x['type']):
            print(f"  {judge['name']:30s} | α: {judge['initial_alpha']:5.1f} → {judge['final_alpha']:5.1f} ({judge['alpha_change']:+.1f}) | β: {judge['initial_beta']:5.1f} → {judge['final_beta']:5.1f} ({judge['beta_change']:+.1f})")

    def plot_results(self, results, save_path=None):
        """Create visualization of results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: True vs Estimated Quality
        ax = axes[0, 0]
        ax.scatter(results['true_qualities'], results['estimated_mus'], alpha=0.6)
        min_val = min(min(results['true_qualities']), min(results['estimated_mus']))
        max_val = max(max(results['true_qualities']), max(results['estimated_mus']))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect ranking')
        ax.set_xlabel('True Quality')
        ax.set_ylabel('Estimated μ')
        ax.set_title(f'True vs Estimated Quality\n(Spearman: {results["spearman_correlation"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Top-K Accuracy
        ax = axes[0, 1]
        k_values = sorted(results['top_k_accuracy'].keys())
        accuracies = [results['top_k_accuracy'][k] for k in k_values]
        ax.bar([f'Top-{k}' for k in k_values], accuracies)
        ax.set_ylabel('Accuracy')
        ax.set_title('Top-K Ranking Accuracy')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        for i, (k, acc) in enumerate(zip(k_values, accuracies)):
            ax.text(i, acc + 0.02, f'{acc:.1%}', ha='center')

        # Plot 3: Judge Type Distribution
        ax = axes[1, 0]
        judge_types = [j['type'] for j in results['judge_analysis']]
        type_counts = {}
        for t in judge_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        ax.bar(type_counts.keys(), type_counts.values())
        ax.set_xlabel('Judge Type')
        ax.set_ylabel('Count')
        ax.set_title('Judge Distribution')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Summary Statistics
        ax = axes[1, 1]
        summary_text = f"""
Simulation Summary

Correlation Metrics:
  Spearman: {results['spearman_correlation']:.3f}
  Kendall:  {results['kendall_tau']:.3f}

Ranking Quality:
  Mean error: {results['mean_rank_error']:.1f} positions

Top-K Accuracy:
"""
        for k, acc in sorted(results['top_k_accuracy'].items()):
            summary_text += f"  Top-{k}: {acc:.1%}\n"

        summary_text += f"""
Dataset:
  {results['num_projects']} projects
  {results['num_judges']} judges
  {results['num_comparisons']} comparisons
"""

        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        else:
            plt.show()

    def export_results(self, results, output_path):
        """Export results to JSON."""
        # Convert to JSON-serializable format
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'spearman_correlation': float(results['spearman_correlation']),
                'kendall_tau': float(results['kendall_tau']),
                'top_k_accuracy': {int(k): float(v) for k, v in results['top_k_accuracy'].items()},
                'top_5_in_top_10': float(results['top_5_in_top_10']),
                'mean_rank_error': float(results['mean_rank_error'])
            },
            'simulation_params': {
                'num_projects': results['num_projects'],
                'num_judges': results['num_judges'],
                'num_comparisons': results['num_comparisons'],
                'judge_distribution': self.judge_distribution
            },
            'judge_analysis': results['judge_analysis']
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo simulation for Gavel')
    parser.add_argument('--num-projects', type=int, default=DEFAULT_NUM_PROJECTS,
                       help=f'Number of projects (default: {DEFAULT_NUM_PROJECTS})')
    parser.add_argument('--num-judges', type=int, default=DEFAULT_NUM_JUDGES,
                       help=f'Number of judges (default: {DEFAULT_NUM_JUDGES})')
    parser.add_argument('--comparisons', type=int, default=DEFAULT_NUM_COMPARISONS,
                       help=f'Number of comparisons to simulate (default: {DEFAULT_NUM_COMPARISONS})')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./simulation_results', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run simulation
    sim = SimulationRunner(
        num_projects=args.num_projects,
        num_judges=args.num_judges,
        seed=args.seed
    )

    print("Setting up database...")
    sim.setup_database()

    sim.run_simulation(num_comparisons=args.comparisons)

    print("\nAnalyzing results...")
    results = sim.analyze_results()

    sim.print_results(results)

    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim.export_results(results, f'{args.output_dir}/results_{timestamp}.json')

    if args.plot:
        sim.plot_results(results, f'{args.output_dir}/plot_{timestamp}.png')


if __name__ == '__main__':
    main()
