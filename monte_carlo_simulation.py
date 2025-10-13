"""
Monte Carlo simulation for testing Gavel's CrowdBT algorithm.

This simulation allows testing how the algorithm responds to:
- Different numbers of projects and judges
- Varying judge reliability (alpha/beta parameters)
- Different voting patterns and biases

Usage:
    python monte_carlo_simulation.py --num-projects 50 --num-judges 20 --num-simulations 100
"""

import numpy as np
from numpy.random import normal, beta, choice, shuffle, random
import gavel.crowd_bt as crowd_bt
from typing import List, Tuple, Dict, Optional
import argparse
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Project:
    """Represents a project in the simulation."""
    id: int
    true_quality: float  # Ground truth quality
    mu: float = crowd_bt.MU_PRIOR
    sigma_sq: float = crowd_bt.SIGMA_SQ_PRIOR
    views: int = 0

    def __repr__(self):
        return f"Project({self.id}, true={self.true_quality:.2f}, mu={self.mu:.2f})"


@dataclass
class Judge:
    """Represents a judge (annotator) in the simulation."""
    id: int
    alpha: float
    beta: float
    reliability_type: str  # 'reliable', 'unreliable', 'biased', 'random'
    bias: float = 0.0  # Bias term added to judgments
    num_votes: int = 0

    def judge_comparison(self, proj_a: Project, proj_b: Project) -> bool:
        """
        Returns True if judge thinks proj_a is better than proj_b.

        The judgment is based on:
        1. True quality scores
        2. Judge's reliability (modeled by alpha/beta)
        3. Random noise
        4. Potential bias
        """
        # Calculate true quality difference with bias
        quality_diff = (proj_a.true_quality + self.bias) - (proj_b.true_quality + self.bias)

        # Model judge reliability: sample from Beta(alpha, beta)
        # Higher alpha/beta ratio means more reliable
        judge_accuracy = beta(self.alpha, self.beta)

        # Probability that judge makes correct decision
        # Use sigmoid to convert quality difference to probability
        true_prob = 1 / (1 + np.exp(-quality_diff))

        # Judge's perceived probability (influenced by their reliability)
        perceived_prob = judge_accuracy * true_prob + (1 - judge_accuracy) * 0.5

        # Make decision based on perceived probability
        return random() < perceived_prob

    def __repr__(self):
        return f"Judge({self.id}, type={self.reliability_type}, α={self.alpha:.1f}, β={self.beta:.1f})"


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    spearman_correlation: float
    kendall_tau: float
    top_k_accuracy: Dict[int, float]  # Accuracy for top-k projects
    ranking_error: float  # Mean absolute rank error
    convergence_iterations: int  # Iterations until convergence
    judge_alpha_evolution: Dict[int, List[float]]
    judge_beta_evolution: Dict[int, List[float]]
    project_mu_evolution: Dict[int, List[float]]
    num_comparisons: int


class MonteCarloSimulation:
    """Monte Carlo simulation of Gavel's ranking algorithm."""

    def __init__(
        self,
        num_projects: int,
        num_judges: int,
        judge_distribution: Optional[Dict[str, float]] = None,
        min_views_per_project: int = 5,
        seed: Optional[int] = None
    ):
        """
        Initialize simulation.

        Args:
            num_projects: Number of projects to simulate
            num_judges: Number of judges to simulate
            judge_distribution: Distribution of judge types
                e.g., {'reliable': 0.7, 'unreliable': 0.2, 'random': 0.1}
            min_views_per_project: Minimum comparisons per project
            seed: Random seed for reproducibility
        """
        self.num_projects = num_projects
        self.num_judges = num_judges
        self.min_views_per_project = min_views_per_project

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

        # Initialize projects and judges
        self.projects: List[Project] = []
        self.judges: List[Judge] = []

        # Tracking
        self.comparison_count = 0
        self.alpha_history: Dict[int, List[float]] = {}
        self.beta_history: Dict[int, List[float]] = {}
        self.mu_history: Dict[int, List[float]] = {}

    def generate_projects(self, distribution: str = 'normal') -> None:
        """
        Generate projects with true quality scores.

        Args:
            distribution: 'normal', 'uniform', or 'exponential'
        """
        self.projects = []

        for i in range(self.num_projects):
            if distribution == 'normal':
                true_quality = normal(0, 1)
            elif distribution == 'uniform':
                true_quality = np.random.uniform(-2, 2)
            elif distribution == 'exponential':
                true_quality = np.random.exponential(1) - 1
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

            project = Project(
                id=i,
                true_quality=true_quality,
                mu=crowd_bt.MU_PRIOR,
                sigma_sq=crowd_bt.SIGMA_SQ_PRIOR
            )
            self.projects.append(project)
            self.mu_history[i] = [crowd_bt.MU_PRIOR]

    def generate_judges(self) -> None:
        """Generate judges with different reliability profiles."""
        self.judges = []

        # Determine number of each type
        judge_types = []
        for judge_type, proportion in self.judge_distribution.items():
            count = int(self.num_judges * proportion)
            judge_types.extend([judge_type] * count)

        # Fill remaining slots with 'reliable' type
        while len(judge_types) < self.num_judges:
            judge_types.append('reliable')

        shuffle(judge_types)

        for i, judge_type in enumerate(judge_types):
            if judge_type == 'reliable':
                # High alpha, low beta = reliable judge
                alpha = np.random.uniform(8, 12)
                beta = np.random.uniform(0.5, 1.5)
                bias = 0.0
            elif judge_type == 'unreliable':
                # Lower alpha, higher beta = unreliable
                alpha = np.random.uniform(3, 6)
                beta = np.random.uniform(2, 4)
                bias = 0.0
            elif judge_type == 'biased':
                # Reliable but biased
                alpha = np.random.uniform(8, 12)
                beta = np.random.uniform(0.5, 1.5)
                bias = np.random.uniform(-0.5, 0.5)
            elif judge_type == 'random':
                # Nearly random judgments
                alpha = np.random.uniform(1, 2)
                beta = np.random.uniform(1, 2)
                bias = 0.0
            else:
                raise ValueError(f"Unknown judge type: {judge_type}")

            judge = Judge(
                id=i,
                alpha=alpha,
                beta=beta,
                reliability_type=judge_type,
                bias=bias
            )
            self.judges.append(judge)
            self.alpha_history[i] = [alpha]
            self.beta_history[i] = [beta]

    def choose_next_comparison(self, judge: Judge, prev_project: Optional[Project]) -> Optional[Project]:
        """
        Choose next project for comparison using epsilon-greedy strategy.

        This mimics Gavel's choose_next function.
        """
        # Filter out projects the judge has already seen too many times
        available = [p for p in self.projects if p.views < self.min_views_per_project * 2]

        if not available:
            available = self.projects

        if prev_project is None:
            return choice(available)

        shuffle(available)

        # Epsilon-greedy selection
        if random() < crowd_bt.EPSILON:
            return available[0]
        else:
            # Choose project that maximizes expected information gain
            best_proj = crowd_bt.argmax(
                lambda p: crowd_bt.expected_information_gain(
                    judge.alpha,
                    judge.beta,
                    prev_project.mu,
                    prev_project.sigma_sq,
                    p.mu,
                    p.sigma_sq
                ),
                available
            )
            return best_proj

    def perform_comparison(self, judge: Judge, proj_a: Project, proj_b: Project) -> None:
        """Perform a single pairwise comparison and update parameters."""
        # Judge makes decision
        a_wins = judge.judge_comparison(proj_a, proj_b)

        if a_wins:
            winner = proj_a
            loser = proj_b
        else:
            winner = proj_b
            loser = proj_a

        # Update using CrowdBT algorithm
        u_alpha, u_beta, u_winner_mu, u_winner_sigma_sq, u_loser_mu, u_loser_sigma_sq = crowd_bt.update(
            judge.alpha,
            judge.beta,
            winner.mu,
            winner.sigma_sq,
            loser.mu,
            loser.sigma_sq
        )

        # Update judge parameters
        judge.alpha = u_alpha
        judge.beta = u_beta
        judge.num_votes += 1

        # Update project parameters
        winner.mu = u_winner_mu
        winner.sigma_sq = u_winner_sigma_sq
        loser.mu = u_loser_mu
        loser.sigma_sq = u_loser_sigma_sq

        # Update views
        proj_a.views += 1
        proj_b.views += 1

        # Track history
        self.alpha_history[judge.id].append(judge.alpha)
        self.beta_history[judge.id].append(judge.beta)
        self.mu_history[winner.id].append(winner.mu)
        self.mu_history[loser.id].append(loser.mu)

        self.comparison_count += 1

    def run_simulation(self, max_comparisons: Optional[int] = None) -> SimulationMetrics:
        """
        Run the simulation.

        Args:
            max_comparisons: Maximum number of comparisons (if None, uses heuristic)

        Returns:
            SimulationMetrics with results
        """
        if max_comparisons is None:
            # Heuristic: ensure each project gets min_views comparisons
            max_comparisons = self.num_projects * self.min_views_per_project * 2

        self.comparison_count = 0

        # Each judge performs comparisons
        for iteration in range(max_comparisons):
            # Select a random judge
            judge = choice(self.judges)

            # Select first project randomly
            prev_project = choice(self.projects)

            # Select next project using algorithm
            next_project = self.choose_next_comparison(judge, prev_project)

            if next_project is None:
                continue

            # Perform comparison
            self.perform_comparison(judge, prev_project, next_project)

            # Check for convergence (optional early stopping)
            if iteration > 0 and iteration % 100 == 0:
                if self._check_convergence():
                    print(f"Converged after {iteration} comparisons")
                    break

        # Calculate metrics
        return self._calculate_metrics()

    def _check_convergence(self, threshold: float = 0.01) -> bool:
        """Check if rankings have converged."""
        # Check if top projects' mu values are stable
        if len(self.mu_history[0]) < 20:
            return False

        # Check variance in recent mu updates for top projects
        top_projects = sorted(self.projects, key=lambda p: p.mu, reverse=True)[:10]

        for proj in top_projects:
            recent_mus = self.mu_history[proj.id][-10:]
            if len(recent_mus) >= 10 and np.std(recent_mus) > threshold:
                return False

        return True

    def _calculate_metrics(self) -> SimulationMetrics:
        """Calculate simulation metrics."""
        from scipy.stats import spearmanr, kendalltau

        # Get true ranking and estimated ranking
        true_ranking = sorted(self.projects, key=lambda p: p.true_quality, reverse=True)
        estimated_ranking = sorted(self.projects, key=lambda p: p.mu, reverse=True)

        true_qualities = [p.true_quality for p in self.projects]
        estimated_mus = [p.mu for p in self.projects]

        # Calculate correlation metrics
        spearman_corr, _ = spearmanr(true_qualities, estimated_mus)
        kendall_corr, _ = kendalltau(true_qualities, estimated_mus)

        # Calculate top-k accuracy
        top_k_accuracy = {}
        for k in [5, 10, 20]:
            if k > len(self.projects):
                continue
            true_top_k = set(p.id for p in true_ranking[:k])
            estimated_top_k = set(p.id for p in estimated_ranking[:k])
            accuracy = len(true_top_k & estimated_top_k) / k
            top_k_accuracy[k] = accuracy

        # Calculate ranking error
        true_rank_map = {p.id: i for i, p in enumerate(true_ranking)}
        estimated_rank_map = {p.id: i for i, p in enumerate(estimated_ranking)}

        rank_errors = [abs(true_rank_map[p.id] - estimated_rank_map[p.id]) for p in self.projects]
        mean_rank_error = np.mean(rank_errors)

        return SimulationMetrics(
            spearman_correlation=spearman_corr,
            kendall_tau=kendall_corr,
            top_k_accuracy=top_k_accuracy,
            ranking_error=mean_rank_error,
            convergence_iterations=self.comparison_count,
            judge_alpha_evolution=self.alpha_history,
            judge_beta_evolution=self.beta_history,
            project_mu_evolution=self.mu_history,
            num_comparisons=self.comparison_count
        )

    def plot_results(self, metrics: SimulationMetrics, save_path: Optional[str] = None) -> None:
        """Generate visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: True quality vs Estimated quality
        ax = axes[0, 0]
        true_qualities = [p.true_quality for p in self.projects]
        estimated_mus = [p.mu for p in self.projects]
        ax.scatter(true_qualities, estimated_mus, alpha=0.6)
        ax.plot([min(true_qualities), max(true_qualities)],
                [min(true_qualities), max(true_qualities)],
                'r--', label='Perfect ranking')
        ax.set_xlabel('True Quality')
        ax.set_ylabel('Estimated μ')
        ax.set_title(f'True vs Estimated Quality\n(Spearman: {metrics.spearman_correlation:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Judge alpha evolution
        ax = axes[0, 1]
        for judge in self.judges[:10]:  # Plot first 10 judges
            ax.plot(self.alpha_history[judge.id],
                   label=f'{judge.reliability_type[:3]}',
                   alpha=0.7)
        ax.set_xlabel('Comparisons')
        ax.set_ylabel('Alpha')
        ax.set_title('Judge Alpha Evolution (first 10 judges)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 3: Judge beta evolution
        ax = axes[0, 2]
        for judge in self.judges[:10]:
            ax.plot(self.beta_history[judge.id],
                   label=f'{judge.reliability_type[:3]}',
                   alpha=0.7)
        ax.set_xlabel('Comparisons')
        ax.set_ylabel('Beta')
        ax.set_title('Judge Beta Evolution (first 10 judges)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 4: Project mu evolution
        ax = axes[1, 0]
        # Plot top 5 and bottom 5 projects by true quality
        sorted_projects = sorted(self.projects, key=lambda p: p.true_quality)
        sample_projects = sorted_projects[:5] + sorted_projects[-5:]
        for proj in sample_projects:
            color = 'green' if proj.true_quality > 0 else 'red'
            ax.plot(self.mu_history[proj.id], alpha=0.6, color=color)
        ax.set_xlabel('Comparisons')
        ax.set_ylabel('Project μ')
        ax.set_title('Project μ Evolution\n(green=high quality, red=low quality)')
        ax.grid(True, alpha=0.3)

        # Plot 5: Judge reliability distribution
        ax = axes[1, 1]
        judge_types = [j.reliability_type for j in self.judges]
        type_counts = {t: judge_types.count(t) for t in set(judge_types)}
        ax.bar(type_counts.keys(), type_counts.values())
        ax.set_xlabel('Judge Type')
        ax.set_ylabel('Count')
        ax.set_title('Judge Distribution')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 6: Ranking accuracy metrics
        ax = axes[1, 2]
        metrics_text = f"""
        Simulation Results:

        Spearman Correlation: {metrics.spearman_correlation:.3f}
        Kendall Tau: {metrics.kendall_tau:.3f}

        Top-K Accuracy:
        """
        for k, acc in sorted(metrics.top_k_accuracy.items()):
            metrics_text += f"  Top-{k}: {acc:.1%}\n"

        metrics_text += f"""
        Mean Rank Error: {metrics.ranking_error:.1f}
        Total Comparisons: {metrics.num_comparisons}

        Projects: {self.num_projects}
        Judges: {self.num_judges}
        """

        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
               verticalalignment='center')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def export_results(self, metrics: SimulationMetrics, output_path: str) -> None:
        """Export simulation results to JSON."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'num_projects': self.num_projects,
                'num_judges': self.num_judges,
                'min_views_per_project': self.min_views_per_project,
                'judge_distribution': self.judge_distribution
            },
            'metrics': {
                'spearman_correlation': metrics.spearman_correlation,
                'kendall_tau': metrics.kendall_tau,
                'top_k_accuracy': metrics.top_k_accuracy,
                'ranking_error': metrics.ranking_error,
                'num_comparisons': metrics.num_comparisons
            },
            'projects': [
                {
                    'id': p.id,
                    'true_quality': p.true_quality,
                    'estimated_mu': p.mu,
                    'sigma_sq': p.sigma_sq,
                    'views': p.views
                }
                for p in self.projects
            ],
            'judges': [
                {
                    'id': j.id,
                    'type': j.reliability_type,
                    'initial_alpha': self.alpha_history[j.id][0],
                    'final_alpha': j.alpha,
                    'initial_beta': self.beta_history[j.id][0],
                    'final_beta': j.beta,
                    'num_votes': j.num_votes
                }
                for j in self.judges
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results exported to {output_path}")


def run_multiple_simulations(
    num_simulations: int,
    num_projects: int,
    num_judges: int,
    judge_distributions: List[Dict[str, float]],
    output_dir: str = './simulation_results'
) -> None:
    """Run multiple simulations with different configurations."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for sim_id in range(num_simulations):
        print(f"\n=== Running Simulation {sim_id + 1}/{num_simulations} ===")

        # Vary judge distribution
        judge_dist = choice(judge_distributions)

        sim = MonteCarloSimulation(
            num_projects=num_projects,
            num_judges=num_judges,
            judge_distribution=judge_dist,
            seed=sim_id
        )

        sim.generate_projects(distribution='normal')
        sim.generate_judges()

        metrics = sim.run_simulation()

        print(f"Spearman correlation: {metrics.spearman_correlation:.3f}")
        print(f"Kendall tau: {metrics.kendall_tau:.3f}")
        print(f"Top-5 accuracy: {metrics.top_k_accuracy.get(5, 0):.1%}")

        # Save individual simulation
        sim.export_results(metrics, f"{output_dir}/simulation_{sim_id:03d}.json")
        sim.plot_results(metrics, f"{output_dir}/simulation_{sim_id:03d}.png")

        all_results.append({
            'sim_id': sim_id,
            'judge_distribution': judge_dist,
            'spearman': metrics.spearman_correlation,
            'kendall_tau': metrics.kendall_tau,
            'top_k_accuracy': metrics.top_k_accuracy,
            'ranking_error': metrics.ranking_error
        })

    # Summary statistics
    print("\n=== Summary Statistics ===")
    spearmans = [r['spearman'] for r in all_results]
    print(f"Spearman correlation: {np.mean(spearmans):.3f} ± {np.std(spearmans):.3f}")

    kendalls = [r['kendall_tau'] for r in all_results]
    print(f"Kendall tau: {np.mean(kendalls):.3f} ± {np.std(kendalls):.3f}")

    # Save summary
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump({
            'num_simulations': num_simulations,
            'results': all_results,
            'summary': {
                'mean_spearman': float(np.mean(spearmans)),
                'std_spearman': float(np.std(spearmans)),
                'mean_kendall': float(np.mean(kendalls)),
                'std_kendall': float(np.std(kendalls))
            }
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo simulation for Gavel algorithm')
    parser.add_argument('--num-projects', type=int, default=50, help='Number of projects')
    parser.add_argument('--num-judges', type=int, default=20, help='Number of judges')
    parser.add_argument('--num-simulations', type=int, default=1, help='Number of simulations to run')
    parser.add_argument('--output-dir', type=str, default='./simulation_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Generate plots')

    args = parser.parse_args()

    # Define different judge distributions to test
    judge_distributions = [
        {'reliable': 0.7, 'unreliable': 0.2, 'biased': 0.05, 'random': 0.05},
        {'reliable': 0.5, 'unreliable': 0.3, 'biased': 0.1, 'random': 0.1},
        {'reliable': 0.9, 'unreliable': 0.05, 'biased': 0.03, 'random': 0.02},
        {'reliable': 0.3, 'unreliable': 0.4, 'biased': 0.2, 'random': 0.1},
    ]

    if args.num_simulations > 1:
        run_multiple_simulations(
            num_simulations=args.num_simulations,
            num_projects=args.num_projects,
            num_judges=args.num_judges,
            judge_distributions=judge_distributions,
            output_dir=args.output_dir
        )
    else:
        # Single simulation
        sim = MonteCarloSimulation(
            num_projects=args.num_projects,
            num_judges=args.num_judges,
            seed=args.seed
        )

        print("Generating projects...")
        sim.generate_projects(distribution='normal')

        print("Generating judges...")
        sim.generate_judges()

        print("\nJudge distribution:")
        for judge_type in set(j.reliability_type for j in sim.judges):
            count = sum(1 for j in sim.judges if j.reliability_type == judge_type)
            print(f"  {judge_type}: {count}")

        print("\nRunning simulation...")
        metrics = sim.run_simulation()

        print("\n=== Results ===")
        print(f"Spearman correlation: {metrics.spearman_correlation:.3f}")
        print(f"Kendall tau: {metrics.kendall_tau:.3f}")
        for k, acc in sorted(metrics.top_k_accuracy.items()):
            print(f"Top-{k} accuracy: {acc:.1%}")
        print(f"Mean rank error: {metrics.ranking_error:.1f}")
        print(f"Total comparisons: {metrics.num_comparisons}")

        if args.plot:
            sim.plot_results(metrics)

        # Export results
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        sim.export_results(metrics, f"{args.output_dir}/results.json")


if __name__ == '__main__':
    main()
