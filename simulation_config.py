"""
Central configuration for Monte Carlo simulations.

All default simulation parameters are defined here.
Change these values to update defaults across all simulation scripts.
"""

# ========================================
# SIMULATION PARAMETERS
# ========================================

# Number of projects to simulate
DEFAULT_NUM_PROJECTS = 100

# Number of judges (annotators) to simulate
DEFAULT_NUM_JUDGES = 50

# Number of pairwise comparisons to perform
DEFAULT_NUM_COMPARISONS = 1000

# Number of Monte Carlo trials for statistical confidence
DEFAULT_NUM_TRIALS = 10

# ========================================
# JUDGE DISTRIBUTION
# ========================================

# Default distribution of judge types
DEFAULT_JUDGE_DISTRIBUTION = {
    'reliable': 0.7,      # 70% reliable judges (high alpha, low beta)
    'unreliable': 0.2,    # 20% unreliable judges (low alpha, high beta)
    'biased': 0.05,       # 5% biased judges (reliable but systematically shifted)
    'random': 0.05        # 5% random judges (near-random decisions)
}

# Judge type parameters
JUDGE_TYPES = {
    'reliable': {
        'alpha_range': (8, 12),
        'beta_range': (0.5, 1.5),
        'bias_range': (0, 0)
    },
    'unreliable': {
        'alpha_range': (3, 6),
        'beta_range': (2, 4),
        'bias_range': (0, 0)
    },
    'biased': {
        'alpha_range': (8, 12),
        'beta_range': (0.5, 1.5),
        'bias_range': (-2, 2)
    },
    'random': {
        'alpha_range': (0.8, 1.2),
        'beta_range': (0.8, 1.2),
        'bias_range': (0, 0)
    }
}

# ========================================
# CONVERGENCE STUDY SETTINGS
# ========================================

# Comparison counts to test in convergence study
CONVERGENCE_COMPARISON_COUNTS = [50]

# ========================================
# SCALE STUDY SETTINGS
# ========================================

# Different scale configurations to test
SCALE_CONFIGS = [
    {'projects': 25, 'judges': 10, 'name': 'Small'},
    # {'projects': 50, 'judges': 20, 'name': 'Medium'},
    # {'projects': 100, 'judges': 30, 'name': 'Large'},
    # {'projects': 150, 'judges': 40, 'name': 'Extra Large'}
]

# ========================================
# PARALLEL SIMULATION SETTINGS
# ========================================

# Default environment variables for docker-compose parallel runs
PARALLEL_ENV_DEFAULTS = {
    'SIM_PROJECTS': DEFAULT_NUM_PROJECTS,
    'SIM_JUDGES': DEFAULT_NUM_JUDGES,
    'SIM_COMPARISONS': DEFAULT_NUM_COMPARISONS,
    'ENABLE_PROJECT_SYNC': 'false'
}
