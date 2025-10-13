#!/bin/bash
# Parallel simulation runner using docker-compose
#
# Usage:
#   ./run_parallel.sh <num_workers> [trials_per_worker]
#
# Examples:
#   ./run_parallel.sh 4 25          # 4 workers, 25 trials each = 100 total
#   ./run_parallel.sh 8 50          # 8 workers, 50 trials each = 400 total

set -e

NUM_WORKERS=${1:-4}
TRIALS_PER_WORKER=${2:-25}
TOTAL_TRIALS=$((NUM_WORKERS * TRIALS_PER_WORKER))

echo "========================================"
echo "PARALLEL GAVEL SIMULATION RUNNER"
echo "========================================"
echo "Workers:             $NUM_WORKERS"
echo "Trials per worker:   $TRIALS_PER_WORKER"
echo "Total trials:        $TOTAL_TRIALS"
echo "========================================"
echo ""

# Export environment variables for docker-compose
# Note: These match DEFAULT_* constants in run_batch_simulations.py
export SIM_STUDY="monte_carlo"
export SIM_TRIALS="$TRIALS_PER_WORKER"
export SIM_PROJECTS="${SIM_PROJECTS:-100}"
export SIM_JUDGES="${SIM_JUDGES:-50}"
export SIM_COMPARISONS="${SIM_COMPARISONS:-1000}"
export ENABLE_PROJECT_SYNC="false"

# Clean up old results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./simulation_results/parallel_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Starting $NUM_WORKERS parallel simulation workers..."
echo ""

# Run simulations in parallel (detached mode)
docker-compose -f docker-compose.parallel.yml up --scale sim=$NUM_WORKERS -d

# Wait for all containers to complete
echo "Waiting for all workers to complete..."
echo ""

# Get container IDs
CONTAINER_IDS=$(docker-compose -f docker-compose.parallel.yml ps -q)

# Monitor and wait for each container
for CONTAINER_ID in $CONTAINER_IDS; do
    CONTAINER_NAME=$(docker inspect --format='{{.Name}}' $CONTAINER_ID | sed 's/\///')
    echo "Waiting for $CONTAINER_NAME..."
    docker wait $CONTAINER_ID > /dev/null 2>&1
    EXIT_CODE=$(docker inspect --format='{{.State.ExitCode}}' $CONTAINER_ID)
    if [ "$EXIT_CODE" -eq "0" ]; then
        echo "  ✓ $CONTAINER_NAME completed successfully"
    else
        echo "  ✗ $CONTAINER_NAME failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "All workers finished. Merging results..."

# Merge results
python3 - <<PYTHON_SCRIPT
import json
import glob
import numpy as np
import os

# Find all monte_carlo_trials.json files in worker directories
results_dir = "$OUTPUT_DIR"
worker_dirs = glob.glob("simulation_results/*/batch_*/monte_carlo_trials.json")

if not worker_dirs:
    print("Warning: No results found to merge")
    exit(0)

print(f"Found {len(worker_dirs)} result files")

all_trials = []
for result_file in worker_dirs:
    try:
        with open(result_file) as f:
            data = json.load(f)
            trials = data.get('trials', [])
            all_trials.extend(trials)
            print(f"  Loaded {len(trials)} trials from {result_file}")
    except Exception as e:
        print(f"  Error loading {result_file}: {e}")

if not all_trials:
    print("No trials found")
    exit(1)

print(f"\nTotal trials loaded: {len(all_trials)}")

# Calculate merged statistics
spearmans = [t['spearman'] for t in all_trials]
kendalls = [t['kendall_tau'] for t in all_trials]
top5s = [t['top_5_accuracy'] for t in all_trials]
top5_in_10s = [t['top_5_in_top_10'] for t in all_trials]
rank_errors = [t['mean_rank_error'] for t in all_trials]

merged = {
    'timestamp': '$TIMESTAMP',
    'num_workers': $NUM_WORKERS,
    'trials_per_worker': $TRIALS_PER_WORKER,
    'total_trials': len(all_trials),
    'trials': all_trials,
    'summary_statistics': {
        'spearman_mean': float(np.mean(spearmans)),
        'spearman_std': float(np.std(spearmans)),
        'spearman_min': float(np.min(spearmans)),
        'spearman_max': float(np.max(spearmans)),
        'kendall_mean': float(np.mean(kendalls)),
        'kendall_std': float(np.std(kendalls)),
        'top5_mean': float(np.mean(top5s)),
        'top5_std': float(np.std(top5s)),
        'top5_in_top10_mean': float(np.mean(top5_in_10s)),
        'top5_in_top10_std': float(np.std(top5_in_10s)),
        'rank_error_mean': float(np.mean(rank_errors)),
        'rank_error_std': float(np.std(rank_errors))
    }
}

# Save merged results
os.makedirs('$OUTPUT_DIR', exist_ok=True)
output_file = '$OUTPUT_DIR/merged_results.json'

with open(output_file, 'w') as f:
    json.dump(merged, f, indent=2)

# Print summary
print("\n" + "="*70)
print("MERGED RESULTS SUMMARY")
print("="*70)
print(f"Total trials:         {len(all_trials)}")
print(f"Spearman correlation: {merged['summary_statistics']['spearman_mean']:.3f} ± {merged['summary_statistics']['spearman_std']:.3f}")
print(f"  Range: [{merged['summary_statistics']['spearman_min']:.3f}, {merged['summary_statistics']['spearman_max']:.3f}]")
print(f"Kendall tau:          {merged['summary_statistics']['kendall_mean']:.3f} ± {merged['summary_statistics']['kendall_std']:.3f}")
print(f"Top-5 accuracy:       {merged['summary_statistics']['top5_mean']:.1%} ± {merged['summary_statistics']['top5_std']:.1%}")
print(f"Top-5 in Top-10:      {merged['summary_statistics']['top5_in_top10_mean']:.1%} ± {merged['summary_statistics']['top5_in_top10_std']:.1%}")
print(f"Mean rank error:      {merged['summary_statistics']['rank_error_mean']:.1f} ± {merged['summary_statistics']['rank_error_std']:.1f}")
print("="*70)
print(f"\nResults saved to: {output_file}")
PYTHON_SCRIPT

# Clean up
echo ""
echo "Cleaning up containers..."
docker-compose -f docker-compose.parallel.yml down

echo ""
echo "✓ Parallel simulation complete!"
echo "✓ Results in: $OUTPUT_DIR/merged_results.json"
