#!/bin/bash
# Run multiple simulation containers in parallel
#
# Usage:
#   ./run_parallel_simulations.sh <num_containers> <trials_per_container>
#
# Example:
#   ./run_parallel_simulations.sh 4 25
#   This runs 4 containers, each doing 25 trials = 100 total trials

NUM_CONTAINERS=${1:-4}
TRIALS_PER_CONTAINER=${2:-25}
OUTPUT_DIR="./simulation_results/parallel_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "PARALLEL SIMULATION RUNNER"
echo "========================================="
echo "Containers: $NUM_CONTAINERS"
echo "Trials per container: $TRIALS_PER_CONTAINER"
echo "Total trials: $((NUM_CONTAINERS * TRIALS_PER_CONTAINER))"
echo "Output: $OUTPUT_DIR"
echo ""

# Get the Docker image name
IMAGE_NAME=$(docker ps --filter "name=gavel" --format "{{.Image}}" | head -1)

if [ -z "$IMAGE_NAME" ]; then
    echo "Error: Could not find running gavel container"
    exit 1
fi

echo "Using image: $IMAGE_NAME"
echo ""

# Array to store container IDs
declare -a CONTAINER_IDS

# Start all containers
for i in $(seq 1 $NUM_CONTAINERS); do
    CONTAINER_NAME="gavel-sim-$i-$$"
    DB_NAME="simulation_$i.db"

    echo "Starting container $i/$NUM_CONTAINERS: $CONTAINER_NAME"

    # Run container in background
    CONTAINER_ID=$(docker run -d \
        --name "$CONTAINER_NAME" \
        -v "$(pwd)/$OUTPUT_DIR:/output" \
        -e ENABLE_PROJECT_SYNC=false \
        -e DATABASE_URL="sqlite:////tmp/$DB_NAME" \
        "$IMAGE_NAME" \
        python run_batch_simulations.py \
            --study monte_carlo \
            --num-trials "$TRIALS_PER_CONTAINER" \
            --output-dir "/output/container_$i")

    CONTAINER_IDS+=("$CONTAINER_ID")
done

echo ""
echo "All containers started. Waiting for completion..."
echo ""

# Wait for all containers to finish
for i in "${!CONTAINER_IDS[@]}"; do
    CONTAINER_ID="${CONTAINER_IDS[$i]}"
    CONTAINER_NUM=$((i + 1))

    echo "Waiting for container $CONTAINER_NUM..."
    docker wait "$CONTAINER_ID" > /dev/null

    # Show logs
    echo "Container $CONTAINER_NUM output:"
    docker logs "$CONTAINER_ID" | tail -20
    echo ""

    # Clean up container
    docker rm "$CONTAINER_ID" > /dev/null
done

echo "========================================="
echo "ALL SIMULATIONS COMPLETE!"
echo "========================================="
echo "Results in: $OUTPUT_DIR"
echo ""
echo "Merging results..."

# Create a Python script to merge results
cat > "$OUTPUT_DIR/merge_results.py" << 'PYTHON_SCRIPT'
import json
import glob
import numpy as np

# Find all monte_carlo_trials.json files
files = glob.glob('container_*/batch_*/monte_carlo_trials.json')

all_trials = []
for f in files:
    with open(f) as fp:
        data = json.load(fp)
        all_trials.extend(data['trials'])

# Calculate combined statistics
spearmans = [t['spearman'] for t in all_trials]
top5s = [t['top_5_accuracy'] for t in all_trials]
top5_in_top10s = [t['top_5_in_top_10'] for t in all_trials]

merged = {
    'num_trials': len(all_trials),
    'num_containers': len(files),
    'trials': all_trials,
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

with open('merged_results.json', 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(all_trials)} trials from {len(files)} containers")
print(f"\nSpearman correlation: {merged['summary_statistics']['spearman_mean']:.3f} ± {merged['summary_statistics']['spearman_std']:.3f}")
print(f"Top-5 accuracy:       {merged['summary_statistics']['top5_mean']:.1%} ± {merged['summary_statistics']['top5_std']:.1%}")
print(f"Top-5 in Top-10:      {merged['summary_statistics']['top5_in_top10_mean']:.1%} ± {merged['summary_statistics']['top5_in_top10_std']:.1%}")
PYTHON_SCRIPT

cd "$OUTPUT_DIR" && python merge_results.py
echo ""
echo "Merged results saved to: $OUTPUT_DIR/merged_results.json"
