#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

# Activate conda environment FSCBI
source ${conda_source}
conda activate ${conda_env}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset> [<dataset2> ...]"
    exit 1
fi

SEED=${seed_options[0]}

for DATASET in "$@"; do
    OUT_DIR=${RESULTS_DIR}/dataset_stats/${DATASET}
    mkdir -p "$OUT_DIR"
    echo "=== $DATASET -> $OUT_DIR ==="

    python scripts/visualize_class_distribution.py \
        --input_dir ${INPUT_DIR}/$DATASET/ \
        --dataset $DATASET \
        --output_dir $OUT_DIR \
        --scratch_dir ${scratch_dir} \
        --min_samples_per_class ${min_samples_per_class} \
        --seed $SEED \
        ${EXTRA_ARGS}
done
