#!/bin/bash
# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

mkdir -p "$RESULTS_DIR"
# Activate conda environment FSCBI
source ${conda_source}
conda activate ${conda_env}

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_name> <data_collection_strategy> <adaptation_method> <dataset>"
    exit 1
fi

MODEL_NAME="$1"
DATA_COLLECTION_STRATEGY="$2"
METHOD="$3"
DATASET="$4"
seed_options=${seed_options} 
ns_options=${ns_options}
SUFFIX="${SUFFIX:-FSC}"

for i in {1..10}; do
    SEED=${seed_options[$((i-1))]} 

    for ns in "${ns_options[@]}"; do

        echo "Run $i with seed: $SEED , ns: $ns"
        DIR=${RESULTS_DIR}/classification/model_scale_analysis/${DATASET}/${MODEL_NAME}_${METHOD}_${DATA_COLLECTION_STRATEGY}_${SUFFIX}/$ns

        STATS_FILE="$DIR/${DATASET}_${MODEL_NAME}_${ns}_seed${SEED}_stats.json"
        if [ -f "$STATS_FILE" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "Already done ($STATS_FILE), skipping. Set FORCE=1 to rerun."
            continue
        fi
        python scripts/run_classification_step_FSC.py \
            --input_dir ${INPUT_DIR}/$DATASET/ \
            --dataset $DATASET \
            --output_dir ${RESULTS_DIR}/classification/model_scale_analysis/$DATASET \
            --scratch_dir ${scratch_dir} \
            --models_dir ${models_dir} \
            --N_samples $ns \
            --seed $SEED \
            --model_name $MODEL_NAME \
            --data_collection_strategy $DATA_COLLECTION_STRATEGY \
            --adaptation_method $METHOD \
            --tag $DATASET \
            --min_samples_per_class  ${min_samples_per_class} \
            --suffix "$SUFFIX" \

    done
done

