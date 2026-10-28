
#!/bin/bash
# Load shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

mkdir -p "$RESULTS_DIR"



# Activate conda environment FSCBI
source ${conda_source}
conda activate ${conda_env}


DATASET="${1:-}"
if [ -z "$DATASET" ]; then
    echo "Error: dataset argument required"
    exit 1
fi
MODE="${2:-FSC}"


FILTER="${3:-}"
if [ -n "$FILTER" ]; then
    FILTER_NAME="${FILTER// /_}"
fi

save_dir="${RESULTS_DIR}/classification/adaptation_method_analysis/figures/$DATASET"

python FSCBI/evaluation/evaluate_classification_step.py \
    --dataset "$DATASET" \
    --results_dir ${RESULTS_DIR}/classification/adaptation_method_analysis/$DATASET \
    --save_dir "$save_dir/$FILTER_NAME" \
    --filter "$FILTER" \
    --mode "$MODE" \
    --show_what "F1+Acc" \
    --top_3_coloring

echo "Opening results directory: $save_dir/$FILTER_NAME"

gio open "$save_dir/$FILTER_NAME"
