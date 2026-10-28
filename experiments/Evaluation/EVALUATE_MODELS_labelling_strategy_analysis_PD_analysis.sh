
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

# 4th argument controls the NEW paired-difference table:
#   none                                      -> disabled (default)
#   all                                       -> all pairwise comparisons
#   "(dinov3_vits16,dinov2_vits14)"          -> selected pair
#   "(random,prioritized)"                    -> selected pair
#   "(LinearSVC,Svc)"                         -> selected pair
ENABLE_PAIRED_DIFFERENCE="${4:-none}"

save_dir="${RESULTS_DIR}/classification/labelling_strategy_analysis/figures/$DATASET"

python FSCBI/evaluation/evaluate_classification_step_PD_analysis.py \
    --dataset "$DATASET" \
    --results_dir ${RESULTS_DIR}/classification/labelling_strategy_analysis/$DATASET \
    --save_dir "$save_dir/$FILTER_NAME" \
    --filter "$FILTER" \
    --mode "$MODE" \
    --show_what "F1+Acc" \
    --top_3_coloring \
    --enable_paired_difference "$ENABLE_PAIRED_DIFFERENCE" \
    --paired \
    --paired_summary \
    --ci

echo "Opening results directory: $save_dir/$FILTER_NAME"

gio open "$save_dir/$FILTER_NAME"
