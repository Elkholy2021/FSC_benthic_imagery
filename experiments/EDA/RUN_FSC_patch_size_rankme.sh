#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

source ${conda_source}
conda activate ${conda_env}

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <model_name> <dataset_prefix> [gt_method] [gt_data_collection_strategy] [gt_model_name]"
    exit 1
fi

MODEL_NAME="$1"
PREFIX="$2"
GT_METHOD="${3:-}"
GT_STRATEGY="${4:-}"
GT_MODEL_NAME="${5:-$MODEL_NAME}"
SUFFIX="FSC"   # same suffix as RUN_FSC_patch_size_analysis.sh


read -ra ALL_SEEDS <<< "${seed_options[*]}"
N_SEEDS=${N_SEEDS:-10}
SEEDS=("${ALL_SEEDS[@]:0:$N_SEEDS}")
read -ra NS_LIST <<< "${ns_options[*]}"
if [ "${#SEEDS[@]}" -lt 1 ]; then
    echo "No seeds found in seed_options (config.sh)"
    exit 1
fi

PATCH_SUFFIXES=(0_01 0_02 0_03 0_04 0_05 0_1 0_15)
DATASETS=()
for p in "${PATCH_SUFFIXES[@]}"; do
    if [ -d "${INPUT_DIR}/${PREFIX}_${p}" ]; then
        DATASETS+=("${PREFIX}_${p}")
    else
        echo "WARNING: ${INPUT_DIR}/${PREFIX}_${p} not found, skipping"
    fi
done
if [ "${#DATASETS[@]}" -lt 2 ]; then
    echo "Need at least 2 patch-size datasets, found ${#DATASETS[@]}"
    exit 1
fi

OUT_DIR=${RESULTS_DIR}/classification/patch_size_analysis/rankme_train_splits/${PREFIX}_${MODEL_NAME}

EMB_DIR=${EMB_DIR:-${RESULTS_DIR}/classification/patch_size_analysis/rankme_train_splits/embeddings}
mkdir -p "$OUT_DIR" "$EMB_DIR"

EXTRA_ARGS=()

# ---- ground truth ----
if [ -n "$GT_METHOD" ] && [ -n "$GT_STRATEGY" ]; then
    # per-seed GT from the FSC runs:
    # ${RESULTS_DIR}/classification/patch_size_analysis/<dataset>/<model>_<method>_<strategy>_FSC/<ns>/*_seed<seed>_stats.json
    GT_RUN="${GT_MODEL_NAME}_${GT_METHOD}_${GT_STRATEGY}_${SUFFIX}"
    EXTRA_ARGS+=(--gt_root "${RESULTS_DIR}/classification/patch_size_analysis" --gt_run "$GT_RUN")
    [ "${#NS_LIST[@]}" -gt 0 ] && EXTRA_ARGS+=(--gt_ns "${NS_LIST[@]}")
    echo "Per-seed GT: ${RESULTS_DIR}/classification/patch_size_analysis/<dataset>/${GT_RUN}/<ns>/"
fi

# Static fallback (same values for every split), only used if no per-seed GT is found.
# Defaults for 16817 = Table S1 averaged over M=20..100.
if [ "$PREFIX" == "16817" ]; then
    : "${REFERENCE_ACC:=16817_0_01:56.74,16817_0_02:61.29,16817_0_03:64.64,16817_0_04:68.06,16817_0_05:65.20,16817_0_1:66.41,16817_0_15:67.98}"
    : "${REFERENCE_F1:=16817_0_01:44.12,16817_0_02:47.00,16817_0_03:52.67,16817_0_04:55.31,16817_0_05:53.31,16817_0_1:54.48,16817_0_15:55.76}"
fi
[ -n "${REFERENCE_ACC}" ] && EXTRA_ARGS+=(--reference "acc=${REFERENCE_ACC}")
[ -n "${REFERENCE_F1}" ]  && EXTRA_ARGS+=(--reference "f1=${REFERENCE_F1}")

echo "Model: $MODEL_NAME"
echo "Seeds (one training split each): ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Embeddings cache: $EMB_DIR"
echo "Output: $OUT_DIR"

python scripts/run_patch_size_rankme.py \
    --input_root ${INPUT_DIR} \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --output_dir "$OUT_DIR" \
    --embeddings_dir "$EMB_DIR" \
    --scratch_dir ${scratch_dir} \
    --models_dir ${models_dir} \
    --model_name $MODEL_NAME \
    --min_samples_per_class ${min_samples_per_class} \
    --bbox_expand_ratio 0.2 \
    --keep_original_background true \
    --enable_split_50_50 true \
    --n_sub ${N_SUB:-50} \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/log.txt"
