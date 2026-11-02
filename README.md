# FSCBI — Few-Shot Classification for Benthic Imagery

Experiment code for few-shot classification (FSC) of benthic imagery with frozen foundation-model backbones, compared against traditional fine-tuning (TFM).

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
  - [Installation](#installation)
  - [Configure paths](#configure-paths)
  - [Download models](#download-models)
  - [Download data](#download-data)
- [Models](#models)
- [Experiments](#experiments)
  1. [Model family analysis](#1-model-family-analysis)
  2. [Model scale analysis](#2-model-scale-analysis)
  3. [Labelling strategy analysis](#3-labelling-strategy-analysis)
  4. [Adaptation method comparison](#4-adaptation-method-comparison)
  5. [Patch size analysis](#5-patch-size-analysis)
  6. [Annotation time analysis](#6-annotation-time-analysis)
- [Paired performance difference analysis](#paired-performance-difference-analysis)

---

## Repository layout

```
FSCBI/                  Python package
├── dataset/            COCO loading, object cropping, 50/50 splits
├── models/             adaptation algorithms (CosSim, kNN, LinearSVC, SVC, fine-tuning) and data selection
├── pipelines/          end-to-end pseudo-labelling pipelines (FSC and traditional fine-tuning)
├── evaluation/         aggregation across seeds, tables, plots, paired statistics
└── utils/              seeding / GPU helpers
scripts/                entry points called by the experiment shell scripts
experiments/
├── Classification/     RUN_* scripts that train/evaluate classifiers
├── Evaluation/         EVALUATE_MODELS_* scripts and time-saving analysis
├── EDA/                dataset statistics and class-distribution plots
└── others/             helper scripts
config/config.sh        shared paths, seeds and sample budgets
results_papers/         logs of the results reported in the paper
```

## Setup

### Installation

1. Clone the repository and move into it:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create the conda environment (named `FSCBI`) from `environment.yml` and activate it:

   ```bash
   conda env create -f environment.yml
   conda activate FSCBI
   ```

3. Install the `FSCBI` package in editable mode, so the scripts can `import FSCBI`:

   ```bash
   pip install -e .
   ```

   `environment.yml` already runs this step (`-e .` in its `pip` section). Run it again if you created the environment some other way, or if `import FSCBI` fails.

4. Make the experiment scripts executable:

   ```bash
   chmod +x experiments/*/*.sh
   ```

### Configure paths

All experiment scripts source `config/config.sh`. **Before running anything, open `config/config.sh` and replace the directory paths with paths on your machine.** The values in the repository are the author's paths and will not exist on your system.

```bash
RESULTS_DIR="/path/to/results"          # where classification results and figures are written
INPUT_DIR="/path/to/datasets"           # root folder of the datasets (one subfolder per dataset)
scratch_dir="/path/to/scratch"          # cache for cropped objects (see "Download data")
models_dir="/path/to/models"            # downloaded model weights
conda_source="/path/to/miniconda3/etc/profile.d/conda.sh"   # conda activation script
conda_env="FSCBI"                       # name of the conda environment
```

| Variable | Purpose |
|---|---|
| `RESULTS_DIR` | Where classification results and figures are written |
| `INPUT_DIR` | Root folder of the datasets (one subfolder per dataset) |
| `scratch_dir` | Cache for cropped objects |
| `models_dir` | Downloaded model weights |
| `conda_source`, `conda_env` | Conda activation script and environment name |
| `seed_options` | The 10 seeds every experiment is repeated over |
| `ns_options` | Annotation budgets (number of labelled samples) |
| `min_samples_per_class` | Classes with fewer samples are dropped |

Keep `seed_options`, `ns_options` and `min_samples_per_class` unchanged to reproduce the paper.

All commands below are run from the repository root.

### Download models
```
# Model weights
aws s3 sync s3://fsc-benthic-imagery/models/ ./models/ --no-sign-request
```
### Download data

The experiments run on object crops cut from the annotated images. You can get them in two ways:

- **Option A (recommended):** download the ready-made crops and start the experiments directly.
- **Option B:** download the original datasets and generate the crops yourself.

| Dataset | Name in the paper | Original source |
|---|---|---|
| `18167` | Dataset A | https://squidle.org/geodata/annotation_set/18167 |
| `18290` | Dataset B | https://squidle.org/geodata/annotation_set/18290 |
| `16817` | Dataset C | https://squidle.org/geodata/annotation_set/16817 |

`16817` is used with several patch sizes (`16817_0_01`, `16817_0_02`, `16817_0_03`, `16817_0_04`, `16817_0_05`, `16817_0_1`, `16817_0_15`). Each patch size is a separate dataset folder. `16817_0_05` is the default in experiments 1–4 and 6.

#### Option A: download the pre-processed crops (recommended)

Download the formatted dataset from and extract it. It contains two parts:
```
# COCO annotations and datasets

aws s3 sync s3://fsc-benthic-imagery/SQ_datasets/ ./SQ_datasets/ --no-sign-request

# Preprocessed object crops

aws s3 sync s3://fsc-benthic-imagery/scratch_space/ ./scratch_space/ --no-sign-request
```

1. The COCO annotation files. Place them under `INPUT_DIR`.
2. The cropped objects. Place them under `scratch_dir`.

```
INPUT_DIR/                                   # config.sh: INPUT_DIR
├── 18167/
│   └── <name>_all.json                      # COCO annotations
├── 18290/
│   └── <name>_all.json
├── 16817_0_05/
│   └── <name>_all.json
└── ...                                      # other 16817_* patch sizes

scratch_dir/                                 # config.sh: scratch_dir
├── 18167_cropped_objects_temp/
│   ├── <Category_Name>/
│   │   ├── obj_<annotation_id>.png
│   │   └── ...
│   └── ...
├── 18290_cropped_objects_temp/
├── 16817_0_05_cropped_objects_temp/
└── ...
```

The original images are not needed for this option. The scripts still read the COCO JSON (to get labels and filter classes), then load the matching crops from `scratch_dir`. Keep the folder names exactly as shown: the crop folder must be `<dataset>_cropped_objects_temp`, and the JSON file name must contain `all`.

With the default setting (`overwrite=False`), existing crops are reused and nothing is regenerated, so no code changes are needed.

#### Option B: generate the crops from the original datasets

Download each dataset from its original source (table above). Arrange it as a COCO JSON plus an `all/` folder that holds the images:

```
INPUT_DIR/
├── 18167/
│   ├── <name>_all.json                      # COCO annotations (file name must contain "all")
│   └── all/                                 # images referenced by "file_name" in the JSON
│       ├── <image>.jpg
│       └── ...
├── 18290/
│   └── ...
└── 16817_0_05/
    └── ...
```

The crops are created by the classification scripts the first time they run. **Crop creation is off by default.** With `overwrite=False`, the scripts only load crops that already exist in `scratch_dir`, so on a fresh setup they would find no samples. To generate the crops:

1. Set `overwrite=True` in the `prepare_cropped_objects(...)` call in `scripts/run_classification_step_FSC.py`:

   ```python
   metadata_all1 = prepare_cropped_objects(
       data_json=data_json_filtered,
       images_root=images_root_all,
       output_root=output_root,
       overwrite=True,          # was False: create the crops from the images
       ...
   )
   ```

   Do the same in `scripts/run_classification_step_TFM.py` if you run the TFM experiments first.

2. Run any classification command once per dataset, for example:

   ```bash
   ./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh dinov3_vits16 random linearsvc 18167
   ```

   This writes the crops to `scratch_dir/<dataset>_cropped_objects_temp/`. Crops are cut from the bounding box, which is enlarged by 20% (`--bbox_expand_ratio 0.2`), with the original background kept.

3. Set `overwrite` back to `False`. Otherwise every run regenerates all crops, which is slow but gives the same result.

## Models

Parameter counts are based on the **vision encoder only**. For CLIP, only the visual backbone (`model.visual`) is counted, not the text encoder.

| Config | Scale | ResNet | DINOv2 | DINOv3 | CLIP (visual only) |
|---|---|---|---|---|---|
| Small | ~20M–40M | ResNet50 (~23.5M) | ViT-S/14 (~21M) | ViT-S/16 (~21M) | RN50 visual (~38M) |
| Medium | ~60M–90M | ResNet152 (~58M) | ViT-B/14 (~86M) | ViT-B/16 (~86M) | ViT-B/16 visual (~86M) |
| Large | >~300M | — | ViT-L/14 (~300M) | ViT-L/16 (~300M) | ViT-L/14 visual (~304M) |

### Recommended comparison sets

| Purpose | Models |
|---|---|
| Most balanced small-scale comparison | ResNet50, DINOv2 ViT-S/14, DINOv3 ViT-S/16, CLIP RN50 visual |
| Best overall fair comparison | ResNet152, DINOv2 ViT-B/14, DINOv3 ViT-B/16, CLIP ViT-B/16 visual |
| Foundation-model scale comparison | DINOv2 ViT-L/14, DINOv3 ViT-L/16, CLIP ViT-L/14 visual |

---

## Experiments

Every experiment has two steps. First, a `RUN_*` script trains and predicts for every seed and budget. Second, an `EVALUATE_MODELS_*` script aggregates the results into tables and figures.

The classification scripts take positional arguments:

```
<model_name> <data_collection_strategy> <adaptation_method> <dataset>
```

The datasets are `18167`, `18290` and `16817_0_05`. The suffix on `16817` sets the patch size; see [experiment 5](#5-patch-size-analysis).

### 1. Model family analysis

| Setting | Value |
|---|---|
| Labelling strategy | random |
| Adaptation algorithm | LinearSVC |
| Models | resnet50, dinov2_vits14, dinov3_vits16, clip_vitb16, clip_rn50 |
| Feature scaling | L2 for resnet50, dinov2_vits14, dinov3_vits16; StandardScaler for clip_vitb16, clip_rn50 |

The feature scaling is applied to the embeddings inside the LinearSVC pipeline. It differs per backbone family and was selected on separate validation sets (see the supplementary materials). There are two scripts:

- `RUN_FSC_model_family_analysis_l2.sh`: L2 normalization, used for ResNet, DINOv2 and DINOv3. Results are written to `<model>_linearsvc_random_FSC_l2/`.
- `RUN_FSC_model_family_analysis_standard.sh`: StandardScaler, used for CLIP. Results are written to `<model>_linearsvc_random_FSC_standard/`.

**Run: 18167**

```bash
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh resnet50 random linearsvc 18167
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh dinov2_vits14 random linearsvc 18167
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh dinov3_vits16 random linearsvc 18167
./experiments/Classification/RUN_FSC_model_family_analysis_standard.sh clip_vitb16 random linearsvc 18167
./experiments/Classification/RUN_FSC_model_family_analysis_standard.sh clip_rn50 random linearsvc 18167
```

**Run: 18290**

```bash
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh resnet50 random linearsvc 18290
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh dinov2_vits14 random linearsvc 18290
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh dinov3_vits16 random linearsvc 18290
./experiments/Classification/RUN_FSC_model_family_analysis_standard.sh clip_vitb16 random linearsvc 18290
./experiments/Classification/RUN_FSC_model_family_analysis_standard.sh clip_rn50 random linearsvc 18290
```

**Run: 16817**

```bash
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh resnet50 random linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh dinov2_vits14 random linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_model_family_analysis_l2.sh dinov3_vits16 random linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_model_family_analysis_standard.sh clip_vitb16 random linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_model_family_analysis_standard.sh clip_rn50 random linearsvc 16817_0_05
```

Both scripts write to the same `model_family_analysis/<dataset>/` folder, so the evaluation below compares all five models together.

**Evaluate**

```bash
./experiments/Evaluation/EVALUATE_MODELS_model_family_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_family_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_family_analysis.sh 16817_0_05 FSC
```

### 2. Model scale analysis

| Setting | Value |
|---|---|
| Labelling strategy | random |
| Adaptation algorithm | LinearSVC |
| Models | dinov3_vits16, dinov3_vitb16, dinov3_vitl16 |

**Run: 18167**

```bash
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vits16 random linearsvc 18167
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vitb16 random linearsvc 18167
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vitl16 random linearsvc 18167
```

**Run: 18290**

```bash
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vits16 random linearsvc 18290
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vitb16 random linearsvc 18290
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vitl16 random linearsvc 18290
```

**Run: 16817**

```bash
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vits16 random linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vitb16 random linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_model_scale_analysis.sh dinov3_vitl16 random linearsvc 16817_0_05
```

**Evaluate**

```bash
./experiments/Evaluation/EVALUATE_MODELS_model_scale_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_scale_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_scale_analysis.sh 16817_0_05 FSC
```

### 3. Labelling strategy analysis

| Setting | Value |
|---|---|
| Labelling strategy | prioritized vs. random |
| Adaptation algorithm | LinearSVC |
| Model | dinov3_vits16 |

**Run: 18167**

```bash
./experiments/Classification/RUN_FSC_labelling_strategy_analysis.sh dinov3_vits16 prioritized linearsvc 18167
./experiments/Classification/RUN_FSC_labelling_strategy_analysis.sh dinov3_vits16 random linearsvc 18167
```

**Run: 18290**

```bash
./experiments/Classification/RUN_FSC_labelling_strategy_analysis.sh dinov3_vits16 prioritized linearsvc 18290
./experiments/Classification/RUN_FSC_labelling_strategy_analysis.sh dinov3_vits16 random linearsvc 18290
```

**Run: 16817**

```bash
./experiments/Classification/RUN_FSC_labelling_strategy_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_labelling_strategy_analysis.sh dinov3_vits16 random linearsvc 16817_0_05
```

**Evaluate**

```bash
./experiments/Evaluation/EVALUATE_MODELS_labelling_strategy_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_labelling_strategy_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_labelling_strategy_analysis.sh 16817_0_05 FSC
```

### 4. Adaptation method comparison

| Setting | Value |
|---|---|
| Labelling strategy | prioritized |
| Adaptation algorithms | linearsvc, svc, cossim, knn |
| Model | dinov3_vits16 |

**Run: 18167**

```bash
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized linearsvc 18167
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized svc 18167
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized cossim 18167
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized knn 18167
```

**Run: 18290**

```bash
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized linearsvc 18290
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized svc 18290
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized cossim 18290
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized knn 18290
```

**Run: 16817**

```bash
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized svc 16817_0_05
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized cossim 16817_0_05
./experiments/Classification/RUN_FSC_adaptation_method_analysis.sh dinov3_vits16 prioritized knn 16817_0_05
```

**Evaluate**

```bash
./experiments/Evaluation/EVALUATE_MODELS_adaptation_method_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_adaptation_method_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_adaptation_method_analysis.sh 16817_0_05 FSC
```

### 5. Patch size analysis

| Setting | Value |
|---|---|
| Dataset | 16817 (patch sizes `0_01` … `0_15`) |
| Labelling strategy | prioritized |
| Adaptation algorithm | LinearSVC |
| Model | dinov3_vits16 |

**Run**

```bash
./experiments/Classification/RUN_FSC_patch_size_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_01
./experiments/Classification/RUN_FSC_patch_size_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_02
./experiments/Classification/RUN_FSC_patch_size_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_03
./experiments/Classification/RUN_FSC_patch_size_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_04
./experiments/Classification/RUN_FSC_patch_size_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_05
./experiments/Classification/RUN_FSC_patch_size_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_1
./experiments/Classification/RUN_FSC_patch_size_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_15
```

**Evaluate**

```bash
./experiments/Evaluation/EVALUATE_MODELS_patch_size_analysis.sh 16817
```

### 6. Annotation time analysis

Compares FSC with traditional fine-tuning (TFM: partial fine-tuning, `pft`) in terms of the annotation time saved. Both runs write to the same `annotation_time_analysis/<dataset>/` folder, and the evaluation compares them there, so run both.

| Setting | FSC | TFM |
|---|---|---|
| Labelling strategy | prioritized | random |
| Adaptation algorithm | LinearSVC | pft (partial fine-tuning) |
| Model | dinov3_vits16 | dinov3_vits16 |
| Labelled samples (MS) | 100 | 1000 training + 100 validation = 1100 |
| Early stopping | — | on validation accuracy |
| Training augmentation | — | on |

**Run: FSC**

```bash
./experiments/Classification/RUN_FSC_annotation_time_analysis.sh dinov3_vits16 prioritized linearsvc 18167

./experiments/Classification/RUN_FSC_annotation_time_analysis.sh dinov3_vits16 prioritized linearsvc 18290

./experiments/Classification/RUN_FSC_annotation_time_analysis.sh dinov3_vits16 prioritized linearsvc 16817_0_05
```

**Run: TFM**

The TFM script reads two environment variables. `EARLY_STOPPING` can be `f1`, `accuracy` or `none`; the default `none` trains for all epochs. `AUGMENT=1` turns on training-crop augmentation. The paper uses early stopping on validation accuracy with augmentation, so set both:

```bash
EARLY_STOPPING=accuracy AUGMENT=1 ./experiments/Classification/RUN_TFM_annotation_time_analysis.sh dinov3_vits16 random pft 18167

EARLY_STOPPING=accuracy AUGMENT=1 ./experiments/Classification/RUN_TFM_annotation_time_analysis.sh dinov3_vits16 random pft 18290

EARLY_STOPPING=accuracy AUGMENT=1 ./experiments/Classification/RUN_TFM_annotation_time_analysis.sh dinov3_vits16 random pft 16817_0_05
```

Results are written to `<model>_pft_random_TFM_accuracy_aug1/`.

**Evaluate: time-saving analysis per dataset**

```bash
python ./experiments/Evaluation/time_saving_analysis.py     --dataset 18167     --results_dir /media/hassan/Data_vault1/datasets_hassan/Paper_results_finalV2/classification/annotation_time_analysis/18167     --save_dir ./plots     --S 3588 

python ./experiments/Evaluation/time_saving_analysis.py     --dataset 18290     --results_dir /media/hassan/Data_vault1/datasets_hassan/Paper_results_finalV2/classification/annotation_time_analysis/18290     --save_dir ./plots     --S 3613 

python ./experiments/Evaluation/time_saving_analysis.py     --dataset 16817_0_05     --results_dir /media/hassan/Data_vault1/datasets_hassan/Paper_results_finalV2/classification/annotation_time_analysis/16817_0_05     --save_dir ./plots     --S 9891 
```

**Evaluate: FSC vs. TFM time saved**

```bash
python experiments/Evaluation/plot_time_saved_fsc_vs_tfm_std.py     
```

**Evaluate: FSC vs. TFM accuracy**

```bash
./experiments/Evaluation/EVALUATE_MODELS_FSC_TFM.sh  18167

./experiments/Evaluation/EVALUATE_MODELS_FSC_TFM.sh  18290

./experiments/Evaluation/EVALUATE_MODELS_FSC_TFM.sh  16817_0_05
```

## Paired performance difference analysis

The paper's paired performance difference analysis (Table 6) and the complete paired results in the supplementary materials are produced by the `*_PD_analysis.sh` scripts in `experiments/Evaluation/`. They read the same results as the `EVALUATE_MODELS_*` scripts of experiments 1–5, so run the corresponding classification scripts first.

The configurations are compared over the 10 matched repetitions (the same seeds give the same 50/50 splits). For every pair of configurations, the scripts report:

- the paired difference in accuracy and macro-F1 at each support-set size M, with a 95% t confidence interval and a paired t-test;
- the paired difference pooled over M = 20–100, as reported in Table 6: accuracy and macro-F1 are first averaged over M within each repetition, then the mean difference and its 95% CI are computed over the 10 repetitions.

Tables are printed to the terminal and saved as CSV files (`<dataset>_paired_differences.csv`, `<dataset>_paired_summary_over_M.csv`) under `RESULTS_DIR/classification/<experiment>/figures/<dataset>/`.

```bash
# Backbone family (experiment 1)
./experiments/Evaluation/EVALUATE_MODELS_model_family_analysis_PD_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_family_analysis_PD_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_family_analysis_PD_analysis.sh 16817_0_05 FSC

# Model scale (experiment 2)
./experiments/Evaluation/EVALUATE_MODELS_model_scale_analysis_PD_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_scale_analysis_PD_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_model_scale_analysis_PD_analysis.sh 16817_0_05 FSC

# Labelling strategy (experiment 3)
./experiments/Evaluation/EVALUATE_MODELS_labelling_strategy_analysis_PD_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_labelling_strategy_analysis_PD_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_labelling_strategy_analysis_PD_analysis.sh 16817_0_05 FSC

# Adaptation method (experiment 4)
./experiments/Evaluation/EVALUATE_MODELS_adaptation_method_analysis_PD_analysis.sh 18167 FSC
./experiments/Evaluation/EVALUATE_MODELS_adaptation_method_analysis_PD_analysis.sh 18290 FSC
./experiments/Evaluation/EVALUATE_MODELS_adaptation_method_analysis_PD_analysis.sh 16817_0_05 FSC

# Patch size (experiment 5, supplementary materials)
./experiments/Evaluation/EVALUATE_MODELS_patch_size_analysis_PD_analysis.sh 16817 FSC
```

The scripts take positional arguments `<dataset> [mode] [filter] [pair]`:

- `mode`: `FSC` (default) uses support-set sizes M = 20–100.
- `filter`: optional expression that keeps only some result folders, e.g. `"keep dinov3 or dinov2"`. Leave it as `""` to compare all folders.
- `pair`: optional extra table for selected pairs, e.g. `all`, `"(random,prioritized)"` or `"(dinov3_vits16,dinov2_vits14)"`. The default is `none`.
