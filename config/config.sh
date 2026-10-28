#!/bin/bash

RESULTS_DIR="/media/hassan/Data_vault1/datasets_hassan/Paper_results"
INPUT_DIR="/media/hassan/Data_vault1/datasets_hassan/SQ_datasets"
scratch_dir="/media/hassan/Data_vault1/datasets_hassan/scratch_space"
models_dir="/media/hassan/Data_vault1/datasets_hassan/models"
conda_source="/home/hassan/miniconda3/etc/profile.d/conda.sh"

conda_env="FSCBI"


seed_options=(4729 8156 2341 9087 5623 1504 7892 3465 6218 9741)
ns_options=(20 30 40 50 60 70 80 90 100)


min_samples_per_class=100
