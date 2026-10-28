import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
import torchvision.transforms as T
import argparse
from FSCBI.dataset.dataset_utils import *
from FSCBI.evaluation.evaluation_utils import *
from FSCBI.utils.gpu_utils import *
from FSCBI.pipelines.backbone import *
from FSCBI.pipelines.backbone_traditional_finetuning import *
import time
import os
import json
from collections import defaultdict
import random
from collections import Counter
import threading
import time
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_dict_to_json(data_dict, output_path):
    """Save a dictionary to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=4)


def gpu_mem_logger(interval=1.0, stop_event=None, stats=None):
    while not stop_event.is_set():
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            shell=True
        ).decode().strip()

        mem = float(out)
        # print("[GPU]", mem, "MiB")

        # initialize on first sample
        if stats["initial"] is None:
            stats["initial"] = mem
            stats["peak"] = mem
        else:
            if mem > stats["peak"]:
                stats["peak"] = mem

        time.sleep(interval)

def get_dino_transform(model_name=""):
    if "clip" in model_name.lower():  # OpenAI CLIP normalisation (also used by PFT/VFT, which reuse this transform)
        return T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToImage(),                               # replaces implicit PIL->image handling
        T.ToDtype(torch.float32, scale=True),      # replaces ToTensor() behavior
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])



def get_clusters_info(num_categ, N_samples):
    n_test_clusters = num_categ * 2
    reps_per_cluster_base = N_samples // n_test_clusters
    remainder = N_samples % n_test_clusters
    reps_per_cluster = [reps_per_cluster_base] * n_test_clusters
    for i in range(remainder):
        reps_per_cluster[i] += 1
    return n_test_clusters, reps_per_cluster
# Print a sample from train_loader
def print_sample_dict(loader, num_samples=1):
    """Print the dictionary keys of samples from a DataLoader."""
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        print(f"\n{'='*50}")
        print(f"Batch {i+1}:")
        print(f"{'='*50}")
        if isinstance(batch, dict):
            print(f"Dictionary keys: {list(batch.keys())}")
        elif isinstance(batch, (tuple, list)):
            print(f"Batch is a {type(batch).__name__} with {len(batch)} items")
        else:
            print(f"Batch type: {type(batch)}")















PTA_dict = {"dinov2": "DINOv2",
            "dinov3": "DINOv3",    
            "clip": "CLIP",
            "resnet":"CE",
            "cnn":"CE",}

AT_dict = {"dino": "VIT",
            "clip": "VIT",
            "resnet" :"CNN",
            "cnn" :"CNN"}



AN_dict = {"vits16": "VIT-16s",
            "vitb16": "VIT-16b",
            "vitl16": "VIT-16l",
            "vits14": "VIT-14s",
            "vitb14": "VIT-14b",
            "vitl14" :"VIT-14l",
            "vit-s16": "VIT-16s",
            "vit-b16": "VIT-16b",
            "vit-l16": "VIT-16l",
            "vit-s14": "VIT-14s",
            "vit-b14": "VIT-14b",
            "vitl14" :"VIT-14l",
            "resnet18": "ResNet18",
            "resnet34" :"ResNet34",
            "resnet50" :"ResNet50",
            "resnet101" :"ResNet101",
            "cnn":"ResNet",
            }
            
DS_dict = {
        "prioritized": "PR",
        "random|":"RA"
        }
                        
AM_dict = {"cossim": "CC",
        "linearsvc": "LR",
        "svc": "NLR",
        "eucdis": "NCC",
        "resnet": "PFT",
        }
            

def filter_json(
    data_json = None,
    min_samples_per_class=None,
    exclude_labels=None,
    plot_histogram=False,
    print_=False,
):
    """
    Analyze class distribution in COCO-format JSON data, optionally filter by min samples/class and exclude labels,
    and optionally plot histograms before and after filtering.

    Args:
        data (dict): COCO-format JSON data.
        min_samples_per_class (int, optional): Minimum number of samples required per class.
        exclude_labels (list, optional): List of label names to exclude.
        plot_histogram (bool, optional): Whether to plot histograms before and after filtering.
    """

    # Build category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in data_json['categories']}
    # Count samples per class
    class_counts = Counter()
    for ann in data_json['annotations']:
        class_counts[cat_id_to_name[ann['category_id']]] += 1
    if print_:
        print("Class distribution (before filtering):")
        for label, count in class_counts.items():
            print(f"{label}: {count}")
        print(f"Total classes: {len(class_counts)}")

    if plot_histogram:
        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title("Class Distribution Before Filtering")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.title(f"Class Distribution Before Filtering, N_classes = {len(class_counts)}, N_samples = {sum(class_counts.values())}")
        plt.show()

    # Filtering
    filtered_labels = set(class_counts.keys())
    if min_samples_per_class is not None:
        filtered_labels = {label for label in filtered_labels if class_counts[label] >= min_samples_per_class}
    if exclude_labels is not None:
        filtered_labels = filtered_labels - set(exclude_labels)

    # Filter annotations
    filtered_annotations = [
        ann for ann in data_json['annotations']
        if cat_id_to_name[ann['category_id']] in filtered_labels
    ]
    filtered_class_counts = Counter()
    for ann in filtered_annotations:
        filtered_class_counts[cat_id_to_name[ann['category_id']]] += 1
    if print_:
        print("\nClass distribution (after filtering):")
        for label, count in filtered_class_counts.items():
            print(f"{label}: {count}")
        print(f"Total classes after filtering: {len(filtered_class_counts)}")

    if plot_histogram:
        plt.figure(figsize=(10, 5))
        plt.bar(filtered_class_counts.keys(), filtered_class_counts.values())
        plt.title("Class Distribution After Filtering")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Update data_json dict with filtered annotations and filtered categories
    filtered_cat_ids = {cat_id for cat_id, name in cat_id_to_name.items() if name in filtered_labels}
    data_json['annotations'] = filtered_annotations
    data_json['categories'] = [cat for cat in data_json['categories'] if cat['id'] in filtered_cat_ids]

    return data_json
def save_metadata_to_json(metadata, output_path):
    """Save a list of dictionaries (metadata) to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_metadata_from_json(json_path):
    """Load a list of dictionaries (metadata) from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def split_train_val(dataset, val_ratio=0.2, seed=42):
    """Split a dataset into training and validation sets."""
    total_size = len(dataset)
    indices = list(range(total_size))
    random.Random(seed).shuffle(indices)

    split_idx = int(total_size * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    return train_subset, val_subset







import random
import math
from collections import defaultdict
import torch



def split_by_sample_count(dataset, sample_count, seed=42, label_key="category_id"):
    """
    Split a dataset into:
      1) sampled_subset
      2) remainder_subset

    using stratified sampling based on class labels found in each dataset item.

    Args:
        dataset: PyTorch dataset where dataset[i] returns a dict containing label_key
        sample_count: int, number of samples to put in sampled_subset
        seed: random seed
        label_key: key in dataset item dict used as class label
                   e.g. 'category_id' or 'category_name'

    Returns:
        sampled_subset, remainder_subset
    """

    total_size = len(dataset)
    if not (0 < sample_count < total_size):
        raise ValueError(
            f"sample_count must be between 1 and {total_size - 1}, got {sample_count}"
        )

    rng = random.Random(seed)

    # Extract labels from dataset items
    labels = []
    for i in range(total_size):
        item = dataset[i]

        if not isinstance(item, dict):
            raise TypeError(
                f"dataset[{i}] is not a dict. Expected dict with key '{label_key}'."
            )

        if label_key not in item:
            raise KeyError(
                f"Key '{label_key}' not found in dataset[{i}]. "
                f"Available keys: {list(item.keys())}"
            )

        labels.append(item[label_key])

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    # Shuffle indices within each class
    for label in class_to_indices:
        rng.shuffle(class_to_indices[label])

    # Count samples per class
    class_counts = {label: len(indices) for label, indices in class_to_indices.items()}
    total = sum(class_counts.values())

    # Ideal number of samples per class in sampled subset
    ideal_alloc = {
        label: (count / total) * sample_count
        for label, count in class_counts.items()
    }

    # Start with floor allocation
    sampled_per_class = {
        label: math.floor(v) for label, v in ideal_alloc.items()
    }

    # Distribute remaining samples according to largest remainder
    assigned = sum(sampled_per_class.values())
    remaining = sample_count - assigned

    remainders = sorted(
        [
            (label, ideal_alloc[label] - sampled_per_class[label])
            for label in class_counts
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    for label, _ in remainders:
        if remaining == 0:
            break

        # Do not take more than available in that class
        if sampled_per_class[label] < class_counts[label]:
            sampled_per_class[label] += 1
            remaining -= 1

    # Build sampled and remainder index lists
    sampled_indices = []
    remainder_indices = []

    for label, indices in class_to_indices.items():
        n_take = sampled_per_class[label]
        sampled_indices.extend(indices[:n_take])
        remainder_indices.extend(indices[n_take:])

    # Shuffle final indices
    rng.shuffle(sampled_indices)
    rng.shuffle(remainder_indices)

    sampled_subset = torch.utils.data.Subset(dataset, sampled_indices)
    remainder_subset = torch.utils.data.Subset(dataset, remainder_indices)

    return sampled_subset, remainder_subset





def compute_validation_size(M):
    """
    Compute validation set size based on total labeled budget M.

    Rule:
        V = min(max(20, 0.2 * M), 100)

    Args:
        M (int): total number of labeled samples

    Returns:
        int: validation set size
    """
    if M <= 0:
        raise ValueError("M must be a positive integer")

    V = int(0.2 * M)
    V = max(20, V)
    V = min(100, V)

    return V





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--json_dir', type=str, required=False, help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory containing output files')
    parser.add_argument('--models_dir', type=str, default="./models", help='Directory containing model weights')
    parser.add_argument('--scratch_dir', type=str, default="/tmp", help='Directory for temporary files')
    parser.add_argument('--config_dir', type=str, default="./config", help='Directory for config files')


    parser.add_argument('--N_samples', type=int, default=10, help='Number of samples per class')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_name', type=str, default="DINOv2_vit16s", help='Architecture name for backbone: DINOv2_vit16s, DINOv2_vit14s, DINOv2_vit14b, DINOv2_vit14l, resnet50')



    parser.add_argument('--adaptation_method', type=str, default="SVM", help='Pseudo-labeling method: # one of: "SVC","LinearSVC", "kNN", "EucDis", "CosSim", "RFC", "SK" ')
    parser.add_argument('--feature_scaling', type=str, default="l2", choices=["l2", "standard", "none"], help='Embedding preprocessing inside kNN / LinearSVC / SVC: "l2" (default), "standard" (StandardScaler fitted on the reference set, previous behaviour) or "none"')
    parser.add_argument('--split_by', type=str, default="object", choices=["object", "image"], help='50/50 train/test split unit: "object" (previous behaviour) or "image" (all objects of an image on the same side, no image overlap between train and test)')
    parser.add_argument('--suffix', type=str, default="", help='Suffix to add to output files to distinguish different runs')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--data_collection_strategy', type=str, default="balanced", help='Sampling strategy for finetuning: "balanced" or "unbalanced"')
    parser.add_argument('--bbox_expand_ratio', type=float, default=0.2, help='Bounding box expansion ratio for cropping objects')
    parser.add_argument('--keep_original_background', type=bool, default=True, help='Whether to keep original background when cropping objects')
    parser.add_argument('--enable_split_50_50',    action=argparse.BooleanOptionalAction,    default=True,    help='Whether to split the dataset into 50% train and 50% test or use the entire dataset for both training and testing')
    

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for fine-tuning (only applicable if adaptation_method is a fine-tuning method)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for fine-tuning (only applicable if adaptation_method is a fine-tuning method)')
    parser.add_argument('--tag', type=str, default="", help='Additional tag to add to output files for better organization and identification of different runs (e.g., "run1", "experimentA", etc.)')
    parser.add_argument('--min_samples_per_class', type=int, default=50, help='Minimum number of samples required per class to be included in the dataset (default: 50)')
    parser.add_argument('--exclude_labels', type=str, nargs='*', default=["Unknown", "Unscorable"], help='List of label names to exclude from the dataset (default: ["Unknown", "Unscorable"])')
    parser.add_argument('--val_size', type=int, default=-1, help='Validation set size (default: 50)')
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=False, help='Augment the fine-tuning training crops (random crop, 90-degree rotations, flips, colour jitter); validation and test crops are not augmented')
    parser.add_argument('--early_stopping', type=str, default="none", choices=["f1", "accuracy", "none"], help='Early stopping of fine-tuning on the validation set: "f1" (macro-F1), "accuracy" or "none" (default: train all num_epochs). With f1/accuracy the best validation epoch is restored')
    args = parser.parse_args()

    set_seed(args.seed)





    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> (1) Loading dataset and preparing cropped objects dataset for training and testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Loading dataset in coco format and preparing cropped objects dataset for training and testing the few shot classification step. 
    # The cropping is done based on the bounding box annotations in the coco json files.
    # The resulting cropped objects dataset is then used for both DINO pseudo-labeling and classifier finetuning and evaluation.
    if args.json_dir is None:
        json_dir = args.input_dir
    else:
        json_dir = args.json_dir

    os.makedirs(args.output_dir, exist_ok=True)
    all_json = get_json_files(json_dir)
    
    

    output_root = f"{args.dataset}_cropped_objects_temp"
    output_root = os.path.join(args.scratch_dir, output_root)

    transform_dino = get_dino_transform(args.model_name)

    json_path_all = os.path.join(json_dir, all_json)
    images_root_all = os.path.join(args.input_dir, "all")



    with open(json_path_all, "r") as f:
        data_json = json.load(f)

    num_categ_ = get_num_categories(data_json)
    print(f"Dataset: {args.dataset}, Number of categories BEFORE filtering: {num_categ_}")
    data_json_filtered = filter_json(
                                data_json = data_json,
                                min_samples_per_class=args.min_samples_per_class,
                                exclude_labels=args.exclude_labels,
                                plot_histogram=False,
                                print_=False,
                            )
    # breakpoint()
    num_categ = get_num_categories(data_json_filtered)
    print(f"Dataset: {args.dataset}, Number of categories AFTER filtering: {num_categ}")
    

    metadata_all1 = prepare_cropped_objects(
        data_json=data_json_filtered,
        images_root=images_root_all,
        output_root=output_root,
        overwrite=False,
        keep_original_background=args.keep_original_background,
        bbox_expand_ratio=args.bbox_expand_ratio,
        seed=args.seed,
    )

    print(f"Total number of samples in all dataset (after filtering): {len(metadata_all1)}")
    print(f"type of metadata_all1: {type(metadata_all1)}")
    print(f"tyoe of metadata_all1[0]: {type(metadata_all1[0])}")
    print(f"Example metadata entry: {metadata_all1[0]}")
    metadata_all = metadata_all1
    # Load metadata_all2 from the saved JSON file
    


    # breakpoint()
    dataset_all = CroppedObjectsDataset(metadata_all, transform=transform_dino, augment=False)
    print(f"Total number of samples in all dataset: {len(dataset_all)}")












    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> (2) splitting dataset into training and testing sets 50% each for adjusted cross validation <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    enable_split_50_50 = args.enable_split_50_50
    if enable_split_50_50:
        training_dataset, test_dataset = split_50_50(dataset_all, seed=args.seed, by=args.split_by) # split dataset into 50% train and 50% test for the quantitative analysis
    else:
        training_dataset = dataset_all          # use the entire dataset for training and testing
        test_dataset = dataset_all



    tradidional_finetuning = True
    if tradidional_finetuning:
        
        training_samples_N = args.N_samples
        # validation_samples_N = compute_validation_size(training_samples_N)
        if args.val_size > 0:
            validation_samples_N = args.val_size
        else:
            validation_samples_N = 100
        # print(f"sample training_dataset item {training_dataset[0]}")
        # breakpoint()
        training_dataset, remaining_dataset = split_by_sample_count(training_dataset, training_samples_N, seed=args.seed)

        if len(remaining_dataset) < validation_samples_N:
            print(f"Warning: Not enough samples left for validation after sampling training dataset. Adjusting validation_samples_N to {len(remaining_dataset)}.")
            validation_samples_N = len(remaining_dataset) - 1  
        validation_dataset, _ = split_by_sample_count(remaining_dataset, validation_samples_N, seed=args.seed)



        # so now 0.4 for training, 0.1 for validation, and 0.5 for testing
        val_loader = DataLoader(
            validation_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
        )


    train_loader = DataLoader(
        training_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    # fig = plot_label_distribution_comparison(training_dataset, test_dataset, title1="Training Split 1", title2="Testing Split 2")

    print(f"Train loader length: {len(train_loader)}")
    if tradidional_finetuning:
        print(f"Validation loader length: {len(val_loader)}")
    print(f"Test loader length: {len(test_loader)}")
    # breakpoint()
    n_test_clusters, reps_per_cluster = get_clusters_info(num_categ, args.N_samples)

    








    #Measure GPU memory before starting DINO pseudo-labeling as well as the time taken for the entire DINO pseudo-labeling process (including feature extraction, pseudo-label generation, and evaluation)
    stop_event = threading.Event()
    memory_stats = {"initial": None, "peak": None}
    t = threading.Thread(target=gpu_mem_logger, args=(1.0, stop_event, memory_stats), daemon=True)
    t.start()
    time_start = time.time()




    # Create output directory for DINO results
    save_dir = os.path.join(args.output_dir,f"{args.model_name}_{args.adaptation_method}_{args.data_collection_strategy}_{args.suffix}",f'{args.N_samples}')
    os.makedirs(save_dir, exist_ok=True)
    csv_path_out = os.path.join(save_dir, f"{args.dataset}_{args.model_name}_{args.N_samples}_seed{args.seed}.csv")


  
    
    stats = run_pseudo_labeling_traditional_finetuning(
        train_loader,
        val_loader,
        test_loader,
        args.N_samples,
        csv_path_out,
        num_categ = num_categ,
        n_test_clusters=n_test_clusters,
        reps_per_cluster=reps_per_cluster,
        args = args,
        tradidional_finetuning= tradidional_finetuning
        )        
    time_end = time.time()
    print(f"{args.model_name} pseudo-labeling took {time_end - time_start:.2f} seconds")

    stop_event.set()
    t.join()

    # summary (only if at least one sample exists)
    if memory_stats["initial"] is not None:
        initial = memory_stats["initial"]
        peak = memory_stats["peak"]
        diff = peak - initial

        print("\033[38;5;208m\n=== GPU memory summary ===\033[0m")
        print(f"\033[38;5;208mDelta   : {diff/1024:.2f} GiB\033[0m")
    else:
        print("\033[38;5;208mNo GPU samples were recorded.\033[0m")

    stats['GPU_memory_used'] = diff/1024 if memory_stats["initial"] is not None else None # in GiB
    stats['time_taken_seconds'] = time_end - time_start


    ## pretraining algorithm [DINOv2, DINOv3, CLIP, CE]
    for key in PTA_dict.keys():
        if key in args.model_name.lower():
            stats['PTA'] = PTA_dict[key]
            break


    ## architecture type [VIT, CNN]
    for key in AT_dict.keys():
            if key in args.model_name.lower():
                stats['AT'] = AT_dict[key]
                break
                

    ## architecture name [DINOv2_vit16s, DINOv2_vit14s, DINOv2_vit14b, DINOv2_vit14l, resnet50, etc..]
    for key in AN_dict.keys():
            if key in args.model_name.lower():
                stats['AN'] = AN_dict[key]
                break
                

    ## data collection strategy [balanced, unbalanced, prioritized, etc..]
    stats['DS'] = args.data_collection_strategy


    ## adaptation method [Cosine Similarity, Linear SVC, Non-linear SVC, Euclidean Distance, ResNet classifier finetuning, etc..]            
    stats['AM'] = args.adaptation_method
    
    results = evaluate_pseudo_labels(
        csv_path_out,
        use_names=True,
        normalize_cm="true",   # rows normalized to 1
        figsize=(8, 6),
        cmap="Blues",
        plot_=False,
    )


    stats.update({
        "num_epochs": args.num_epochs,
        "accuracy": results["accuracy"],
        "f1_macro": results["f1_macro"],
        "f1_micro": results["f1_micro"],
        "f1_weighted": results["f1_weighted"],
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "labels": results["labels"],
        "tag": args.tag,
        "min_samples_per_class": args.min_samples_per_class,
        "exclude_labels": args.exclude_labels,
    })
    stats["split_by"] = args.split_by
    stats["augment"] = args.augment
    stats["early_stopping"] = args.early_stopping
    save_dict_to_json(stats, os.path.join(save_dir, f"{args.dataset}_{args.model_name}_{args.N_samples}_seed{args.seed}_stats.json"))




































