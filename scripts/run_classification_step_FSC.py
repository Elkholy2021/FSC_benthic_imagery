import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
import torchvision.transforms as T
import argparse
from FSCBI.dataset.dataset_utils import *

from FSCBI.evaluation.evaluation_utils import *
from FSCBI.utils.gpu_utils import *
from FSCBI.pipelines.backbone import *
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

def get_transform(model_name):
    if "clip" in model_name.lower():
        print("Using CLIP preprocessing for model:", model_name)
        return T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToImage(),                               # replaces implicit PIL->image handling
            T.ToDtype(torch.float32, scale=True),      # replaces ToTensor() behavior
            T.Normalize(                                #openai clip
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    else: #dinov2, dinov3, resnet
        print("Using ImageNet preprocessing for model:", model_name)
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToImage(),                               # replaces implicit PIL->image handling
            T.ToDtype(torch.float32, scale=True),      # replaces ToTensor() behavior
            T.Normalize(                              #imagenet
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    

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

def split_80_20(dataset, seed=42):
    set_seed(seed)
    random.seed(seed)

    # Group metadata by category_id
    category_groups = defaultdict(list)
    for item in dataset.metadata:
        category_groups[item["category_id"]].append(item)

    # Split each category 80-20
    split1_metadata = []
    split2_metadata = []

    for category_id, items in category_groups.items():
        # Shuffle items within category for randomness
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)

        # Split 80% / 20%
        split_point = int(len(shuffled_items) * 0.8)

        split1_metadata.extend(shuffled_items[:split_point])
        split2_metadata.extend(shuffled_items[split_point:])

    # Create new dataset instances with same transform/augment settings
    dataset1 = CroppedObjectsDataset(
        split1_metadata,
        transform=dataset.transform,
        augment=dataset.augment,
        seed=seed
    )

    dataset2 = CroppedObjectsDataset(
        split2_metadata,
        transform=dataset.transform,
        augment=dataset.augment,
        seed=seed
    )

    return dataset1, dataset2






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
    parser.add_argument('--evaluate_validation', action=argparse.BooleanOptionalAction, default=False, help='Whether to evaluate on the validation set')

    args = parser.parse_args()

    set_seed(args.seed)








    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> (1) Loading dataset and preparing cropped objects dataset for training and testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<


    if args.json_dir is None:
        json_dir = args.input_dir
    else:
        json_dir = args.json_dir

    os.makedirs(args.output_dir, exist_ok=True)
    all_json = get_json_files(json_dir)
    
    

    output_root = f"{args.dataset}_cropped_objects_temp"
    output_root = os.path.join(args.scratch_dir, output_root)

    transform = get_transform(args.model_name)

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
    


    # breakpoint()
    dataset_all = CroppedObjectsDataset(metadata_all, transform=transform, augment=False)
    print(f"Total number of samples in all dataset: {len(dataset_all)}")









    evaluate_validation = args.evaluate_validation #False
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> (2) splitting dataset into training and testing sets 50% each for adjusted cross validation <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    enable_split_50_50 = args.enable_split_50_50
    if enable_split_50_50:
        training_dataset0, test_dataset = split_50_50(dataset_all, seed=args.seed, by=args.split_by) # split dataset into 50% train and 50% test for the quantitative analysis
        if evaluate_validation:
            print(f"Evaluating on the validation dataset (20% of the training dataset) for hyperparameter tuning and model selection.")
            training_dataset, val_dataset = split_80_20(training_dataset0, seed=args.seed)
            test_dataset = val_dataset  
        else:
            print(f"Evaluating on the test dataset (50% of the original dataset) without a separate validation set.")
            training_dataset = training_dataset0
            test_dataset = test_dataset
    else:
        print(f"Using the entire dataset for both training and testing without splitting.")
        training_dataset = dataset_all          # use the entire dataset for training and testing
        test_dataset = dataset_all


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


    print(f"Train loader length: {len(train_loader)}")
    print(f"Test loader length: {len(test_loader)}")

    args.elbow_cache_path = os.path.join(args.scratch_dir, "elbow_cache", f"{args.dataset}_{args.model_name}.json")

    








    stop_event = threading.Event()
    memory_stats = {"initial": None, "peak": None}
    t = threading.Thread(target=gpu_mem_logger, args=(1.0, stop_event, memory_stats), daemon=True)
    t.start()
    time_start = time.time()




    save_dir = os.path.join(args.output_dir,f"{args.model_name}_{args.adaptation_method}_{args.data_collection_strategy}_{args.suffix}",f'{args.N_samples}')
    os.makedirs(save_dir, exist_ok=True)
    csv_path_out = os.path.join(save_dir, f"{args.dataset}_{args.model_name}_{args.N_samples}_seed{args.seed}.csv")

    stats = run_pseudo_labeling(
            train_loader,
            test_loader,
            args.N_samples,
            csv_path_out,
            num_categ = num_categ,
            args = args,
            )        
    time_end = time.time()
    print(f"{args.model_name} pseudo-labeling took {time_end - time_start:.2f} seconds")

    stop_event.set()
    t.join()

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


    for key in PTA_dict.keys():
        if key in args.model_name.lower():
            stats['PTA'] = PTA_dict[key]
            break


    for key in AT_dict.keys():
            if key in args.model_name.lower():
                stats['AT'] = AT_dict[key]
                break
                

    for key in AN_dict.keys():
            if key in args.model_name.lower():
                stats['AN'] = AN_dict[key]
                break
                

    stats['DS'] = args.data_collection_strategy
    if args.data_collection_strategy == "prioritized":
        stats['selection'] = stats.get('prioritized_selection', {}).get('selection')


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
        "evaluate_validation": evaluate_validation
    })
    stats["split_by"] = args.split_by
    save_dict_to_json(stats, os.path.join(save_dir, f"{args.dataset}_{args.model_name}_{args.N_samples}_seed{args.seed}_stats.json"))




































