
import time
import csv
import random
import numpy as np
import torch
from tqdm import tqdm
from thop import profile, clever_format

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from FSCBI.utils.gpu_utils import set_seed
from FSCBI.models.adaptation_algorithm import *
from FSCBI.models.data_selection import *

# Optional: only needed if you enable augmentation
try:
    # import torchvision.transforms as T
    from torchvision.transforms import v2 as T

except Exception:
    T = None
import os
import torch.hub as hub
import contextlib
from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnet34, ResNet34_Weights,
        resnet50, ResNet50_Weights,
        resnet101, ResNet101_Weights,
        resnet152, ResNet152_Weights,
    )
import torch.nn as nn
import clip
def compute_embeddings(loader, split_name, device="cuda", model=None,model_name="dinov2_vitl14"):
    all_emb = []
    all_meta_out = []
    print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
    print(f"model_name: {model_name}")
    print(f"Computing embeddings for model: {model_name} ")
    if "dino" in model_name.lower():
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Computing DINO embeddings ({split_name})"):
                imgs = batch["image"].to(device)
                feats_dict = model.forward_features(imgs)
                features = feats_dict["x_norm_clstoken"]  # [B, D]

                all_emb.append(features.cpu())
                bsize = features.size(0)

                for i in range(bsize):
                    all_meta_out.append({
                        "object_id": batch["object_id"][i],
                        "image_id": batch["image_id"][i],
                        "crop_path": batch["crop_path"][i],
                        "category_id": batch["category_id"][i],
                        "category_name": batch["category_name"][i],
                        "image_file": batch["image_file"][i],
                    })


    elif "resnet" in model_name.lower():
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Computing embeddings ({split_name})"):
                imgs = batch["image"].to(device)
                features = model(imgs)  # [B, D]

                all_emb.append(features.cpu())
                bsize = features.size(0)

                for i in range(bsize):
                    all_meta_out.append({
                        "object_id": batch["object_id"][i],
                        "image_id": batch["image_id"][i],
                        "crop_path": batch["crop_path"][i],
                        "category_id": batch["category_id"][i],
                        "category_name": batch["category_name"][i],
                        "image_file": batch["image_file"][i],
                    })
    elif "clip" in model_name.lower():
        if hasattr(model, 'visual') and model.visual is not None:
            clip_dtype = next(model.visual.parameters()).dtype
        else:
            clip_dtype = next(model.parameters()).dtype
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Computing CLIP embeddings ({split_name})"):
                imgs = batch["image"].to(device=device, dtype=clip_dtype)
                if hasattr(model, 'visual') and model.visual is not None:
                    features = model.encode_image(imgs)
                else:
                    features = model(imgs)
                
                features = features.float()

                all_emb.append(features.cpu())
                bsize = features.size(0)

                for i in range(bsize):
                    all_meta_out.append({
                        "object_id": batch["object_id"][i],
                        "image_id": batch["image_id"][i],
                        "crop_path": batch["crop_path"][i],
                        "category_id": batch["category_id"][i],
                        "category_name": batch["category_name"][i],
                        "image_file": batch["image_file"][i],
                    })
    if len(all_emb) == 0:
        return torch.empty(0), []
    all_emb = torch.cat(all_emb, dim=0)
    return all_emb, all_meta_out
def load_resnet(model_name="resnet50", device="cuda", models_dir='.', load_downloaded_weights=False):
    """
    Load ResNet model with ImageNet pretrained weights as a feature extractor.

    Args:
        model_name: Name of the ResNet variant
        device: Device to load the model on
        models_dir: Directory containing downloaded model weights
        load_downloaded_weights: If True, load weights from models_dir instead of downloading

    Returns:
        model: ResNet model whose fc is nn.Identity, so model(x) returns the
               global-average-pooled features [B, model.num_features]
    """
    print(f"Loading {model_name} (ImageNet pretrained) ...")
    
    # Ensure model_name is a string, not a list
    if isinstance(model_name, list):
        model_name = model_name[0]
    
    model_dict = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_dict.keys())}")
    
    model_fn, weights = model_dict[model_name]
    
    if load_downloaded_weights:
        # Load model without pretrained weights first
        model = model_fn(weights=None)
        
        # Load weights from local file
        # Find the first file that starts with model_name
        matching_files = [f for f in os.listdir(models_dir) if f.startswith(model_name)]
        if not matching_files:
            raise FileNotFoundError(f"No weights file found starting with '{model_name}' in {models_dir}")
        weights_path = os.path.join(models_dir, matching_files[0])
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Download and load pretrained weights
        model = model_fn(weights=weights)
    
    # Drop the ImageNet classifier so model(x) returns the pooled backbone features.
    # A task head is attached by adapt_finetuning2 only for fine-tuning (PFT/VFT).
    model.num_features = model.fc.in_features
    model.fc = nn.Identity()

    model = model.to(device)
    return model


def load_dinov2(model_name="dinov2_vitl14", device="cuda"):
    valid_models = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
    if model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}, got {model_name}")
    
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
                model = hub.load(
                    "facebookresearch/dinov2",
                    model_name
                ).to(device)

        model.eval()
    print(f">>> Loaded DINOv2 model: {model_name}")
    return model

def load_dinov3(
        REPO_DIR = "/home/hassan/codes/installed_packages/dinov3/",
        Checkpoint_dir='/media/hassan/Data_vault1/datasets_hassan/models',
        device="cuda", 
        model_name="dinov3_vits16"

        ):
    dino3_dict = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "dinov3_vith7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
        }
    
    CheckpointPath = os.path.join(Checkpoint_dir, dino3_dict[model_name])
    model = torch.hub.load(REPO_DIR, model_name, source='local', weights=CheckpointPath)
    model.to(device)
    print(f">>> Loaded DINOv3 model {model_name} from {CheckpointPath}")


    return model

def load_clip_model(path="models",model_name="clip_vitb16"):
    clip_dict = {"clip_vitb16": "ViT-B/16",
                    "clip_vitb32": "ViT-B/32",
                    "clip_vitl14": "ViT-L/14",
                    "clip_rn50": "RN50",
                    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = clip_dict.get(model_name.lower())
    print(f"Loading CLIP model {clip_model_name} ...")
    model, preprocess = clip.load(
        clip_model_name,
        device=device,
        download_root=path
    )

    # Extract only the visual backbone
    visual = model.visual


    return visual, preprocess, device
def plot_ref_meta(ref_meta,data_collection_strategy = '', figsize=(15, 10)):
    """
    Plot reference set images in subplots, colored by category.
    
    Args:
        ref_meta: List of dicts with keys: crop_path, category_name, etc.
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    from PIL import Image
    n_samples = len(ref_meta)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create color map for categories
    unique_categories = sorted(set(m["category_name"] for m in ref_meta))
    colors = plt.cm.tab20(range(len(unique_categories)))
    category_colors = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx, meta in enumerate(ref_meta):
        ax = axes[idx]
        try:
            img = Image.open(meta["crop_path"]).convert('RGB')
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading:\n{e}", ha='center', va='center')
        
        category = meta["category_name"]
        color = category_colors[category]
        ax.set_title(category, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['right'].set_color(color)
        for spine in ax.spines.values():
            spine.set_linewidth(3)
    
    # Hide unused subplots
    for idx in range(len(ref_meta), len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    legend_patches = [mpatches.Patch(facecolor=category_colors[cat], label=cat) 
                        for cat in unique_categories]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                ncol=min(5, len(unique_categories)), frameon=True)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.suptitle(f"Reference Set Samples selected using {data_collection_strategy} strategy", fontsize=16, y=0.98)
    plt.show()

def loader_transform(loader):
    """Transform of the dataset behind a DataLoader, unwrapping torch Subset wrappers (e.g. from split_by_sample_count)."""
    ds = loader.dataset
    while not hasattr(ds, "transform") and hasattr(ds, "dataset"):
        ds = ds.dataset
    return getattr(ds, "transform", None)


def run_pseudo_labeling_traditional_finetuning(
    train_loader,
    val_loader,
    test_loader,
    N_samples,  # NOW USED: to subsample the train ref set per class (None/0 -> use all)
    csv_out_path,
    n_test_clusters=15,
    reps_per_cluster=1,
    num_categ = None,
    args = None,
    tradidional_finetuning = False
):


    if isinstance(args.device, tuple):
        device = args.device[0]
    else:
        device = args.device
    if isinstance(args.seed, tuple):
        seed = args.seed[0]
    else:
        seed = args.seed
    if isinstance(args.model_name, tuple):    
        model_name = args.model_name[0]
    else:
        model_name = args.model_name
    if isinstance(args.adaptation_method, tuple):
        adaptation_method = args.adaptation_method[0]
    else:
        adaptation_method = args.adaptation_method
    if isinstance(args.data_collection_strategy, tuple):
        data_collection_strategy = args.data_collection_strategy[0]  
    else:
        data_collection_strategy = args.data_collection_strategy

    
    
    # ------------------ method knobs ------------------
    # LinearSVC / SVC
    svm_C=1.0
    svm_class_weight="balanced"
    rep_selection_mode="center+diverse"
    # kNN
    knn_k=1
    knn_weights="distance"  # "uniform" or "distance"
    # Preprocessing inside the kNN / LinearSVC / SVC pipelines: "l2" (default), "standard" (old StandardScaler) or "none"
    feature_scaling = str(getattr(args, "feature_scaling", "l2")).lower()
    # Random Forest
    rfc_n_estimators=300
    rfc_max_depth=None
    rfc_min_samples_leaf=1
    rfc_n_jobs=-1
    # Unknown thresholds (optional; disabled by default)
    unknown_threshold=None
    


    

    print(f"\033[1;33m>>>>>>>>>>>>>>>>>>>>>>> Running {model_name} pseudo-labeling with TRAIN ref-set + {adaptation_method} using {data_collection_strategy} data collection strategy...\033[0m") 
    t0 = time.time()

    # ------------------ Reproducibility ------------------
    import random
    import numpy as np
    import torch
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------ Load DINO ------------------
    if "dinov2" in model_name.lower():
        model = load_dinov2(device=device, model_name=model_name)
    elif "dinov3" in model_name.lower():
        model = load_dinov3(device=device, model_name=model_name)
    elif "resnet" in model_name.lower():
        model = load_resnet(
            model_name=model_name, device=device,
            models_dir=args.models_dir, load_downloaded_weights=True
        )
    elif "clip" in model_name.lower():
        model, preprocess, device = load_clip_model(path=args.models_dir, model_name=model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model.eval()
    print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&6")
    # print_model(model)
    # ------------------ Calculate Model Stats ------------------
    from thop import profile, clever_format
    from PIL import Image
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    if hasattr(model, "visual"):  # CLIP
        target_model = model.visual
    else:
        target_model = model

    # THOP is usually safest in float32
    target_model = target_model.float()
    dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)

    flops, params = profile(target_model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Model: {model_name}")
    print(f"Parameters: {params}")
    print(f"FLOPs: {flops}")
    t_train0 = time.time()
    # =================================================================
    #  Compute TRAIN embeddings (reference set)
    # =================================================================

    
    train_embeddings, train_meta = compute_embeddings(train_loader, split_name="train(ref)", device=device, model=model, model_name=model_name)

    if tradidional_finetuning:
        val_embeddings, val_meta = compute_embeddings(val_loader, split_name="validation", device=device, model=model, model_name=model_name)

        if val_embeddings.shape[0] == 0:
            raise ValueError("No validation embeddings computed; check val_loader.")
        X_val = val_embeddings.numpy().astype(np.float32)  # [N_val, D]
        y_val = np.array([m["category_id"] for m in val_meta])
        N_val = X_val.shape[0]
    if train_embeddings.shape[0] == 0:
        raise ValueError("No train embeddings computed; check train_loader.")
    print(f"Train embeddings shape: {list(train_embeddings.shape)}")
    # Build category vocabulary from TRAIN (reference labels come from train)
    cat_names = sorted({m["category_name"] for m in train_meta})
    num_classes = len(cat_names)
    if num_classes == 0:
        raise ValueError("No category names found in train_meta.")
    


    if tradidional_finetuning:
        ref_embeddings, ref_label_names, ref_label_ids, ref_meta = all_subsample(train_embeddings, 
                                                                                   train_meta, 
                                                                                   N_samples, 
                                                                                   cat_names, 
                                                                                   seed=seed)
        

    else:
        if data_collection_strategy == "random":
            ref_embeddings, ref_label_names, ref_label_ids, ref_meta = random_subsample(train_embeddings, 
                                                                                    train_meta, 
                                                                                    N_samples, 
                                                                                    cat_names, 
                                                                                    seed=seed)

        elif data_collection_strategy == "prioritized":
            ref_embeddings, ref_label_names, ref_label_ids, ref_meta = prioritized_subsample(train_embeddings, 
                                                                                train_meta, 
                                                                                N_samples, 
                                                                                cat_names,
                                                                                n_test_clusters, 
                                                                                reps_per_cluster, 
                                                                                rep_selection_mode, 
                                                                                seed=seed)

    plot_ref_samples = False
    if plot_ref_samples:
        plot_ref_meta(ref_meta,data_collection_strategy=data_collection_strategy, figsize=(15, 10))



    if tradidional_finetuning:
            print(f"\033[1;33m>>>>>>>>> Not using any data collection strategy for reference set. Using all dataset for traditional finetuning <<<<<<<\033[0m")
    else:
        print(f"\033[1;33m>>>>>>>>> Using {data_collection_strategy} data collection strategy for reference set. <<<<<<<\033[0m")


    print(f"\033[1;32mReference set size (base): {list(ref_embeddings.shape)}\033[0m")
    print(f"\033[1;32mValidation set size (base): {list(X_val.shape)}\033[0m")
    unique_ids, counts = np.unique(ref_label_ids, return_counts=True)
    print("\033[1;33m")  # Start yellow text

    name_to_pseudo_id = {name: i for i, name in enumerate(cat_names)}
    id_to_name = {i: name for name, i in name_to_pseudo_id.items()}
    label_dist = {id_to_name[int(uid)]: int(count) for uid, count in zip(unique_ids, counts)}
    print(f"Reference labels distribution: {label_dist}")
    print("\033[0m")  # Reset to default text

    X_ref = ref_embeddings.numpy().astype(np.float32)  # [R, D]
    y_ref = ref_label_ids.astype(int)                  # [R]

    t_test0 = time.time()
    test_embeddings, test_meta = compute_embeddings(test_loader, "test", device=device, model=model, model_name=model_name)
    print(f"Test embeddings shape: {list(test_embeddings.shape)}")
    if test_embeddings.shape[0] == 0:
        raise ValueError("No test embeddings computed; check test_loader.")
    # Without the 50/50 split, train and test are the same dataset: drop the reference samples from
    # the test set so they are not scored on themselves
    if not args.enable_split_50_50:
        ref_crop_paths = {m["crop_path"] for m in ref_meta}
        keep = [i for i, m in enumerate(test_meta) if m["crop_path"] not in ref_crop_paths]
        print(f"\033[1;33mRemoving {len(test_meta) - len(keep)} reference samples from the test set "
              f"(enable_split_50_50=False)\033[0m")
        test_embeddings = test_embeddings[keep]
        test_meta = [test_meta[i] for i in keep]
        print(f"Test embeddings shape (without reference samples): {list(test_embeddings.shape)}")
    X_test = test_embeddings.numpy().astype(np.float32)  # [N, D]
    N_test = X_test.shape[0]


    










    # =================================================================
    # 3) Assign pseudo labels for ALL test using TRAIN ref set
    # =================================================================
    adaptation_method = str(adaptation_method).strip()
    method_upper = adaptation_method.upper()

    pseudo_ids = None
    pseudo_names = None
    category_likelihoods = None

    # ---- Helper for per-class likelihoods from similarities/distances ----
    # (assumes you already have per_class_average_from_pairwise in your codebase)
    print(f">>>>>>>>>>>>>>>>>> X_test type: {type(X_test)}, X_ref type: {type(X_ref)}, y_ref type: {type(y_ref)}")
    print(f">>>>>>>>>>>>>>>>>> X_test dtype: {X_test.dtype}, X_ref dtype: {X_ref.dtype}, y_ref dtype: {y_ref.dtype}")
    print(f">>>>>>>>>>>>>>>>>> X_test shape: {X_test.shape}, X_ref shape: {X_ref.shape}, y_ref shape: {y_ref.shape}")

    # print(f"ref_label_names, {ref_label_names}")
    # print(f"ref_label_ids, {ref_label_ids}")
    # print(f"ref_meta, {ref_meta}")
    print(f"type(ref_label_ids): {type(ref_label_ids)}, ref_label_ids dtype: {ref_label_ids.dtype},  ref_label_ids shape: {ref_label_ids.shape}")
    print(f"type(ref_label_names): {type(ref_label_names)}, ref_label_names dtype: {ref_label_names.dtype},  ref_label_names shape: {ref_label_names.shape}")
    print(f"type(ref_meta): {type(ref_meta)}, ref_meta length: {len(ref_meta)}")
    print(f"type(train_meta): {type(train_meta)}, train_meta length: {len(train_meta)}")
    print(f"type(test_meta): {type(test_meta)}, test_meta length: {len(test_meta)}")
    # print(f"ref_meta {ref_meta}")
    if method_upper == "COSSIM":
        pseudo_ids, pseudo_names, category_likelihoods = adapt_cossim(X_test, X_ref, y_ref, id_to_name, unknown_threshold, num_classes, feature_scaling=feature_scaling)

    elif method_upper == "KNN":
        pseudo_ids, pseudo_names, category_likelihoods = adapt_knn(X_test, X_ref, y_ref, id_to_name, unknown_threshold, num_classes, knn_k=knn_k, knn_weights=knn_weights, feature_scaling=feature_scaling)

    elif method_upper == "LINEARSVC":
        pseudo_ids, pseudo_names, category_likelihoods = adapt_linearsvc(X_test, X_ref, y_ref, id_to_name, unknown_threshold, seed=seed, svm_C=svm_C, svm_class_weight=svm_class_weight, num_classes=num_classes, feature_scaling=feature_scaling)

    elif method_upper == "SVC":
        pseudo_ids, pseudo_names, category_likelihoods = adapt_svc(X_test, X_ref, y_ref, id_to_name, unknown_threshold, num_classes, seed=seed, svm_C=svm_C, svm_class_weight=svm_class_weight, feature_scaling=feature_scaling)

    elif method_upper in ["PFT", "VFT", "LNT", "LORA"]: #Partial / full / LayerNorm / LoRA fine tuning
        pseudo_ids, pseudo_names, category_likelihoods = adapt_finetuning2( ref_meta,
                                                                    test_meta,
                                                                    id_to_name,
                                                                    unknown_threshold,
                                                                    num_classes,
                                                                    FT_method = method_upper,
                                                                    seed=seed,
                                                                    lr=1e-4,
                                                                    weight_decay=1e-4,
                                                                    num_epochs=args.num_epochs,
                                                                    batch_size=args.batch_size,
                                                                    device=device,
                                                                    model=model,
                                                                    model_name=model_name,
                                                                    val_meta = val_meta if tradidional_finetuning else None,
                                                                    transform=loader_transform(train_loader),  # same preprocessing as the embeddings
                                                                    augment=getattr(args, "augment", False),
                                                                    early_stopping=getattr(args, "early_stopping", "none"),
                                                                    ) 
    


    else:
        raise ValueError("method must be one of: 'LinearSVC', 'kNN', 'EucDis', 'CosSim', 'RFC', 'SK', 'SVC', 'PFT', 'VFT', 'LNT', 'LoRA'")

    print(f"unique names in ref_meta: {set(m['category_name'] for m in ref_meta)}")
    print(f"unique names in train_meta: {set(m['category_name'] for m in train_meta)}")
    # print(f"unique ids in test_meta: {set(m['category_id'] for m in test_meta)}")
    print(f"unique names in test_meta: {set(m['category_name'] for m in test_meta)}")

    print(f"unique pseudo ids: {np.unique(pseudo_ids)}")
    print(f"unique pseudo names: {set(pseudo_names)}")
    # breakpoint()

    # Safety checks
    if pseudo_ids is None or len(pseudo_ids) != N_test:
        print(f"Error: pseudo_ids is None or has length {len(pseudo_ids)} but expected {N_test}")
        raise RuntimeError("Pseudo labeling failed: invalid pseudo_ids produced.")
    pseudo_ids = np.array(pseudo_ids, dtype=int)
    if np.isnan(pseudo_ids.astype(float)).any():
        raise RuntimeError("Pseudo labeling produced NaN ids (unexpected).")



    # =================================================================
    # 4) Save CSV (same format; inference on TEST)
    # =================================================================
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "object_id",
            "crop_path",
            "image_id",
            "image_file",
            "gt_category_id",
            "gt_category_name",
            "pseudo_category_id",
            "pseudo_category_name",
            "likelihood",
        ])

        for i, m in enumerate(test_meta):
            writer.writerow([
                int(m["object_id"]),
                m["crop_path"],
                int(m["image_id"]),
                m["image_file"],
                int(m["category_id"]),
                m["category_name"],
                int(pseudo_ids[i]),
                pseudo_names[i],
                category_likelihoods[i],
            ])

    print(f"Saved pseudo labels for TEST split → {csv_out_path}")

    t1 = time.time()
    #print(f"\033[1;32mSpc used for ref set: {spc}\033[0m") 
    def get_class_distribution_test(test_meta):
        class_distribution_test = {}

        for meta in test_meta:
            cat_name = str(meta["category_name"])
            class_distribution_test[cat_name] = class_distribution_test.get(cat_name, 0) + 1

        return class_distribution_test
    class_distribution_test = get_class_distribution_test(test_meta)
    stats = {
        "flops": flops,
        "params": params,
        "model_name": model_name,
        "seed": seed,
        "N_samples": N_samples,
        "ref_size": int(X_ref.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(N_test),
        "class_distribution_ref" : label_dist,
        "class_distribution_test": class_distribution_test,
        
        "num_classes": int(num_classes),
        "unknown_threshold": unknown_threshold,
        "feature_scaling": feature_scaling,
    }
    return stats








