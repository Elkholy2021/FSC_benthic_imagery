import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Dataset
import os

def per_class_average_from_pairwise(pairwise, labels_ref, n_classes, mode="mean"):
    """
    pairwise: [N, R] similarities OR negative distances (higher=better)
    returns: list of length N, each is list length C
    """
    out = []
    for i in range(pairwise.shape[0]):
        row = []
        for c in range(n_classes):
            mask = (labels_ref == c)
            if mask.sum() > 0:
                v = pairwise[i, mask].mean()
            else:
                v = 0.0
            row.append(float(v))
        out.append(row)
    return out


def make_feature_scaler(feature_scaling="l2"):
    """
    Feature preprocessing fitted inside the classifier pipeline (kNN / LinearSVC / SVC).
      "l2"       -> L2-normalise each embedding (stateless, so nothing is estimated from the few labelled samples)
      "standard" -> StandardScaler fitted on the reference set (previous behaviour; per-dimension mean/std
                    estimated from only N_samples points, which is unstable when D >> N_samples)
      "none"     -> raw embeddings
    """
    feature_scaling = str(feature_scaling).lower()
    if feature_scaling == "l2":
        return Normalizer(norm="l2")
    if feature_scaling == "standard":
        return StandardScaler(with_mean=True, with_std=True)
    if feature_scaling == "none":
        return FunctionTransformer()
    raise ValueError(f"Unknown feature_scaling: {feature_scaling} (use 'l2', 'standard' or 'none')")


def scores_to_all_classes(scores, classes, num_classes, fill=0.0):
    """
    Place per-class scores [N, len(classes)] into [N, num_classes] columns indexed by class id,
    so likelihood columns line up with class ids even when some classes have no reference samples.
    """
    scores = np.asarray(scores, dtype=float)
    if num_classes is None:
        return scores
    full = np.full((scores.shape[0], num_classes), fill, dtype=float)
    for j, c in enumerate(classes):
        full[:, int(c)] = scores[:, j]
    return full


def adapt_cossim(X_test, X_ref, y_ref, id_to_name, unknown_threshold, num_classes, feature_scaling="l2"):
    # Same preprocessing as kNN / LinearSVC / SVC (fitted on the reference set); cosine similarity
    # still L2-normalises afterwards, so "l2" and "none" give identical results
    scaler = make_feature_scaler(feature_scaling).fit(X_ref)
    X_test = scaler.transform(X_test)
    X_ref = scaler.transform(X_ref)
    all_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-12)
    ref_norm = X_ref / (np.linalg.norm(X_ref, axis=1, keepdims=True) + 1e-12)
    sim = all_norm @ ref_norm.T  # [N, R]

    best = np.argmax(sim, axis=1)
    max_sim = sim[np.arange(sim.shape[0]), best]

    pseudo_ids = y_ref[best]
    if unknown_threshold is not None:
        pseudo_ids = np.where(max_sim >= float(unknown_threshold), pseudo_ids, -1)

    pseudo_names = np.array(
        [id_to_name.get(int(i), "unknown") if int(i) != -1 else "unknown" for i in pseudo_ids],
        dtype=object
    )
    category_likelihoods = per_class_average_from_pairwise(sim, y_ref, num_classes)
    return pseudo_ids, pseudo_names, category_likelihoods


def adapt_knn(X_test, X_ref, y_ref, id_to_name, unknown_threshold, num_classes, knn_k=1, knn_weights="distance", feature_scaling="l2"):
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except Exception as e:
        raise ImportError("kNN option requires scikit-learn (pip install scikit-learn).") from e

    clf = make_pipeline(
        make_feature_scaler(feature_scaling),
        KNeighborsClassifier(n_neighbors=int(knn_k), weights=str(knn_weights))
    )
    clf.fit(X_ref, y_ref)

    pred = clf.predict(X_test).astype(int)
    pseudo_ids = pred

    if hasattr(clf[-1], "predict_proba"):
        probs = clf.predict_proba(X_test)  # [N, C_seen]
        if probs.shape[1] != num_classes:
            full = np.zeros((probs.shape[0], num_classes), dtype=float)
            present = clf[-1].classes_
            for j, lab in enumerate(present):
                full[:, int(lab)] = probs[:, j]
            probs = full
        max_prob = probs.max(axis=1)
        if unknown_threshold is not None:
            pseudo_ids = np.where(max_prob >= float(unknown_threshold), pseudo_ids, -1)
        category_likelihoods = probs.tolist()
    else:
        all_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-12)
        ref_norm = X_ref / (np.linalg.norm(X_ref, axis=1, keepdims=True) + 1e-12)
        sim = all_norm @ ref_norm.T
        category_likelihoods = per_class_average_from_pairwise(sim, y_ref, num_classes)

    pseudo_names = np.array(
        [id_to_name.get(int(i), "unknown") if int(i) != -1 else "unknown" for i in pseudo_ids],
        dtype=object
    )

    return pseudo_ids, pseudo_names, category_likelihoods


def adapt_linearsvc(
    X_test,
    X_ref,
    y_ref,
    id_to_name,
    unknown_threshold,
    seed=42,
    svm_C=1.0,
    svm_class_weight="balanced",
    num_classes=None,
    feature_scaling="l2",
):
    import numpy as np

    try:
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError(
            "LinearSVC option requires scikit-learn (pip install scikit-learn)."
        ) from e

    print(f"X_ref shape: {X_ref.shape}, y_ref shape: {y_ref.shape}")

    unique_classes = np.unique(y_ref)

    # Handle the one-class case explicitly
    if len(unique_classes) == 1:
        only_class = int(unique_classes[0])

        pred_ids = np.full(X_test.shape[0], only_class, dtype=int)

        # In the one-class case, there is no real decision boundary.
        # We use a constant score of 1.0 for the single class so downstream
        # code still receives category_likelihoods in a compatible shape.
        scores = np.ones((X_test.shape[0], 1), dtype=float)

        pseudo_ids = pred_ids
        if unknown_threshold is not None:
            margin = scores.max(axis=1)
            pseudo_ids = np.where(margin >= float(unknown_threshold), pseudo_ids, -1)

        pseudo_names = np.array(
            [
                id_to_name.get(int(i), "unknown") if int(i) != -1 else "unknown"
                for i in pseudo_ids
            ],
            dtype=object,
        )
        category_likelihoods = scores_to_all_classes(scores, unique_classes, num_classes).tolist()
        return pseudo_ids, pseudo_names, category_likelihoods

    svm = make_pipeline(
        make_feature_scaler(feature_scaling),
        LinearSVC(C=float(svm_C), class_weight=svm_class_weight, random_state=seed),
    )

    svm.fit(X_ref, y_ref)

    pred_ids = svm.predict(X_test).astype(int)
    scores = svm.decision_function(X_test)

    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T

    pseudo_ids = pred_ids
    if unknown_threshold is not None:
        margin = scores.max(axis=1)
        pseudo_ids = np.where(margin >= float(unknown_threshold), pseudo_ids, -1)

    pseudo_names = np.array(
        [id_to_name.get(int(i), "unknown") if int(i) != -1 else "unknown" for i in pseudo_ids],
        dtype=object,
    )
    category_likelihoods = scores_to_all_classes(scores, svm[-1].classes_, num_classes).tolist()
    return pseudo_ids, pseudo_names, category_likelihoods


def adapt_svc(X_test, X_ref, y_ref, id_to_name, unknown_threshold, num_classes, seed=42, svm_C=1.0, svm_class_weight="balanced", feature_scaling="l2"):
    try:
        from sklearn.svm import SVC
    except Exception as e:
        raise ImportError("SVC option requires scikit-learn (pip install scikit-learn).") from e

    # optional: allow overriding via outer scope if you had it before
    svm_kernel = locals().get("svm_kernel", "rbf")
    svm_gamma = locals().get("svm_gamma", "scale")

    svm = make_pipeline(
        make_feature_scaler(feature_scaling),
        SVC(
            C=float(svm_C),
            kernel=svm_kernel,
            gamma=svm_gamma,
            class_weight=svm_class_weight,
            decision_function_shape="ovr",
            random_state=seed,
        )
    )
    svm.fit(X_ref, y_ref)

    pred_ids = svm.predict(X_test).astype(int)
    scores = svm.decision_function(X_test)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T

    pseudo_ids = pred_ids
    if unknown_threshold is not None:
        margin = scores.max(axis=1)
        pseudo_ids = np.where(margin >= float(unknown_threshold), pseudo_ids, -1)

    pseudo_names = np.array(
        [id_to_name.get(int(i), "unknown") if int(i) != -1 else "unknown" for i in pseudo_ids],
        dtype=object
    )
    category_likelihoods = scores_to_all_classes(scores, svm[-1].classes_, num_classes).tolist()
    return pseudo_ids, pseudo_names, category_likelihoods



def get_last_block(model, model_name: str):
    """
    Returns (block_name: str, block_module: nn.Module)

    Supports:
        - resnet18
        - resnet50
        - dinov2
        - dinov3
    """

    name = model_name.lower()
    # ---------------- CLIP ----------------
    if "clip" in name or hasattr(model, "encode_image"):

        if hasattr(model, 'visual') and model.visual is not None:
            visual = model.visual
        else:
            visual = model
        

        # ViT-based CLIP
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            return "visual.transformer.resblocks[-1]", visual.transformer.resblocks[-1]

        # ResNet-based CLIP
        if hasattr(visual, "layer4"):
            return "visual.layer4", visual.layer4

        raise ValueError("CLIP detected but could not find known last visual block.")


    # ---------------- ResNet family ----------------
    if "resnet" in name:
        if hasattr(model, "layer4"):
            return "layer4", model.layer4
        else:
            raise ValueError("ResNet detected but 'layer4' not found.")

    # ---------------- DINOv2 / DINOv3 (ViT-based) ----------------
    elif "dino" in name:
        if hasattr(model, "blocks"):
            return "blocks[-1]", model.blocks[-1]
        else:
            raise ValueError("DINO model detected but 'blocks' not found.")

    else:
        raise ValueError(f"Unsupported model type: {model_name}")




def print_model(model):
    print("Model architecture:")
    for name, param in model.named_parameters():
        print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")


class RandomRotate90:
    """Rotate a PIL image by a random multiple of 90 degrees (top-down imagery has no canonical orientation)."""
    def __call__(self, img):
        k = random.randint(0, 3)
        return img.rotate(90 * k, expand=True) if k else img


class RandomCropKeepSize:
    """
    Random crop covering `scale` of the image area, without resizing: the base transform resizes
    the crop afterwards, so this only changes the framing of the object.
    """
    def __init__(self, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3)):
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
        return transforms.functional.crop(img, i, j, h, w)


def make_train_augmentation(base_transform):
    """Training-only augmentation (on the PIL crop) followed by the usual preprocessing."""
    return transforms.Compose([
        RandomCropKeepSize(scale=(0.7, 1.0)),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        base_transform,
    ])


def get_norm_parameters(model):
    """
    Affine parameters of every normalization layer (LayerNorm / RMSNorm / GroupNorm / BatchNorm) of the model,
    i.e. what LayerNorm tuning (LNT) trains. ResNets have no LayerNorm, so their BatchNorm layers are tuned instead.
    """
    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)
    params = []
    for m in model.modules():
        if isinstance(m, norm_types) or type(m).__name__.endswith("Norm"):  # e.g. DINOv3 RMSNorm, CLIP LayerNorm
            params.extend(m.parameters(recurse=False))
    return params


class LoRAWeight(nn.Module):
    """
    Low-rank update of a frozen weight, registered as a parametrization: W -> W + (alpha / r) * B @ A.
    B starts at zero so training starts from the pretrained model. Works for nn.Linear.weight and for the fused
    nn.MultiheadAttention.in_proj_weight used by CLIP.
    """
    def __init__(self, weight, r=8, alpha=16):
        super().__init__()
        out_features, in_features = weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=weight.dtype, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=weight.dtype, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scaling = alpha / r

    def forward(self, W):
        return W + self.scaling * (self.lora_B @ self.lora_A)


def get_lora_targets(model, model_name):
    """
    (module, parameter name) pairs of the attention weights that get a LoRA update:
        - DINOv2 / DINOv3: blocks[i].attn.qkv.weight and blocks[i].attn.proj.weight
        - CLIP ViT: transformer.resblocks[i].attn.in_proj_weight and .attn.out_proj.weight
    Convolutional backbones (ResNet, CLIP RN50) are not supported.
    """
    name = model_name.lower()
    targets = []
    if "clip" in name or hasattr(model, "encode_image"):
        visual = model.visual if getattr(model, "visual", None) is not None else model
        if not (hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks")):
            raise ValueError(f"LoRA is only implemented for ViT backbones, got {model_name}")
        for block in visual.transformer.resblocks:
            targets.append((block.attn, "in_proj_weight"))
            targets.append((block.attn.out_proj, "weight"))
    elif "dino" in name:
        for block in model.blocks:
            targets.append((block.attn.qkv, "weight"))
            targets.append((block.attn.proj, "weight"))
    else:
        raise ValueError(f"LoRA is only implemented for ViT backbones (DINOv2 / DINOv3 / CLIP ViT), got {model_name}")
    return targets


def add_lora(model, model_name, r=8, alpha=16):
    """Freeze the given weights and add a trainable low-rank update to each (see LoRAWeight). Returns the number of LoRA parameters."""
    import torch.nn.utils.parametrize as parametrize
    n_params = 0
    for module, param_name in get_lora_targets(model, model_name):
        getattr(module, param_name).requires_grad = False
        lora = LoRAWeight(getattr(module, param_name), r=r, alpha=alpha)
        parametrize.register_parametrization(module, param_name, lora)
        n_params += lora.lora_A.numel() + lora.lora_B.numel()
    return n_params


def adapt_finetuning2(
    ref_meta,
    test_meta,
    id_to_name,
    unknown_threshold,
    num_classes,
    FT_method="PFT",
    seed=42,
    lr=1e-4,
    weight_decay=1e-4,
    num_epochs=20,
    batch_size=16,
    device=None,
    model=None,
    model_name="resnet",
    val_meta = None,
    transform = None,
    augment = False,
    early_stopping = "none",
    early_stopping_patience = 10,
    lora_r = 8,
    lora_alpha = 16,
):
    """
    FT_method: "VFT" (full finetuning), "PFT" (last block + final norm + head), "LNT" (normalization layers +
    head) or "LORA" (rank-`lora_r` updates of the attention weights + head, ViT backbones only).
    transform: preprocessing for the crops. Pass the same transform used for embedding extraction so
    fine-tuned and frozen-feature methods see identical inputs; None falls back to a 224x224 resize.
    augment: apply random crop / 90-degree rotation / flips / colour jitter to the training crops only
    (validation and test use `transform` unchanged).
    early_stopping: "f1" (validation macro-F1), "accuracy" (validation accuracy) or "none" (train all
    num_epochs). With "f1"/"accuracy" (requires val_meta), training stops after `early_stopping_patience`
    epochs without improvement and the weights of the best validation epoch are restored.
    """
    early_stopping = str(early_stopping).lower()
    if early_stopping not in ("f1", "accuracy", "none"):
        raise ValueError(f"Unknown early_stopping: {early_stopping} (use 'f1', 'accuracy' or 'none')")
    if early_stopping != "none" and val_meta is None:
        raise ValueError(f"early_stopping='{early_stopping}' needs a validation set (val_meta)")
    def _to_python_scalar(x):
        if torch.is_tensor(x):
            if x.numel() == 1:
                return x.item()
            return x.detach().cpu().tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x

    def _to_python_str(x):
        x = _to_python_scalar(x)
        return str(x)

    def _replace_classification_head(model, model_name, out_dim):
        # ResNet family (load_resnet sets fc to nn.Identity and stores the feature dim in num_features)
        if hasattr(model, "fc") and isinstance(model.fc, (nn.Linear, nn.Identity)):
            if isinstance(model.fc, nn.Linear):
                in_features = model.fc.in_features
            else:
                in_features = model.num_features
            model.fc = nn.Linear(in_features, out_dim)
            return "fc"

        # ViT / DINO style existing head
        if hasattr(model, "head"):
            if isinstance(model.head, nn.Linear):
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, out_dim)
                return "head"

            if isinstance(model.head, nn.Identity):
                in_features = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
                if in_features is None:
                    raise ValueError(f"Could not infer embed dim for model_name={model_name}")
                model.head = nn.Linear(in_features, out_dim)
                return "head"

            if isinstance(model.head, nn.Sequential) and len(model.head) > 0:
                for i in range(len(model.head) - 1, -1, -1):
                    if isinstance(model.head[i], nn.Linear):
                        in_features = model.head[i].in_features
                        model.head[i] = nn.Linear(in_features, out_dim)
                        return f"head.{i}"

        # DINO / ViT backbone with no head at all
        in_features = None
        for attr in ["embed_dim", "num_features", "feature_dim"]:
            if hasattr(model, attr):
                in_features = int(getattr(model, attr))
                break

        if in_features is not None:
            model.head = nn.Linear(in_features, out_dim)
            return "head"
        
        # CLIP-style models with encode_image but no head
        model_name_l = str(model_name).lower()
        if "clip" in model_name_l or hasattr(model, "encode_image"):
            in_features = None

            if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
                in_features = int(model.visual.output_dim)

            elif hasattr(model, "visual") and hasattr(model.visual, "proj"):
                proj = model.visual.proj
                if proj is not None and hasattr(proj, "shape"):
                    if len(proj.shape) == 2:
                        in_features = int(proj.shape[1])
                    else:
                        in_features = int(proj.shape[0])
            else:
                in_features = int(model.output_dim)
            if in_features is None:
                raise ValueError(f"Could not infer CLIP image embedding dim for model_name={model_name}")

            model.head = nn.Linear(in_features, out_dim)
            return "head"



        raise ValueError(f"Could not find or create a classification head for model_name={model_name}")

    def _get_head_module(model, model_name=None):
        # CLIP: custom classification head attached manually
        if "clip" in str(model_name).lower() or hasattr(model, "encode_image"):
            if hasattr(model, "head"):
                return model.head, "head"
            raise ValueError("CLIP model detected, but no custom head found. Call _replace_classification_head(...) first.")

        if hasattr(model, "fc"):
            return model.fc, "fc"
        if hasattr(model, "head"):
            return model.head, "head"
        raise ValueError("Could not find classification head module.")

    def _forward_logits(model, x, model_name=None):
        model_name_l = str(model_name).lower() if model_name is not None else ""

        # ---------------- CLIP special case ----------------
        if "clip" in model_name_l or hasattr(model, "encode_image"):
            head_module, _ = _get_head_module(model, model_name=model_name)
            if hasattr(model, "visual"):   
                clip_dtype = next(model.visual.parameters()).dtype
                feats = model.encode_image(x.to(dtype=clip_dtype))
            else:
                clip_dtype = next(model.parameters()).dtype
                feats = model(x.to(dtype=clip_dtype))
            
            feats = feats.float()

            return head_module(feats)

        # ---------------- default behavior ----------------
        out = model(x)

        # already logits
        if torch.is_tensor(out) and out.ndim == 2:
            head_module = None
            try:
                head_module, _ = _get_head_module(model, model_name=model_name)
            except Exception:
                head_module = None

            if isinstance(head_module, nn.Linear) and out.shape[1] == head_module.out_features:
                return out

            if isinstance(head_module, nn.Sequential):
                # if already class logits, keep them
                if out.shape[1] == train_num_classes:
                    return out

            # otherwise treat as features and pass through head
            if head_module is not None:
                return head_module(out)

        raise ValueError("Could not convert model output to logits.")

        # dict outputs from some models
        if isinstance(out, dict):
            for key in ["x_norm_clstoken", "x_cls_token", "cls_token", "pooler_output", "features"]:
                if key in out:
                    feats = out[key]
                    head_module, _ = _get_head_module(model)
                    return head_module(feats)

        raise ValueError("Could not convert model output to logits.")

    class CropPathDataset(Dataset):
        def __init__(self, meta_list, transform, cat_id_to_label_idx=None, with_labels=False):
            self.meta_list = meta_list
            self.transform = transform
            self.cat_id_to_label_idx = cat_id_to_label_idx
            self.with_labels = with_labels

        def __len__(self):
            return len(self.meta_list)

        def __getitem__(self, idx):
            meta = self.meta_list[idx]
            crop_path = _to_python_str(meta["crop_path"])

            if not os.path.exists(crop_path):
                raise FileNotFoundError(f"crop_path does not exist: {crop_path}")

            img = Image.open(crop_path).convert("RGB")
            img = self.transform(img)

            if self.with_labels:
                cat_id = int(_to_python_scalar(meta["category_id"]))
                label_idx = self.cat_id_to_label_idx[cat_id]
                return img, label_idx

            return img

    def _set_train_mode(model):
        """
        model.train(), but BatchNorm layers whose parameters are all frozen stay in eval mode, so their
        ImageNet running statistics are not overwritten by the few reference images (matters for PFT on ResNet).
        """
        model.train()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                params = list(m.parameters())
                if params and not any(p.requires_grad for p in params):
                    m.eval()

    # -------------------- seeds --------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------- device --------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None:
        raise ValueError("model must not be None")

    if not isinstance(ref_meta, list) or not isinstance(test_meta, list):
        raise ValueError("ref_meta and test_meta must both be lists")

    if len(ref_meta) == 0:
        raise ValueError("ref_meta is empty")
    if len(test_meta) == 0:
        raise ValueError("test_meta is empty")

    # -------------------- build label mapping from ref_meta --------------------
    ref_cat_ids = []
    for m in ref_meta:
        ref_cat_ids.append(int(_to_python_scalar(m["category_id"])))

    class_ids = np.array(sorted(set(ref_cat_ids)), dtype=np.int64)
    cat_id_to_label_idx = {int(cid): i for i, cid in enumerate(class_ids)}
    label_idx_to_cat_id = {i: int(cid) for i, cid in enumerate(class_ids)}
    train_num_classes = len(class_ids)

    # -------------------- replace/add head to match ref classes --------------------
    replaced_head_name = _replace_classification_head(model, model_name, train_num_classes)
    print(f"[adapt_finetuning] Replaced classification head: {replaced_head_name} -> out_dim={train_num_classes}")

    # -------------------- freeze/unfreeze according to method --------------------
    FT_method = FT_method.upper()
    # print(f"Model BEFORE")
    # print_model(model)
    if FT_method == "VFT":
        # full finetuning
        for p in model.parameters():
            p.requires_grad = True

    elif FT_method == "PFT":
        # partial finetuning: last block + head (+ norm if exists)
        last_block_name, last_block_module = get_last_block(model, model_name)

        for p in model.parameters():
            p.requires_grad = False

        for p in last_block_module.parameters():
            p.requires_grad = True

        if hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True

        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True

        if hasattr(model, "norm"):
            for p in model.norm.parameters():
                p.requires_grad = True

    elif FT_method == "LNT":
        # LayerNorm tuning: normalization parameters + head
        head_module, _ = _get_head_module(model, model_name=model_name)
        for p in model.parameters():
            p.requires_grad = False

        for p in get_norm_parameters(model):
            p.requires_grad = True

        for p in head_module.parameters():
            p.requires_grad = True

    elif FT_method == "LORA":
        # LoRA: low-rank updates of the attention weights + head (pretrained weights stay frozen)
        head_module, _ = _get_head_module(model, model_name=model_name)
        for p in model.parameters():
            p.requires_grad = False

        n_lora = add_lora(model, model_name, r=lora_r, alpha=lora_alpha)
        print(f"[adapt_finetuning] Added LoRA (r={lora_r}, alpha={lora_alpha}): {n_lora} parameters")

        for p in head_module.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown FT_method={FT_method}. Expected one of: VFT, PFT, LNT, LORA")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[adapt_finetuning] {FT_method}: {n_trainable} / {n_total} trainable parameters ({100 * n_trainable / n_total:.2f}%)")
    # print(f"Model AFTER")
    # print_model(model)
    model = model.to(device)

    # -------------------- image transform --------------------
    if transform is not None:
        pass
    elif "clip" in model_name.lower():
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    else:

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],       #imagenet
                std=[0.229, 0.224, 0.225],
            )



        ])

    train_transform = make_train_augmentation(transform) if augment else transform
    print(f"[adapt_finetuning] Training augmentation: {'on' if augment else 'off'}")

    # -------------------- dataloaders --------------------
    ds_ref = CropPathDataset(
        meta_list=ref_meta,
        transform=train_transform,
        cat_id_to_label_idx=cat_id_to_label_idx,
        with_labels=True,
    )
    ds_test = CropPathDataset(
        meta_list=test_meta,
        transform=transform,
        with_labels=False,
    )

    if val_meta is not None:
        ds_val = CropPathDataset(
            meta_list=val_meta,
            transform=transform,
            cat_id_to_label_idx=cat_id_to_label_idx,
            with_labels=True,
        )
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)


    loader_ref = DataLoader(ds_ref, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)


    # -------------------- optimizer --------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # -------------------- finetuning on ref_meta --------------------
    best_val_score, best_epoch, best_state, epochs_no_improve = None, None, None, 0
    _set_train_mode(model)
    for epoch in tqdm(range(1, num_epochs + 1), desc=f"Finetuning ({FT_method})"):
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader_ref:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(model, xb, model_name=model_name)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)







        if early_stopping != "none":
            # evaluate on val set
            model.eval()
            val_true, val_pred = [], []
            with torch.no_grad():
                for xb_val, yb_val in loader_val:
                    xb_val = xb_val.to(device)
                    logits_val = _forward_logits(model, xb_val, model_name=model_name)
                    val_pred.extend(logits_val.argmax(dim=1).cpu().tolist())
                    val_true.extend(yb_val.tolist())
            val_acc = accuracy_score(val_true, val_pred)
            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
            print(f"[adapt_finetuning] Epoch {epoch:03d}/{num_epochs} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f} | val_acc={val_acc:.4f} | val_f1_macro={val_f1:.4f}")

            # early stopping on the chosen validation metric; keep a copy of the best weights
            val_score = val_f1 if early_stopping == "f1" else val_acc
            if best_val_score is None or val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"[adapt_finetuning] Early stopping: validation {early_stopping} not improved for {early_stopping_patience} epochs")
                break

            _set_train_mode(model)

    if early_stopping != "none":
        model.load_state_dict(best_state)
        print(f"[adapt_finetuning] Restored best weights from epoch {best_epoch} (validation {early_stopping} = {best_val_score:.4f})")


    # -------------------- inference on test_meta --------------------
    model.eval()
    probs_all = []

    with torch.no_grad():
        for xb in tqdm(loader_test, desc="Inference"):
            xb = xb.to(device)
            logits = _forward_logits(model, xb, model_name=model_name)
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs.cpu())

    probs = torch.cat(probs_all, dim=0).numpy()

    pred_label_idx = probs.argmax(axis=1)
    pred_cat_ids = np.array([label_idx_to_cat_id[int(i)] for i in pred_label_idx], dtype=np.int64)

    # Head output index -> COCO category_id -> category name -> class id used by every other method
    # (id_to_name is indexed by the sorted category names of the train pool, not by category_id).
    cat_id_to_name = {}
    for m in ref_meta:
        cat_id_to_name[int(_to_python_scalar(m["category_id"]))] = _to_python_str(m["category_name"])
    name_to_id = {name: i for i, name in id_to_name.items()}

    pseudo_names = np.array([cat_id_to_name[int(c)] for c in pred_cat_ids], dtype=object)
    pseudo_ids = np.array([name_to_id[n] for n in pseudo_names], dtype=np.int64)
    head_class_ids = [name_to_id[cat_id_to_name[label_idx_to_cat_id[i]]] for i in range(train_num_classes)]

    print(f"label_idx_to_cat_id: {label_idx_to_cat_id}")
    print(f"unique pseudo_names: {np.unique(pseudo_names)}, length: {len(pseudo_names)}")
    print("-------------------------------------------------------------")
    category_likelihoods = scores_to_all_classes(probs, head_class_ids, num_classes).tolist()
    return pseudo_ids, pseudo_names, category_likelihoods