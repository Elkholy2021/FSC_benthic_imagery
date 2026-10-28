

import argparse
import glob
import hashlib
import importlib
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr


# ---------------------------------------------------------------------------------------
# Dataset helpers (identical behaviour to run_classification_step_FSC.py)
# ---------------------------------------------------------------------------------------
def filter_json(data_json=None, min_samples_per_class=None, exclude_labels=None):
    """Same filtering as filter_json() in run_classification_step_FSC.py."""
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data_json["categories"]}
    class_counts = Counter(cat_id_to_name[a["category_id"]] for a in data_json["annotations"])

    filtered_labels = set(class_counts.keys())
    if min_samples_per_class is not None:
        filtered_labels = {l for l in filtered_labels if class_counts[l] >= min_samples_per_class}
    if exclude_labels is not None:
        filtered_labels = filtered_labels - set(exclude_labels)

    data_json["annotations"] = [
        a for a in data_json["annotations"] if cat_id_to_name[a["category_id"]] in filtered_labels
    ]
    keep_ids = {cid for cid, name in cat_id_to_name.items() if name in filtered_labels}
    data_json["categories"] = [c for c in data_json["categories"] if c["id"] in keep_ids]
    return data_json


def get_dino_transform():
    import torch
    import torchvision.transforms.v2 as T

    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def patch_size_from_name(dataset_name):
    """'16817_0_01' -> 0.01, '16817_0_1' -> 0.1, '16817_0_15' -> 0.15"""
    m = re.search(r"_(\d+)_(\d+)$", dataset_name)
    if m is None:
        return float("nan")
    return float(f"{m.group(1)}.{m.group(2)}")


def fscbi_set_seed(seed):
    for mod in ("FSCBI.pipelines.backbone", "FSCBI.utils.gpu_utils",
                "FSCBI.evaluation.evaluation_utils", "FSCBI.dataset.dataset_utils"):
        try:
            fn = getattr(importlib.import_module(mod), "set_seed", None)
        except Exception:  # noqa: BLE001
            continue
        if callable(fn):
            fn(seed)
            return mod
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return "fallback"


def build_metadata(dataset_name, seed, args):
    """Cropped-objects metadata, built exactly like run_classification_step_FSC.py."""
    from FSCBI.dataset.dataset_utils import get_json_files, prepare_cropped_objects

    input_dir = os.path.join(args.input_root, dataset_name)
    json_path = os.path.join(input_dir, get_json_files(input_dir))
    images_root = os.path.join(input_dir, "all")
    with open(json_path, "r") as f:
        data_json = json.load(f)

    data_json = filter_json(
        data_json=data_json,
        min_samples_per_class=None if args.no_class_filter else args.min_samples_per_class,
        exclude_labels=args.exclude_labels,
    )
    output_root = os.path.join(args.scratch_dir, f"{dataset_name}_cropped_objects_temp")
    return prepare_cropped_objects(
        data_json=data_json,
        images_root=images_root,
        output_root=output_root,
        overwrite=False,  # crops already on scratch are reused (same as the FSC script)
        keep_original_background=args.keep_original_background,
        bbox_expand_ratio=args.bbox_expand_ratio,
        seed=seed,
    )


def resolve_indices(ds, base):
    """Indices into `base` of a (possibly nested) torch Subset."""
    idx = None
    while ds is not base:
        if hasattr(ds, "indices") and hasattr(ds, "dataset"):
            cur = list(ds.indices)
            idx = cur if idx is None else [cur[i] for i in idx]
            ds = ds.dataset
        else:
            raise RuntimeError(f"cannot resolve indices of {type(ds).__name__}")
    return list(range(len(base))) if idx is None else [int(i) for i in idx]


def training_split(dataset_name, seed, args):
    """Metadata entries of the FSC training split for this seed (same steps, same order)."""
    from FSCBI.dataset.dataset_utils import CroppedObjectsDataset, split_50_50

    fscbi_set_seed(seed)
    metadata = build_metadata(dataset_name, seed, args)
    dataset_all = CroppedObjectsDataset(metadata, transform=get_dino_transform(), augment=False)
    if not args.enable_split_50_50:
        return list(metadata)
    train_ds, _ = split_50_50(dataset_all, seed=seed)

    if isinstance(train_ds, (list, tuple)):
        return list(train_ds)
    try:
        return [metadata[i] for i in resolve_indices(train_ds, dataset_all)]
    except RuntimeError:
        md = getattr(train_ds, "metadata", None)
        if md is None:
            raise RuntimeError(
                "split_50_50 returned an object whose crops cannot be identified "
                "(no Subset .indices and no .metadata). Adapt training_split()."
            )
        return list(md)


def crop_key(entry):
    """Stable identifier of a crop = its full metadata entry."""
    try:
        return json.dumps(entry, sort_keys=True, default=str)
    except TypeError:
        return repr(entry)


def settings_tag(args):
    """Short hash of everything that changes which crops exist or how they look."""
    s = json.dumps({
        "model": args.model_name,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "keep_original_background": args.keep_original_background,
        "min_samples_per_class": None if args.no_class_filter else args.min_samples_per_class,
        "exclude_labels": sorted(args.exclude_labels or []),
    }, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


class EmbeddingCache:
    """Per-crop embedding cache: <dataset>_<model>_<tag>.npz with arrays `keys` and `feats`."""

    def __init__(self, path):
        self.path = path
        self.keys, self.feats, self.index = [], None, {}
        if os.path.isfile(path):
            d = np.load(path, allow_pickle=False)
            self.keys = [str(k) for k in d["keys"].tolist()]
            self.feats = d["feats"]
            self.index = {k: i for i, k in enumerate(self.keys)}

    def __len__(self):
        return len(self.keys)

    def add(self, keys, feats):
        feats = np.asarray(feats, dtype=np.float32)
        self.feats = feats if self.feats is None else np.concatenate([self.feats, feats], axis=0)
        for k in keys:
            self.index[k] = len(self.keys)
            self.keys.append(k)

    def get(self, keys):
        return self.feats[[self.index[k] for k in keys]]

    def save(self):
        tmp = self.path[:-4] + ".tmp.npz"
        np.savez(tmp, keys=np.array(self.keys, dtype=str), feats=self.feats)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------------------
# Backbone loading / embedding
# ---------------------------------------------------------------------------------------
HF_IDS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
}


def load_backbone(model_name, models_dir, device, source="auto"):

    import torch

    errors = []
    family = "dinov3" if "dinov3" in model_name.lower() else "dinov2"

    weights = [os.environ["DINOV3_WEIGHTS"]] if os.environ.get("DINOV3_WEIGHTS") else []
    for ext in ("pth", "pt"):
        weights += sorted(glob.glob(os.path.join(models_dir, "**", f"{model_name}*.{ext}"), recursive=True))
    weights = [w for w in weights if os.path.isfile(w)]
    print(f"[backbone] weights found: {weights if weights else 'none'} (searched {models_dir})")
    if weights and family != "dinov3":
        print("[backbone] note: local weights are only passed to torch.hub for DINOv3")

    if source in ("auto", "hub"):
        hub_dir = torch.hub.get_dir()
        repo_candidates = [os.environ.get("DINOV3_REPO", "")]
        repo_candidates += [os.path.join(models_dir, d) for d in (family, f"{family}-main", f"{family}_main")]
        repo_candidates += sorted(glob.glob(os.path.join(hub_dir, f"*{family}*")))
        repo_candidates += sorted(glob.glob(os.path.join(models_dir, "**", "hubconf.py"), recursive=True))
        repo_candidates = [os.path.dirname(r) if r.endswith("hubconf.py") else r for r in repo_candidates]
        repo_candidates = list(dict.fromkeys(r for r in repo_candidates if r))

        kwargs = {"weights": weights[0]} if (weights and family == "dinov3") else {}
        for repo in repo_candidates:
            if not os.path.isfile(os.path.join(repo, "hubconf.py")):
                errors.append(f"hub-local: no hubconf.py in {repo}")
                continue
            try:
                model = torch.hub.load(repo, model_name, source="local", **kwargs)
                print(f"[backbone] torch.hub local repo={repo} {kwargs}")
                return model.to(device).eval()
            except Exception as e:  # noqa: BLE001
                errors.append(f"hub-local({repo}): {e}")

        try:
            model = torch.hub.load(f"facebookresearch/{family}", model_name, **kwargs)
            print(f"[backbone] torch.hub facebookresearch/{family} {kwargs}")
            return model.to(device).eval()
        except Exception as e:  # noqa: BLE001
            errors.append(f"hub-github(facebookresearch/{family}): {e}")

        if source == "hub":
            raise RuntimeError("Could not load backbone via torch.hub:\n  " + "\n  ".join(errors))

    if source in ("auto", "hf"):
        try:
            from transformers import AutoModel

            hf_id = HF_IDS.get(model_name, model_name)
            local = os.path.join(models_dir, hf_id.split("/")[-1])
            hf_model = AutoModel.from_pretrained(local if os.path.isdir(local) else hf_id)

            class HFWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    out = self.m(pixel_values=x)
                    if getattr(out, "pooler_output", None) is not None:
                        return out.pooler_output
                    return out.last_hidden_state[:, 0]

            print(f"[backbone] HuggingFace {hf_id}")
            return HFWrapper(hf_model).to(device).eval()
        except Exception as e:  # noqa: BLE001
            errors.append(f"hf: {e}")

    raise RuntimeError(
        "Could not load backbone. Tried:\n  " + "\n  ".join(errors)
        + "\nSet DINOV3_REPO=<dir with hubconf.py> and DINOV3_WEIGHTS=<.pth>, "
          "or plug your FSCBI loader into load_backbone()."
    )


def init_backbone(args):
    import torch

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = load_backbone(args.model_name, args.models_dir, device, args.backbone_source)
    return model, device


def unpack_batch(batch):
    """Works for dict batches and (img, label, ...) tuples."""
    import torch

    imgs = None
    if isinstance(batch, dict):
        for k in ("image", "images", "img", "pixel_values", "x"):
            if k in batch:
                imgs = batch[k]
                break
        if imgs is None:
            imgs = next(v for v in batch.values() if torch.is_tensor(v) and v.ndim == 4)
    else:
        imgs = batch[0]
    return imgs


def embed_entries(entries, args, model, device, desc):
    """Embed a list of metadata entries (same dataset class / transform as the FSC script)."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from FSCBI.dataset.dataset_utils import CroppedObjectsDataset

    dataset = CroppedObjectsDataset(entries, transform=get_dino_transform(), augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            out = model(unpack_batch(batch).to(device, non_blocking=True))
            if isinstance(out, dict):  # some DINO forwards return dicts
                out = out.get("x_norm_clstoken", next(iter(out.values())))
            feats.append(out.float().cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    if len(feats) != len(entries):
        raise RuntimeError(f"{desc}: got {len(feats)} embeddings for {len(entries)} crops")
    return feats


# ---------------------------------------------------------------------------------------
# RankMe
# ---------------------------------------------------------------------------------------
SCORE = "rankme"
SCORE_LABEL = "RankMe"


def rankme(Z, eps=1e-7):
    """
    RankMe(Z) = exp(-sum p_k log p_k),  p_k = sigma_k / ||sigma||_1 + eps  (Garrido et al. 2023),
    computed on L2-normalised, mean-centred embeddings.
    """
    Z = Z.astype(np.float64)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    Z = Z - Z.mean(0)
    s = np.sqrt(np.clip(np.linalg.eigvalsh(Z.T @ Z), 0, None))  # singular values via D x D Gram
    p = s / (s.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------------------
REFERENCE_LABELS = {"acc": "Accuracy (%)", "f1": "Macro-F1 (%)"}


def parse_reference(s):
    """'16817_0_01:55.11,16817_0_02:60.44'  or a path to a JSON {dataset: score}."""
    if os.path.isfile(s):
        with open(s) as f:
            return {k: float(v) for k, v in json.load(f).items()}
    return {kv.split(":")[0].strip(): float(kv.split(":")[1]) for kv in s.split(",") if kv.strip()}


def load_gt_runs(gt_root, gt_run, datasets, ns_filter=None):
    """
    Per-seed GT from run_classification_step_FSC.py outputs:
      <gt_root>/<dataset>/<gt_run>/<N_samples>/<dataset>_<model>_<N_samples>_seed<seed>_stats.json
    Returns a long DataFrame: dataset, seed, ns, acc, f1 (in %).
    """
    rows = []
    for ds in datasets:
        pattern = os.path.join(gt_root, ds, gt_run, "*", f"{ds}_*_seed*_stats.json")
        for f in sorted(glob.glob(pattern)):
            m = re.search(r"_seed(-?\d+)_stats\.json$", f)
            ns = os.path.basename(os.path.dirname(f))
            if m is None or (ns_filter and ns not in ns_filter):
                continue
            with open(f) as fh:
                st = json.load(fh)
            rows.append({"dataset": ds, "seed": int(m.group(1)), "ns": ns,
                         "acc": st.get("accuracy"), "f1": st.get("f1_macro")})
    df = pd.DataFrame(rows, columns=["dataset", "seed", "ns", "acc", "f1"])
    for c in ("acc", "f1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if len(df) and df[c].max() <= 1.0:  # stored as fractions -> %
            df[c] = df[c] * 100
    return df


def agreement(score, ref):
    """How well a label-free score (higher = better) predicts the reference ranking."""
    rho, p_rho = spearmanr(score, ref)
    tau, p_tau = kendalltau(score, ref)
    order_s = list(score.sort_values(ascending=False).index)
    order_r = list(ref.sort_values(ascending=False).index)
    pick = order_s[0]
    return {
        "spearman_rho": rho, "spearman_p": p_rho,
        "kendall_tau": tau, "kendall_p": p_tau,
        "top1_hit": int(pick == order_r[0]),
        "top2_overlap": len(set(order_s[:2]) & set(order_r[:2])) / 2,
        "picked": pick,
        "picked_true_rank": order_r.index(pick) + 1,
        "best": order_r[0],
        "regret": float(ref.max() - ref[pick]),          # points lost by trusting the pick
        "random_pick_regret": float(ref.max() - ref.mean()),
        "worst_pick_regret": float(ref.max() - ref.min()),
    }


def attach_refs(df, refs):
    """refs: {name: {dataset: value}} -> ref_<name>, rank_<name> columns."""
    for nm, r in refs.items():
        df[f"ref_{nm}"] = pd.to_numeric(df.index.map(lambda d: r.get(d, np.nan)), errors="coerce")
        df[f"rank_{nm}"] = df[f"ref_{nm}"].rank(ascending=False, method="min")
    return df


def agreement_table(df, score_col, ref_names, split):
    rows, stats = [], {}
    for nm in ref_names:
        have = df[[score_col, f"ref_{nm}"]].dropna()
        if len(have) < 2:
            continue
        st = agreement(have[score_col], have[f"ref_{nm}"])
        stats[nm] = st
        rows.append({"split": split, "reference": nm, "n_patch_sizes": len(have), **st})
    return rows, stats


# ---------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------
def fmt_ps(p):
    return f"{p:g}"


def plot_score_vs_patch(d, ycol, errcol, ylabel, title, out_path, dpi, label, per_seed=None):
    d = d.sort_values("patch_size")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if per_seed is not None:
        for j, s in enumerate(per_seed.columns):
            ax.plot(x, per_seed.loc[d.index, s].values, "-", color="grey", alpha=0.35, lw=1,
                    label="individual seeds" if j == 0 else None)
    yerr = d[errcol] if (errcol and errcol in d) else None
    ax.errorbar(x, d[ycol], yerr=yerr, fmt="o-", capsize=4, lw=2, color="C0", label=label)
    best = list(d.index).index(d[ycol].idxmax())
    ax.annotate("best", (best, d[ycol].iloc[best]), xytext=(0, 10), textcoords="offset points",
                ha="center", color="green", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_ps(p) for p in d["patch_size"]])
    ax.set_xlabel("patch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_vs_references(df, xcol, xerrcol, xlabel, refs, stats, title, out_path, dpi):
    fig, axes = plt.subplots(1, len(refs), figsize=(6 * len(refs), 4.8), squeeze=False)
    for ax, nm in zip(axes[0], refs):
        col = f"ref_{nm}"
        d = df.dropna(subset=[col])
        xerr = d[xerrcol] if (xerrcol and xerrcol in d) else None
        yerr = d[f"{col}_std"] if f"{col}_std" in d else None
        ax.errorbar(d[xcol], d[col], xerr=xerr, yerr=yerr, fmt="o", capsize=3)
        for _, row in d.iterrows():
            ax.annotate(fmt_ps(row["patch_size"]), (row[xcol], row[col]),
                        xytext=(4, 4), textcoords="offset points", fontsize=9)
        st = stats.get(nm)
        # sub = (f"\nSpearman ρ={st['spearman_rho']:+.2f}  Kendall τ={st['kendall_tau']:+.2f}  "
        #        f"regret={st['regret']:.2f} pts") if st else ""
        sub = ""
        ax.set_title(f"{REFERENCE_LABELS.get(nm, nm)}{sub}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(REFERENCE_LABELS.get(nm, nm))
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="INPUT_DIR (needed to rebuild the splits)")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True,
                    help="Seeds of the FSC runs; each seed defines one training split")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embeddings_dir", default="",
                    help="Per-crop embedding cache (default: <output_dir>/embeddings)")
    ap.add_argument("--scratch_dir", default="/tmp", help="Must be the scratch_dir of the FSC runs")
    ap.add_argument("--models_dir", default="./models")
    ap.add_argument("--model_name", default="dinov3_vits16")
    ap.add_argument("--backbone_source", default="auto", choices=["auto", "hub", "hf"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)

    # must match run_classification_step_FSC.py
    ap.add_argument("--bbox_expand_ratio", type=float, default=0.2)
    ap.add_argument("--keep_original_background", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--min_samples_per_class", type=int, default=50)
    ap.add_argument("--exclude_labels", nargs="*", default=["Unknown", "Unscorable"])
    ap.add_argument("--no_class_filter", action="store_true",
                    help="Skip the min_samples_per_class filter (then splits differ from the FSC runs)")
    ap.add_argument("--enable_split_50_50", type=lambda s: s.lower() in ("1", "true", "yes"), default=True,
                    help="Same as the FSC script (its bool flag is effectively always True)")

    ap.add_argument("--n_sub", type=int, default=50, help="Subsamples per split for within-split spread (0 = off)")
    ap.add_argument("--sub_frac", type=float, default=0.5, help="Fraction of training crops per subsample")

    ap.add_argument("--gt_root", default="", help="RESULTS_DIR/classification/patch_size_analysis")
    ap.add_argument("--gt_run", default="", help="<model>_<method>_<strategy>_FSC folder of the FSC runs")
    ap.add_argument("--gt_ns", nargs="*", default=None, help="N_samples to average GT over (default: all found)")
    ap.add_argument("--reference", action="append", default=[],
                    help="Static GT (same for every split), only used if no per-seed GT is found: "
                         "NAME=ds1:v1,ds2:v2 or NAME=/path.json (NAME=acc / NAME=f1)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--analysis_only", action="store_true", help="Fail instead of extracting embeddings")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_dir = os.path.join(args.output_dir, "per_seed")
    comb_dir = os.path.join(args.output_dir, "combined")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(comb_dir, exist_ok=True)
    emb_dir = args.embeddings_dir or os.path.join(args.output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    tag = settings_tag(args)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "cache_tag": tag}, f, indent=4)

    # ---------------- ground truth ----------------
    gt_long = pd.DataFrame()
    if args.gt_root and args.gt_run:
        gt_long = load_gt_runs(args.gt_root, args.gt_run, args.datasets, args.gt_ns)
        gt_long = gt_long[gt_long["seed"].isin(args.seeds)]
        if gt_long.empty:
            print(f"[GT] no *_stats.json found under {args.gt_root}/<dataset>/{args.gt_run}/<ns>/")
        else:
            gt_long.to_csv(os.path.join(args.output_dir, "gt_per_seed_per_ns.csv"), index=False)
            counts = gt_long.groupby(["dataset", "seed"]).size().unstack("seed")
            print(f"[GT] per-seed results found ({len(gt_long)} files); N_samples values per dataset x seed:")
            print(counts.to_string())
            vals = counts.values.astype(float).ravel()
            if np.isnan(vals).any() or len(set(vals[~np.isnan(vals)])) > 1:
                print("[GT] WARNING: not every dataset/seed has the same number of N_samples results; "
                      "GT averages are over different N_samples sets")
    static_refs = {}
    for item in args.reference:
        nm, spec = item.split("=", 1)
        static_refs[nm.strip()] = parse_reference(spec.strip())

    def refs_for_seed(seed):
        if not gt_long.empty:
            g = gt_long[gt_long["seed"] == seed].groupby("dataset")[["acc", "f1"]].mean()
            return {nm: g[nm].dropna().to_dict() for nm in ("acc", "f1") if g[nm].notna().any()}
        return static_refs

    gt_source = "per-seed FSC results" if not gt_long.empty else ("static --reference" if static_refs else "none")
    print(f"[GT] source: {gt_source}")

    # ---------------- per seed: split -> embeddings -> RankMe ----------------
    caches = {n: EmbeddingCache(os.path.join(emb_dir, f"{n}_{args.model_name}_{tag}.npz")) for n in args.datasets}
    backbone = {"model": None, "device": None}
    manifest, per_seed, all_ag, seed_stats = [], {}, [], {}

    for seed in args.seeds:
        print(f"\n================ seed {seed} ================")
        Z_by, n_train = {}, {}
        for name in args.datasets:
            entries = training_split(name, seed, args)
            keys = [crop_key(e) for e in entries]
            if len(set(keys)) != len(keys):
                print(f"[{name} seed {seed}] WARNING: {len(keys) - len(set(keys))} duplicate crop entries")
            cache = caches[name]
            missing = {}
            for k, e in zip(keys, entries):
                if k not in cache.index and k not in missing:
                    missing[k] = e
            if missing:
                if args.analysis_only:
                    raise FileNotFoundError(f"--analysis_only but {len(missing)} crops of {name} seed {seed} "
                                            f"are not in {cache.path}")
                if backbone["model"] is None:
                    backbone["model"], backbone["device"] = init_backbone(args)
                t0 = time.time()
                f_new = embed_entries(list(missing.values()), args, backbone["model"], backbone["device"],
                                      f"embedding {name} seed {seed}")
                cache.add(list(missing.keys()), f_new)
                cache.save()
                print(f"[{name} seed {seed}] embedded {len(missing)} new crops in {time.time() - t0:.1f}s "
                      f"(cache now {len(cache)})")
            Z_by[name] = cache.get(keys)
            n_train[name] = len(keys)
            manifest.append({"seed": seed, "dataset": name, "n_train": len(keys),
                             "split_md5": hashlib.md5("\n".join(sorted(keys)).encode()).hexdigest()})
            print(f"[{name} seed {seed}] training split: {len(keys)} crops")

        # same number of crops for every patch size (RankMe grows slightly with N)
        rng = np.random.default_rng(seed)
        n_common = min(n_train.values())
        n_sub = int(args.sub_frac * n_common)
        sub_idx = [np.sort(rng.choice(n_common, size=n_sub, replace=False)) for _ in range(args.n_sub)]
        rows, subs = {}, {}
        for name in args.datasets:
            Z = Z_by[name]
            if len(Z) > n_common:
                Z = Z[np.sort(rng.choice(len(Z), size=n_common, replace=False))]
            row = {"seed": seed, "patch_size": patch_size_from_name(name),
                   "n_train": n_train[name], "n_used": n_common}
            row[SCORE] = rankme(Z)
            if args.n_sub > 0:
                subs[name] = np.array([rankme(Z[b]) for b in sub_idx])
                row[f"{SCORE}_sub_mean"] = subs[name].mean()
                row[f"{SCORE}_sub_std"] = subs[name].std()
            rows[name] = row

        df = pd.DataFrame(rows).T.apply(pd.to_numeric)
        df.index.name = "dataset"
        df["seed"] = df["seed"].astype(int)
        df[f"rank_{SCORE}"] = df[SCORE].rank(ascending=False, method="min").astype(int)
        if args.n_sub > 0:
            B = np.stack([subs[n] for n in df.index], axis=0)
            winners = Counter(np.array(df.index)[B.argmax(0)])
            df[f"{SCORE}_sub_top1_freq"] = [winners.get(n, 0) / args.n_sub for n in df.index]

        refs = refs_for_seed(seed)
        df = attach_refs(df, refs)
        rows_ag, stats = agreement_table(df, SCORE, list(refs), f"seed {seed}")
        for r in rows_ag:
            r["seed"] = seed
        all_ag += rows_ag
        seed_stats[seed] = stats
        per_seed[seed] = df

        # ---- per-seed outputs ----
        out = os.path.join(seed_dir, f"seed_{seed}")
        os.makedirs(out, exist_ok=True)
        df.sort_values(f"rank_{SCORE}").to_csv(os.path.join(out, "rankme_ranking.csv"))
        show = ["patch_size", "n_used", SCORE, f"rank_{SCORE}"] + [c for nm in refs for c in (f"ref_{nm}", f"rank_{nm}")]
        print(f"\nseed {seed}: ranking by {SCORE_LABEL} (GT: {gt_source})")
        print(df.sort_values(f"rank_{SCORE}")[show].round(3).to_string())
        for r in rows_ag:
            print(f"  vs {r['reference']}: Spearman={r['spearman_rho']:+.2f} Kendall={r['kendall_tau']:+.2f} "
                  f"top1_hit={r['top1_hit']} picked={r['picked']} (true rank {r['picked_true_rank']}) "
                  f"regret={r['regret']:.2f}")

        sub_std = f"{SCORE}_sub_std" if args.n_sub > 0 else None
        plot_score_vs_patch(df, SCORE, sub_std, SCORE_LABEL,
                            f"seed {seed}: {SCORE_LABEL} on the training split",
                            os.path.join(out, f"{SCORE}_vs_patch_size.png"), args.dpi,
                            label=f"all training crops ± subsample std ({args.sub_frac:g})")
        if refs:
            plot_vs_references(df, SCORE, sub_std, SCORE_LABEL, list(refs), stats,
                               f"seed {seed}: {SCORE_LABEL} vs ground truth (one dot per patch size)",
                               os.path.join(out, f"{SCORE}_vs_acc_f1.png"), args.dpi)

    pd.DataFrame(manifest).to_csv(os.path.join(args.output_dir, "train_split_manifest.csv"), index=False)

    # ---------------- all seeds, long format ----------------
    long = pd.concat([d.reset_index() for d in per_seed.values()], ignore_index=True)
    long.to_csv(os.path.join(args.output_dir, "per_seed_rankings.csv"), index=False)
    ag = pd.DataFrame(all_ag)
    ref_names = [nm for nm in ("acc", "f1") if f"ref_{nm}" in long] + \
                [nm for nm in static_refs if nm not in ("acc", "f1") and f"ref_{nm}" in long]
    if not ag.empty:
        ag.to_csv(os.path.join(args.output_dir, "per_seed_agreement.csv"), index=False)

    # ---------------- combined over seeds ----------------
    g = long.groupby("dataset")
    comb = pd.DataFrame({"patch_size": g["patch_size"].first(), "n_seeds": g["seed"].nunique()})
    comb[f"{SCORE}_mean"] = g[SCORE].mean()
    comb[f"{SCORE}_std"] = g[SCORE].std(ddof=1) if len(args.seeds) > 1 else 0.0
    comb[f"{SCORE}_mean_rank"] = g[f"rank_{SCORE}"].mean()
    comb[f"{SCORE}_seeds_top1"] = g[f"rank_{SCORE}"].apply(lambda r: float((r == 1).mean()))
    comb[f"rank_{SCORE}"] = comb[f"{SCORE}_mean"].rank(ascending=False, method="min").astype(int)
    for nm in ref_names:
        comb[f"ref_{nm}"] = g[f"ref_{nm}"].mean()
        comb[f"ref_{nm}_std"] = g[f"ref_{nm}"].std(ddof=1) if len(args.seeds) > 1 else 0.0
        comb[f"ref_{nm}_mean_rank"] = g[f"rank_{nm}"].mean()
        comb[f"rank_{nm}"] = comb[f"ref_{nm}"].rank(ascending=False, method="min")
    comb_rows, comb_stats = agreement_table(comb, f"{SCORE}_mean", ref_names, "combined")

    comb = comb.sort_values(f"rank_{SCORE}")
    comb.to_csv(os.path.join(args.output_dir, "combined_ranking.csv"))
    print(f"\n================ combined over {len(args.seeds)} seeds ================")
    show = ["patch_size", f"{SCORE}_mean", f"{SCORE}_std", f"rank_{SCORE}", f"{SCORE}_mean_rank", f"{SCORE}_seeds_top1"]
    show += [c for nm in ref_names for c in (f"ref_{nm}", f"rank_{nm}")]
    print(comb[show].round(3).to_string())

    if not ag.empty:
        summary = ag.groupby("reference").agg(
            n_splits=("split", "count"),
            spearman_mean=("spearman_rho", "mean"), spearman_std=("spearman_rho", "std"),
            kendall_mean=("kendall_tau", "mean"), kendall_std=("kendall_tau", "std"),
            top1_hit_rate=("top1_hit", "mean"), top2_overlap_mean=("top2_overlap", "mean"),
            regret_mean=("regret", "mean"), regret_std=("regret", "std"),
            random_pick_regret_mean=("random_pick_regret", "mean"),
        ).reset_index()
        cdf = pd.DataFrame(comb_rows)
        summary = summary.merge(
            cdf[["reference", "spearman_rho", "kendall_tau", "top1_hit", "picked", "regret"]]
            .rename(columns=lambda c: c if c == "reference" else f"combined_{c}"),
            on="reference", how="left")
        summary.to_csv(os.path.join(args.output_dir, "agreement_summary.csv"), index=False)
        print("\nAgreement with ground truth (per-split mean ± std, and the combined ranking):")
        print(summary.round(3).to_string(index=False))

    # ---------------- combined figures ----------------
    per_seed_vals = long.pivot(index="dataset", columns="seed", values=SCORE)
    plot_score_vs_patch(comb, f"{SCORE}_mean", f"{SCORE}_std", SCORE_LABEL,
                        f"{SCORE_LABEL} on the training splits ({len(args.seeds)} seeds)",
                        os.path.join(comb_dir, f"{SCORE}_vs_patch_size.png"), args.dpi,
                        label="mean over seeds ± std", per_seed=per_seed_vals)
    if ref_names:
        plot_vs_references(comb, f"{SCORE}_mean", f"{SCORE}_std", SCORE_LABEL, ref_names, comb_stats,
                           f"Combined: {SCORE_LABEL} vs ground truth (mean ± std over seeds)",
                           os.path.join(comb_dir, f"{SCORE}_vs_acc_f1.png"), args.dpi)
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
