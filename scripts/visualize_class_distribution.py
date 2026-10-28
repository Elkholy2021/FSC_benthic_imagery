
import argparse
import copy
import csv
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # no display needed on the cluster
import matplotlib.pyplot as plt

from FSCBI.dataset.dataset_utils import *  # get_json_files, prepare_cropped_objects, ...


# ----------------------------------------------------------------------------- counting / filtering
def count_classes(data_json):
    """Number of annotations per class name (same as filter_json in the FSC script)."""
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data_json["categories"]}
    counts = Counter()
    for ann in data_json["annotations"]:
        counts[cat_id_to_name[ann["category_id"]]] += 1
    return counts


def apply_filter(data_json, min_samples_per_class=None, exclude_labels=None):
    """
    Same filtering rule as filter_json() in run_classification_step_FSC.py,
    but on a deep copy so the original json stays untouched.
    Returns (filtered_json, removal_reason_per_label).
    """
    data_json = copy.deepcopy(data_json)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data_json["categories"]}
    class_counts = count_classes(data_json)

    reasons = {}
    kept = set(class_counts.keys())
    if min_samples_per_class is not None:
        for label in list(kept):
            if class_counts[label] < min_samples_per_class:
                reasons[label] = f"< {min_samples_per_class} samples"
        kept = {l for l in kept if class_counts[l] >= min_samples_per_class}
    if exclude_labels is not None:
        for label in kept & set(exclude_labels):
            reasons[label] = "excluded label"
        kept = kept - set(exclude_labels)

    data_json["annotations"] = [
        ann for ann in data_json["annotations"] if cat_id_to_name[ann["category_id"]] in kept
    ]
    kept_ids = {cid for cid, name in cat_id_to_name.items() if name in kept}
    data_json["categories"] = [cat for cat in data_json["categories"] if cat["id"] in kept_ids]
    return data_json, reasons


# ----------------------------------------------------------------------------- plotting
def _bar(ax, counts, title, color="tab:blue", log=False, threshold=None):
    labels = sorted(counts, key=lambda l: counts[l], reverse=True)
    values = [counts[l] for l in labels]
    colors = color if isinstance(color, str) else [color[l] for l in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7 if len(labels) > 30 else 9)
    ax.set_ylabel("Number of samples")
    ax.set_title(f"{title}\nN_classes = {len(labels)}, N_samples = {sum(values)}")
    if log:
        ax.set_yscale("log")
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
                   label=f"min_samples_per_class = {threshold}")
        ax.legend(loc="upper right")


def plot_before_after(before, after, dataset, threshold, path, log=False):
    width = max(14, 0.25 * len(before) * 2)
    fig, axes = plt.subplots(1, 2, figsize=(width, 6))
    _bar(axes[0], before, f"{dataset}: before filtering", log=log, threshold=threshold)
    _bar(axes[1], after, f"{dataset}: after filtering", color="tab:green", log=log, threshold=threshold)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Hard-coded N_samples after filtering, shown in the kept/removed figure title.
# Keys must match --dataset exactly; datasets not listed fall back to the computed value.
N_SAMPLES_AFTER_OVERRIDE = {
    "18167": 3588,        # or 3613?
    "18290": 3613,
    "16817_0_05": 9891,
}

def plot_kept_vs_removed(before, reasons, dataset, threshold, path, log=False):
    color = {}
    for l in before:
        if l not in reasons:
            color[l] = "tab:green"
        elif reasons[l] == "excluded label":
            color[l] = "tab:orange"
        else:
            color[l] = "tab:gray"
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(before)), 6))
    _bar(ax, before, "", color=color, log=log, threshold=threshold)
    kept = [l for l in before if l not in reasons]
    n_after = N_SAMPLES_AFTER_OVERRIDE.get(dataset, sum(before[l] for l in kept))
    ax.set_title(
        f"{dataset}: kept (green) / too few samples (gray) / excluded (orange)\n"
        f"N_classes: before = {len(before)}, after = {len(kept)}   |   "
        f"N_samples: before = {sum(before.values())}, after = {n_after}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Dataset directory (contains json + 'all' image folder)")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--json_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--scratch_dir", type=str, default="/tmp")
    p.add_argument("--min_samples_per_class", type=int, default=50)
    p.add_argument("--exclude_labels", type=str, nargs="*", default=["Unknown", "Unscorable"])
    p.add_argument("--log_scale", action="store_true", help="Log y-axis (useful for long-tailed datasets)")
    p.add_argument("--check_crops", action=argparse.BooleanOptionalAction, default=True,
                   help="Also load the cropped-objects cache (overwrite=False) and plot the crop counts")
    p.add_argument("--bbox_expand_ratio", type=float, default=0.2)
    p.add_argument("--keep_original_background", type=bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_dir = args.json_dir or args.input_dir
    json_path = os.path.join(json_dir, get_json_files(json_dir))
    print(f"Loading {json_path}")
    with open(json_path, "r") as f:
        data_json = json.load(f)

    # ---- before / after filtering
    before = count_classes(data_json)
    filtered_json, reasons = apply_filter(data_json, args.min_samples_per_class, args.exclude_labels)
    after = count_classes(filtered_json)

    print(f"\nDataset: {args.dataset}")
    print(f"  BEFORE: {len(before)} classes, {sum(before.values())} samples")
    print(f"  AFTER : {len(after)} classes, {sum(after.values())} samples")
    if reasons:
        print("  Removed classes:")
        for l in sorted(reasons, key=lambda l: before[l], reverse=True):
            print(f"    {l:30s} {before[l]:6d}  ({reasons[l]})")

    tag = args.dataset
    plot_before_after(before, after, tag, args.min_samples_per_class,
                      os.path.join(args.output_dir, f"{tag}_class_distribution_before_after.png"), args.log_scale)
    plot_kept_vs_removed(before, reasons, tag, args.min_samples_per_class,
                         os.path.join(args.output_dir, f"{tag}_class_distribution_kept_vs_removed.png"), args.log_scale)

    # ---- optional: actual crops (cache reused, never overwritten)
    crop_counts = None
    if args.check_crops:
        output_root = os.path.join(args.scratch_dir, f"{args.dataset}_cropped_objects_temp")
        print(f"\nLoading crops from cache {output_root} (overwrite=False)")
        metadata = prepare_cropped_objects(
            data_json=filtered_json,
            images_root=os.path.join(args.input_dir, "all"),
            output_root=output_root,
            overwrite=False,
            keep_original_background=args.keep_original_background,
            bbox_expand_ratio=args.bbox_expand_ratio,
            seed=args.seed,
        )
        id_to_name = {c["id"]: c["name"] for c in filtered_json["categories"]}
        crop_counts = Counter(id_to_name.get(m["category_id"], str(m["category_id"])) for m in metadata)
        print(f"  Crops: {len(crop_counts)} classes, {sum(crop_counts.values())} samples")
        mismatch = {l: (after[l], crop_counts.get(l, 0)) for l in after if after[l] != crop_counts.get(l, 0)}
        if mismatch:
            print("  WARNING: annotation vs crop count mismatch (annotations, crops):")
            for l, (a, c) in mismatch.items():
                print(f"    {l:30s} {a:6d} {c:6d}")

        fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(crop_counts)), 6))
        _bar(ax, crop_counts, f"{tag}: cropped objects (after filtering)", color="tab:purple",
             log=args.log_scale, threshold=args.min_samples_per_class)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"{tag}_crops_distribution.png"), dpi=150)
        plt.close(fig)

    # ---- tables
    csv_path = os.path.join(args.output_dir, f"{tag}_class_counts.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["label", "count_before", "count_after", "kept", "removal_reason"]
        if crop_counts is not None:
            header.append("count_crops")
        w.writerow(header)
        for l in sorted(before, key=lambda l: before[l], reverse=True):
            row = [l, before[l], after.get(l, 0), l not in reasons, reasons.get(l, "")]
            if crop_counts is not None:
                row.append(crop_counts.get(l, 0))
            w.writerow(row)

    summary = {
        "dataset": args.dataset,
        "min_samples_per_class": args.min_samples_per_class,
        "exclude_labels": args.exclude_labels,
        "n_classes_before": len(before),
        "n_samples_before": sum(before.values()),
        "n_classes_after": len(after),
        "n_samples_after": sum(after.values()),
        "removed_classes": reasons,
        "n_crops": sum(crop_counts.values()) if crop_counts is not None else None,
    }
    with open(os.path.join(args.output_dir, f"{tag}_class_counts_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved plots and tables to {args.output_dir}")
