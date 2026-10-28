import os
import re
import json
import argparse
from token import AT
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from rich.console import Console
from rich.table import Table
from rich import box
import numpy as np

from rich.console import Console
from rich.table import Table
from rich import box
import numpy as np
from rich.table import Table

import re
from scipy import stats
from rich.markup import escape


# --------------------------
# Statistics helpers (confidence intervals + paired comparisons)
# --------------------------
def _seed_key_from_path(path: str) -> str:
    """
    Build a key identifying the random split/seed of a result file, so that runs of
    different configurations can be PAIRED (same split -> same key).

    Looks for 'seed<N>', 'seed_<N>' or 'seed-<N>' anywhere in the path; otherwise
    falls back to the file name without extension.
    """
    p = path.replace("\\", "/").lower()
    m = re.findall(r"seed[_\-]?(\d+)", p)
    if m:
        return f"seed{int(m[-1])}"
    return os.path.splitext(os.path.basename(path))[0]


def mean_confidence_interval(values, level: float = 0.95, method: str = "t",
                             n_boot: int = 10000, rng_seed: int = 0):
    """
    Confidence interval for the MEAN of `values` (e.g. the 10 repeated runs).

    method="t"         : Student-t interval, mean ± t_{n-1} * s / sqrt(n)  (s uses ddof=1)
    method="bootstrap" : percentile bootstrap of the mean (n_boot resamples)

    Returns (low, high); (nan, nan) if fewer than 2 finite values.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (np.nan, np.nan)

    if method == "bootstrap":
        rng = np.random.default_rng(rng_seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = x[idx].mean(axis=1)
        alpha = 1.0 - level
        lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
        return (float(lo), float(hi))

    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.5 + level / 2.0, df=n - 1)
    return (float(m - tcrit * se), float(m + tcrit * se))


def paired_difference_stats(values_a: dict, values_b: dict, level: float = 0.95, method: str = "t"):
    """
    Paired comparison of two configurations evaluated on the same repeated splits.

    values_a / values_b : {seed_key: metric_value}
    Differences are d_i = a_i - b_i for each shared split i.

    Pairing:
      - by seed key when both configs share >= 2 keys ("seed-key")
      - otherwise by run order when both have the same number of runs ("order", warn)

    Returns dict (or None if pairing impossible) with:
      n, pairing, mean_a, mean_b, mean_diff, sd_diff, ci_low, ci_high,
      t_stat, p_ttest (paired t-test), p_wilcoxon (Wilcoxon signed-rank), cohens_dz
    """
    common = [k for k in values_a.keys() if k in values_b]
    if len(common) >= 2:
        a = np.array([values_a[k] for k in common], dtype=float)
        b = np.array([values_b[k] for k in common], dtype=float)
        pairing = "seed-key"
    elif len(values_a) == len(values_b) and len(values_a) >= 2:
        a = np.array(list(values_a.values()), dtype=float)
        b = np.array(list(values_b.values()), dtype=float)
        pairing = "order"
    else:
        return None

    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 2:
        return None

    d = a - b
    mean_diff = float(d.mean())
    sd_diff = float(d.std(ddof=1))
    ci_low, ci_high = mean_confidence_interval(d, level=level, method=method)

    if sd_diff > 0:
        t_res = stats.ttest_rel(a, b)
        t_stat, p_ttest = float(t_res.statistic), float(t_res.pvalue)
        cohens_dz = mean_diff / sd_diff
    else:
        # all differences identical -> test undefined
        t_stat, p_ttest, cohens_dz = np.nan, np.nan, np.nan

    try:
        p_wilcoxon = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        # e.g. all differences are zero
        p_wilcoxon = np.nan

    return {
        "n": int(n),
        "pairing": pairing,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
        "cohens_dz": cohens_dz,
    }


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjustment. NaNs are kept as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    m = valid.size
    if m == 0:
        return out
    order = valid[np.argsort(p[valid])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, adj)
        out[idx] = running_max
    return out


# --------------------------
# JSON helpers
# --------------------------
def load_stats_from_json(json_path: str) -> dict:
    """Load run statistics from a single JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def _get_first_key(d: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def aggregate_stats_across_seeds(json_paths):
    """
    Aggregate JSON statistics across multiple seeds.
    Keeps only:
      - total_time from time_taken_seconds
      - gpu_memory from GPU_memory_used
      - params, flops (first seed)
    """
    per_seed = [load_stats_from_json(p) for p in json_paths]

    total_times = np.array(
        [
            float(
                _get_first_key(
                    d,
                    ("time_taken_seconds", "time_taken_sec", "total_time_sec", "total_time_seconds"),
                    np.nan,
                )
            )
            for d in per_seed
        ],
        dtype=float,
    )

    gpu_mem = np.array(
        [
            float(
                _get_first_key(
                    d,
                    ("GPU_memory_used", "gpu_memory_used", "gpu_mem_used", "gpu_memory", "GPU_mem_used"),
                    np.nan,
                )
            )
            for d in per_seed
        ],
        dtype=float,
    )

    params = _get_first_key(per_seed[0], ("params", "Params", "num_params", "param_count"), None)
    flops = _get_first_key(per_seed[0], ("flops", "FLOPs", "gflops", "GFLOPs", "macs", "MACs"), None)

    return {
        "n_seeds": len(per_seed),
        "total_time_mean": float(np.nanmean(total_times)),
        "total_time_std": float(np.nanstd(total_times)),
        "gpu_memory_mean": float(np.nanmean(gpu_mem)),
        "gpu_memory_std": float(np.nanstd(gpu_mem)),
        "params": params,
        "flops": flops,
    }


# --------------------------
# Plotting: Total time + GPU memory (from JSON)
# --------------------------
def _collect_all_n_samples(all_models_stats: dict):
    all_n = set()
    for model_stats in all_models_stats.values():
        all_n.update(model_stats.keys())
    return sorted(all_n)


def plot_total_time_from_json(all_models_stats, dataset_name, save_dir=None, save_figures=False, plot_figures=True, show_only_model_name=False):
    """Plot total time comparison across models using JSON field time_taken_seconds."""
    if not all_models_stats:
        print("No data to plot")
        return

    n_samples_list = _collect_all_n_samples(all_models_stats)
    if not n_samples_list:
        print("No sample sizes found")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models_stats)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    plt.figure(figsize=(10, 6))
    for idx, (model_name, model_stats) in enumerate(all_models_stats.items()):
        n_samples = [n for n in n_samples_list if n in model_stats]
        means = np.array([model_stats[n].get("total_time_mean", np.nan) for n in n_samples], dtype=float)
        stds = np.array([model_stats[n].get("total_time_std", np.nan) for n in n_samples], dtype=float)
        if show_only_model_name:
            model_name = get_model_name(model_name)
        plt.plot(
            n_samples,
            means,
            marker=markers[idx % len(markers)],
            label=model_name,
            linewidth=2,
            color=colors[idx],
        )
        plt.fill_between(n_samples, means - stds, means + stds, alpha=0.2, color=colors[idx])

    plt.xlabel("Number of samples", fontsize=12)
    plt.ylabel("Total Time (seconds)", fontsize=12)
    plt.title(f"{dataset_name} - Total Time (time_taken_seconds) Comparison", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xticks(n_samples_list)
    plt.tight_layout()

    if save_figures and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_total_time_time_taken_seconds.png"), dpi=150)

    if plot_figures:
        plt.show()
    else:
        plt.close()

def get_model_name(full_name):
    if full_name.startswith("resnet"):
        model_name = full_name.split("_")[0]
    else:
        model_name = full_name.split("_")[0] +'_'  + full_name.split("_")[1] 
    return model_name

def plot_gpu_memory_from_json(all_models_stats, dataset_name, save_dir=None, save_figures=False, plot_figures=True,show_only_model_name=False):
    """Plot GPU memory used comparison across models using JSON field GPU_memory_used."""
    if not all_models_stats:
        print("No data to plot")
        return

    n_samples_list = _collect_all_n_samples(all_models_stats)
    if not n_samples_list:
        print("No sample sizes found")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models_stats)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    plt.figure(figsize=(10, 6))
    for idx, (model_name, model_stats) in enumerate(all_models_stats.items()):
        n_samples = [n for n in n_samples_list if n in model_stats]
        means = np.array([model_stats[n].get("gpu_memory_mean", np.nan) for n in n_samples], dtype=float)
        stds = np.array([model_stats[n].get("gpu_memory_std", np.nan) for n in n_samples], dtype=float)
        if show_only_model_name:
            model_name = get_model_name(model_name)
        plt.plot(
            n_samples,
            means,
            marker=markers[idx % len(markers)],
            label=model_name,
            linewidth=2,
            color=colors[idx],
        )
        plt.fill_between(n_samples, means - stds, means + stds, alpha=0.2, color=colors[idx])

    plt.xlabel("Number of samples", fontsize=12)
    plt.ylabel("GPU Memory Used", fontsize=12)
    plt.title(f"{dataset_name} - GPU Memory Used (GPU_memory_used) Comparison", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xticks(n_samples_list)
    plt.tight_layout()

    if save_figures and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_gpu_memory_used.png"), dpi=150)

    if plot_figures:
        plt.show()
    else:
        plt.close()


# --------------------------
# File discovery
# --------------------------
def find_all_json_by_samples(results_dir: str, model_folder: str, dataset_name: str, n_cap: int = None):
    """
    Return dict: {N_samples: [json_paths]}
    Structure: results_dir/model_folder/N_samples/*.json
    """
    model_path = os.path.join(results_dir, model_folder)
    if not os.path.isdir(model_path):
        return {}

    samples_dict = {}
    for n_samples_dir in os.listdir(model_path):
        n_samples_path = os.path.join(model_path, n_samples_dir)
        if not os.path.isdir(n_samples_path):
            continue

        try:
            n_samples = int(n_samples_dir)
        except ValueError:
            continue

        if n_cap is not None and n_samples > n_cap:
            continue

        json_files = [os.path.join(n_samples_path, fn) for fn in os.listdir(n_samples_path) if fn.endswith(".json")]
        if json_files:
            samples_dict[n_samples] = sorted(json_files)

    return samples_dict


def find_all_seed_csvs_by_samples(results_dir: str, model_folder: str, dataset_name: str, n_cap: int = None):
    """
    Return dict: {N_samples: [csv_paths]}
    Structure: results_dir/model_folder/N_samples/*.csv
    """
    model_path = os.path.join(results_dir, model_folder)
    if not os.path.isdir(model_path):
        return {}

    samples_dict = {}
    for n_samples_dir in os.listdir(model_path):
        n_samples_path = os.path.join(model_path, n_samples_dir)
        if not os.path.isdir(n_samples_path):
            continue

        try:
            n_samples = int(n_samples_dir)
        except ValueError:
            continue

        if n_cap is not None and n_samples > n_cap:
            continue

        csv_files = [os.path.join(n_samples_path, fn) for fn in os.listdir(n_samples_path) if fn.endswith(".csv")]
        if csv_files:
            samples_dict[n_samples] = sorted(csv_files)

    return samples_dict


# --------------------------
# Core eval on ONE CSV (NO top-k, NO MRR)
# --------------------------
def evaluate_pseudo_labels_single(csv_path, use_names=True):
    df = pd.read_csv(csv_path)

    if use_names:
        y_true_raw = df["gt_category_name"].astype(str).values
        y_pred_raw = df["pseudo_category_name"].astype(str).values
        label_type = "names"
    else:
        y_true_raw = df["gt_category_id"].astype(int).values
        y_pred_raw = df["pseudo_category_id"].astype(int).values
        label_type = "ids"

    le = LabelEncoder()
    le.fit(np.concatenate([y_true_raw, y_pred_raw], axis=0))

    y_true = le.transform(y_true_raw)
    y_pred = le.transform(y_pred_raw)
    labels = le.classes_.tolist()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "label_type": label_type,
        "labels": labels,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
    }


def aggregate_metrics_across_seeds(csv_paths, use_names=True, ci_level=0.95, ci_method="t"):
    """
    Aggregate metrics across multiple seed CSVs for a single N_samples value.

    Besides mean/std, this also stores:
      - <metric>_ci_low / <metric>_ci_high : CI of the mean across runs
      - per_seed[<metric>] = {seed_key: value} : raw per-run values (used for paired tests)
    """
    per_seed = [evaluate_pseudo_labels_single(p, use_names=use_names) for p in csv_paths]

    seed_keys = [_seed_key_from_path(p) for p in csv_paths]
    if len(set(seed_keys)) != len(seed_keys):
        # keys not unique -> fall back to run index (pairing will then be by order)
        seed_keys = [f"run{i}" for i in range(len(csv_paths))]

    accs = np.array([d["accuracy"] for d in per_seed], dtype=float)
    f1m = np.array([d["f1_macro"] for d in per_seed], dtype=float)
    f1mi = np.array([d["f1_micro"] for d in per_seed], dtype=float)
    f1w = np.array([d["f1_weighted"] for d in per_seed], dtype=float)
    precisions = np.array([d["precision"] for d in per_seed], dtype=float)
    recalls = np.array([d["recall"] for d in per_seed], dtype=float)

    out = {
        "n_seeds": len(per_seed),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "f1_macro_mean": float(f1m.mean()),
        "f1_macro_std": float(f1m.std()),
        "f1_micro_mean": float(f1mi.mean()),
        "f1_micro_std": float(f1mi.std()),
        "f1_weighted_mean": float(f1w.mean()),
        "f1_weighted_std": float(f1w.std()),
        "precision_mean": float(precisions.mean()),
        "precision_std": float(precisions.std()),
        "recall_mean": float(recalls.mean()),
        "recall_std": float(recalls.std()),
    }

    # ---- Confidence intervals of the mean + raw per-run values ----
    out["ci_level"] = ci_level
    out["ci_method"] = ci_method
    out["per_seed"] = {}
    for name, arr in (
        ("accuracy", accs),
        ("f1_macro", f1m),
        ("f1_micro", f1mi),
        ("f1_weighted", f1w),
        ("precision", precisions),
        ("recall", recalls),
    ):
        lo, hi = mean_confidence_interval(arr, level=ci_level, method=ci_method)
        out[f"{name}_ci_low"] = lo
        out[f"{name}_ci_high"] = hi
        out["per_seed"][name] = dict(zip(seed_keys, arr.tolist()))

    return out


# --------------------------
# Plot: Compare models across metrics (NO top-k, NO MRR)
# --------------------------
def plot_models_comparison(all_models_metrics, dataset_name, save_dir=None, save_figures=False, plot_figures=True, scale_yaxis=False,show_only_model_name=False):
    """
    Plots:
      1) Accuracy
      2) F1 Macro
      3) F1 Weighted
      4) Precision
      5) Recall
    """
    if not all_models_metrics:
        print("No data to plot")
        return

    all_n_samples = set()
    for model_metrics in all_models_metrics.values():
        all_n_samples.update(model_metrics.keys())
    n_samples_list = sorted(all_n_samples)

    if not n_samples_list:
        print("No sample sizes found")
        return

    model_families = {
        "dinov2": "#1f77b4",
        "dinov3": "#2ca02c",
        "resnet": "#d62728",
    }

    all_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X", "P", "d", "8"]
    marker_colors = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#9467bd",
                     "#8c564b", "#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]

    model_styles = {}
    family_marker_idx = {family: 0 for family in model_families.keys()}
    global_marker_idx = 0

    for model_name in all_models_metrics.keys():
        family = None
        model_lower = model_name.lower()
        for family_name in model_families.keys():
            if family_name in model_lower:
                family = family_name
                break

        if family is None:
            line_color = "#7f7f7f"
        else:
            line_color = model_families[family]
            family_marker_idx[family] += 1

        marker = all_markers[global_marker_idx % len(all_markers)]
        marker_color = marker_colors[global_marker_idx % len(marker_colors)]
        model_styles[model_name] = {"line_color": line_color, "marker": marker, "marker_color": marker_color}
        global_marker_idx += 1

    sorted_model_names = sorted(all_models_metrics.keys())

    def _plot_metric(metric_key_mean, metric_key_std, y_label, title_suffix, filename, show_only_model_name=False):
        plt.figure(figsize=(10, 6))
        all_means = []

        for model_name in sorted_model_names:
            model_metrics = all_models_metrics[model_name]
            n_samples = [n for n in n_samples_list if n in model_metrics]
            means = np.array([model_metrics[n].get(metric_key_mean, np.nan) for n in n_samples], dtype=float)
            stds = np.array([model_metrics[n].get(metric_key_std, np.nan) for n in n_samples], dtype=float)
            all_means.extend(means[~np.isnan(means)].tolist())

            style = model_styles[model_name]
            if show_only_model_name:
                model_name = get_model_name(model_name)
            plt.plot(
                n_samples,
                means,
                marker=style["marker"],
                label=model_name,
                linewidth=2,
                color=style["line_color"],
                markerfacecolor=style["marker_color"],
                markeredgecolor=style["marker_color"],
                markersize=5,
            )
            plt.fill_between(n_samples, means - stds, means + stds, alpha=0.2, color=style["line_color"])
        datasets_dict = {"18167": "A","18290": "B","16817": "C"}
        for key, value in datasets_dict.items():
            if key in str(dataset_name):
                dataset_code = value
                break

        plt.xlabel("Number of samples", fontsize=18)
        plt.ylabel(y_label, fontsize=18)
        plt.title(f"Dataset {dataset_code} - {dataset_name} - {title_suffix}", fontsize=20)
        plt.legend(loc="best", fontsize=14)
        plt.grid(True, alpha=0.3)

        if scale_yaxis and len(all_means) > 0:
            min_val = float(np.min(all_means))
            max_val = float(np.max(all_means))
            plt.ylim([0.9 * min_val, 1.1 * max_val])
        else:
            plt.ylim([0, 1])

        plt.xticks(n_samples_list)
        plt.tight_layout()

        if save_figures and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename), dpi=150)

        if plot_figures:
            plt.show()
        else:
            plt.close()

    _plot_metric("accuracy_mean", "accuracy_std", "Accuracy", "Accuracy Comparison", f"{dataset_name}_accuracy_comparison.png", show_only_model_name=show_only_model_name)
    _plot_metric("f1_macro_mean", "f1_macro_std", "F1 Macro", "F1 Macro Comparison", f"{dataset_name}_f1_macro_comparison.png", show_only_model_name=show_only_model_name)
    _plot_metric("f1_weighted_mean", "f1_weighted_std", "F1 Weighted", "F1 Weighted Comparison", f"{dataset_name}_f1_weighted_comparison.png", show_only_model_name=show_only_model_name)
    _plot_metric("precision_mean", "precision_std", "Precision", "Precision Comparison", f"{dataset_name}_precision_comparison.png", show_only_model_name=show_only_model_name)
    _plot_metric("recall_mean", "recall_std", "Recall", "Recall Comparison", f"{dataset_name}_recall_comparison.png", show_only_model_name=show_only_model_name)


# --------------------------
# Summary table (NO train/test columns)
# --------------------------
def show_models_summary_table_at_n(
    all_models_metrics: dict,
    n_samples: int,
    sort_by: str = "f1_macro",
    ascending: bool = False,
    display: bool = True,
    save_csv_path: str | None = None,
    keep_human_readable: bool = True,
    show_pretty_units: bool = True,
    show_only_model_name: bool = False,
    dataset_name: str = "",
):
    """
    Summary table @ fixed N_samples_per_class.

    sort_by options:
      - "model_alpha"
      - "flops"
      - "params"
      - "total_time"
      - "gpu_memory"
      - "accuracy"
      - "f1_macro"
      - "f1_weighted"
      - "precision"
      - "recall"
    """

    def _parse_human_number(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if not isinstance(x, str):
            return np.nan

        s = x.strip().replace(",", "")
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*([KMGTP]?)$", s, flags=re.IGNORECASE)
        if not m:
            return np.nan

        val = float(m.group(1))
        suf = m.group(2).upper()
        mult = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}[suf]
        return val * mult

    def _get_first_present(d: dict, keys: tuple[str, ...], default=np.nan):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    def _format_g(x):
        return "-" if pd.isna(x) else f"{x / 1e9:.2f}G"

    def _format_m(x):
        return "-" if pd.isna(x) else f"{x / 1e6:.0f}M"

    rows = []
    for model_name, per_n in all_models_metrics.items():
        if n_samples not in per_n:
            continue
        d = per_n[n_samples]

        flops_raw = _get_first_present(d, ("flops", "FLOPs", "gflops", "GFLOPs", "macs", "MACs"))
        params_raw = _get_first_present(d, ("params", "Params", "num_params", "n_params", "param_count"))
        if show_only_model_name:
            model_name = get_model_name(model_name)
        row = {
            "model": model_name,
            "flops": _parse_human_number(flops_raw),
            "params": _parse_human_number(params_raw),
            "total_time_sec": d.get("total_time_mean", np.nan),
            "gpu_memory_used": d.get("gpu_memory_mean", np.nan),
            "accuracy": d.get("accuracy_mean", np.nan),
            "f1_macro": d.get("f1_macro_mean", np.nan),
            "f1_weighted": d.get("f1_weighted_mean", np.nan),
            "precision": d.get("precision_mean", np.nan),
            "recall": d.get("recall_mean", np.nan),
        }

        if keep_human_readable:
            row["flops_str"] = flops_raw
            row["params_str"] = params_raw

        rows.append(row)

    df = pd.DataFrame(rows)

    sort_map = {
        "flops": "flops",
        "params": "params",
        "total_time": "total_time_sec",
        "gpu_memory": "gpu_memory_used",
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision": "precision",
        "recall": "recall",
    }

    if sort_by == "model_alpha":
        df = df.sort_values(by="model", key=lambda s: s.str.lower(), ascending=ascending, na_position="last").reset_index(drop=True)
    else:
        sort_key = sort_map.get(sort_by, "f1_macro")
        if sort_key in df.columns:
            df = df.sort_values(sort_key, ascending=ascending, na_position="last").reset_index(drop=True)

    for c in ["flops", "params", "total_time_sec", "gpu_memory_used", "accuracy", "f1_macro", "f1_weighted", "precision", "recall"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if show_pretty_units:
        df["FLOPs (G)"] = df["flops"].apply(_format_g)
        df["Params (M)"] = df["params"].apply(_format_m)

        display_cols = [
            "model",
            "FLOPs (G)",
            "Params (M)",
            "total_time_sec",
            "gpu_memory_used",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision",
            "recall",
        ]
        df_out = df[display_cols]
    else:
        df_out = df

    if display:
        title = f"\n{dataset_name} Model Summary @ Number of samples = {n_samples}\n"
        print(title + df_out.to_string(index=False))

    if save_csv_path is not None:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_out.to_csv(save_csv_path, index=False)

    return df_out






def show_f1_and_accuracy_table(
    all_models_metrics: dict,
    n_samples_list: list[int] | None = None,
    sort_models: bool = True,
    decimals: int = 1,
    include_only: list[str] | None = None,  # e.g. ["dinov2", "random"]
    color_monotonic: bool = True,
    show_only_model_name: bool = False,
    show_what: str = "F1+Acc",
    dataset_name: str = "",
    top_3_coloring: bool = False,
    show_ci: bool = False,
):
    """
    Table columns:
      Config. Label, PTA, AT, AN, DS, AM, TAG, [Model name], Other info, M=20..100

    "Other info" contains the model key (model_name) from all_models_metrics.

    Each M cell has:
      F1:  xx.x±yy.y
      Acc: xx.x±yy.y

    If show_ci=True, each value gets an extra line with the confidence interval
    of the mean across runs:
      F1:  xx.x±yy.y
      [lo, hi]
    """
    console = Console()

    def _get_first_present(d: dict, keys: tuple[str, ...], default="-"):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    def _as_percent(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x) * 100.0

    def _basename_tokens(s: str) -> str:
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        if not s:
            return ""
        s = s.replace("\\", "/")
        return s.split("/")[-1].strip().lower()

    def _row_matches_include_only(PTA: str, AT: str, AN: str, DS: str, AM: str, TAG: str) -> bool:
        if include_only is None:
            return True
        tokens = [t.strip().lower() for t in include_only if str(t).strip()]
        if not tokens:
            return True

        PTA_b = _basename_tokens(PTA)
        AT_b  = _basename_tokens(AT)
        AN_b  = _basename_tokens(AN)
        DS_b  = _basename_tokens(DS)
        AM_b  = _basename_tokens(AM)
        TAG_b = _basename_tokens(TAG)

        return any(
            (t in PTA_b) or (t in AT_b) or (t in AN_b) or (t in DS_b) or (t in AM_b) or (t in TAG_b)
            for t in tokens
        )

    def _fmt(mean_p, std_p):
        return f"{mean_p:.{decimals}f}±{std_p:.{decimals}f}"

    def _fmt_ci(lo_p, hi_p):
        if np.isnan(lo_p) or np.isnan(hi_p):
            return "[-]"
        # escape() so rich never interprets the brackets as markup
        return escape(f"[{lo_p:.{decimals}f}, {hi_p:.{decimals}f}]")

    def _delta_color(curr: float, prev: float) -> str | None:
        if not color_monotonic:
            return None
        if np.isnan(curr) or np.isnan(prev):
            return None
        eps = 1e-9
        if curr > prev + eps:
            return "green"
        if curr < prev - eps:
            return "red"
        return "grey66"

    # ---- Force M columns ----
    if n_samples_list is None:
        n_samples_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        # keep only those requested, but preserve order as given
        n_samples_list = list(n_samples_list)

    if not n_samples_list:
        console.print("[red]No sample sizes found.[/red]")
        return

    # ---- Model filtering + metadata ----
    model_names = list(all_models_metrics.keys())  # NEW: model names from the dict keys
    model_order = sorted(model_names) if sort_models else model_names
    model_meta: dict[str, dict] = {}
    kept_models: list[str] = []

    for model in model_order:
        per_n = all_models_metrics.get(model, {})
        first_d = next(iter(per_n.values())) if per_n else {}

        PTA = _get_first_present(first_d, ("PTA",), default="-")
        AT  = _get_first_present(first_d, ("AT",), default="-")
        AN  = _get_first_present(first_d, ("AN",), default="-")
        DS  = _get_first_present(first_d, ("DS",), default="-")
        AM  = _get_first_present(first_d, ("AM",), default="-")
        TAG = _get_first_present(first_d, ("TAG",), default="-")
        if _row_matches_include_only(PTA, AT, AN, DS, AM, TAG):
            kept_models.append(model)
            model_meta[model] = {"PTA": PTA, "AT": AT, "AN": AN, "DS": DS, "AM": AM, "TAG": TAG}

    if not kept_models:
        console.print("[yellow]No rows matched include_only filter.[/yellow]")
        return

    # ---- Column-wise maxima ----
    col_max_f1 = {n: -np.inf for n in n_samples_list}
    col_max_acc = {n: -np.inf for n in n_samples_list}

    for model in kept_models:
        for n in n_samples_list:
            d = all_models_metrics[model].get(n, {})
            f1m = _as_percent(d.get("f1_macro_mean", np.nan))
            accm = _as_percent(d.get("accuracy_mean", np.nan))
            if not np.isnan(f1m):
                col_max_f1[n] = max(col_max_f1[n], f1m)
            if not np.isnan(accm):
                col_max_acc[n] = max(col_max_acc[n], accm)

    # ---- Top-3 per column (optional) ----
    top3_f1_per_col = {n: [] for n in n_samples_list}
    top3_acc_per_col = {n: [] for n in n_samples_list}

    if top_3_coloring:
        for n in n_samples_list:
            f1_vals, acc_vals = [], []
            for model in kept_models:
                d = all_models_metrics[model].get(n, {})
                f1m = _as_percent(d.get("f1_macro_mean", np.nan))
                accm = _as_percent(d.get("accuracy_mean", np.nan))
                if not np.isnan(f1m):
                    f1_vals.append(f1m)
                if not np.isnan(accm):
                    acc_vals.append(accm)

            top3_f1_per_col[n] = sorted(set(f1_vals), reverse=True)[:3]
            top3_acc_per_col[n] = sorted(set(acc_vals), reverse=True)[:3]

    def _get_top3_color(val: float, top3_list: list[float]) -> str | None:
        if not top3_list or np.isnan(val):
            return None
        eps = 1e-9
        if abs(val - top3_list[0]) < eps:
            return "green"
        if len(top3_list) > 1 and abs(val - top3_list[1]) < eps:
            return "blue"
        if len(top3_list) > 2 and abs(val - top3_list[2]) < eps:
            return "orange"
        return None

    # ---- Build table ----
    if show_what == "F1+Acc":
        title_suffix = "F1-Macro & Accuracy"
    elif show_what == "F1":
        title_suffix = "F1-Macro"
    elif show_what == "Acc":
        title_suffix = "Accuracy"
    else:
        title_suffix = "F1-Macro & Accuracy"

    table_title = f"{dataset_name} {title_suffix} Table (%)"
    if show_ci:
        # pick CI settings from the data itself
        ci_level, ci_method, n_runs = None, None, set()
        for model in kept_models:
            for d in all_models_metrics[model].values():
                ci_level = d.get("ci_level", ci_level)
                ci_method = d.get("ci_method", ci_method)
                if "n_seeds" in d:
                    n_runs.add(d["n_seeds"])
        n_txt = ",".join(str(x) for x in sorted(n_runs)) if n_runs else "?"
        lvl_txt = f"{100 * ci_level:g}%" if ci_level is not None else ""
        table_title += f"\nmean±std, {lvl_txt} CI of the mean ({ci_method}, n={n_txt} runs)"

    table = Table(
        title=table_title,
        box=box.DOUBLE,
        show_lines=True,
        header_style="bold cyan",
    )

    table.add_column("Config.\nLabel", justify="center", style="bold")
    table.add_column("PTA", justify="center")
    table.add_column("AT", justify="center")
    table.add_column("AN", justify="center")
    table.add_column("DS", justify="center")
    table.add_column("AM", justify="center")
    table.add_column("TAG", justify="center")

    if show_only_model_name:
        table.add_column("Model\nname", justify="center")

    table.add_column("Other info", justify="center", overflow="fold")  # NEW

    for n in n_samples_list:
        if n == 1000:
            n += 100
        table.add_column(f"M={n}", justify="center")

    # ---- Fill rows ----
    for idx, model_key in enumerate(kept_models, start=1):
        per_n = all_models_metrics.get(model_key, {})
        PTA = model_meta[model_key]["PTA"]
        AT  = model_meta[model_key]["AT"]
        AN  = model_meta[model_key]["AN"]
        DS  = model_meta[model_key]["DS"]
        AM  = model_meta[model_key]["AM"]
        TAG = model_meta[model_key]["TAG"]

        row = [f"A{idx}", str(PTA), str(AT), str(AN), str(DS), str(AM), str(TAG)]

        if show_only_model_name:
            row.append(str(get_model_name(model_key)))

        # NEW: corresponding model_name (key in all_models_metrics);
        # escape() so brackets in names aren't parsed as rich markup
        row.append(escape(str(model_key)))

        prev_f1_mean = np.nan
        prev_acc_mean = np.nan

        for n in n_samples_list:
            d = per_n.get(n, {})

            f1_mean = _as_percent(d.get("f1_macro_mean", np.nan))
            f1_std  = _as_percent(d.get("f1_macro_std", np.nan))
            acc_mean = _as_percent(d.get("accuracy_mean", np.nan))
            acc_std  = _as_percent(d.get("accuracy_std", np.nan))

            f1_col = _delta_color(f1_mean, prev_f1_mean)
            acc_col = _delta_color(acc_mean, prev_acc_mean)

            if top_3_coloring:
                c = _get_top3_color(f1_mean, top3_f1_per_col[n])
                if c is not None:
                    f1_col = c
                c = _get_top3_color(acc_mean, top3_acc_per_col[n])
                if c is not None:
                    acc_col = c

            # F1 line
            if np.isnan(f1_mean) or np.isnan(f1_std):
                f1_line_core = "F1: -"
            else:
                f1_txt = _fmt(f1_mean, f1_std)
                is_best_f1 = np.isfinite(col_max_f1[n]) and abs(f1_mean - col_max_f1[n]) < 1e-9
                f1_line_core = f"[bold underline]F1: {f1_txt}[/bold underline]" if is_best_f1 else f"F1: {f1_txt}"
                if show_ci:
                    f1_lo = _as_percent(d.get("f1_macro_ci_low", np.nan))
                    f1_hi = _as_percent(d.get("f1_macro_ci_high", np.nan))
                    f1_line_core += "\n" + _fmt_ci(f1_lo, f1_hi)
            f1_line = f"[{f1_col}]{f1_line_core}[/{f1_col}]" if f1_col else f1_line_core

            # Acc line
            if np.isnan(acc_mean) or np.isnan(acc_std):
                acc_line_core = "Acc: -"
            else:
                acc_txt = _fmt(acc_mean, acc_std)
                is_best_acc = np.isfinite(col_max_acc[n]) and abs(acc_mean - col_max_acc[n]) < 1e-9
                acc_line_core = f"[bold underline]Acc: {acc_txt}[/bold underline]" if is_best_acc else f"Acc: {acc_txt}"
                if show_ci:
                    acc_lo = _as_percent(d.get("accuracy_ci_low", np.nan))
                    acc_hi = _as_percent(d.get("accuracy_ci_high", np.nan))
                    acc_line_core += "\n" + _fmt_ci(acc_lo, acc_hi)
            acc_line = f"[{acc_col}]{acc_line_core}[/{acc_col}]" if acc_col else acc_line_core

            if show_what == "F1+Acc":
                row.append(f"{f1_line}\n{acc_line}")
            elif show_what == "F1":
                row.append(f"{f1_line}")
            elif show_what == "Acc":
                row.append(f"{acc_line}")

            if not np.isnan(f1_mean):
                prev_f1_mean = f1_mean
            if not np.isnan(acc_mean):
                prev_acc_mean = acc_mean

        table.add_row(*row)

    console.print(table)



def _config_labels(all_models_metrics: dict, sort_models: bool = True) -> dict:
    """Same A1, A2, ... labelling as show_f1_and_accuracy_table (no include_only filter)."""
    order = sorted(all_models_metrics.keys()) if sort_models else list(all_models_metrics.keys())
    return {m: f"A{i}" for i, m in enumerate(order, start=1)}


def _resolve_config(ref: str, labels: dict) -> str:
    """Resolve a reference given as a label ('A2') or a unique substring of a folder name."""
    ref_l = ref.strip().lower()
    for model, lab in labels.items():
        if lab.lower() == ref_l:
            return model
    hits = [m for m in labels if ref_l in m.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"--paired_ref '{ref}' matches no configuration. Available: "
                         + ", ".join(f"{lab}={m}" for m, lab in labels.items()))
    raise ValueError(f"--paired_ref '{ref}' is ambiguous, matches: {hits}. Use the A<k> label instead.")


def save_mean_std_ci_csv(all_models_metrics: dict, n_samples_list: list[int], save_csv_path: str,
                         metrics=("f1_macro", "accuracy")):
    """Long-format CSV: one row per (config, M, metric) with mean, std, CI (all in %)."""
    labels = _config_labels(all_models_metrics)
    rows = []
    for model, lab in labels.items():
        per_n = all_models_metrics[model]
        for n in n_samples_list:
            d = per_n.get(n)
            if not d:
                continue
            for met in metrics:
                rows.append({
                    "config": lab,
                    "folder": model,
                    "PTA": d.get("PTA"), "AT": d.get("AT"), "AN": d.get("AN"),
                    "DS": d.get("DS"), "AM": d.get("AM"), "TAG": d.get("TAG"),
                    "M": n,
                    "metric": met,
                    "n_runs": d.get("n_seeds"),
                    "mean": 100 * d.get(f"{met}_mean", np.nan),
                    "std_ddof0": 100 * d.get(f"{met}_std", np.nan),
                    "ci_low": 100 * d.get(f"{met}_ci_low", np.nan),
                    "ci_high": 100 * d.get(f"{met}_ci_high", np.nan),
                    "ci_level": d.get("ci_level"),
                    "ci_method": d.get("ci_method"),
                })
    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(save_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    print(f"Saved mean/std/CI table to {save_csv_path}")
    return df


def show_paired_differences_table(
    all_models_metrics: dict,
    n_samples_list: list[int],
    reference: str | None = None,
    show_what: str = "F1+Acc",
    ci_level: float = 0.95,
    ci_method: str = "t",
    p_adjust: str = "none",
    alpha: float = 0.05,
    decimals: int = 1,
    dataset_name: str = "",
    save_csv_path: str | None = None,
):
    """
    Paired comparison between configurations evaluated on the SAME repeated splits.

    For every pair (X, Y) and every M, the per-run differences d_i = X_i - Y_i are used to report:
      - mean difference and its CI (paired; t-based or bootstrap)
      - paired t-test p-value and Wilcoxon signed-rank p-value (CSV)
      - Cohen's d_z (CSV)

    reference=None -> all pairs (Ai - Aj, i<j); otherwise every config minus the reference.

    Cell text:  ΔF1 +2.1 [+0.3, +3.9]*
      green  : CI entirely above 0 (row config better)
      red    : CI entirely below 0 (row config worse)
      plain  : CI contains 0 (difference not distinguishable from split-to-split variation)
      *      : paired t-test p < alpha (after Holm adjustment if p_adjust='holm')
    """
    console = Console()
    labels = _config_labels(all_models_metrics)
    models = list(labels.keys())

    if len(models) < 2:
        console.print("[yellow]Paired comparison needs at least 2 configurations.[/yellow]")
        return None

    if reference is not None:
        ref_model = _resolve_config(reference, labels)
        pairs = [(m, ref_model) for m in models if m != ref_model]
    else:
        pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]

    if show_what == "F1":
        metrics = [("f1_macro", "ΔF1")]
    elif show_what == "Acc":
        metrics = [("accuracy", "ΔAcc")]
    else:
        metrics = [("f1_macro", "ΔF1"), ("accuracy", "ΔAcc")]

    # ---- compute all comparisons ----
    rows = []
    warned_order = set()
    for a, b in pairs:
        for n in n_samples_list:
            da = all_models_metrics[a].get(n)
            db = all_models_metrics[b].get(n)
            if not da or not db:
                continue
            for met, _ in metrics:
                va = da.get("per_seed", {}).get(met, {})
                vb = db.get("per_seed", {}).get(met, {})
                res = paired_difference_stats(va, vb, level=ci_level, method=ci_method)
                if res is None:
                    continue
                if res["pairing"] == "order" and (a, b) not in warned_order:
                    console.print(f"[yellow]Warning: could not match run/seed ids between {labels[a]} and "
                                  f"{labels[b]}; pairing runs by sorted file order. Make sure run i uses the "
                                  f"same split in both configurations.[/yellow]")
                    warned_order.add((a, b))
                rows.append({
                    "comparison": f"{labels[a]} - {labels[b]}",
                    "config_a": labels[a], "config_b": labels[b],
                    "folder_a": a, "folder_b": b,
                    "M": n, "metric": met,
                    **{k: (100 * v if k in ("mean_a", "mean_b", "mean_diff", "sd_diff", "ci_low", "ci_high")
                           else v) for k, v in res.items()},
                    "ci_level": ci_level, "ci_method": ci_method,
                })

    if not rows:
        console.print("[yellow]No paired comparisons could be computed.[/yellow]")
        return None

    df = pd.DataFrame(rows)

    # ---- multiple-comparison adjustment (family = all comparisons of one metric in this table) ----
    df["p_adj"] = df["p_ttest"]
    if p_adjust == "holm":
        for met in df["metric"].unique():
            mask = df["metric"] == met
            df.loc[mask, "p_adj"] = holm_adjust(df.loc[mask, "p_ttest"].values)
    df["p_adjust_method"] = p_adjust
    df["significant"] = df["p_adj"] < alpha

    # ---- rich table ----
    ref_txt = f"vs reference {labels[_resolve_config(reference, labels)]}" if reference else "all pairs"
    adj_txt = ", Holm-adjusted" if p_adjust == "holm" else ""
    table = Table(
        title=(f"{dataset_name} Paired differences ({ref_txt}), percentage points\n"
               f"mean Δ [{100 * ci_level:g}% CI, {ci_method}], * = paired t-test p<{alpha:g}{adj_txt}"),
        box=box.DOUBLE,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Comparison", justify="center", style="bold")
    ms = [n for n in n_samples_list if n in set(df["M"])]
    for n in ms:
        table.add_column(f"M={n}", justify="center")

    def _cell(r, short):
        txt = escape(f"{short} {r['mean_diff']:+.{decimals}f} "
                     f"[{r['ci_low']:+.{decimals}f}, {r['ci_high']:+.{decimals}f}]")
        if r["significant"]:
            txt += "*"
        if np.isfinite(r["ci_low"]) and r["ci_low"] > 0:
            return f"[green]{txt}[/green]"
        if np.isfinite(r["ci_high"]) and r["ci_high"] < 0:
            return f"[red]{txt}[/red]"
        return txt

    for comp in df["comparison"].drop_duplicates():
        sub = df[df["comparison"] == comp]
        row = [comp]
        for n in ms:
            lines = []
            for met, short in metrics:
                r = sub[(sub["M"] == n) & (sub["metric"] == met)]
                lines.append(_cell(r.iloc[0], short) if len(r) else f"{short} -")
            row.append("\n".join(lines))
        table.add_row(*row)

    console.print(table)
    legend = {m: lab for m, lab in labels.items()}
    console.print("Config labels: " + ", ".join(f"{lab}={m}" for m, lab in legend.items()))

    if save_csv_path is not None:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        console.print(f"Saved paired comparison table to {save_csv_path}")

    return df


def _paired_diffs_by_key(values_a: dict, values_b: dict):
    """
    Per-split differences {key: a - b} for one M, using the same pairing rule as
    paired_difference_stats (seed-key if >= 2 shared keys, otherwise run order).
    Returns (diffs_dict, pairing) or (None, None).
    """
    common = [k for k in values_a.keys() if k in values_b]
    if len(common) >= 2:
        return {k: float(values_a[k]) - float(values_b[k]) for k in common}, "seed-key"
    if len(values_a) == len(values_b) and len(values_a) >= 2:
        va, vb = list(values_a.values()), list(values_b.values())
        return {f"run{i}": float(x) - float(y) for i, (x, y) in enumerate(zip(va, vb))}, "order"
    return None, None


def show_paired_summary_table(
    all_models_metrics: dict,
    n_samples_list: list[int],
    reference: str | None = None,
    show_what: str = "F1+Acc",
    ci_level: float = 0.95,
    ci_method: str = "t",
    p_adjust: str = "none",
    alpha: float = 0.05,
    decimals: int = 1,
    dataset_name: str = "",
    save_csv_path: str | None = None,
):
    """
    ONE overall paired comparison per pair of configurations, pooled over all labelling budgets M.

    For each run/split key k:
        dbar_k = (1/|M|) * sum_M ( a_{k,M} - b_{k,M} )      # the run's average advantage over M
    Then the usual paired statistics are computed on the n values dbar_k:
        overall Δ = mean_k dbar_k   (= the average of the per-M mean Δs)
        CI        = mean ± t_{n-1} * sd(dbar_k) / sqrt(n)   (or bootstrap)
        p         = one-sample t-test of dbar_k against 0 (+ Wilcoxon in CSV)

    This is the statistically valid way to "average over M". Simply averaging the per-M CI
    bounds is NOT a confidence interval (it ignores that differences at different M are
    correlated and averaged), so it is not used.

    Only runs present at every included M are used, so every dbar_k averages the same M set.
    """
    console = Console()
    labels = _config_labels(all_models_metrics)
    models = list(labels.keys())
    if len(models) < 2:
        console.print("[yellow]Paired summary needs at least 2 configurations.[/yellow]")
        return None

    if reference is not None:
        ref_model = _resolve_config(reference, labels)
        pairs = [(m, ref_model) for m in models if m != ref_model]
    else:
        pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]

    if show_what == "F1":
        metrics = [("f1_macro", "ΔF1")]
    elif show_what == "Acc":
        metrics = [("accuracy", "ΔAcc")]
    else:
        metrics = [("f1_macro", "ΔF1"), ("accuracy", "ΔAcc")]

    rows = []
    for a, b in pairs:
        for met, _ in metrics:
            per_m = {}          # M -> {key: d}
            pairings = set()
            per_m_sig_pos, per_m_sig_neg = 0, 0
            for n in n_samples_list:
                da = all_models_metrics[a].get(n)
                db = all_models_metrics[b].get(n)
                if not da or not db:
                    continue
                va = da.get("per_seed", {}).get(met, {})
                vb = db.get("per_seed", {}).get(met, {})
                diffs, pairing = _paired_diffs_by_key(va, vb)
                if diffs is None:
                    continue
                per_m[n] = diffs
                pairings.add(pairing)
                # how many individual M are significant on their own (same CI rule as the per-M table)
                lo, hi = mean_confidence_interval(list(diffs.values()), level=ci_level, method=ci_method)
                if np.isfinite(lo) and lo > 0:
                    per_m_sig_pos += 1
                elif np.isfinite(hi) and hi < 0:
                    per_m_sig_neg += 1

            if not per_m:
                continue

            # keep only runs present at every included M
            keys = set.intersection(*(set(d.keys()) for d in per_m.values()))
            keys = [k for k in next(iter(per_m.values())).keys() if k in keys]

            # reference value: plain average of the per-M mean differences (uses ALL runs at each M;
            # identical to the per-M table). Equals the pooled Δ when no run is dropped.
            avg_of_per_m = float(np.mean([np.mean(list(per_m[n].values())) for n in sorted(per_m)]))
            dropped = {n: sorted(set(per_m[n].keys()) - set(keys)) for n in sorted(per_m)}
            dropped = {n: ks for n, ks in dropped.items() if ks}
            if dropped:
                console.print(
                    f"[yellow]Warning ({labels[a]} - {labels[b]}, {met}): only {len(keys)} run(s) exist at ALL M, "
                    f"so the pooled Δ differs from the average of the per-M table. Runs excluded -> "
                    + "; ".join(f"M={n}: {', '.join(ks)}" for n, ks in dropped.items())
                    + "[/yellow]"
                )
            if len(keys) < 2:
                console.print(f"[yellow]{labels[a]} - {labels[b]} ({met}): fewer than 2 runs shared across "
                              f"all M; skipped.[/yellow]")
                continue

            Ms = sorted(per_m.keys())
            dbar = np.array([np.mean([per_m[n][k] for n in Ms]) for k in keys], dtype=float)
            n_runs = dbar.size
            mean_d = float(dbar.mean())
            sd_d = float(dbar.std(ddof=1))
            lo, hi = mean_confidence_interval(dbar, level=ci_level, method=ci_method)
            if sd_d > 0:
                t_res = stats.ttest_1samp(dbar, 0.0)
                t_stat, p_t = float(t_res.statistic), float(t_res.pvalue)
                dz = mean_d / sd_d
            else:
                t_stat, p_t, dz = np.nan, np.nan, np.nan
            try:
                p_w = float(stats.wilcoxon(dbar).pvalue)
            except ValueError:
                p_w = np.nan

            n_pos_M = int(sum(np.mean(list(per_m[n].values())) > 0 for n in Ms))

            rows.append({
                "comparison": f"{labels[a]} - {labels[b]}",
                "config_a": labels[a], "config_b": labels[b],
                "folder_a": a, "folder_b": b,
                "metric": met,
                "M_included": ",".join(str(m) for m in Ms),
                "n_M": len(Ms),
                "n_runs": n_runs,
                "n_runs_dropped": int(sum(len(v) for v in dropped.values())),
                "avg_of_per_M_mean_diffs": 100 * avg_of_per_m,
                "pairing": "+".join(sorted(pairings)),
                "mean_diff": 100 * mean_d,
                "sd_run_avg_diff": 100 * sd_d,
                "ci_low": 100 * lo,
                "ci_high": 100 * hi,
                "t_stat": t_stat,
                "p_ttest": p_t,
                "p_wilcoxon": p_w,
                "cohens_dz": dz,
                "n_M_mean_positive": n_pos_M,
                "n_M_sig_positive": per_m_sig_pos,
                "n_M_sig_negative": per_m_sig_neg,
                "ci_level": ci_level,
                "ci_method": ci_method,
            })

    if not rows:
        console.print("[yellow]No pooled paired comparisons could be computed.[/yellow]")
        return None

    df = pd.DataFrame(rows)
    df["p_adj"] = df["p_ttest"]
    if p_adjust == "holm":
        for met in df["metric"].unique():
            mask = df["metric"] == met
            df.loc[mask, "p_adj"] = holm_adjust(df.loc[mask, "p_ttest"].values)
    df["p_adjust_method"] = p_adjust
    df["significant"] = df["p_adj"] < alpha

    # ---- rich table ----
    ref_txt = f"vs reference {labels[_resolve_config(reference, labels)]}" if reference else "all pairs"
    adj_txt = ", Holm-adjusted" if p_adjust == "holm" else ""
    Ms_all = sorted({int(m) for s in df["M_included"] for m in s.split(",")})
    m_txt = f"M={Ms_all[0]}..{Ms_all[-1]}" if Ms_all else ""
    table = Table(
        title=(f"{dataset_name} Paired differences pooled over labelling budgets ({m_txt}, {ref_txt}), "
               f"percentage points\nper run: average Δ over M, then mean Δ [{100 * ci_level:g}% CI, {ci_method}] "
               f"over runs, * = t-test p<{alpha:g}{adj_txt}"),
        box=box.DOUBLE,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Comparison", justify="center", style="bold")
    for _, short in metrics:
        table.add_column(f"Overall {short}", justify="center")
    table.add_column("Runs × M", justify="center")

    def _cell(r):
        txt = escape(f"{r['mean_diff']:+.{decimals}f} [{r['ci_low']:+.{decimals}f}, {r['ci_high']:+.{decimals}f}]")
        if r["significant"]:
            txt += "*"
        if np.isfinite(r["ci_low"]) and r["ci_low"] > 0:
            txt = f"[green]{txt}[/green]"
        elif np.isfinite(r["ci_high"]) and r["ci_high"] < 0:
            txt = f"[red]{txt}[/red]"
        p_show = r["p_adj"] if p_adjust == "holm" else r["p_ttest"]
        p_txt = f"p={p_show:.3g}" if np.isfinite(p_show) else "p=-"
        dz_txt = f"dz={r['cohens_dz']:+.2f}" if np.isfinite(r["cohens_dz"]) else "dz=-"
        detail = (f"{p_txt}, {dz_txt}\n"
                  f"Δ>0 at {r['n_M_mean_positive']}/{r['n_M']} M; "
                  f"sig +{r['n_M_sig_positive']} / -{r['n_M_sig_negative']}")
        return f"{txt}\n[grey62]{detail}[/grey62]"

    for comp in df["comparison"].drop_duplicates():
        sub = df[df["comparison"] == comp]
        row = [comp]
        for met, _ in metrics:
            r = sub[sub["metric"] == met]
            row.append(_cell(r.iloc[0]) if len(r) else "-")
        r0 = sub.iloc[0]
        row.append(f"{r0['n_runs']} × {r0['n_M']}")
        table.add_row(*row)

    console.print(table)
    console.print("Config labels: " + ", ".join(f"{lab}={m}" for m, lab in labels.items()))

    if save_csv_path is not None:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        console.print(f"Saved pooled paired comparison table to {save_csv_path}")

    return df


def filter_folders(model_folders, command: str):
    """
    Filter folder names with boolean logic over keywords.

    Supported syntax (case-insensitive):
      - keep <expr>
      - remove <expr>

    <expr> supports:
      - keywords (alphanumerics/underscore)
      - and, or, not
      - parentheses: ( )

    Examples:
      keep CLIP or CosSim not LinearSVC
      keep (CLIP or CosSim) and not LinearSVC
      remove LinearSVC
    """
    print(f"Filtering model folders with command: '{command}'")
    cmd = command.strip()

    m = re.match(r"^\s*(keep|remove)\s+(.+?)\s*$", cmd, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Command must start with 'keep' or 'remove' followed by an expression")

    mode = m.group(1).lower()
    expr = m.group(2).strip().lower()

    # If user writes "... CosSim not LinearSVC" (missing 'and'),
    # interpret it as "... CosSim and not LinearSVC"
    expr = re.sub(r"\bnot\b", " and not ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()

    # Tokenize: words or parentheses
    tokens = re.findall(r"\(|\)|\w+", expr)

    allowed_ops = {"and", "or", "not", "(", ")"}

    def matches(folder_name: str) -> bool:
        fname = folder_name.lower()
        py_tokens = []

        for tok in tokens:
            if tok in allowed_ops:
                py_tokens.append(tok)
            else:
                # keyword match: substring membership
                py_tokens.append(str(tok in fname))

        py_expr = " ".join(py_tokens)

        # Safety: expression should only contain True/False and operators/parens
        if not re.fullmatch(r"[truefalsenotandor()\s]+", py_expr.replace("True","true").replace("False","false")):
            raise ValueError(f"Unsafe expression generated: {py_expr}")

        return bool(eval(py_expr, {"__builtins__": {}}, {}))

    if mode == "keep":
        return [f for f in model_folders if matches(f)]
    else:  # remove
        return [f for f in model_folders if not matches(f)]




# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pseudo labels across seeds and sample sizes")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--results_dir", type=str, default="./models_results", help="Output directory for results")
    parser.add_argument("--N_cap", type=int, default=None, help="Maximum N_samples_per_class to include in evaluation")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--config_dir", type=str, default="./config", help="Directory containing models.json")
    parser.add_argument("--filter", type=str, default="", help="Filter command for model folders (e.g., 'keep dinov2 and resnet')")
    parser.add_argument("--show_only_model_name", action="store_true", help="Show only model name in plots (no extra metadata)")
    parser.add_argument("--show_what", type=str, default="F1+Acc", help="What to show in the table: 'F1+Acc', 'F1', or 'Acc'")
    parser.add_argument("--color_monotonic", action="store_true", help="Color monotonic changes in table (green/red/grey)")
    parser.add_argument("--top_3_coloring", action="store_true", help="Color top 3 values in each column (green/blue/red)")
    parser.add_argument("--mode", default="FSC", help="Evaluation mode (e.g., FSC)")
    # ---- Statistical uncertainty (all OFF by default) ----
    parser.add_argument("--ci", action="store_true",
                        help="Add the confidence interval of the mean (across runs) under each F1/Acc value in the table")
    parser.add_argument("--ci_level", type=float, default=0.95,
                        help="Confidence level for all CIs (default: 0.95)")
    parser.add_argument("--ci_method", type=str, default="t", choices=["t", "bootstrap"],
                        help="CI method: 't' (Student-t, default) or 'bootstrap' (percentile, 10k resamples)")
    parser.add_argument("--paired", action="store_true",
                        help="Also print paired-difference table (mean Δ with CI, paired t-test / Wilcoxon) "
                             "between configurations evaluated on the same splits")
    parser.add_argument("--paired_ref", type=str, default=None,
                        help="Reference config for --paired, as label (e.g. 'A2') or unique folder-name substring. "
                             "Default: all pairwise comparisons")
    parser.add_argument("--p_adjust", type=str, default="none", choices=["none", "holm"],
                        help="Multiple-comparison correction for --paired p-values (default: none)")
    parser.add_argument("--paired_summary", action="store_true",
                        help="Also print ONE overall paired difference per comparison, pooled over all M "
                             "(per run: average Δ over M; then CI/t-test over runs). Uses --paired_ref, "
                             "--ci_level, --ci_method, --p_adjust, --alpha")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level used to mark '*' in the paired table (default: 0.05)")
    args = parser.parse_args()
    if not (0.0 < args.ci_level < 1.0):
        parser.error("--ci_level must be in (0, 1), e.g. 0.95")
    show_only_model_name = args.show_only_model_name
    dataset_name = args.dataset
    results_dir = args.results_dir
    n_cap = args.N_cap
    config_dir = args.config_dir
    show_what = args.show_what
    filter_command = args.filter
    color_monotonic = args.color_monotonic
    top_3_coloring = args.top_3_coloring

    # Load model folders from models.json (fallback to dir listing as in original code)
    models_json_path = os.path.join(config_dir, "models.json")
    models_config = {}
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as f:
            models_config = json.load(f)

    
    model_folders0 = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
     
    print(f"Found {len(model_folders0)} model folders: {model_folders0}")
    if filter_command == "":
        model_folders = model_folders0
        print("No filtering applied to model folders.")
    else:
        model_folders = filter_folders(model_folders0, filter_command)
        print(f"Filtered {len(model_folders)} model folders: {model_folders}")
    if n_cap is not None:
        print(f"Limiting evaluation to N_samples <= {n_cap}")

    all_models_metrics = {}

    for model in model_folders:
        samples_dict = find_all_seed_csvs_by_samples(results_dir, model, dataset_name, n_cap=n_cap)
        json_samples_dict = find_all_json_by_samples(results_dir, model, dataset_name, n_cap=n_cap)

        if len(samples_dict) == 0:
            print(f"\n>>>>>> {model}: no seed CSVs found for {dataset_name}")
            continue
        if len(json_samples_dict) == 0:
            print(f"\n>>>>>> {model}: no JSON stats found for {dataset_name}")
            continue

        print(f"\n>>>>>> Evaluating {model} for {dataset_name} <<<<<<")


        samples_metrics = {}
        for n_samples in tqdm(sorted(samples_dict.keys()), desc=f"Processing {model}", unit="sample_size"):
            if n_samples not in json_samples_dict:
                continue

            csv_paths = samples_dict[n_samples]
            json_paths = json_samples_dict[n_samples]

            # ---- aggregate CSV metrics ----
            agg_metrics = aggregate_metrics_across_seeds(csv_paths, use_names=True,
                                                         ci_level=args.ci_level, ci_method=args.ci_method)

            # ---- aggregate JSON metrics ----
            json_metrics = aggregate_stats_across_seeds(json_paths)
            agg_metrics.update(json_metrics)

            # ---- extract metadata from first JSON (same for all seeds) ----
            with open(json_paths[0], "r") as f:
                meta = json.load(f)

            agg_metrics["Feature Learning"] = meta.get("Feature Learning", meta.get("feature_learning", "-"))
            agg_metrics["Data selection"] = meta.get("Data selection", meta.get("data_selection", "-"))
            agg_metrics["Classifier"] = meta.get("Classifier", meta.get("classifier", "-"))
            # print(55555555555555555)
            # print(meta.get("Feature Learning", meta.get("feature_learning", "-")))
            PTA_dict = {"dinov2": "DINOv2",
                        "dinov3": "DINOv3",    
                        "clip": "CLIP",
                        "resnet":"CE",
                        "cnn":"CE"}
        
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
            



            fl = meta.get("Feature Learning", meta.get("feature_learning", "-")).lower()
            ds_ = meta.get("Data selection", meta.get("data_selection", "-")).lower()
            cl_ = meta.get("Classifier", meta.get("classifier", "-")).lower()

            PTA = meta.get("PTA", meta.get("PTA", "-")).lower()
            AT = meta.get("AT", meta.get("AT", "-")).lower()
            AN = meta.get("AN", meta.get("AN", "-")).lower()
            DS = meta.get("DS", meta.get("DS", "-")).lower()
            AM = meta.get("AM", meta.get("AM", "-")).lower()
            TAG = meta.get("tag", meta.get("tag", "-")).lower()

            
            agg_metrics["TAG"] = TAG


            if PTA and PTA != "-":
                agg_metrics["PTA"] = PTA
            else:
                pta = "Unknown"
                for key in PTA_dict.keys():
                    if key in fl.lower():
                        pta = PTA_dict[key]
                        break
                agg_metrics["PTA"] = pta
            

            if AT and AT != "-":
                agg_metrics["AT"] = AT
            else:
                at = "Unknown"
                for key in AT_dict.keys():
                    if key in fl.lower():
                        at = AT_dict[key]
                        break
                agg_metrics["AT"] = at

            if AN and AN != "-":
                agg_metrics["AN"] = AN
            else:
                an = "Unknown"
                for key in AN_dict.keys():
                    if key in fl.lower():
                        an = AN_dict[key]
                        break
                agg_metrics["AN"] = an




            if DS and DS != "-":
                agg_metrics["DS"] = DS
            else:
                ds = "Unknown"
                for key in DS_dict.keys():
                    if key in ds_.lower():
                        ds = DS_dict[key]
                        break
                agg_metrics["DS"] = ds







            if AM and AM != "-":
                agg_metrics["AM"] = AM
            else:
                am = "Unknown"
                for key in AM_dict.keys():
                    if key in cl_.lower():
                        am = AM_dict[key]
                        break
                agg_metrics["AM"] = am




            
            # ---- extract folder name (parent of n_samples folder) ----
            # json_paths[0] structure: results_dir/model_folder/n_samples/seed/file.json
            # We want the model_folder basename
            agg_metrics["Folder name"] = model

            samples_metrics[n_samples] = agg_metrics



        # print(samples_metrics)
        if samples_metrics:
            all_models_metrics[model] = samples_metrics

    print("\n>>>>>> Generating comparison plots <<<<<<")
    plot_figures = False
    save_figures = True
    save_dir = args.save_dir
    if save_dir is not None:
        os.path.exists(save_dir) or os.makedirs(save_dir)

    plot_models_comparison(
        all_models_metrics,
        dataset_name,
        save_dir=save_dir,
        save_figures=save_figures,
        plot_figures=plot_figures,
        scale_yaxis=True,
        show_only_model_name=show_only_model_name,
    )

    # New plots requested (from JSON fields)
    plot_total_time_from_json(
        all_models_metrics,
        dataset_name,
        save_dir=save_dir,
        save_figures=save_figures,
        plot_figures=plot_figures,
        show_only_model_name=show_only_model_name
    )
    plot_gpu_memory_from_json(
        all_models_metrics,
        dataset_name,
        save_dir=save_dir,
        save_figures=save_figures,
        plot_figures=plot_figures,
        show_only_model_name=show_only_model_name
    )

    # Keep your summary calls (now without train/test columns)
    try:
        show_models_summary_table_at_n(all_models_metrics, n_samples=50, sort_by="params", ascending=True,show_only_model_name=show_only_model_name,dataset_name=dataset_name)
    except Exception as e:
        print(f"Error showing summary table at N=50: {e}")
    try:
        show_models_summary_table_at_n(all_models_metrics, n_samples=100, sort_by="params", ascending=True,show_only_model_name=show_only_model_name,dataset_name=dataset_name)
    except Exception as e:
        print(f"Error showing summary table at N=100: {e}")
    if args.mode.upper() == "TFM":
        n_samples_list = [100,250,500,750,1000,1250,1500]
    elif args.mode.upper() == "FSC":
        n_samples_list = [20,30,40,50,60,70,80,90,100]
    elif args.mode.upper() == "ALL":
        n_samples_list = [20,30,40,50,60,70,80,90,100,250,500,750,1000,1250,1500]
    elif args.mode.upper() == "FSC+TFM":
        n_samples_list = [20,30,40,50,60,70,80,90,100,1000]
    else:
        n_samples_list = [20,30,40,50,60,70,80,90,100]
    

    try:
        show_f1_and_accuracy_table(all_models_metrics, 
                                   n_samples_list = n_samples_list,
                                   decimals=1,
                                   color_monotonic=color_monotonic,
                                   show_only_model_name=show_only_model_name,
                                   show_what=show_what,
                                   dataset_name=dataset_name,
                                   top_3_coloring=top_3_coloring,
                                   show_ci=args.ci,
                                   )
    except Exception as e:
        print(f"Error showing F1 and accuracy table: {e}")

    if args.ci and save_dir is not None:
        try:
            save_mean_std_ci_csv(all_models_metrics, n_samples_list,
                                 os.path.join(save_dir, f"{dataset_name}_mean_std_ci.csv"))
        except Exception as e:
            print(f"Error saving mean/std/CI CSV: {e}")

    if args.paired:
        try:
            show_paired_differences_table(
                all_models_metrics,
                n_samples_list=n_samples_list,
                reference=args.paired_ref,
                show_what=show_what,
                ci_level=args.ci_level,
                ci_method=args.ci_method,
                p_adjust=args.p_adjust,
                alpha=args.alpha,
                decimals=1,
                dataset_name=dataset_name,
                save_csv_path=(os.path.join(save_dir, f"{dataset_name}_paired_differences.csv")
                               if save_dir is not None else None),
            )
        except Exception as e:
            print(f"Error showing paired differences table: {e}")

    if args.paired_summary:
        try:
            show_paired_summary_table(
                all_models_metrics,
                n_samples_list=n_samples_list,
                reference=args.paired_ref,
                show_what=show_what,
                ci_level=args.ci_level,
                ci_method=args.ci_method,
                p_adjust=args.p_adjust,
                alpha=args.alpha,
                decimals=1,
                dataset_name=dataset_name,
                save_csv_path=(os.path.join(save_dir, f"{dataset_name}_paired_summary_over_M.csv")
                               if save_dir is not None else None),
            )
        except Exception as e:
            print(f"Error showing pooled paired summary table: {e}")