import os
import re
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
dataset_name_mapping = {"18167": "Dataset A",
                        "18290": "Dataset B",
                        "16817": "Dataset C",
                        "16817_0_05": "Dataset C"}

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


def get_model_name(full_name):
    if full_name.startswith("resnet"):
        model_name = full_name.split("_")[0]
    else:
        model_name = full_name.split("_")[0] +'_'  + full_name.split("_")[1] 
    return model_name

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



from collections import Counter



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
# Time-saving analysis wrt manual annotation
# --------------------------
def aggregate_time_saving_inputs_across_seeds(json_paths,S=3599):
    """
    Aggregate the inputs needed for time-saving analysis across multiple JSON files.

    Reads:
      - Tp from time_taken_seconds-like fields
      - acc from accuracy-like fields

    Returns:
      {
        "n_seeds": int,
        "tp_mean": float,
        "tp_std": float,
        "acc_mean": float,
        "acc_std": float,
      }
    """
    per_seed = [load_stats_from_json(p) for p in json_paths]

    tp_vals = np.array(
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

    acc_vals = np.array(
        [
            float(
                _get_first_key(
                    d,
                    ("accuracy", "acc", "Accuracy"),
                    np.nan,
                )
            )
            for d in per_seed
        ],
        dtype=float,
    )

    MS = np.array(
        [
            int(
                _get_first_key(
                    d,
                    ("ref_size","ref size"),
                    np.nan,
                )
            )
            for d in per_seed
        ],
        dtype=int,
    )


    MS_val = np.array(
        [
            int(val) if not np.isnan(val) else np.nan
            for val in [
                float(
                    _get_first_key(
                        d,
                        ("val_size","val size"),
                        np.nan,
                    )
                )
                for d in per_seed
            ]
        ],
        dtype=float,
    )
    # Use the first value if all are the same, otherwise use the most frequent value
    if len(set(MS)) == 1:
        ms = MS[0]
    else:
        ms = Counter(MS).most_common(1)[0][0]

    if MS_val is not None and not np.isnan(MS_val).all():
        if len(set(MS_val)) == 1:
            ms_val = MS_val[0]
        else:
            ms_val = Counter(MS_val).most_common(1)[0][0]
    else:
        ms_val = 0
    ms = ms + ms_val
    return {
        "n_seeds": len(per_seed),
        "tp_mean": float(np.nanmean(tp_vals)),
        "tp_std": float(np.nanstd(tp_vals)),
        "acc_mean": float(np.nanmean(acc_vals)),
        "acc_std": float(np.nanstd(acc_vals)),
        "acc_min": float(np.nanmin(acc_vals)),
        "acc_max": float(np.nanmax(acc_vals)),
        "ms": ms,
    }


def collect_time_saving_inputs_by_samples(results_dir: str, model_folder: str, dataset_name: str, n_cap: int = None, S: int = 3599):
    """
    For one model, collect {n_samples: aggregated_inputs} from JSON files.

    Output example:
      {
        20: {"tp_mean": ..., "acc_mean": ..., ...},
        30: {"tp_mean": ..., "acc_mean": ..., ...},
        ...
      }
    """
    json_samples_dict = find_all_json_by_samples(results_dir, model_folder, dataset_name, n_cap=n_cap)

    out = {}
    for n_samples, json_paths in sorted(json_samples_dict.items()):
        if not json_paths:
            continue
        out[n_samples] = aggregate_time_saving_inputs_across_seeds(json_paths,S=S)

    return out


def compute_time_saving_ratios_for_all_ta(
    tp: float,
    acc: float,
    n_samples: int,
    scenario: str,
    ta_values=None,
    S: int = 3599,
    k: float = 0.25,
    ms: int = None,
    acc_std: float = 0.0,
    acc_min: float = None,
    acc_max: float = None,
    band_mode: str = "std2",
):
    """
    Compute total times and time-saved ratios for one scenario across all Ta values.

    Definitions:
      Manual = Ta * S + Tr * S
      FSC    = MS * Ta + Tp + Tr * S + Tm * (1 - acc) * (S-MS)
      TFM    = MS * Ta + Tp + Tr * S + Tm * (1 - acc) * (S-MS)

    where:
      Tr = k * Ta
      Tm = Ta

    Uncertainty band: total_method is linear in acc, so evaluating the cost
    model at a low and a high accuracy gives the exact lower/upper bounds of
    the time-saved ratio (higher acc -> less correction -> more saving).

    band_mode controls how those low/high accuracies are chosen:
      "minmax" -> [acc_min, acc_max]  : full spread across seeds. ASYMMETRIC about
                  the mean in general, because the worst/best seed are rarely
                  equidistant from the average (a single low outlier drags the
                  lower edge far down while the upper edge stays near the mean).
      "std"    -> [acc - std, acc + std]  : symmetric, +/- 1 std.
      "std2"   -> [acc - 2*std, acc + 2*std] : symmetric, +/- 2 std.
    If min/max are unavailable, falls back to the "std" band.
    """
    if ta_values is None:
        ta_values = list(range(1, 11))

    scenario_upper = str(scenario).strip().upper()

    if scenario_upper not in {"FSC", "TFM"}:
        raise ValueError("scenario must be either 'FSC' or 'TFM'")

    ta_values = np.array(list(ta_values), dtype=float)

    tr = float(k) * ta_values
    total_manual = ta_values * int(S) + tr * int(S)

    tm = ta_values

    time_support_training = ms * ta_values
    time_processing = float(tp)
    time_review = tr * int(S)

    def _method_total(a: float) -> np.ndarray:
        time_modification = tm * (1.0 - float(a)) * (int(S) - ms)
        return time_support_training + time_processing + time_review + time_modification

    acc = float(acc)
    acc_std = float(acc_std) if acc_std is not None and not np.isnan(acc_std) else 0.0

    # --- choose the accuracy band bounds ---
    mode = str(band_mode).strip().lower()
    have_minmax = not (
        acc_min is None or acc_max is None
        or (isinstance(acc_min, float) and np.isnan(acc_min))
        or (isinstance(acc_max, float) and np.isnan(acc_max))
    )

    if mode == "minmax" and have_minmax:
        acc_lo = float(acc_min)   # worst seed  -> asymmetric in general
        acc_hi = float(acc_max)   # best seed
    elif mode == "std2":
        acc_lo = acc - 2.0 * acc_std   # symmetric, +/- 2 std
        acc_hi = acc + 2.0 * acc_std
    else:  # "std" (also the fallback when min/max are unavailable)
        acc_lo = acc - acc_std         # symmetric, +/- 1 std
        acc_hi = acc + acc_std

    # order then clip to a valid accuracy range
    lo, hi = (acc_lo, acc_hi) if acc_lo <= acc_hi else (acc_hi, acc_lo)
    acc_lo = max(0.0, lo)
    acc_hi = min(1.0, hi)

    total_method = _method_total(acc)
    total_method_at_acc_lo = _method_total(acc_lo)  # worst case (more corrections)
    total_method_at_acc_hi = _method_total(acc_hi)  # best case (fewer corrections)

    ratio = (total_manual - total_method) / total_manual
    ratio_low = (total_manual - total_method_at_acc_lo) / total_manual
    ratio_high = (total_manual - total_method_at_acc_hi) / total_manual

    time_modification = tm * (1.0 - acc) * (int(S) - ms)
    print(">>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<")
    print(f"Scenario: {scenario_upper}, S: {S}, acc: {acc:.4f} (std {acc_std:.4f}), "
          f"band[{mode}]: [{acc_lo:.4f}, {acc_hi:.4f}] "
          f"(below {acc - acc_lo:+.4f}, above {acc_hi - acc:+.4f}), tp: {tp:.2f}, ms: {ms}")
    print(f"Ta_values: {ta_values}")
    print(f"total_manual: {total_manual}")
    print(f"Support training time (MS * Ta): {time_support_training}")
    print(f"Processing time (TP): {time_processing}")
    print(f"Review time (TR * S): {time_review}")
    print(f"Modification time (TM * (1 - ACC) * (S - MS)): {time_modification}")
    print(f"total_method: {total_method}")
    print(">>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<")

    return {
        "scenario": scenario_upper,
        "n_samples": int(n_samples),
        "ms_effective": int(ms),
        "ta_values": ta_values,
        "total_manual": total_manual,
        "total_method": total_method,
        "time_saved_ratio": ratio,
        "time_saved_ratio_low": ratio_low,
        "time_saved_ratio_high": ratio_high,
        "time_saved_absolute": total_manual - total_method,
        "time_saved_absolute_low": total_manual - total_method_at_acc_lo,
        "time_saved_absolute_high": total_manual - total_method_at_acc_hi,
        "acc_mean": acc,
        "acc_std": acc_std,
        "acc_min": acc_lo,
        "acc_max": acc_hi,
        "band_mode": mode,
        "n_samples_effective": ms,
    }


# --------------------------
# Build curves for MULTIPLE models
# --------------------------
def build_time_saving_curves_for_models(
    results_dir: str,
    model_folders: list[str],
    dataset_name: str,
    n_cap: int = None,
    ta_values=None,
    S: int = 3599,
    k: float = 0.25,
    band_mode: str = "std2",
):
    """
    Returns:
      {
        model_name: {
            "FSC": {n_samples: curve_dict},
            "TFM": {n_samples: curve_dict},
        },
        ...
      }
    """
    if ta_values is None:
        ta_values = list(range(1, 11))

    all_models_curves = {}

    for model in model_folders:
        inputs_by_samples = collect_time_saving_inputs_by_samples(
            results_dir=results_dir,
            model_folder=model,
            dataset_name=dataset_name,
            n_cap=n_cap,
            S= S
        )

        curves = {}

        for n_samples, d in sorted(inputs_by_samples.items()):
            tp = d.get("tp_mean", np.nan)
            acc = d.get("acc_mean", np.nan)
            acc_std = d.get("acc_std", 0.0)
            acc_min = d.get("acc_min", np.nan)
            acc_max = d.get("acc_max", np.nan)
            ms = d.get("ms", np.nan)
            if np.isnan(tp) or np.isnan(acc):
                continue
            if "FSC" in model.upper():
                scenario = "FSC"
                n_samples_effective = n_samples
            elif "TFM" in model.upper():
                scenario = "TFM"
                n_samples_effective = n_samples
            else:
                scenario = "Unknown"  # default to FSC if not clear from name
            curves[ms] = compute_time_saving_ratios_for_all_ta(
                tp=tp,
                acc=acc,
                n_samples=n_samples,
                scenario=scenario,
                ta_values=ta_values,
                S=S,
                k=k,
                ms = ms,
                acc_std=acc_std,
                acc_min=acc_min,
                acc_max=acc_max,
                band_mode=band_mode,
            )

        print(f"model.upper = {model.upper()}, scenario = {scenario}, curves keys = {list(curves.keys())}")
        all_models_curves[model] = {
                scenario: curves
            }

    return all_models_curves


def plot_time_saved_ratio_all_models_one_graph(
    all_models_curves: dict,
    dataset_name: str,
    save_dir: str = None,
    save_figures: bool = False,
    plot_figures: bool = True,
    show_only_model_name: bool = False,
):
    """
    Plot everything in ONE graph:
      - all model folders
      - FSC curves
      - TFM curves
      - shaded band around each curve = accuracy spread across seeds (band_mode)

    Legend examples:
      MS = 20, dinov3_vits16_TFM
      MS = 20, dinov3_vits16_FSC
    """
    if not all_models_curves:
        print("No curves to plot")
        return

    plt.figure(figsize=(14, 8))

    total_curves = sum(
        len(model_curves.get("FSC", {})) + len(model_curves.get("TFM", {}))
        for model_curves in all_models_curves.values()
    )

    colors = plt.cm.tab20(np.linspace(0, 1, max(total_curves, 1)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X", "P", "d", "8"]
    linestyles = {"FSC": "-", "TFM": "--"}

    curve_idx = 0

    for model_name, model_curves in all_models_curves.items():
        model_label = get_model_name(model_name) if show_only_model_name else model_name

        for scenario in ["FSC", "TFM"]:
            curves_by_n = model_curves.get(scenario, {})

            for n_samples, d in sorted(curves_by_n.items()):
                x = np.array(d["ta_values"], dtype=float)
                y = np.array(d["time_saved_ratio"], dtype=float) * 100.0
                ms = d.get("n_samples_effective", n_samples)
                model_label = "dinov3_vits16" #for simiplicity, TODO: add option to show only model name without extra metadata

                color = colors[curve_idx % len(colors)]
                label = f"MS = {ms}, {model_label}_{scenario}"

                # Shaded band from accuracy spread across seeds (band_mode)
                y_low = d.get("time_saved_ratio_low")
                y_high = d.get("time_saved_ratio_high")
                if y_low is not None and y_high is not None:
                    plt.fill_between(
                        x,
                        np.array(y_low, dtype=float) * 100.0,
                        np.array(y_high, dtype=float) * 100.0,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                        zorder=1,
                    )

                plt.plot(
                    x,
                    y,
                    marker=markers[curve_idx % len(markers)],
                    linestyle=linestyles[scenario],
                    linewidth=2,
                    color=color,
                    label=label,
                    zorder=2,
                )


                curve_idx += 1

    plt.xlabel("Ta - Average annotation time per point (seconds)", fontsize=20)
    plt.ylabel("Time Saved Ratio wrt Manual Annotation Time (%)", fontsize=20)
    plt.title(f"{dataset_name} - Time Saved Ratio (All Models, FSC + TFM)", fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=20)
    plt.xticks(list(range(1, 11)))
    plt.tight_layout()

    if save_figures and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{dataset_name}_all_models_fsc_tfm_time_saved_ratio.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)

    if plot_figures:
        plt.show()
    else:
        plt.close()


def plot_time_saved_abs_all_models_one_graph(
    all_models_curves: dict,
    dataset_name: str,
    save_dir: str = None,
    save_figures: bool = False,
    plot_figures: bool = True,
    show_only_model_name: bool = False,
):
    """
    Plot everything in ONE graph:
      - all model folders
      - FSC curves
      - TFM curves
      - shaded band around each curve = accuracy spread across seeds (band_mode)

    Legend examples:
      MS = 20, dinov3_vits16_TFM
      MS = 20, dinov3_vits16_FSC
    """
    if not all_models_curves:
        print("No curves to plot")
        return

    plt.figure(figsize=(14, 8))

    total_curves = sum(
        len(model_curves.get("FSC", {})) + len(model_curves.get("TFM", {}))
        for model_curves in all_models_curves.values()
    )

    colors = plt.cm.tab20(np.linspace(0, 1, max(total_curves, 1)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X", "P", "d", "8"]
    linestyles = {"FSC": "-", "TFM": "--"}

    curve_idx = 0

    for model_name, model_curves in all_models_curves.items():
        model_label = get_model_name(model_name) if show_only_model_name else model_name

        for scenario in ["FSC", "TFM"]:
            curves_by_n = model_curves.get(scenario, {})

            for n_samples, d in sorted(curves_by_n.items()):
                x = np.array(d["ta_values"], dtype=float)
                y = np.array(d["time_saved_absolute"], dtype=float) / 60.0
                ms = d.get("n_samples_effective", n_samples)
                model_label = "dinov3_vits16" #for simiplicity, TODO: add option to show only model name without extra metadata

                color = colors[curve_idx % len(colors)]
                label = f"MS = {ms}, {model_label}_{scenario}"

                # Shaded band from accuracy spread across seeds (band_mode)
                y_low = d.get("time_saved_absolute_low")
                y_high = d.get("time_saved_absolute_high")
                if y_low is not None and y_high is not None:
                    plt.fill_between(
                        x,
                        np.array(y_low, dtype=float) / 60.0,
                        np.array(y_high, dtype=float) / 60.0,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                        zorder=1,
                    )

                plt.plot(
                    x,
                    y,
                    marker=markers[curve_idx % len(markers)],
                    linestyle=linestyles[scenario],
                    linewidth=2,
                    color=color,
                    label=label,
                    zorder=2,
                )

                curve_idx += 1

    plt.xlabel("Ta - Average annotation time per point (seconds)", fontsize=20)
    plt.ylabel("Time Saved (minutes) wrt. Manual Annotation time", fontsize=20)
    plt.title(f"{dataset_name} - Time Saved (minutes) (All Models, FSC + TFM)", fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=20)
    plt.xticks(list(range(1, 11)))
    plt.tight_layout()

    if save_figures and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{dataset_name}_all_models_fsc_tfm_time_saved_abs.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)

    if plot_figures:
        plt.show()
    else:
        plt.close()


# --------------------------
# Curve selection
# --------------------------
def filter_curves_by_scenario_ms(all_models_curves: dict, keep) -> dict:
    """
    Keep only curves whose (scenario, MS) is in `keep`.

    keep: iterable of (scenario, ms) tuples, e.g. {("FSC", 100), ("TFM", 1100)}.
          None or empty -> keep everything.
    """
    if not keep:
        return all_models_curves

    keep_norm = {(str(s).strip().upper(), int(m)) for s, m in keep}

    out = {}
    for model_name, scen_dict in all_models_curves.items():
        new_scen = {}
        for scenario, curves_by_ms in scen_dict.items():
            kept = {
                ms: d
                for ms, d in curves_by_ms.items()
                if (str(scenario).strip().upper(), int(ms)) in keep_norm
            }
            if kept:
                new_scen[scenario] = kept
        if new_scen:
            out[model_name] = new_scen

    if not out:
        print(f"Warning: plot_only filter {sorted(keep_norm)} matched no curves; nothing will be plotted.")
    return out


def parse_plot_only_arg(s: str):
    """
    Parse a CLI string like 'FSC:100,TFM:1100' into {("FSC", 100), ("TFM", 1100)}.
    Empty string -> None (plot all curves).
    """
    s = (s or "").strip()
    if not s:
        return None
    keep = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        scenario, ms = part.split(":")
        keep.add((scenario.strip().upper(), int(float(ms))))
    return keep


# --------------------------
# Convenience wrapper
# --------------------------
def run_time_saving_analysis_for_models(
    results_dir: str,
    model_folders: list[str],
    dataset_name: str,
    n_cap: int = None,
    ta_values=None,
    S: int = 3599,
    k: float = 0.25,
    save_dir: str = None,
    save_figures: bool = False,
    plot_figures: bool = True,
    show_only_model_name: bool = False,
    plot_only=None,
    band_mode: str = "std2",
):
    """
    Full pipeline:
      - build curves for all models
      - optionally keep only selected (scenario, MS) curves via `plot_only`,
        e.g. {("FSC", 100), ("TFM", 1100)}
      - plot the selected FSC and TFM curves in ONE graph

    band_mode: "minmax" (full spread, asymmetric), "std" (+/-1 std, symmetric),
               or "std2" (+/-2 std, symmetric).
    """
    all_models_curves = build_time_saving_curves_for_models(
        results_dir=results_dir,
        model_folders=model_folders,
        dataset_name=dataset_name,
        n_cap=n_cap,
        ta_values=ta_values,
        S=S,
        k=k,
        band_mode=band_mode,
    )

    all_models_curves = filter_curves_by_scenario_ms(all_models_curves, plot_only)

    plot_time_saved_ratio_all_models_one_graph(
        all_models_curves=all_models_curves,
        dataset_name=dataset_name,
        save_dir=save_dir,
        save_figures=save_figures,
        plot_figures=plot_figures,
        show_only_model_name=show_only_model_name,
    )


    plot_time_saved_abs_all_models_one_graph(
        all_models_curves=all_models_curves,
        dataset_name=dataset_name,
        save_dir=save_dir,
        save_figures=save_figures,
        plot_figures=plot_figures,
        show_only_model_name=show_only_model_name,
    )


    return all_models_curves


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
    parser.add_argument("--S", type=int, default=3599, help="Number of samples in the dataset")
    parser.add_argument(
        "--plot_only",
        type=str,
        default="FSC:100,TFM:1100",
        help="Comma-separated SCENARIO:MS pairs to plot, e.g. 'FSC:100,TFM:1100'. Empty string plots all curves.",
    )
    parser.add_argument(
        "--band_mode",
        type=str,
        default="std2",
        choices=["minmax", "std", "std2"],
        help="Shaded band: 'minmax' = full spread across seeds (asymmetric); "
             "'std' = mean +/- 1 std (symmetric); 'std2' = mean +/- 2 std (symmetric).",
    )

    args = parser.parse_args()
    show_only_model_name = args.show_only_model_name
    dataset_name = args.dataset
    dataset_ID = dataset_name_mapping[dataset_name ] if dataset_name in dataset_name_mapping else dataset_name 
    dataset_name = f"{dataset_ID} ({dataset_name})"
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



    run_time_saving_analysis_for_models(
        results_dir=results_dir,
        model_folders=model_folders,
        dataset_name=dataset_name,
        n_cap=100000,
        save_dir=args.save_dir,
        save_figures=True,
        S = args.S,
        plot_only=parse_plot_only_arg(args.plot_only),
        band_mode=args.band_mode,
    )
