

import hashlib
import json
import math
import os

import numpy as np
from sklearn.cluster import KMeans


# Elbow settings (fixed)
ELBOW_K_MIN = 2              # smallest number of clusters searched
MIN_SAMPLES_PER_CLUSTER = 2  # k_max = floor(M / 2) -> at least 2 samples per cluster
ELBOW_N_INIT = 3
ELBOW_MAX_POINTS = 20000
_ELBOW_CACHE_VERSION = "v1"

SELECTION_NAME = "elbow_centroid_farthest_first"


# =============================================================================
# Elbow method
# =============================================================================
def _embedding_fingerprint(X):
    """Short hash of the embedding matrix, so the elbow cache is only reused for identical embeddings."""
    h = hashlib.sha1()
    h.update(str(X.shape).encode())
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def kneedle(ks, wcss):
    """
    Offline Kneedle for a decreasing convex curve.
    Returns (k_knee, strength); (None, 0.0) if no knee can be computed.
    """
    ks = np.asarray(ks, dtype=float)
    y = np.minimum.accumulate(np.asarray(wcss, dtype=float))   # monotone non-increasing
    if ks.size < 3:
        return None, 0.0
    x_rng = ks[-1] - ks[0]
    y_rng = y[0] - y[-1]
    if x_rng <= 0 or y_rng <= 0:
        return None, 0.0
    x_n = (ks - ks[0]) / x_rng
    y_n = (y - y[-1]) / y_rng
    d = (1.0 - x_n) - y_n
    i = int(np.argmax(d))          # first maximum -> smallest k on ties
    return int(ks[i]), float(d[i])


def _wcss_input(X, seed, max_points=ELBOW_MAX_POINTS):
    """Rows used for the WCSS sweep (seeded subset if N > max_points)."""
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    if max_points is not None and N > max_points:
        rng = np.random.default_rng(seed)
        return X[np.sort(rng.choice(N, size=int(max_points), replace=False))]
    return X


def compute_wcss(X_fit, ks, seed=42, n_init=ELBOW_N_INIT):
    """K-means inertia (WCSS) for every k in ks. Returns {k: wcss}."""
    out = {}
    for k in ks:
        km = KMeans(n_clusters=int(k), random_state=seed, init="k-means++", n_init=n_init)
        km.fit(X_fit)
        out[int(k)] = float(km.inertia_)
    return out


def _load_or_compute_wcss(X, ks, seed, cache_path):
    """
    WCSS for the requested ks. Values are cached per k (WCSS(k) does not depend on the budget M),
    so a run with a larger M only computes the missing k values.
    """
    key = (f"{_ELBOW_CACHE_VERSION}_{_embedding_fingerprint(X)}_seed{seed}"
           f"_ninit{ELBOW_N_INIT}_max{ELBOW_MAX_POINTS}")
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            cache = {}
    stored = {int(k): v for k, v in cache.get(key, {}).items()}
    missing = [k for k in ks if k not in stored]
    if missing:
        stored.update(compute_wcss(_wcss_input(X, seed), missing, seed=seed))
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                cache[key] = {str(k): v for k, v in sorted(stored.items())}
                tmp_path = f"{cache_path}.tmp{os.getpid()}"
                with open(tmp_path, "w") as f:
                    json.dump(cache, f)
                os.replace(tmp_path, cache_path)
            except OSError as e:
                print(f"[elbow] could not write cache {cache_path}: {e}")
    else:
        print(f"\033[1;36m[elbow] using cached WCSS values ({cache_path})\033[0m")
    return [stored[k] for k in ks]


def elbow_number_of_clusters(X, M, seed=42, cache_path=None):
    """
    K from the elbow method searched over k = 2 .. min(floor(M/2), N-1), so that every cluster
    receives at least 2 samples. Returns (K, info).
    """
    N = X.shape[0]
    M = int(M)
    k_max = min(M // MIN_SAMPLES_PER_CLUSTER, N - 1)

    if k_max < ELBOW_K_MIN:
        # M < 4: two clusters cannot both receive 2 samples -> a single cluster
        K = 1
        print(f"\033[1;33m[elbow] M={M} too small for 2 clusters with >= {MIN_SAMPLES_PER_CLUSTER} "
              f"samples each; K = 1\033[0m")
        return K, {"elbow_k_range": None, "k_elbow": None, "elbow_knee_strength": 0.0,
                   "elbow_no_curve_reason": "budget_too_small", "k_max": int(max(k_max, 0))}

    ks = list(range(ELBOW_K_MIN, k_max + 1))
    wcss = _load_or_compute_wcss(X, ks, seed, cache_path) if len(ks) >= 3 else []
    k_knee, strength = kneedle(ks, wcss) if wcss else (None, 0.0)

    if k_knee is None:
        reason = "too_few_candidate_k" if len(ks) < 3 else "flat_wcss_curve"
        K = ELBOW_K_MIN
        print(f"\033[1;33m[elbow] no WCSS curve to analyse ({reason}); K = {ELBOW_K_MIN}\033[0m")
    else:
        reason = None
        K = k_knee

    print(f"\033[1;36m[elbow] k range [{ks[0]}, {ks[-1]}] (k_max = floor(M/2) = {M // 2}, N={N}), "
          f"K = {K} (knee strength {strength:.3f}), >= {M // K} samples per cluster\033[0m")

    info = {
        "elbow_k_range": [ks[0], ks[-1]],
        "k_elbow": int(K),
        "elbow_knee_strength": strength,
        "elbow_no_curve_reason": reason,
        "k_max": int(k_max),
    }
    return K, info


def split_budget_first_clusters(n_clusters, N_samples):
    """N_samples // K for each cluster, remainder +1 to the first clusters."""
    reps_per_cluster_base = N_samples // n_clusters
    remainder = N_samples % n_clusters
    reps_per_cluster = [reps_per_cluster_base] * n_clusters
    for i in range(remainder):
        reps_per_cluster[i] += 1
    return reps_per_cluster


# =============================================================================
# Centroid-initialized farthest-first sampling
# =============================================================================
def _select_centroid_farthest_first(X_train, train_cluster_ids, centers, K, Ms):
    selected_global_indices = []

    for c in range(K):
        M = Ms[c]

        idxs = np.where(train_cluster_ids == c)[0]
        if idxs.size == 0:
            continue

        # 1) start from the member closest to the cluster centroid
        diffs = X_train[idxs] - centers[c][None, :]
        dist2 = np.sum(diffs * diffs, axis=1)

        center_local = int(np.argmin(dist2))
        center_idx = int(idxs[center_local])
        chosen = [center_idx]

        if M == 1 or idxs.size == 1:
            selected_global_indices.extend(chosen)
            continue

        # 2) farthest-first: add the member farthest from all already-selected members
        chosen_set = set(chosen)
        V = X_train[idxs]
        global_to_local = {int(g): li for li, g in enumerate(idxs.tolist())}
        chosen_locals = [global_to_local[center_idx]]

        for _ in range(M - 1):
            cand_globals = [int(g) for g in idxs.tolist() if int(g) not in chosen_set]
            if not cand_globals:
                break
            cand_locals = [global_to_local[int(g)] for g in cand_globals]
            Cc = V[cand_locals]
            min_d2 = None
            for cl in chosen_locals:
                d = Cc - V[cl][None, :]
                d2 = np.sum(d * d, axis=1)
                min_d2 = d2 if min_d2 is None else np.minimum(min_d2, d2)
            pick = int(np.argmax(min_d2))
            picked_global = int(cand_globals[pick])
            chosen.append(picked_global)
            chosen_set.add(picked_global)
            chosen_locals.append(global_to_local[picked_global])

        selected_global_indices.extend(chosen)
    return selected_global_indices


# =============================================================================
# Public selection strategies
# =============================================================================
def random_subsample(train_embeddings, train_meta, N_samples, cat_names, seed=42):


    num_classes = len(cat_names)

    name_to_pseudo_id = {name: i for i, name in enumerate(cat_names)}
    id_to_name = {i: name for name, i in name_to_pseudo_id.items()}
    print(f"Found TRAIN category names ({num_classes}): {cat_names}")

    # Optional: subsample reference set per class (classic / traditional approach for stability)
    # samples_per_class: int or None/0 -> use all
    if N_samples is None:
        spc = []
    else:
        print(f"Subsampling reference set to {N_samples} samples randomly ...")
        spc_base = N_samples // num_classes
        spc_remainder = N_samples % num_classes
        spc = [spc_base] * num_classes
        for i in range(spc_remainder):
            spc[i] += 1

    if len(spc) > 0:
        # group indices by class
        by_class = {c: [] for c in cat_names}
        for idx, m in enumerate(train_meta):
            by_class[m["category_name"]].append(idx)

        rng = np.random.default_rng(seed)
        selected = []
        for cname, idxs in by_class.items():
            if len(idxs) == 0:
                continue
            if len(idxs) <=   spc[cat_names.index(cname)]:
                chosen = idxs
            else:
                chosen = rng.choice(idxs, size=spc[cat_names.index(cname)], replace=False).tolist()
            selected.extend(chosen)

        selected = sorted(set(selected))
        ref_embeddings = train_embeddings[selected]
        ref_label_names = np.array([train_meta[i]["category_name"] for i in selected], dtype=object)
    else:
        selected = list(range(len(train_meta)))
        ref_embeddings = train_embeddings
        ref_label_names = np.array([m["category_name"] for m in train_meta], dtype=object)

    ref_label_ids = np.array([name_to_pseudo_id[n] for n in ref_label_names], dtype=int)
    ref_meta = [train_meta[i] for i in selected]

    return ref_embeddings, ref_label_names, ref_label_ids, ref_meta


def prioritized_subsample(train_embeddings, train_meta, N_samples, cat_names, seed=42,
                          elbow_cache_path=None, selection_info=None):
    """
    Elbow + Centroid-Initialized Farthest-First Sampling (see module docstring).
    Returns ref_embeddings, rep_label_names, ref_label_ids, ref_meta.
    """
    if N_samples is None or int(N_samples) <= 0:
        raise ValueError("prioritized_subsample needs a positive N_samples (annotation budget M).")
    N_samples = int(N_samples)

    num_classes = len(cat_names)
    name_to_pseudo_id = {name: i for i, name in enumerate(cat_names)}
    print(f"Subsampling reference set to {N_samples} samples using prioritized labeling "
          f"(elbow K + centroid-initialized farthest-first sampling) ...")

    N_train = train_embeddings.shape[0]

    print(f"Found train category names ({num_classes}): {cat_names}")

    X_train = train_embeddings.numpy().astype(np.float32)  # [N, D]

    # =========================
    # 1) Number of clusters K (elbow) and samples per cluster
    # =========================
    K, elbow_info = elbow_number_of_clusters(X_train, N_samples, seed=seed, cache_path=elbow_cache_path)
    Ms = split_budget_first_clusters(K, N_samples)

    # =========================
    # 2) Cluster train embeddings
    # =========================
    kmeans = KMeans(n_clusters=K, random_state=seed, init="k-means++")
    train_cluster_ids = kmeans.fit_predict(X_train)  # [N]
    centers = kmeans.cluster_centers_.astype(np.float32)  # [K, D]
    cluster_sizes = np.bincount(train_cluster_ids, minlength=K)

    # =========================
    # 3) Centroid-initialized farthest-first sampling per cluster
    # =========================
    selected_global_indices = _select_centroid_farthest_first(X_train, train_cluster_ids, centers, K, Ms)

    # Deduplicate (stable order)
    seen = set()
    selected_global_indices = [i for i in selected_global_indices if not (i in seen or seen.add(i))]
    print(f"Selected representatives: {len(selected_global_indices)} (K={K}, M={Ms})")
    print(f"\033[1;34mselected_global_indices: {selected_global_indices}\033[0m")
    if len(selected_global_indices) == 0:
        raise RuntimeError("No representatives were selected; check clustering.")
    if len(selected_global_indices) < N_samples:
        print(f"\033[1;33m[prioritized] WARNING: only {len(selected_global_indices)} of {N_samples} samples selected "
              f"(some clusters had fewer members than their share).\033[0m")

    # =========================
    # 4) Build ref set from representatives (using their known labels from train_meta)
    # =========================
    rep_label_names = np.array([train_meta[i]["category_name"] for i in selected_global_indices], dtype=object)
    rep_label_ids = np.array([name_to_pseudo_id[n] for n in rep_label_names], dtype=int)

    ref_embeddings = train_embeddings[selected_global_indices]  # CPU tensor [R, D]
    ref_label_ids = rep_label_ids
    ref_meta = [train_meta[i] for i in selected_global_indices]

    if selection_info is not None:
        selection_info.update({
            "selection": SELECTION_NAME,
            "budget_M": N_samples,
            "n_clusters": int(K),
            "cluster_sizes": [int(s) for s in cluster_sizes],
            "samples_per_cluster": [int(m) for m in Ms],
            "n_selected": int(len(selected_global_indices)),
            **elbow_info,
        })

    return ref_embeddings, rep_label_names, ref_label_ids, ref_meta


def all_subsample(train_embeddings, train_meta, N_samples, cat_names, seed=42):
    num_classes = len(cat_names)
    name_to_pseudo_id = {name: i for i, name in enumerate(cat_names)}
    id_to_name = {i: name for name, i in name_to_pseudo_id.items()}
    print(f"Found TRAIN category names ({num_classes}): {cat_names}")

    # Use all samples
    selected = np.arange(len(train_meta))
    ref_embeddings = train_embeddings[selected]
    ref_label_names = np.array([train_meta[i]["category_name"] for i in selected], dtype=object)
    ref_label_ids = np.array([name_to_pseudo_id[n] for n in ref_label_names], dtype=int)
    ref_meta = [train_meta[i] for i in selected]

    return ref_embeddings, ref_label_names, ref_label_ids, ref_meta
