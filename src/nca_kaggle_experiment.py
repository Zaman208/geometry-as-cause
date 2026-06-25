#!/usr/bin/env python3
"""
For each dataset this script computes:
  - alpha            : neural scaling exponent via MLP learning curves
  - fisher_ratio     : between/within class scatter ratio
  - silhouette       : cluster coherence (shifted to [0,1])
  - delta            : FR x silhouette (the geometric Delta-metric)
  - intrinsic_dim    : TwoNN dimensionality estimate
  - linear_probe_difficulty : logistic regression 5-fold CV log-loss
  - class_entropy, label_noise, class_imbalance
  - n_samples, n_features, n_classes

"""

import os, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import openml
warnings.filterwarnings("ignore")

# -- Output directory -----------------------------------------------------------
OUT = "/kaggle/working/nca_out" if os.path.exists("/kaggle") else "./nca_out"
os.makedirs(OUT, exist_ok=True)

# -- Reproducibility ------------------------------------------------------------
MASTER_SEED = 42
np.random.seed(MASTER_SEED)

# -- Hyperparameters ------------------------------------------------------------
MIN_TRAIN    = 40
MIN_POINTS   = 4
MIN_R2       = 0.20
N_SEEDS      = 3
BOOTSTRAP_N  = 500
MAX_SAMPLES  = 8000
MAX_FEATURES = 100
FRACS        = np.array([0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00])

# -- Exact same 103 datasets as causal experiment ------------------------------
DATASETS = {
    "blood-transfusion":        1464,
    "banknote-authentication":  1462,
    "bank-marketing":           1461,
    "eeg-eye-state":            1471,
    "adult":                    1590,
    "electricity":              151,
    "magic-telescope":          1120,
    "qsar-biodeg":              1494,
    "phoneme":                  1489,
    "ozone":                    1487,
    "hill-valley":              1479,
    "wall-robot-navigation":    1497,
    "cnae-9":                   1468,
    "nomao":                    1486,
    "cardiotocography":         1466,
    "kc2":                      1063,
    "kc1":                      1067,
    "breast-w":                 15,
    "pc1":                      1068,
    "haberman":                 43,
    "ecoli":                    40671,
    "diabetes":                 37,
    "heart-statlog":            53,
    "ilpd":                     1480,
    "yeast":                    181,
    "dermatology":              35,
    "liver-disorders":          8,
    "balance-scale":            11,
    "arrhythmia":               5,
    "credit-g":                 31,
    "glass":                    41,
    "kr-vs-kp":                 3,
    "nursery":                  26,
    "mushroom":                 24,
    "segment":                  36,
    "soybean":                  42,
    "sonar":                    40,
    "splice":                   46,
    "vehicle":                  54,
    "vowel":                    307,
    "cmc":                      23,
    "credit-approval":          29,
    "letter":                   6,
    "mfeat-factors":            12,
    "mfeat-fourier":            14,
    "optdigits":                28,
    "satimage":                 182,
    "waveform-5000":            60,
    "pendigits":                32,
    "spambase":                 44,
    "madelon":                  1485,
    "steel-plates-fault":       1504,
    "hepatitis":                55,
    "autos":                    9,
    "anneal":                   2,
    "primary-tumor":            171,
    "mfeat-morphological":      16,
    "mfeat-karhunen":           13,
    "mfeat-zernike":            22,
    "mfeat-pixel":              20,
    "semeion":                  1501,
    "twonorm":                  1507,
    "libras":                   299,
    "ringnorm":                 1496,
    "isolet":                   300,
    "PhishingWebsites":         1019,
    "amazon-commerce-reviews":  1457,
    "kc3":                      1069,
    "pc2":                      1066,
    "mnist":                    554,
    "GesturePhaseSegmentation": 1472,
    "fashion-mnist":            40996,
    "sick":                     38,
    "jm1":                      1053,
    "spect":                    336,
    "dna":                      40670,
    "musk1":                    1116,
    "monks-2":                  334,
    "tic-tac-toe":              50,
    "colic":                    25,
    "page-blocks":              30,
    "wilt":                     40983,
    "har":                      1478,
    "cpu-small":                560,
    "analcatdata-dmft":         962,
    "analcatdata-authorship":   458,
    "analcatdata-lawsuit":      299,
    "robot-failures-lp2":       1498,
    "robot-failures-lp4":       1500,
    "kin8nm":                   189,
    "cpu-act":                  197,
    "shuttle":                  40685,
    "pol":                      722,
    "ele-1":                    1037,
    "2dplanes":                 727,
    "user-knowledge":           1508,
    "parity5+5":                1016,
    "monks-1":                  333,
    "census-income":            4535,
    "connect-4":                40668,
    "hypothyroid":              40,
    "mofn-3-7-10":              1015,
    "pc3":                      1050,
    "porto-seguro":             42742,
}

# Deduplicate by OpenML ID
_seen, _deduped = set(), {}
for k, v in DATASETS.items():
    if v not in _seen:
        _seen.add(v)
        _deduped[k] = v
DATASETS = _deduped
print(f"Dataset registry: {len(DATASETS)} unique datasets")


# PREPROCESSING
def preprocess(X_raw, y_raw):
    df = X_raw.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(df.median(numeric_only=True))
    X = df.values.astype(np.float32)

    y = LabelEncoder().fit_transform(y_raw.astype(str))

    # Drop classes with fewer than 5 samples
    cls, cnt = np.unique(y, return_counts=True)
    valid = cls[cnt >= 5]
    if len(valid) < 2:
        return None, None
    mask = np.isin(y, valid)
    X, y = X[mask], y[mask]
    y = LabelEncoder().fit_transform(y)

    # Cap size
    if len(y) > MAX_SAMPLES:
        idx = np.random.default_rng(MASTER_SEED).choice(len(y), MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]

    # Cap features by variance
    if X.shape[1] > MAX_FEATURES:
        top = np.argsort(X.var(axis=0))[::-1][:MAX_FEATURES]
        X = X[:, top]

    X = RobustScaler().fit_transform(X).astype(np.float32)
    return X, y


# ALPHA ESTIMATION
def estimate_alpha(X, y):
    n = len(y)
    K = len(np.unique(y))
    if n < 80 or K < 2:
        return np.nan, np.nan, np.nan, np.nan

    sizes = np.unique(np.maximum(MIN_TRAIN, (FRACS * int(0.8 * n)).astype(int)))
    sizes = sizes[sizes <= int(0.8 * n)]
    if len(sizes) < MIN_POINTS:
        return np.nan, np.nan, np.nan, np.nan

    losses = []
    for size in sizes:
        fold_losses = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + MASTER_SEED)
            idx_all   = rng.permutation(n)
            train_idx = idx_all[:size]
            test_size = max(30, int(0.15 * n))
            test_idx  = idx_all[max(0, n - test_size):]
            if len(test_idx) < 5:
                continue
            Xtr, ytr = X[train_idx], y[train_idx]
            Xte, yte = X[test_idx],  y[test_idx]
            if len(np.unique(ytr)) < 2:
                continue
            try:
                mlp = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    random_state=seed,
                    learning_rate_init=1e-3,
                    alpha=1e-4,
                )
                mlp.fit(Xtr, ytr)
                proba = mlp.predict_proba(Xte)
                full  = np.zeros((len(yte), K))
                for j, c in enumerate(mlp.classes_):
                    full[:, c] = proba[:, j]
                full = np.clip(full, 1e-15, 1)
                full /= full.sum(axis=1, keepdims=True)
                fold_losses.append(log_loss(yte, full, labels=list(range(K))))
            except Exception:
                continue
        losses.append(np.nanmean(fold_losses) if fold_losses else np.nan)

    valid = [(s, l) for s, l in zip(sizes, losses)
             if np.isfinite(l) and l > 0 and s > 0]
    if len(valid) < MIN_POINTS:
        return np.nan, np.nan, np.nan, np.nan

    D = np.array([v[0] for v in valid], dtype=float)
    L = np.array([v[1] for v in valid], dtype=float)
    logD, logL = np.log(D), np.log(L)

    slope, _, r, _, _ = stats.linregress(logD, logL)
    alpha_hat = float(-slope)
    r2 = float(r ** 2)

    if r2 < MIN_R2:
        return np.nan, r2, np.nan, np.nan

    # Bootstrap CI
    rng_b = np.random.default_rng(MASTER_SEED)
    boots = []
    for _ in range(BOOTSTRAP_N):
        idx_b = rng_b.integers(0, len(D), len(D))
        if len(np.unique(idx_b)) < 2:
            continue
        try:
            s_b, *_ = stats.linregress(logD[idx_b], logL[idx_b])
            boots.append(-s_b)
        except Exception:
            continue
    ci_lo = float(np.percentile(boots, 2.5))  if len(boots) > 10 else np.nan
    ci_hi = float(np.percentile(boots, 97.5)) if len(boots) > 10 else np.nan

    return alpha_hat, r2, ci_lo, ci_hi


# FEATURE FUNCTIONS
def compute_fisher_ratio(X, y):
    classes = np.unique(y)
    n, d = X.shape
    grand_mean = X.mean(axis=0)
    try:
        sb = sum(
            np.sum(y == c) * np.outer(
                X[y == c].mean(axis=0) - grand_mean,
                X[y == c].mean(axis=0) - grand_mean,
            ) for c in classes
        )
        sw = sum(
            np.cov(X[y == c].T, bias=True) * np.sum(y == c)
            if np.sum(y == c) > 1 else np.zeros((d, d))
            for c in classes
        )
        return float(np.trace(sb)) / (float(np.trace(sw)) + 1e-10)
    except Exception:
        return np.nan


def compute_silhouette(X, y):
    n = len(y)
    K = len(np.unique(y))
    if K < 2 or K >= n:
        return np.nan
    try:
        if n > 2000:
            idx = np.random.default_rng(MASTER_SEED).choice(n, 2000, replace=False)
            Xs, ys = X[idx], y[idx]
            if len(np.unique(ys)) < 2:
                Xs, ys = X, y
        else:
            Xs, ys = X, y
        raw = silhouette_score(Xs, ys, random_state=MASTER_SEED)
        return float((raw + 1.0) / 2.0)
    except Exception:
        return np.nan


def compute_intrinsic_dim(X):
    """TwoNN estimator (Facco et al. 2017)."""
    n = X.shape[0]
    if n > 3000:
        idx = np.random.default_rng(MASTER_SEED).choice(n, 3000, replace=False)
        X = X[idx]
    try:
        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        dists, _ = nbrs.kneighbors(X)
        r1 = dists[:, 1]
        r2 = dists[:, 2]
        mask = (r1 > 0) & (r2 > r1)
        mu   = r2[mask] / r1[mask]
        if mask.sum() < 10:
            return float(X.shape[1])
        d_hat = 1.0 / (np.mean(np.log(mu)) + 1e-10)
        return float(np.clip(d_hat, 1.0, X.shape[1]))
    except Exception:
        return float(X.shape[1])


def compute_probe_difficulty(X, y):
    K = len(np.unique(y))
    try:
        lr = LogisticRegression(max_iter=400, random_state=MASTER_SEED, C=1.0)
        cv = min(5, K, len(y) // 20 + 2)
        scores = cross_val_score(lr, X, y, cv=cv,
                                 scoring="neg_log_loss", error_score=np.nan)
        return float(np.nanmean(-scores))
    except Exception:
        return np.nan


def compute_label_noise(X, y):
    try:
        k = min(5, len(y) // 10)
        if k < 1:
            return np.nan
        knn = KNeighborsClassifier(n_neighbors=k + 1)
        knn.fit(X, y)
        _, idxs = knn.kneighbors(X, n_neighbors=k + 1)
        neighbor_y = y[idxs[:, 1:]]
        return float(np.mean(neighbor_y != y[:, None]))
    except Exception:
        return np.nan


def extract_all_features(X, y):
    n, d = X.shape
    K    = len(np.unique(y))
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    fr   = compute_fisher_ratio(X, y)
    sil  = compute_silhouette(X, y)
    dint = compute_intrinsic_dim(X)
    prob = compute_probe_difficulty(X, y)
    ln   = compute_label_noise(X, y)
    ent  = float(-np.sum(probs * np.log(probs + 1e-10)))
    imb  = float(counts.max() / (counts.min() + 1))
    delta = float(fr * sil) if (np.isfinite(fr) and np.isfinite(sil)) else np.nan

    return {
        "fisher_ratio":            fr,
        "silhouette":              sil,
        "delta":                   delta,
        "intrinsic_dim":           dint,
        "linear_probe_difficulty": prob,
        "label_noise":             ln,
        "class_entropy":           ent,
        "class_imbalance":         imb,
        "n_samples":               int(n),
        "n_features":              int(d),
        "n_classes":               int(K),
    }


# MAIN LOOP
def run_experiment():
    results  = []
    failed   = []
    t_start  = time.time()
    names    = list(DATASETS.keys())
    n_total  = len(names)

    for i, name in enumerate(names):
        did = DATASETS[name]
        print(f"\n[{i+1:3d}/{n_total}]  {name}  (OpenML ID={did})")

        try:
            t0 = time.time()

            ds = openml.datasets.get_dataset(
                did,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            X_raw, y_raw, _, _ = ds.get_data(
                dataset_format="dataframe",
                target=ds.default_target_attribute,
            )
            if y_raw is None:
                raise ValueError("No target column")

            X, y = preprocess(X_raw, y_raw)
            if X is None:
                raise ValueError("Preprocessing failed: fewer than 2 valid classes")

            n, d = X.shape
            K    = len(np.unique(y))
            print(f"  preprocessed: n={n}  d={d}  K={K}")

            print(f"  computing features...")
            feats = extract_all_features(X, y)
            print(f"  FR={feats['fisher_ratio']:.4f}  sil={feats['silhouette']:.4f}"
                  f"  delta={feats['delta']:.4f}  d_int={feats['intrinsic_dim']:.2f}"
                  f"  probe={feats['linear_probe_difficulty']:.4f}")

            print(f"  estimating alpha  (N_SEEDS={N_SEEDS})...")
            t_alp = time.time()
            alpha, r2, ci_lo, ci_hi = estimate_alpha(X, y)
            print(f"  alpha={alpha:.4f}  R2={r2:.3f}  "
                  f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  ({time.time()-t_alp:.1f}s)")

            if not np.isfinite(alpha):
                failed.append({
                    "dataset":   name,
                    "openml_id": did,
                    "reason":    f"alpha quality gate failed (R2={r2:.3f})",
                })
                print(f"  FAILED quality gate")
                continue

            row = {
                "dataset":       name,
                "openml_id":     did,
                "alpha":         round(alpha, 6),
                "alpha_r2":      round(r2, 4),
                "alpha_ci_low":  round(ci_lo, 6) if np.isfinite(ci_lo) else np.nan,
                "alpha_ci_high": round(ci_hi, 6) if np.isfinite(ci_hi) else np.nan,
                **{k: round(v, 6) if isinstance(v, float) else v
                   for k, v in feats.items()},
                "elapsed_s": round(time.time() - t0, 1),
            }
            results.append(row)
            print(f"  ACCEPTED  alpha={alpha:.4f}  elapsed={time.time()-t0:.1f}s")

            # Rolling checkpoint
            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(
                    os.path.join(OUT, "results_checkpoint.csv"), index=False
                )
                print(f"  -- checkpoint: {len(results)} accepted --")

        except Exception as e:
            failed.append({"dataset": name, "openml_id": did, "reason": str(e)})
            print(f"  ERROR: {e}")

    # -- Final outputs ----------------------------------------------------------
    df_res  = pd.DataFrame(results)
    df_fail = pd.DataFrame(failed)

    df_res.to_csv(os.path.join(OUT,  "results.csv"), index=False)
    df_fail.to_csv(os.path.join(OUT, "failed.csv"),  index=False)

    elapsed = time.time() - t_start
    summary = {
        "total_attempted": n_total,
        "total_accepted":  len(results),
        "total_failed":    len(failed),
        "accept_rate":     round(len(results) / n_total, 3),
        "alpha_mean":      round(float(df_res["alpha"].mean()), 4) if len(results) else None,
        "alpha_median":    round(float(df_res["alpha"].median()), 4) if len(results) else None,
        "alpha_std":       round(float(df_res["alpha"].std()), 4) if len(results) else None,
        "alpha_range":     [round(float(df_res["alpha"].min()), 4),
                             round(float(df_res["alpha"].max()), 4)] if len(results) else None,
        "elapsed_min":     round(elapsed / 60, 1),
        "output_dir":      OUT,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"COMPLETE  |  accepted={len(results)}  failed={len(failed)}"
          f"  time={elapsed/60:.1f}min")
    print(f"results.csv -> {OUT}/results.csv")
    print("=" * 65)

    if len(results):
        print("\nTop 5 datasets by alpha:")
        print(df_res.nlargest(5, "alpha")[
            ["dataset", "alpha", "delta", "linear_probe_difficulty", "n_samples"]
        ].to_string(index=False))
        print("\nBottom 5 datasets by alpha:")
        print(df_res.nsmallest(5, "alpha")[
            ["dataset", "alpha", "delta", "linear_probe_difficulty", "n_samples"]
        ].to_string(index=False))

    return df_res, df_fail


df_results, df_failed = run_experiment()
