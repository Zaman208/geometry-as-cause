#!/usr/bin/env python3
"""
nca_causal_intervention

Interventions & Hypotheses
--------------------------
  label_noise_10   : flip 10% labels  -> alpha DOWN  (label_noise UP)
  label_noise_20   : flip 20% labels  -> alpha DOWN  (label_noise UP more)
  feature_noise_03 : Gauss noise s=.3 -> alpha DOWN  (probe_difficulty UP)
  feature_noise_07 : Gauss noise s=.7 -> alpha DOWN  (probe_difficulty UP more)
  smote_balance    : SMOTE rebalance  -> alpha UP    (class_entropy UP)
  class_imbalance  : subsample minority 10x -> alpha DOWN (class_entropy DOWN)

Statistical Validation (per intervention)
------------------------------------------
  - Paired Wilcoxon signed-rank test  (H0: median(Δα)=0)
  - Sign accuracy  (fraction of datasets with Δα in predicted direction)
  - Spearman rho(ΔFeature, Δα)  -- causal dose-response signal
  - Bootstrap 95% CI on mean Δα  (1000 iterations)
  - Cohen's d effect size
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import log_loss
import openml
warnings.filterwarnings("ignore")

# -- imblearn (SMOTE) ----------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: imblearn not available, smote_balance will be skipped")

# -- Output dir ----------------------------------------------------------------
OUT = "/kaggle/working/causal_out" if os.path.exists("/kaggle") else "./causal_out"
os.makedirs(OUT, exist_ok=True)

# -- Reproducibility -----------------------------------------------------------
MASTER_SEED = 42
RNG = np.random.default_rng(MASTER_SEED)

# -- Alpha estimation settings -------------------------------------------------
MIN_TRAIN    = 40       # minimum training samples per size point
MIN_POINTS   = 4        # minimum valid size points for log-log fit
MIN_R2       = 0.15     # quality gate (slightly lower for modified datasets)
N_SEEDS      = 2        # seeds for MLP stability
BOOTSTRAP_N  = 1000     # bootstrap iterations for CI
FRACS        = np.array([0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00])

# -- Dataset registry (115 datasets from prior run -- all accepted OpenML sets) -
# Selected: n >= 200, n_classes >= 2, clean loading, passed alpha quality gate
DATASETS = {
    # -- Core 50 (original) ----------------------------------------------------
    "blood-transfusion":              1464,
    "banknote-authentication":        1462,
    "bank-marketing":                 1461,
    "eeg-eye-state":                  1471,
    "adult":                          1590,
    "electricity":                    151,
    "magic-telescope":                1120,
    "qsar-biodeg":                    1494,
    "phoneme":                        1489,
    "ozone":                          1487,
    "hill-valley":                    1479,
    "wall-robot-navigation":          1497,
    "cnae-9":                         1468,
    "nomao":                          1486,
    "cardiotocography":               1466,
    "kc2":                            1063,
    "kc1":                            1067,
    "breast-w":                       15,
    "pc1":                            1068,
    "haberman":                       43,
    "ecoli":                          40671,
    "diabetes":                       37,
    "heart-statlog":                  53,
    "ilpd":                           1480,
    "yeast":                          181,
    "dermatology":                    35,
    "liver-disorders":                8,
    "balance-scale":                  11,
    "arrhythmia":                     5,
    "credit-g":                       31,
    "glass":                          41,
    "kr-vs-kp":                       3,
    "nursery":                        26,
    "mushroom":                       24,
    "segment":                        36,
    "soybean":                        42,
    "sonar":                          40,
    "splice":                         46,
    "vehicle":                        54,
    "vowel":                          307,
    "cmc":                            23,
    "credit-approval":                29,
    "letter":                         6,
    "mfeat-factors":                  12,
    "mfeat-fourier":                  14,
    "optdigits":                      28,
    "satimage":                       182,
    "waveform-5000":                  60,
    "pendigits":                      32,
    "spambase":                       44,
    "madelon":                        1485,
    "steel-plates-fault":             1504,
    "hepatitis":                      55,
    "autos":                          9,
    "anneal":                         2,
    "primary-tumor":                  171,
    "mfeat-morphological":            16,
    "mfeat-karhunen":                 13,
    "mfeat-zernike":                  22,
    "mfeat-pixel":                    20,
    "semeion":                        1501,
    "twonorm":                        1507,
    "libras":                         299,
    "ringnorm":                       1496,
    "isolet":                         300,
    "PhishingWebsites":               1019,
    "amazon-commerce-reviews":        1457,
    "kc3":                            1069,
    "pc2":                            1066,
    "mnist":                          554,
    "GesturePhaseSegmentation":       1472,
    "fashion-mnist":                  40996,
    "sick":                           38,
    "jm1":                            1053,
    "spect":                          336,
    "dna":                            40670,
    "musk1":                          1116,
    "kr-vs-kp":                       3,
    "chess-krvkp":                    3,
    "monks-2":                        334,
    "tic-tac-toe":                    50,
    "colic":                          25,
    "page-blocks":                    30,
    "wilt":                           40983,
    "har":                            1478,
    "cpu-small":                      560,
    "musk2":                          1116,
    "analcatdata-dmft":               962,
    "analcatdata-authorship":         458,
    "analcatdata-lawsuit":            299,
    "robot-failures-lp1":             1497,
    "robot-failures-lp2":             1498,
    "robot-failures-lp4":             1500,
    "kin8nm":                         189,
    "cpu-act":                        197,
    "shuttle":                        40685,
    "pol":                            722,
    "ele-1":                          1037,
    "2dplanes":                       727,
    "collins":                        40671,
    "user-knowledge":                 1508,
    "parity5+5":                      1016,
    "monks-1":                        333,
    "kr-vs-k":                        3,
    "census-income":                  4535,
    "connect-4":                      40668,
    "hypothyroid":                    40,
    "sick-euthyroid":                 38,
    "mofn-3-7-10":                    1015,
    "breast-cancer-wisconsin":        15,
    "pc3":                            1050,
    "dresses-sales":                  23381,
    "porto-seguro":                   42742,
}

# Deduplicate by OpenML ID
_seen, _deduped = set(), {}
for k, v in DATASETS.items():
    if v not in _seen:
        _seen.add(v); _deduped[k] = v
DATASETS = _deduped

# -- Preprocessing -------------------------------------------------------------
def preprocess(X_raw, y_raw, max_samples=5000, max_features=60):
    """Standard preprocessing: encode, impute, scale, cap size."""
    df = X_raw.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    df = df.fillna(df.median(numeric_only=True))
    X = df.values.astype(np.float32)

    le_y = LabelEncoder()
    y = le_y.fit_transform(y_raw.astype(str))

    # Remove classes with < 5 samples
    cls, cnt = np.unique(y, return_counts=True)
    valid_cls = cls[cnt >= 5]
    if len(valid_cls) < 2:
        return None, None
    mask = np.isin(y, valid_cls)
    X, y = X[mask], y[mask]
    # Re-encode to 0..K-1
    y = LabelEncoder().fit_transform(y)

    # Cap samples
    if len(y) > max_samples:
        idx = np.random.default_rng(MASTER_SEED).choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]

    # Cap features via variance-based selection
    if X.shape[1] > max_features:
        vars = X.var(axis=0)
        top = np.argsort(vars)[::-1][:max_features]
        X = X[:, top]

    # Scale
    scaler = RobustScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, y


# -- Feature extraction --------------------------------------------------------
def extract_features(X, y):
    """Extract the features predictive of alpha (from IEEE paper)."""
    feats = {}
    n, d = X.shape
    classes = np.unique(y)
    K = len(classes)

    # Linear probe difficulty
    lr = LogisticRegression(max_iter=300, random_state=MASTER_SEED, C=1.0)
    try:
        cv_scores = cross_val_score(lr, X, y, cv=min(5, K, 10),
                                    scoring="neg_log_loss",
                                    error_score=np.nan)
        probe_err = float(np.nanmean(-cv_scores))
    except Exception:
        probe_err = np.nan
    feats["probe_difficulty"] = probe_err

    # Class entropy
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    feats["class_entropy"] = float(-np.sum(probs * np.log(probs + 1e-10)))

    # Label noise (neighbor disagreement)
    try:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        neighbor_labels = knn.kneighbors(X, n_neighbors=2, return_distance=False)[:, 1]
        feats["label_noise"] = float(np.mean(y != knn._y[neighbor_labels]))
    except Exception:
        feats["label_noise"] = np.nan

    # Fisher Ratio
    try:
        grand_mean = X.mean(axis=0)
        sb = sum(np.sum(y == c) * np.outer(X[y == c].mean(axis=0) - grand_mean,
                                            X[y == c].mean(axis=0) - grand_mean)
                 for c in classes)
        sw = sum(np.cov(X[y == c].T, bias=True) * np.sum(y == c)
                 if np.sum(y == c) > 1 else np.zeros((d, d))
                 for c in classes)
        tr_sb = float(np.trace(sb))
        tr_sw = float(np.trace(sw))
        feats["fisher_ratio"] = tr_sb / (tr_sw + 1e-10)
    except Exception:
        feats["fisher_ratio"] = np.nan

    # Class imbalance ratio
    feats["class_imbalance"] = float(counts.max() / (counts.min() + 1))
    feats["n_samples"]       = int(n)
    feats["n_features"]      = int(d)
    feats["n_classes"]       = int(K)

    return feats


# -- Alpha estimation ----------------------------------------------------------
def estimate_alpha(X, y, verbose=False):
    """
    Estimate scaling exponent alpha from MLP learning curves.
    Returns (alpha, r2, ci_low, ci_high) or (nan, nan, nan, nan) if failed.
    """
    n = len(y)
    K = len(np.unique(y))
    if n < 80 or K < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Build size grid
    sizes = np.unique(np.maximum(MIN_TRAIN,
                                  (FRACS * int(0.8 * n)).astype(int)))
    sizes = sizes[sizes <= int(0.8 * n)]
    if len(sizes) < MIN_POINTS:
        return np.nan, np.nan, np.nan, np.nan

    # Fixed test set (20%)
    rng_split = np.random.default_rng(MASTER_SEED)
    all_idx = np.arange(n)
    test_n = max(20, int(0.2 * n))
    # Stratified split
    from sklearn.model_selection import train_test_split
    try:
        tr_idx, te_idx = train_test_split(all_idx, test_size=test_n,
                                           random_state=MASTER_SEED,
                                           stratify=y)
    except Exception:
        tr_idx, te_idx = train_test_split(all_idx, test_size=test_n,
                                           random_state=MASTER_SEED)
    X_te, y_te = X[te_idx], y[te_idx]
    X_tr, y_tr = X[tr_idx], y[tr_idx]

    losses = []
    valid_sizes = []

    for sz in sizes:
        if sz > len(tr_idx):
            sz = len(tr_idx)
        seed_losses = []
        for seed in range(N_SEEDS):
            rng_s = np.random.default_rng(seed + 100)
            # Stratified subsample
            try:
                sub_idx, _ = train_test_split(np.arange(len(tr_idx)),
                                               train_size=sz,
                                               random_state=seed,
                                               stratify=y_tr)
            except Exception:
                sub_idx = rng_s.choice(len(tr_idx), sz, replace=False)
            Xs, ys = X_tr[sub_idx], y_tr[sub_idx]

            # Skip if any class has < 2 samples
            _, cnts = np.unique(ys, return_counts=True)
            if cnts.min() < 2:
                continue

            mlp = MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                max_iter=80,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.15,
                learning_rate_init=1e-3,
            )
            try:
                mlp.fit(Xs, ys)
                proba = mlp.predict_proba(X_te)
                # Add small epsilon to avoid log(0)
                proba = np.clip(proba, 1e-7, 1 - 1e-7)
                loss = log_loss(y_te, proba, labels=np.unique(y))
                if np.isfinite(loss):
                    seed_losses.append(loss)
            except Exception:
                continue

        if len(seed_losses) >= 1:
            losses.append(np.mean(seed_losses))
            valid_sizes.append(sz)

    if len(valid_sizes) < MIN_POINTS:
        return np.nan, np.nan, np.nan, np.nan

    # Log-log regression
    log_D = np.log(valid_sizes)
    log_L = np.log(np.array(losses))
    slope, intercept, r, p, se = stats.linregress(log_D, log_L)
    alpha = -slope
    r2 = r ** 2

    if r2 < MIN_R2 or not np.isfinite(alpha):
        return np.nan, r2, np.nan, np.nan

    # Bootstrap CI
    boot_alphas = []
    rng_b = np.random.default_rng(MASTER_SEED + 1)
    for _ in range(BOOTSTRAP_N):
        idx_b = rng_b.integers(0, len(valid_sizes), len(valid_sizes))
        if len(np.unique(idx_b)) < 3:
            continue
        s_b, *_ = stats.linregress(log_D[idx_b], log_L[idx_b])
        if np.isfinite(-s_b):
            boot_alphas.append(-s_b)
    ci_low  = float(np.percentile(boot_alphas, 2.5))  if boot_alphas else np.nan
    ci_high = float(np.percentile(boot_alphas, 97.5)) if boot_alphas else np.nan

    if verbose:
        print(f"    alpha={alpha:.4f}  R2={r2:.3f}  CI=[{ci_low:.3f},{ci_high:.3f}]"
              f"  n_pts={len(valid_sizes)}")

    return float(alpha), float(r2), ci_low, ci_high


# -- Intervention functions ----------------------------------------------------
def iv_label_noise(X, y, rate, seed=42):
    """Flip `rate` fraction of labels to a random different class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    y_new = y.copy()
    n_flip = max(1, int(len(y) * rate))
    flip_idx = rng.choice(len(y), n_flip, replace=False)
    for i in flip_idx:
        others = classes[classes != y[i]]
        if len(others) > 0:
            y_new[i] = rng.choice(others)
    return X.copy(), y_new


def iv_feature_noise(X, y, sigma_scale, seed=42):
    """Add Gaussian noise N(0, sigma_scale * feature_std) to each feature."""
    rng = np.random.default_rng(seed)
    stds = X.std(axis=0) + 1e-8
    noise = rng.normal(0, sigma_scale, X.shape).astype(np.float32) * stds
    return (X + noise).astype(np.float32), y.copy()


def iv_smote_balance(X, y, seed=42):
    """SMOTE oversample to fully balance all classes."""
    if not HAS_SMOTE:
        return None, None
    K = len(np.unique(y))
    # Need at least 6 samples per class for SMOTE k=5
    cls, cnts = np.unique(y, return_counts=True)
    if cnts.min() < 6 or K < 2:
        return None, None
    try:
        sm = SMOTE(random_state=seed, k_neighbors=min(5, cnts.min() - 1))
        X_bal, y_bal = sm.fit_resample(X, y)
        return X_bal.astype(np.float32), y_bal
    except Exception:
        return None, None


def iv_class_imbalance(X, y, imbalance_factor=10, seed=42):
    """
    Subsample minority classes to 1/imbalance_factor of majority size.
    Creates severe class imbalance, reducing class entropy.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None, None
    dominant_cls = classes[np.argmax(counts)]
    dominant_n   = counts.max()
    target_minority = max(10, dominant_n // imbalance_factor)

    keep_idx = list(np.where(y == dominant_cls)[0])
    for c in classes:
        if c == dominant_cls:
            continue
        c_idx = np.where(y == c)[0]
        n_keep = min(target_minority, len(c_idx))
        if n_keep < 5:
            continue
        keep_idx.extend(rng.choice(c_idx, n_keep, replace=False).tolist())

    if len(keep_idx) < 80:
        return None, None
    keep_idx = np.array(keep_idx)
    # Check at least 2 classes remain
    if len(np.unique(y[keep_idx])) < 2:
        return None, None
    return X[keep_idx].copy(), y[keep_idx].copy()


# -- Statistical analysis helpers ----------------------------------------------
def cohens_d(a, b=None):
    """Cohen's d for paired differences (a = Δα array) or two-sample."""
    if b is None:
        # one-sample vs 0
        return float(np.mean(a) / (np.std(a, ddof=1) + 1e-10))
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-10))


def bootstrap_mean_ci(arr, n_boot=1000, seed=42):
    """Bootstrap 95% CI for mean of arr."""
    rng = np.random.default_rng(seed)
    arr = np.array(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return np.nan, np.nan
    boot = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_statistical_tests(delta_alpha, delta_feature, predicted_direction,
                          intervention_name):
    """
    Full statistical battery for one intervention.

    Parameters
    ----------
    delta_alpha       : array of (alpha_modified - alpha_original)
    delta_feature     : array of matching feature changes
    predicted_direction : +1 (expect alpha UP) or -1 (expect alpha DOWN)
    intervention_name : string label

    Returns dict with all test results.
    """
    da = np.array(delta_alpha,   dtype=float)
    df = np.array(delta_feature, dtype=float)

    # Remove NaN pairs
    mask = np.isfinite(da) & np.isfinite(df)
    da, df = da[mask], df[mask]
    n = len(da)

    result = {"intervention": intervention_name, "n_datasets": n}

    if n < 5:
        result["insufficient_data"] = True
        return result

    # 1. Mean and std of Δα
    result["mean_delta_alpha"] = float(np.mean(da))
    result["std_delta_alpha"]  = float(np.std(da, ddof=1))
    result["median_delta_alpha"] = float(np.median(da))

    # 2. Paired Wilcoxon signed-rank test (H0: median(Δα) = 0)
    try:
        stat_w, p_w = wilcoxon(da, alternative="less" if predicted_direction == -1
                                                       else "greater")
        result["wilcoxon_statistic"] = float(stat_w)
        result["wilcoxon_p"]         = float(p_w)
    except Exception as e:
        result["wilcoxon_p"] = np.nan

    # 3. One-sample t-test (parametric complement)
    t_stat, p_t = stats.ttest_1samp(da, 0,
                                     alternative="less" if predicted_direction == -1
                                                        else "greater")
    result["ttest_p"] = float(p_t)

    # 4. Sign accuracy: fraction matching predicted direction
    if predicted_direction == -1:
        sign_correct = np.sum(da < 0)
    else:
        sign_correct = np.sum(da > 0)
    result["sign_accuracy"]      = float(sign_correct / n)
    result["sign_correct_n"]     = int(sign_correct)
    result["sign_p_binomial"]    = float(stats.binomtest(sign_correct, n, 0.5, alternative="greater").pvalue)

    # 5. Spearman rho(ΔFeature, Δα) -- dose-response causal signal
    try:
        rho, p_rho = spearmanr(df, da)
        result["spearman_rho_feature_alpha"] = float(rho)
        result["spearman_p"]                 = float(p_rho)
    except Exception:
        result["spearman_rho_feature_alpha"] = np.nan
        result["spearman_p"]                 = np.nan

    # 6. Bootstrap 95% CI on mean Δα
    ci_lo, ci_hi = bootstrap_mean_ci(da, n_boot=BOOTSTRAP_N)
    result["bootstrap_ci_low"]  = ci_lo
    result["bootstrap_ci_high"] = ci_hi

    # 7. Cohen's d (effect size)
    result["cohens_d"] = cohens_d(da)

    # 8. Summary verdict
    p_threshold = 0.05
    strong = (result.get("wilcoxon_p", 1) < p_threshold and
              result.get("sign_accuracy", 0) >= 0.60)
    result["verdict"] = "SUPPORTED" if strong else (
        "PARTIAL" if result.get("sign_accuracy", 0) >= 0.55 else "NOT SUPPORTED"
    )

    return result


# -- Main experiment -----------------------------------------------------------
INTERVENTIONS = [
    # (name, function, kwargs, predicted_direction, tracking_feature)
    ("label_noise_10",   iv_label_noise,     {"rate": 0.10}, -1, "label_noise"),
    ("label_noise_20",   iv_label_noise,     {"rate": 0.20}, -1, "label_noise"),
    ("feature_noise_03", iv_feature_noise,   {"sigma_scale": 0.30}, -1, "probe_difficulty"),
    ("feature_noise_07", iv_feature_noise,   {"sigma_scale": 0.70}, -1, "probe_difficulty"),
    ("smote_balance",    iv_smote_balance,   {}, +1, "class_entropy"),
    ("class_imbalance",  iv_class_imbalance, {"imbalance_factor": 10}, -1, "class_entropy"),
]


def run_experiment():
    rows         = []   # per-dataset × intervention
    stat_inputs  = {iv[0]: {"delta_alpha": [], "delta_feature": [],
                             "direction": iv[3], "feature_key": iv[4]}
                    for iv in INTERVENTIONS}

    ds_names = list(DATASETS.keys())
    n_total  = len(ds_names)

    print(f"\n{'='*60}")
    print(f"Causal Intervention Experiment")
    print(f"Datasets: {n_total}   Interventions: {len(INTERVENTIONS)}")
    print(f"{'='*60}\n")

    accepted_baseline = 0
    t_start = time.time()

    for ds_i, ds_name in enumerate(ds_names):
        ds_id = DATASETS[ds_name]
        t0 = time.time()
        print(f"[{ds_i+1:3d}/{n_total}]  {ds_name}  (openml={ds_id})")

        # -- Load --------------------------------------------------------------
        try:
            ds = openml.datasets.get_dataset(ds_id,
                                              download_data=True,
                                              download_qualities=False,
                                              download_features_meta_data=False)
            X_raw, y_raw, _, _ = ds.get_data(
                dataset_format="dataframe",
                target=ds.default_target_attribute)
            if y_raw is None or len(y_raw) < 100:
                print("  skip: too small")
                continue
        except Exception as e:
            print(f"  skip: load error -- {e}")
            continue

        # -- Preprocess --------------------------------------------------------
        X, y = preprocess(X_raw, y_raw)
        if X is None or len(np.unique(y)) < 2 or len(y) < 100:
            print("  skip: preprocess failed")
            continue

        print(f"  shape={X.shape}  classes={len(np.unique(y))}")

        # -- Baseline alpha ----------------------------------------------------
        alpha_orig, r2_orig, ci_lo_orig, ci_hi_orig = estimate_alpha(X, y, verbose=True)
        if not np.isfinite(alpha_orig):
            print("  skip: baseline alpha failed")
            continue
        accepted_baseline += 1

        # -- Baseline features -------------------------------------------------
        feats_orig = extract_features(X, y)

        # -- Run each intervention ---------------------------------------------
        for iv_name, iv_fn, iv_kwargs, direction, track_feat in INTERVENTIONS:
            # Apply intervention
            try:
                X_mod, y_mod = iv_fn(X, y, **iv_kwargs)
            except Exception as e:
                print(f"    [{iv_name}] intervention error: {e}")
                continue

            if X_mod is None or y_mod is None:
                continue
            if len(np.unique(y_mod)) < 2 or len(y_mod) < 60:
                continue

            # Estimate modified alpha
            alpha_mod, r2_mod, ci_lo_mod, ci_hi_mod = estimate_alpha(X_mod, y_mod)
            if not np.isfinite(alpha_mod):
                continue

            # Extract modified features
            feats_mod = extract_features(X_mod, y_mod)

            # Compute deltas
            delta_alpha   = alpha_mod - alpha_orig
            feat_orig_val = feats_orig.get(track_feat, np.nan)
            feat_mod_val  = feats_mod.get(track_feat, np.nan)
            delta_feature = feat_mod_val - feat_orig_val

            direction_correct = (delta_alpha < 0) if direction == -1 else (delta_alpha > 0)

            print(f"    [{iv_name}]  alpha: {alpha_orig:.4f} -> {alpha_mod:.4f}"
                  f"  Δα={delta_alpha:+.4f}  Δfeat={delta_feature:+.4f}"
                  f"  {'✓' if direction_correct else '✗'}")

            # Store row
            rows.append({
                "dataset":          ds_name,
                "openml_id":        ds_id,
                "n_samples_orig":   len(y),
                "n_samples_mod":    len(y_mod),
                "n_classes":        len(np.unique(y)),
                "intervention":     iv_name,
                "alpha_orig":       alpha_orig,
                "r2_orig":          r2_orig,
                "alpha_mod":        alpha_mod,
                "r2_mod":           r2_mod,
                "delta_alpha":      delta_alpha,
                "direction_pred":   direction,
                "direction_correct": int(direction_correct),
                "feature_key":      track_feat,
                "feature_orig":     feat_orig_val,
                "feature_mod":      feat_mod_val,
                "delta_feature":    delta_feature,
                **{f"feat_orig_{k}": v for k, v in feats_orig.items()},
                **{f"feat_mod_{k}":  v for k, v in feats_mod.items()},
            })

            # Accumulate for statistical tests
            stat_inputs[iv_name]["delta_alpha"].append(delta_alpha)
            stat_inputs[iv_name]["delta_feature"].append(delta_feature)

        elapsed = time.time() - t0
        print(f"  done ({elapsed:.1f}s)\n")

    # -- Save raw results ------------------------------------------------------
    df_results = pd.DataFrame(rows)
    df_results.to_csv(f"{OUT}/intervention_results.csv", index=False)
    print(f"\nSaved {len(df_results)} rows to intervention_results.csv")

    # -- Statistical tests -----------------------------------------------------
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    test_rows = []
    for iv_name, data in stat_inputs.items():
        result = run_statistical_tests(
            delta_alpha=data["delta_alpha"],
            delta_feature=data["delta_feature"],
            predicted_direction=data["direction"],
            intervention_name=iv_name,
        )
        test_rows.append(result)

        print(f"\n[{iv_name}]  n={result.get('n_datasets',0)}")
        print(f"  mean Δα = {result.get('mean_delta_alpha', np.nan):.4f}"
              f"  ({result.get('bootstrap_ci_low', np.nan):.4f},"
              f"  {result.get('bootstrap_ci_high', np.nan):.4f})")
        print(f"  Wilcoxon p = {result.get('wilcoxon_p', np.nan):.4f}")
        print(f"  Sign accuracy = {result.get('sign_accuracy', np.nan):.3f}"
              f"  ({result.get('sign_correct_n',0)}/{result.get('n_datasets',0)})")
        print(f"  Spearman rho = {result.get('spearman_rho_feature_alpha', np.nan):.3f}"
              f"  p={result.get('spearman_p', np.nan):.4f}")
        print(f"  Cohen's d = {result.get('cohens_d', np.nan):.3f}")
        print(f"  Verdict: {result.get('verdict','?')}")

    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(f"{OUT}/causal_tests.csv", index=False)

    # -- Feature delta correlations detail -------------------------------------
    if len(df_results) > 0:
        feat_rows = []
        for iv_name in df_results["intervention"].unique():
            sub = df_results[df_results["intervention"] == iv_name].copy()
            sub = sub.dropna(subset=["delta_alpha", "delta_feature"])
            if len(sub) >= 5:
                rho, p = spearmanr(sub["delta_feature"], sub["delta_alpha"])
                feat_rows.append({
                    "intervention": iv_name,
                    "n": len(sub),
                    "spearman_rho": round(rho, 4),
                         "p_value": round(p, 6),
                    "mean_delta_alpha": round(sub["delta_alpha"].mean(), 5),
                    "mean_delta_feature": round(sub["delta_feature"].mean(), 5),
                })
        df_feat = pd.DataFrame(feat_rows)
        df_feat.to_csv(f"{OUT}/feature_deltas.csv", index=False)
        print(f"\nFeature delta correlations saved.")

    # -- Summary JSON ----------------------------------------------------------
    total_time = time.time() - t_start
    summary = {
        "n_datasets_attempted": n_total,
        "n_datasets_accepted_baseline": accepted_baseline,
        "n_total_observations": len(df_results),
        "interventions": {},
        "total_time_sec": round(total_time, 1),
    }
    for row in test_rows:
        iv = row["intervention"]
        summary["interventions"][iv] = {
            "n": row.get("n_datasets", 0),
            "mean_delta_alpha": round(row.get("mean_delta_alpha", float("nan")), 5),
            "wilcoxon_p": round(row.get("wilcoxon_p", float("nan")), 6),
            "sign_accuracy": round(row.get("sign_accuracy", float("nan")), 4),
            "spearman_rho": round(row.get("spearman_rho_feature_alpha", float("nan")), 4),
            "cohens_d": round(row.get("cohens_d", float("nan")), 4),
            "bootstrap_ci": [round(row.get("bootstrap_ci_low", float("nan")), 4),
                              round(row.get("bootstrap_ci_high", float("nan")), 4)],
            "verdict": row.get("verdict", "?"),
        }

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"  Total time     : {total_time/60:.1f} min")
    print(f"  Accepted DS    : {accepted_baseline}/{n_total}")
    print(f"  Observations   : {len(df_results)}")
    print(f"  Files saved to : {OUT}")
    print(f"{'='*60}")

    print("\nSUMMARY TABLE")
    print(f"{'Intervention':<22} {'n':>4} {'meanΔα':>8} {'Wilcox-p':>10}"
          f" {'SignAcc':>8} {'SpRho':>7} {'Verdict'}")
    print("-" * 80)
    for row in test_rows:
        print(f"{row['intervention']:<22}"
              f" {row.get('n_datasets',0):>4}"
              f" {row.get('mean_delta_alpha', float('nan')):>+8.4f}"
              f" {row.get('wilcoxon_p', float('nan')):>10.4f}"
              f" {row.get('sign_accuracy', float('nan')):>8.3f}"
              f" {row.get('spearman_rho_feature_alpha', float('nan')):>7.3f}"
              f"  {row.get('verdict','?')}")

    return df_results, df_tests


# -- Entry point (runs directly as Kaggle notebook cell) ---------------------
try:
    import imblearn
    HAS_SMOTE = True
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "imbalanced-learn", "-q"])
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True

df_results, df_tests = run_experiment()
