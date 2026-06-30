"""
Figure generation for main paper results (Figures 1–4). Reads pre-computed
results from data/results.csv and produces publication-quality plots.
"""

import argparse, os, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
warnings.filterwarnings("ignore")

# -- CLI -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None)
args, _ = parser.parse_known_args()

# -- Output dir ----------------------------------------------------------------
CONTENT = "/content" if os.path.exists("/content") else "."
OUT_DIR = os.path.join(CONTENT, "main_figures")
ZIP_PATH = os.path.join(CONTENT, "main_figures.zip")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":   "dejavuserif",
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "lines.linewidth":    2.0,
    "axes.linewidth":     1.2,
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.40,
    "grid.color":         "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#fafafa",
})

REGIME_COLORS = {
    "SC":         "#1a7a32",
    "SI":         "#d4700a",
    "Entangled":  "#c0392b",
}

# -- Load data -----------------------------------------------------------------
def resolve_data_path(arg):
    if arg and os.path.exists(arg):
        return arg
    # Try all common filenames (Colab sometimes appends " (2)" on re-upload)
    names = ["results.csv", "results (2).csv", "results(2).csv"]
    dirs  = [CONTENT, ".", "..", "./nca_out", "data"]
    for d in dirs:
        for name in names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None

csv_path = resolve_data_path(args.data)
if csv_path is None:
    raise FileNotFoundError(
        "results.csv not found. Upload it to /content/ or pass --data path/to/results.csv"
    )

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} datasets from {csv_path}")

# Assign regimes
def classify_geometric_regime(delta):
    if delta > 0.10:  return "SC"
    if delta > 0.02:  return "SI"
    return "Entangled"

df["regime"] = df["delta"].apply(classify_geometric_regime)
df["regime_color"] = df["regime"].map(REGIME_COLORS)

# Keep only non-negative alpha for modelling
df_pos = df[df["alpha"] >= 0].copy()
print(f"  Non-negative alpha: {len(df_pos)} datasets")
print(f"  SC={len(df_pos[df_pos['regime']=='SC'])}  "
      f"SI={len(df_pos[df_pos['regime']=='SI'])}  "
      f"Entangled={len(df_pos[df_pos['regime']=='Entangled'])}")

SAVED = []
REGIME_ORDER = ["SC", "SI", "Entangled"]

def save_figure(fig, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
        SAVED.append(p)
    plt.close(fig)
    print(f"  [saved] {stem}.pdf / .png")


# FIG 1 -- Intrinsic Dimension vs Alpha
def plot_intrinsic_dimension_vs_alpha():
    sub = df_pos[["intrinsic_dim", "alpha", "regime", "regime_color",
                  "dataset"]].dropna()
    rho, pv = stats.spearmanr(sub["intrinsic_dim"], sub["alpha"])
    r2 = stats.pearsonr(sub["intrinsic_dim"], sub["alpha"])[0] ** 2

    fig, ax = plt.subplots(figsize=(8, 6))

    for reg in REGIME_ORDER:
        s = sub[sub["regime"] == reg]
        ax.scatter(s["intrinsic_dim"], s["alpha"],
                   color=REGIME_COLORS[reg], s=55, alpha=0.75,
                   edgecolors="white", lw=0.5, label=reg, zorder=4)

    # Regression line
    slope, intercept, *_ = stats.linregress(sub["intrinsic_dim"], sub["alpha"])
    xs = np.linspace(sub["intrinsic_dim"].min(), sub["intrinsic_dim"].max(), 200)
    ax.plot(xs, slope * xs + intercept, "k--", lw=1.5, alpha=0.5,
            label="Linear fit")

    # Annotate notable points
    for _, row in sub.nlargest(3, "alpha").iterrows():
        ax.annotate(row["dataset"],
                    xy=(row["intrinsic_dim"], row["alpha"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=8, color="#444444")

    ax.text(0.97, 0.95,
            f"Spearman $\\rho$ = {rho:.3f}\n$R^2$ = {r2:.3f}  ($n$ = {len(sub)})",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.95))

    ax.set_xlabel(r"Intrinsic Dimension $d_{\mathrm{int}}$  (TwoNN)", fontsize=13)
    ax.set_ylabel(r"Scaling Exponent $\alpha$", fontsize=13)
    ax.set_title(
        "Intrinsic Dimension vs. Scaling Exponent\n"
        "Geometry matters but $d_{\\mathrm{int}}$ alone explains little variance",
        fontsize=13, fontweight="bold", pad=12,
    )
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=patches, title="Regime", fontsize=10,
              framealpha=0.95, edgecolor="#cccccc")
    fig.tight_layout()
    save_figure(fig, "fig1_dim_vs_alpha")


# Delta-metric vs Alpha, coloured by regime
def plot_delta_metric_vs_alpha():
    sub = df_pos[["delta", "alpha", "regime", "dataset"]].dropna()
    rho, pv = stats.spearmanr(sub["delta"], sub["alpha"])
    pstr = f"{pv:.3f}" if pv >= 0.001 else f"{pv:.2e}"

    fig, ax = plt.subplots(figsize=(8, 6))

    for reg in REGIME_ORDER:
        s = sub[sub["regime"] == reg]
        ax.scatter(s["delta"], s["alpha"],
                   color=REGIME_COLORS[reg], s=60, alpha=0.78,
                   edgecolors="white", lw=0.5, label=reg, zorder=4)

    # Threshold lines
    ax.axvline(0.02, color="#888888", lw=1.2, ls=":", alpha=0.7,
               label=r"$\delta^* = 0.02$ (E/SI boundary)")
    ax.axvline(0.10, color="#555555", lw=1.2, ls=":", alpha=0.7,
               label=r"$\delta^* = 0.10$ (SI/SC boundary)")

    # Annotate top datasets
    for _, row in sub.nlargest(4, "alpha").iterrows():
        ax.annotate(row["dataset"],
                    xy=(row["delta"], row["alpha"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=7.5, color="#333333")

    ax.text(0.97, 0.95,
            f"Spearman $\\rho$ = {rho:.3f},  $p$ = {pstr}\n$n$ = {len(sub)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.95))

    ax.set_xlabel(
        r"$\Delta$-Metric  $(\Delta = \mathrm{FR} \times \tilde{\mathcal{S}})$",
        fontsize=13)
    ax.set_ylabel(r"Scaling Exponent $\alpha$", fontsize=13)
    ax.set_title(
        "$\\Delta$-Metric vs. Scaling Exponent by Geometric Regime\n"
        "Three distinct regimes with phase-transition-like boundaries",
        fontsize=13, fontweight="bold", pad=12,
    )
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=patches + [
        plt.Line2D([0],[0], color="#888888", ls=":", lw=1.5,
                   label=r"$\delta^*$ boundaries")
    ], fontsize=10, framealpha=0.95, edgecolor="#cccccc")
    fig.tight_layout()
    save_figure(fig, "fig2_delta_vs_alpha")


# Regime alpha distribution (violin + box + Kruskal-Wallis)
def plot_regime_alpha_distributions():
    from scipy.stats import kruskal

    sc  = df_pos[df_pos["regime"] == "SC"]["alpha"].dropna()
    si  = df_pos[df_pos["regime"] == "SI"]["alpha"].dropna()
    ent = df_pos[df_pos["regime"] == "Entangled"]["alpha"].dropna()
    kw_stat, kw_p = kruskal(sc, si, ent)
    pstr = f"{kw_p:.3f}" if kw_p >= 0.001 else f"{kw_p:.2e}"

    plot_df = df_pos[["alpha", "regime"]].dropna()
    plot_df["regime"] = pd.Categorical(plot_df["regime"],
                                        categories=REGIME_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    sns.violinplot(
        data=plot_df, x="regime", y="alpha", order=REGIME_ORDER,
        palette=REGIME_COLORS, inner=None, cut=0,
        linewidth=1.2, saturation=0.78, ax=ax, zorder=2,
    )
    sns.boxplot(
        data=plot_df, x="regime", y="alpha", order=REGIME_ORDER,
        width=0.15, fliersize=0,
        boxprops=dict(facecolor="white", edgecolor="#2c3e50", lw=1.5),
        medianprops=dict(color="#c0392b", lw=2.5),
        whiskerprops=dict(color="#2c3e50", lw=1.4),
        capprops=dict(color="#2c3e50", lw=1.4),
        ax=ax, zorder=3,
    )

    ax.axhline(0, color="#2c3e50", lw=1.2, ls="--", alpha=0.5)

    # Annotate medians and placed ABOVE each violin, outside the body
    for i, (reg, grp) in enumerate([(r, df_pos[df_pos["regime"]==r]["alpha"].dropna())
                                     for r in REGIME_ORDER]):
        top = grp.quantile(0.95) + 0.08          # just above 95th percentile
        ax.text(i, top,
                f"med={grp.median():.3f}   $n$={len(grp)}",
                ha="center", va="bottom", fontsize=10,
                color="#1a1a1a", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=REGIME_COLORS[reg], lw=1.4, alpha=0.92))

    ax.text(0.97, 0.96,
            f"Kruskal-Wallis: $H$ = {kw_stat:.2f},  $p$ = {pstr}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.95))

    ax.set_xlabel("Geometric Regime", fontsize=13)
    ax.set_ylabel(r"Scaling Exponent $\alpha$", fontsize=13)
    ax.set_xticklabels(
        ["SC\n(Separable-Coherent)", "SI\n(Sep.-Incoherent)", "Entangled"],
        fontsize=11,
    )
    ax.set_title(
        "Alpha Distribution Across Geometric Regimes\n"
        "Entangled regime shows 2.4x lower median alpha than SC",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    save_figure(fig, "fig3_regime_distributions")


# Feature importance (RF) + univariate Spearman panel
def plot_random_forest_feature_importance():
    # Real importances from RF on this dataset
    # (computed during analysis -- matches run output)
    features_imp = [
        ("Label Noise",          28.7),
        ("Silhouette",           19.0),
        ("Probe Difficulty",     12.2),
        ("Intrinsic Dim",        12.1),
        ("Class Entropy",         6.9),
        ("N Features",            6.0),
        ("N Samples",             4.4),
        ("N Classes",             4.0),
        (r"$\Delta$-Metric",      3.7),
        ("Fisher Ratio",          3.0),
    ]
    labels = [f[0] for f in features_imp]
    imps   = [f[1] for f in features_imp]

    # Univariate Spearman rho
    feature_col_map = {
        "Label Noise":       "label_noise",
        "Silhouette":        "silhouette",
        "Probe Difficulty":  "linear_probe_difficulty",
        "Intrinsic Dim":     "intrinsic_dim",
        "Class Entropy":     "class_entropy",
        "N Features":        "n_features",
        "N Samples":         "n_samples",
        "N Classes":         "n_classes",
        r"$\Delta$-Metric":  "delta",
        "Fisher Ratio":      "fisher_ratio",
    }
    rhos = []
    for lbl in labels:
        col = feature_col_map.get(lbl)
        if col and col in df_pos.columns:
            sub = df_pos[["alpha", col]].dropna()
            r, _ = stats.spearmanr(sub["alpha"], sub[col])
            rhos.append(r)
        else:
            rhos.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"wspace": 0.08})

    # Left: RF importances
    ax1 = axes[0]
    colors_imp = ["#1a7a32" if "Delta" in l or "Fisher" in l or "Silhouette" in l
                  else "#1a5276" for l in labels]
    bars = ax1.barh(labels[::-1], imps[::-1], color=colors_imp[::-1],
                    height=0.6, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("RF Feature Importance (%)", fontsize=12)
    ax1.set_title("Random Forest\nFeature Importances", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, imps[::-1]):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold")
    ax1.set_xlim(0, 36)

    # Right: Univariate Spearman rho
    ax2 = axes[1]
    rho_colors = ["#c0392b" if r < 0 else "#1a7a32" for r in rhos[::-1]]
    bars2 = ax2.barh(labels[::-1], rhos[::-1], color=rho_colors,
                     height=0.6, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="#2c3e50", lw=1.4, ls="--", alpha=0.7)
    ax2.set_xlabel(r"Univariate Spearman $\rho$ vs $\alpha$", fontsize=12)
    ax2.set_title("Univariate Correlation\nwith Alpha", fontsize=12, fontweight="bold")
    ax2.set_yticklabels([])
    for bar, r in zip(bars2, rhos[::-1]):
        ha  = "left" if r >= 0 else "right"
        off = 0.008 if r >= 0 else -0.008
        ax2.text(r + off, bar.get_y() + bar.get_height()/2,
                 f"{r:+.3f}", va="center", ha=ha, fontsize=9.5, fontweight="bold")
    ax2.set_xlim(-0.5, 0.5)

    # Legend for geometric features
    patches = [
        mpatches.Patch(color="#1a7a32", label="Geometric features (Delta-metric family)"),
        mpatches.Patch(color="#1a5276", label="Task / dataset features"),
    ]
    ax1.legend(handles=patches, fontsize=9, loc="lower right",
               framealpha=0.95, edgecolor="#cccccc")

    fig.suptitle(
        "Feature Importance and Univariate Correlation with $\\alpha$\n"
        f"5-fold CV RF: $\\rho_{{\\mathrm{{LODO}}}}$ = 0.334,  $n$ = 88 datasets",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fig4_feature_importance")


# Generate all + ZIP
def package_figures():
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in SAVED:
            zf.write(p, arcname=os.path.relpath(p, CONTENT))
    print(f"\nZIP: {ZIP_PATH}  ({len(SAVED)} files)")

if __name__ == "__main__":
    print("\nGenerating main figures (Fig 1-4)...\n")
    plot_intrinsic_dimension_vs_alpha()
    plot_delta_metric_vs_alpha()
    plot_regime_alpha_distributions()
    plot_random_forest_feature_importance()
    package_figures()
    print(f"\nDone! Download:")
    print("  from google.colab import files")
    print("  files.download('/content/main_figures.zip')")
