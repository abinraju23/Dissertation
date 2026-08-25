"""
What it does
    - Uses the TRAINING period only (safest for exploration; the frozen test
      set is never opened for this).
    - Draws box plot grids: all price features, all sentiment features, and
      the six most important graph dimensions (from Task 1's output if it
      exists, otherwise the first six).
    - Writes a class-separation table (Cohen's d for every feature) -- the
    effect-size summaries for the class distributions.

Notes
    - Outliers are hidden in the plots (showfliers=False) so the boxes stay
      readable; full distributions are in the CSV.
        - Set DOWNSAMPLE below to reduce plotting time if required. Statistics
            always use all training rows.

Run from the project root:   python scripts\diag_box_plots.py
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diag_common import (
    DIAG, FIGURES, RANDOM_SEED, TRAIN_END,
    detect_feature_groups, ensure_dirs, load_features,
)

DOWNSAMPLE = None      # e.g. 5000 to plot a random subset of train rows
N_GRAPH_DIMS = 6       # how many graph/embedding dims to plot individually


def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.nan
    s0, s1 = np.std(x0, ddof=1), np.std(x1, ddof=1)
    pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x0)) / pooled


def grid_boxplot(df, cols, label_col, d_lookup, fname, title, ncols=4):
    n = len(cols)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows), squeeze=False)
    for i, c in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        d0 = df.loc[df[label_col] == 0, c].dropna()
        d1 = df.loc[df[label_col] == 1, c].dropna()
        ax.boxplot([d0, d1], showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Down (0)", "Up (1)"], fontsize=8)
        d = d_lookup.get(c, np.nan)
        ax.set_title(f"{c}\nd = {d:+.3f}", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[boxplots] wrote {out}")


def main():
    ensure_dirs()
    df, colmap = load_features()
    groups = detect_feature_groups(df, colmap)
    date_col, label_col = colmap["date"], colmap["label"]

    train = df[df[date_col] <= pd.Timestamp(TRAIN_END)].copy()
    print(f"[boxplots] training rows: {len(train)} "
          f"(class balance: up = {train[label_col].mean():.3f})")

    all_feats = groups["price"] + groups["sentiment"] + groups["graph"]

    # ---- class-separation table on ALL train rows ----
    y = train[label_col].astype(int).to_numpy()
    records = []
    for c in all_feats:
        x = train[c].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        d = cohens_d(x[ok & (y == 0)], x[ok & (y == 1)])
        records.append({
            "feature": c,
            "group": ("price" if c in groups["price"]
                      else "sentiment" if c in groups["sentiment"] else "graph"),
            "mean_down": np.nanmean(x[y == 0]),
            "mean_up": np.nanmean(x[y == 1]),
            "cohens_d": d,
            "abs_d": abs(d) if not np.isnan(d) else np.nan,
        })
    table = pd.DataFrame(records).sort_values("abs_d", ascending=False)
    out_csv = DIAG / "class_separation_cohens_d.csv"
    table.to_csv(out_csv, index=False)
    print(f"[boxplots] wrote {out_csv}")

    print("\n  largest class separations (train period):")
    for _, r in table.head(10).iterrows():
        print(f"    {r.feature:<30} d = {r.cohens_d:+.4f}  ({r.group})")
    max_d = table["abs_d"].max()
    print(f"\n  max |d| across all {len(table)} features = {max_d:.4f}")
    print("  (for reference: |d| >= 0.20 is conventionally a 'small' effect --")
    print("   values an order of magnitude below that mean the classes overlap almost completely)")

    # ---- plots (optionally down-sampled) ----
    plot_df = train
    if DOWNSAMPLE and len(train) > DOWNSAMPLE:
        plot_df = train.sample(DOWNSAMPLE, random_state=RANDOM_SEED)
        print(f"[boxplots] plotting a random subset of {DOWNSAMPLE} rows")

    d_lookup = dict(zip(table["feature"], table["cohens_d"]))

    grid_boxplot(plot_df, groups["price"], label_col, d_lookup,
                 "boxplots_price_features.png",
                 "Price features by next-day direction (train period)", ncols=4)
    grid_boxplot(plot_df, groups["sentiment"], label_col, d_lookup,
                 "boxplots_sentiment_features.png",
                 "Sentiment features by next-day direction (train period)", ncols=5)

    graph_cols = groups["graph"]
    if graph_cols:
        imp_path = DIAG / "feature_importance_all.csv"
        chosen = None
        if imp_path.exists():
            imp = pd.read_csv(imp_path)
            sub = imp[(imp["config"] == "config3") & (imp["model"] == "XGBoost")
                      & (imp["feature"].isin(graph_cols))]
            if len(sub):
                chosen = sub.nlargest(N_GRAPH_DIMS, "abs_importance")["feature"].tolist()
                print(f"[boxplots] graph dims picked by XGBoost config3 importance: {chosen}")
        if not chosen:
            chosen = graph_cols[:N_GRAPH_DIMS]
            print(f"[boxplots] Feature-importance output not found -- plotting first {len(chosen)} graph dims")
        grid_boxplot(plot_df, chosen, label_col, d_lookup,
                     "boxplots_graph_features.png",
                     "Graph features by next-day direction (train period)", ncols=3)

    print("\n[boxplots] done. Expect diffuse importances with no dominant feature --")
    print("class-separation plots and statistics are complete.")


if __name__ == "__main__":
    main()