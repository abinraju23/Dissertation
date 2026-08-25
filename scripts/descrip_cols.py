import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(BASE_DIR, "data", "config3_features.csv")
OUTDIR = os.path.join(BASE_DIR, "data", "describe")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# ---- feature groups + plain-language meaning ----
PRICE = {
    "ret_1d": "today's return", "ret_2d": "2-day return", "ret_5d": "5-day return",
    "ret_10d": "10-day return", "ret_lag1": "return 1 day ago", "ret_lag2": "return 2 days ago",
    "ret_lag3": "return 3 days ago", "close_to_sma5": "price vs 5-day avg",
    "close_to_sma10": "price vs 10-day avg", "close_to_sma20": "price vs 20-day avg",
    "sma5_to_sma20": "short vs long trend", "vol_5": "5-day volatility",
    "vol_10": "10-day volatility", "vol_20": "20-day volatility",
    "hl_range": "intraday high-low range", "vol_ratio_5": "volume vs 5-day avg",
}
SENT = {
    "sent_finbert": "FinBERT daily sentiment (signed)", "sent_av": "Alpha Vantage sentiment",
    "news_count_100h": "articles in 100h window", "news_mass_decayed": "decay-weighted news mass",
    "days_since_news": "days since last news (staleness)",
}
GRAPH = {
    "graph_degree": "# graph neighbours", "graph_wdegree": "weighted connectivity",
    "graph_macro_links": "# macro-hub links", "graph_stale": "carried-forward flag",
}


def describe_block(df, cols_desc, title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    print(f"{'feature':<20}{'meaning':<32}{'mean':>9}{'std':>9}{'min':>8}{'max':>8}")
    rows = []
    for c, meaning in cols_desc.items():
        if c not in df.columns:
            continue
        s = df[c].astype(float)
        print(f"{c:<20}{meaning[:31]:<32}{s.mean():>9.3f}{s.std():>9.3f}{s.min():>8.2f}{s.max():>8.2f}")
        rows.append({"feature": c, "group": title, "meaning": meaning,
                     "mean": s.mean(), "median": s.median(), "std": s.std(),
                     "min": s.min(), "max": s.max(), "n_missing": int(s.isna().sum())})
    return rows


def main():
    df = pd.read_csv(CFG, parse_dates=["date"])
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    print("=" * 70)
    print("DATASET DESCRIPTION  --  config3_features.csv")
    print("=" * 70)
    print(f"Rows                : {len(df):,}")
    print(f"Stocks              : {df['ticker'].nunique()}")
    print(f"Date range          : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Embedding dims       : {len(emb_cols)}")
    print(f"Total feature columns: {len(df.columns)}")

    # ---- class balance ----
    print("\n" + "-" * 70)
    print("CLASS BALANCE  (share of UP days)")
    print("-" * 70)
    for sp in ["train", "val", "test", None]:
        s = df["label"] if sp is None else df.loc[df["split"] == sp, "label"]
        name = "OVERALL" if sp is None else sp
        print(f"  {name:<8} up-share {s.mean():.3f}   ({len(s):,} rows)   "
              f"majority baseline acc = {max(s.mean(), 1-s.mean()):.3f}")
    print("  Interpretation: only mild imbalance -> report F1 & AUC-ROC alongside accuracy;")
    print("  AUC = 0.50 is the no-skill line regardless of accuracy.")

    # ---- per-split distribution shift (a couple of key features) ----
    print("\n" + "-" * 70)
    print("DISTRIBUTION SHIFT ACROSS SPLITS  (chronological split -> expect some drift)")
    print("-" * 70)
    for c in ["ret_1d", "vol_20", "sent_finbert", "news_count_100h", "graph_degree"]:
        if c in df.columns:
            m = df.groupby("split")[c].mean()
            print(f"  {c:<16} train {m.get('train',float('nan')):+.3f} | "
                  f"val {m.get('val',float('nan')):+.3f} | test {m.get('test',float('nan')):+.3f}")

    # ---- feature stats by group ----
    all_rows = []
    all_rows += describe_block(df, PRICE, "PRICE FEATURES (stationary: returns/ratios)")
    all_rows += describe_block(df, SENT,  "SENTIMENT FEATURES (Config 2)")
    all_rows += describe_block(df, GRAPH, "GRAPH FEATURES (Config 3 interpretable)")

    # ---- sentiment-specific reads ----
    quiet = (df["news_count_100h"] == 0).mean()
    live = df[df["news_count_100h"] > 0]
    print("\nSentiment notes:")
    print(f"  quiet days (news_count=0)      : {quiet:.1%}")
    if len(live) > 10:
        print(f"  FinBERT vs AlphaVantage corr   : {live['sent_finbert'].corr(live['sent_av']):+.3f} "
              f"(on {len(live):,} news days)  <- independent sanity check")
    print(f"  sentiment mean is positive ({df['sent_finbert'].mean():+.3f}) -> financial news skews upbeat (expected)")

    # ---- embedding block summary (don't dump 64 rows) ----
    E = df[emb_cols].values.astype(float)
    norms = np.linalg.norm(E, axis=1)
    print("\n" + "-" * 70)
    print(f"EMBEDDING BLOCK  ({len(emb_cols)} dims) -- summarised, not listed")
    print("-" * 70)
    print(f"  mean of all embedding values   : {E.mean():+.4f}  (near 0 = centred coordinates)")
    print(f"  mean per-row vector length     : {norms.mean():.3f}  (sd {norms.std():.3f})")
    print(f"  stale rows (carried-forward)   : {(df['graph_stale']==1).sum():,}")
    print("  Note: the 64 dims are coordinates -- meaningful collectively (distances),")
    print("  not individually. The interpretable graph features above are their readable proxy.")

    # ---- label association preview (TRAIN ONLY -> no peeking at test) ----
    print("\n" + "-" * 70)
    print("LABEL ASSOCIATION  (corr with next-day direction, TRAIN ONLY)")
    print("-" * 70)
    tr = df[df["split"] == "train"]
    feat_cols = list(PRICE) + list(SENT) + list(GRAPH)
    assoc = {c: tr[c].corr(tr["label"]) for c in feat_cols
             if c in tr.columns and tr[c].std() > 0}
    emb_live = [c for c in emb_cols if tr[c].std() > 0]   # skip any constant (stale) dims
    emb_abs = np.array([abs(tr[c].corr(tr["label"])) for c in emb_live]) if emb_live else np.array([0.0])
    top = sorted(assoc.items(), key=lambda kv: -abs(kv[1]))[:8]
    for c, r in top:
        print(f"  {c:<18} corr {r:+.3f}")
    print(f"  embeddings (64 dims)  max|corr| {np.nanmax(emb_abs):.3f}  mean|corr| {np.nanmean(emb_abs):.3f}")
    print("  Interpretation: individual correlations near 0 are EXPECTED for next-day")
    print("  direction (a near-coin-flip). The models look for weak signal jointly; this")
    print("  preview just shows no single feature is doing it alone -- which is normal,")
    print("  Computed on the training period only.")

    # ---- save report tables ----
    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUTDIR, "feature_stats.csv"), index=False)
    pd.DataFrame([{"feature": c, "corr_with_label_train": r} for c, r in assoc.items()]
                 ).to_csv(os.path.join(OUTDIR, "label_assoc.csv"), index=False)
    print(f"\nSaved: feature_stats.csv, label_assoc.csv  -> {OUTDIR}")

    # ---- figures ----
    if HAVE_PLT:
        figdir = os.path.join(OUTDIR, "figures"); os.makedirs(figdir, exist_ok=True)
        # class balance
        fig, ax = plt.subplots(figsize=(6, 4))
        order = ["train", "val", "test"]
        ups = [df.loc[df["split"] == s, "label"].mean() for s in order]
        ax.bar(order, ups, color="#2E5496"); ax.axhline(0.5, ls="--", color="gray")
        ax.set_ylim(0.4, 0.6); ax.set_ylabel("share of UP days")
        ax.set_title("Class balance by split (0.50 = balanced)")
        for i, v in enumerate(ups): ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "class_balance.png"), dpi=150); plt.close(fig)
        # sentiment distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["sent_finbert"], bins=50, color="#2E7D32", alpha=0.8)
        ax.axvline(0, ls="--", color="gray"); ax.set_xlabel("FinBERT daily sentiment")
        ax.set_title("Sentiment distribution (slight positive tilt)")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "sentiment_dist.png"), dpi=150); plt.close(fig)
        # label association bars
        fig, ax = plt.subplots(figsize=(7, 5))
        items = sorted(assoc.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color="#6A4C93")
        ax.axvline(0, color="gray"); ax.set_xlabel("corr with next-day direction (train)")
        ax.set_title("Feature association with the label")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "label_assoc.png"), dpi=150); plt.close(fig)
        print(f"Figures -> {figdir}")


if __name__ == "__main__":
    main()