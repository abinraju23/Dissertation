import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG3 = os.path.join(BASE_DIR, "data", "config3_features.csv")
OUTDIR = os.path.join(BASE_DIR, "data", "sector_check")

SECTORS = {
    "AAPL":"InfoTech","MSFT":"InfoTech","NVDA":"InfoTech","AMD":"InfoTech",
    "GOOGL":"CommSvc","META":"CommSvc","NFLX":"CommSvc","DIS":"CommSvc",
    "AMZN":"ConsDisc","TSLA":"ConsDisc","MCD":"ConsDisc","HD":"ConsDisc",
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","MS":"Financials",
    "JNJ":"HealthCare","PFE":"HealthCare","MRK":"HealthCare","LLY":"HealthCare",
    "WMT":"Staples","KO":"Staples","PEP":"Staples","PG":"Staples",
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "BA":"Industrials","GE":"Industrials","NEM":"Materials",
}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def cosine_matrix(M):
    n = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def intra_inter(emb, tickers):
    """Mean cosine within-sector vs across-sector for one date's embeddings."""
    C = cosine_matrix(emb)
    intra, inter = [], []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            same = SECTORS[tickers[i]] == SECTORS[tickers[j]]
            (intra if same else inter).append(C[i, j])
    return np.mean(intra), np.mean(inter)


def pca_2d(M):
    X = M - M.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def main():
    df = pd.read_csv(CFG3, parse_dates=["date"])
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df = df[df["graph_stale"] == 0]                      # skip pre-history rows
    dates = sorted(df["date"].unique())
    # sample ~40 dates evenly across the timeline for a robust estimate
    sample = dates[::max(1, len(dates) // 40)]

    intra_list, inter_list, wins = [], [], 0
    for D in sample:
        sub = df[df["date"] == D]
        if sub["ticker"].nunique() < 25:
            continue
        sub = sub.drop_duplicates("ticker").set_index("ticker")
        tickers = [t for t in SECTORS if t in sub.index]
        emb = sub.loc[tickers, emb_cols].values.astype(float)
        a, b = intra_inter(emb, tickers)
        intra_list.append(a); inter_list.append(b)
        wins += int(a > b)

    print("=" * 56)
    print("SECTOR-RECOVERY CHECK  (Config 3 embeddings)")
    print("=" * 56)
    print(f"Dates sampled            : {len(intra_list)}")
    print(f"Mean intra-sector cosine : {np.mean(intra_list):+.3f}")
    print(f"Mean inter-sector cosine : {np.mean(inter_list):+.3f}")
    print(f"Separation (intra-inter) : {np.mean(intra_list)-np.mean(inter_list):+.3f}")
    print(f"Dates where intra > inter: {wins}/{len(intra_list)}")
    if np.mean(intra_list) > np.mean(inter_list):
        print("\nVERDICT: embeddings recover sector structure -- the graph carries")
        print("real relational signal (same-sector stocks embed closer).")
    else:
        print("\nVERDICT: no clear sector separation -- worth investigating before")
        print("leaning on the embeddings.")

    if not HAVE_PLT:
        print("\n(matplotlib not installed -> skipping figures; pip install matplotlib)")
        return

    os.makedirs(OUTDIR, exist_ok=True)
    # --- figures from the most recent well-populated date ---
    D = dates[-1]
    sub = df[df["date"] == D].drop_duplicates("ticker").set_index("ticker")
    tickers = [t for t in SECTORS if t in sub.index]
    emb = sub.loc[tickers, emb_cols].values.astype(float)
    secs = [SECTORS[t] for t in tickers]
    uniq = sorted(set(secs))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(uniq)))
    color = {s: cmap[i] for i, s in enumerate(uniq)}

    # scatter
    P = pca_2d(emb)
    fig, ax = plt.subplots(figsize=(9, 7))
    for s in uniq:
        idx = [i for i, x in enumerate(secs) if x == s]
        ax.scatter(P[idx, 0], P[idx, 1], color=color[s], label=s, s=80)
    for i, t in enumerate(tickers):
        ax.annotate(t, (P[i, 0], P[i, 1]), fontsize=7, alpha=0.8)
    ax.set_title(f"Config 3 embeddings by sector (PCA, {pd.Timestamp(D).date()})")
    ax.legend(fontsize=8, loc="best"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "sector_scatter.png"), dpi=150)
    plt.close(fig)

    # heatmap ordered by sector
    order = sorted(range(len(tickers)), key=lambda i: (secs[i], tickers[i]))
    emb2 = emb[order]; tk2 = [tickers[i] for i in order]
    C = cosine_matrix(emb2)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tk2))); ax.set_yticks(range(len(tk2)))
    ax.set_xticklabels(tk2, rotation=90, fontsize=6); ax.set_yticklabels(tk2, fontsize=6)
    ax.set_title(f"Embedding cosine similarity, ordered by sector ({pd.Timestamp(D).date()})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "sector_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"\nFigures written to: {OUTDIR}")
    print("  - sector_scatter.png   (look for sector-coloured clusters)")
    print("  - sector_heatmap.png   (look for bright blocks on the diagonal)")


if __name__ == "__main__":
    main()