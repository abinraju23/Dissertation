import os, glob, json, re, pickle, time
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PRICE_FILE = os.path.join(BASE_DIR, "data", "config2_features.csv")  # price+sentiment
EMB_CACHE  = os.path.join(BASE_DIR, "data", "graph_emb_cache.pkl")
OUT_FILE   = os.path.join(BASE_DIR, "data", "config3_features.csv")

TICKERS = [
    "AAPL","MSFT","NVDA","AMD","GOOGL","META","NFLX","DIS","AMZN","TSLA",
    "MCD","HD","JPM","BAC","WFC","MS","JNJ","PFE","MRK","LLY",
    "WMT","KO","PEP","PG","XOM","CVX","COP","BA","GE","NEM",
]
TARGET = set(TICKERS)

WINDOW_DAYS = 90          # sliding window (~one quarter / ~63 trading days)
EMB_DIM     = 64          # Node2Vec dimensions (matches proposal)
MIN_CTX_WDEG = 0.5        # prune context nodes below this weighted degree
RELEVANCE_MIN = 0.0       # min AV relevance to count a ticker tag

# Node2Vec params (modest, since we run one per trading day)
N2V = dict(walk_length=15, num_walks=30, workers=1, seed=42, quiet=True, weight_key="weight")
N2V_FIT = dict(window=8, min_count=1, batch_words=64, seed=42)

CHECKPOINT_EVERY = 25     # flush embedding cache every N dates

# ----------------------------------------------------------------------------
# MACRO HUB LEXICON (mirrors macro_entities.py / the connectivity probe)
# ----------------------------------------------------------------------------
MACRO_LEXICON = {
    "OIL": ["oil","crude","brent","wti","opec","barrel","petroleum","gasoline","refinery"],
    "GOLD": ["gold","bullion","precious metal","safe haven","spot gold"],
    "FED_RATES": ["federal reserve","the fed","interest rate","rate hike","rate cut","fomc","powell","monetary policy","basis points"],
    "INFLATION": ["inflation","cpi","consumer price","ppi","deflation","cost of living"],
    "RECESSION": ["recession","slowdown","downturn","contraction","soft landing","hard landing","gdp"],
    "GEOPOLITICS": ["war","ukraine","russia","middle east","israel","geopolitical","sanctions","taiwan"],
    "TRADE_TARIFFS": ["tariff","trade war","import duty","export ban","trade deal","supply chain","export control"],
    "MARKET_RISK": ["volatility","vix","sell-off","selloff","market crash","correction","bear market","risk-off"],
}
MACRO_PATTERNS = {h: [re.compile(r"\b"+re.escape(p)+r"\b", re.I) for p in ph]
                  for h, ph in MACRO_LEXICON.items()}
MACRO_HUBS = list(MACRO_LEXICON.keys())
PERSISTENT = set(TICKERS) | set(MACRO_HUBS)   # nodes used to anchor alignment


# ============================================================================
# LOAD + DEDUP  -> per-article (date, entity set)
# ============================================================================
def _iter(raw_dir):
    for fp in glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True):
        try:
            obj = json.load(open(fp, "r", encoding="utf-8"))
        except Exception:
            continue
        resp = obj.get("response") if isinstance(obj, dict) else None
        feed = resp.get("feed") if isinstance(resp, dict) else None
        if isinstance(feed, list):
            for a in feed:
                if isinstance(a, dict):
                    yield a


def macro_tags(text):
    return {h for h, pats in MACRO_PATTERNS.items() if any(p.search(text) for p in pats)}


def load_articles(raw_dir):
    seen = set()
    arts = []
    for a in _iter(raw_dir):
        tickers = set()
        has_target = False
        for ts_ in a.get("ticker_sentiment", []) or []:
            tk = ts_.get("ticker")
            if not tk:
                continue
            try:
                rel = float(ts_.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                rel = 0.0
            if rel >= RELEVANCE_MIN:
                tickers.add(tk)
                if tk in TARGET:
                    has_target = True
        if not has_target:
            continue
        url = a.get("url") or (str(a.get("title")) + str(a.get("time_published")))
        if url in seen:
            continue
        seen.add(url)
        d = str(a.get("time_published", ""))[:8]
        try:
            adate = datetime.strptime(d, "%Y%m%d").date()
        except Exception:
            continue
        text = str(a.get("title", "")) + ". " + str(a.get("summary", ""))
        ents = tickers | macro_tags(text)
        if len(ents) >= 2:
            arts.append((adate, frozenset(ents)))
    arts.sort(key=lambda x: x[0])
    print(f"Loaded {len(arts):,} usable articles (>=2 entities, >=1 target)")
    return arts


# ============================================================================
# WINDOW GRAPH
# ============================================================================
def build_window_graph(window_arts):
    """Roundup-discounted co-occurrence graph from articles in a window."""
    w = {}
    for _, ents in window_arts:
        k = len(ents)
        if k < 2:
            continue
        inc = 1.0 / (k - 1)            # roundup discount
        for u, v in combinations(sorted(ents), 2):
            w[(u, v)] = w.get((u, v), 0.0) + inc
    G = nx.Graph()
    for (u, v), ww in w.items():
        G.add_edge(u, v, weight=ww)
    # prune weakly-connected CONTEXT nodes; always keep targets + hubs
    drop = [n for n in G.nodes
            if n not in PERSISTENT and G.degree(n, weight="weight") < MIN_CTX_WDEG]
    G.remove_nodes_from(drop)
    G.remove_nodes_from([n for n in G.nodes if G.degree(n) == 0 and n not in PERSISTENT])
    return G


def embed_graph(G):
    if G.number_of_edges() == 0:
        return {}
    n2v = Node2Vec(G, dimensions=EMB_DIM, **N2V)
    model = n2v.fit(**N2V_FIT)
    return {str(n): np.asarray(model.wv[str(n)], dtype=float) for n in G.nodes()}


# ============================================================================
# STAGE A -- compute + cache raw embeddings per date (resumable)
# ============================================================================
def stage_a(arts, dates):
    cache = {}
    if os.path.exists(EMB_CACHE):
        cache = pickle.load(open(EMB_CACHE, "rb"))
        print(f"Stage A: {len(cache):,} dates already cached")
    art_dates = np.array([a[0] for a in arts])

    todo = [d for d in dates if d.isoformat() not in cache]
    print(f"Stage A: {len(todo):,} dates to embed")
    for i, D in enumerate(todo, 1):
        d = D.date() if hasattr(D, "date") else D
        lo = d - timedelta(days=WINDOW_DAYS)
        # window [lo, d)  -- strictly before day d (no look-ahead)
        hi_i = np.searchsorted(art_dates, d, side="left")
        lo_i = np.searchsorted(art_dates, lo, side="left")
        window = arts[lo_i:hi_i]
        G = build_window_graph(window)
        cache[d.isoformat()] = {"vecs": embed_graph(G),
                                "deg": dict(G.degree()),
                                "wdeg": dict(G.degree(weight="weight")),
                                "macro": {n: sum(1 for nb in G.neighbors(n) if nb in MACRO_HUBS)
                                          for n in G.nodes() if n in TARGET}}
        if i % CHECKPOINT_EVERY == 0:
            pickle.dump(cache, open(EMB_CACHE, "wb"))
            print(f"   embedded {i:,}/{len(todo):,} dates")
    pickle.dump(cache, open(EMB_CACHE, "wb"))
    print("Stage A: done")
    return cache


# ============================================================================
# STAGE B -- Procrustes alignment across consecutive dates
# ============================================================================
def procrustes(curr, prev_aligned):
    """Rotate current vecs into previous frame using shared persistent nodes."""
    shared = [n for n in curr if n in prev_aligned and n in PERSISTENT]
    if len(shared) < 3:
        return dict(curr)  # not enough anchors -> leave as is
    A = np.stack([curr[n] for n in shared])
    B = np.stack([prev_aligned[n] for n in shared])
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return {n: v @ R for n, v in curr.items()}


def stage_b(cache, dates):
    aligned = {}
    prev = None
    for D in dates:
        d = (D.date() if hasattr(D, "date") else D).isoformat()
        vecs = cache.get(d, {}).get("vecs", {})
        if not vecs:
            aligned[d] = {}
            continue
        aligned[d] = vecs if prev is None else procrustes(vecs, prev)
        prev = {**(prev or {}), **aligned[d]}   # carry frame forward, update nodes
    return aligned


# ============================================================================
# STAGE C -- extract per-stock features, carry forward, join
# ============================================================================
def stage_c(cache, aligned, price):
    last_vec = {t: np.zeros(EMB_DIM) for t in TICKERS}
    seen_any = {t: False for t in TICKERS}
    rows = []
    dates = sorted(price["date"].unique())
    for D in dates:
        d = pd.Timestamp(D).date().isoformat()
        al = aligned.get(d, {})
        meta = cache.get(d, {})
        for t in TICKERS:
            if t in al:
                vec = al[t]
                last_vec[t] = vec
                seen_any[t] = True
                stale = 0
                deg = meta.get("deg", {}).get(t, 0)
                wdeg = meta.get("wdeg", {}).get(t, 0.0)
                mac = meta.get("macro", {}).get(t, 0)
            else:
                vec = last_vec[t]                 # carry forward
                stale = 0 if seen_any[t] else 1
                deg = wdeg = mac = 0
            row = {"ticker": t, "date": pd.Timestamp(D)}
            for j in range(EMB_DIM):
                row[f"emb_{j:03d}"] = float(vec[j])
            row["graph_degree"] = int(deg)
            row["graph_wdegree"] = float(wdeg)
            row["graph_macro_links"] = int(mac)
            row["graph_stale"] = int(stale)
            rows.append(row)
    gdf = pd.DataFrame(rows)
    merged = price.merge(gdf, on=["ticker", "date"], how="left")
    return merged


def main():
    print("Loading panel:", PRICE_FILE)
    price = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    dates = sorted(price["date"].unique())
    print(f"{len(dates):,} distinct trading dates, {price['ticker'].nunique()} tickers")

    arts = load_articles(RAW_DIR)
    cache = stage_a(arts, [pd.Timestamp(d) for d in dates])
    aligned = stage_b(cache, [pd.Timestamp(d) for d in dates])
    merged = stage_c(cache, aligned, price)
    merged.to_csv(OUT_FILE, index=False)
    summarise(merged)


def summarise(merged):
    emb_cols = [c for c in merged.columns if c.startswith("emb_")]
    print("\n" + "=" * 60)
    print("CONFIG 3 GRAPH BLOCK BUILT")
    print("=" * 60)
    print(f"Rows                 : {len(merged):,}")
    print(f"Embedding dims       : {len(emb_cols)}")
    print(f"Stale (pre-first) rows: {(merged['graph_stale']==1).sum():,}")
    print(f"Mean graph_degree    : {merged['graph_degree'].mean():.1f}")
    print(f"Mean macro links     : {merged['graph_macro_links'].mean():.2f}")
    # embedding stability: avg day-to-day change per ticker (lower = more stable)
    drift = []
    for t, g in merged.sort_values("date").groupby("ticker"):
        E = g[emb_cols].values
        if len(E) > 1:
            num = np.linalg.norm(np.diff(E, axis=0), axis=1)
            den = np.linalg.norm(E[:-1], axis=1) + np.linalg.norm(E[1:], axis=1) + 1e-9
            drift.append(np.mean(num / den))
    if drift:
        print(f"Mean embedding drift : {np.mean(drift):.4f}  (Singer et al. stability error)")
    print(f"\nWritten: {OUT_FILE}")


if __name__ == "__main__":
    main()