"""
============================================================================
CO-OCCURRENCE CONNECTIVITY PROBE
============================================================================
Purpose:
    Before freezing the final 30 stocks, check whether they actually FORM A
    CONNECTED GRAPH in the news -- not just whether they have enough articles.
    Also produces the graph descriptive-statistics (node count, edge count,
    density, components) you need for the report.

What it measures (using data already on disk -- NO FinBERT required):
    1. DIRECT links  : two target stocks tagged in the same article
                       (from Alpha Vantage's own `ticker_sentiment` list).
    2. INDIRECT links: two stocks that both attach to the same MACRO HUB
                       (OIL, FED_RATES, RECESSION, ...) via phrase-matching
                       the title+summary. This mirrors your macro_entities.py
                       idea and is how XOM and JPM end up "related" without
                       ever sharing a headline.

IMPORTANT -- this is a STATIC, full-period graph built for SELECTION and
DESCRIPTIVE STATS ONLY. It is deliberately NOT the modelling graph. Your
Config 3 graph is the daily sliding-window version built strictly from past
data to avoid look-ahead. Do not feed this static graph into the model.

How to run:
    1. Set RAW_DIR below to the folder holding your raw news JSON.
    2. (Optional) edit CANDIDATES to your final list / swaps.
    3. python co_occurrence_probe.py
    4. Read the console verdict; CSVs + heatmap land in OUTPUT_DIR.
============================================================================
"""

import os
import re
import csv
import glob
import json
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd
import networkx as nx

# matplotlib is optional -- the probe still works without it (no heatmap)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ============================================================================
# CONFIG  -- edit these
# ============================================================================

# Folder containing your raw Alpha Vantage news. The loader searches it
# recursively for .json / .jsonl files, so per-slice or per-ticker layouts
# both work. EDIT THIS to point at your raw layer.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "probe_out")

# Only count a ticker as "present" in an article if AV's relevance_score is
# at least this. 0.0 = count every tag. Raise to ~0.15 to ignore trivial
# passing mentions. Start at 0.0, then re-run at 0.15 to see how robust the
# graph is to noise.
RELEVANCE_MIN = 0.15

# Your candidate 30, grouped by sector (edit freely -- e.g. swap NEM <-> GS).
CANDIDATES = {
    "Information Technology":  ["AAPL", "MSFT", "NVDA", "AMD"],
    "Communication Services":  ["GOOGL", "META", "NFLX", "DIS"],
    "Consumer Discretionary":  ["AMZN", "TSLA", "MCD", "HD"],
    "Financials":              ["JPM", "BAC", "WFC", "MS"],
    "Health Care":             ["JNJ", "PFE", "MRK", "LLY"],
    "Consumer Staples":        ["WMT", "KO", "PEP", "PG"],
    "Energy":                  ["XOM", "CVX", "COP"],
    "Industrials":             ["BA", "GE"],
    "Materials":               ["NEM"],
}

TARGETS = [t for tickers in CANDIDATES.values() for t in tickers]
SECTOR_OF = {t: s for s, tickers in CANDIDATES.items() for t in tickers}


# ----------------------------------------------------------------------------
# MACRO HUB LEXICON
# Self-contained phrase lexicon mirroring your macro_entities.py (8 hubs).
# Replace with `from macro_entities import ...` later if you prefer your own.
# Phrases are matched case-insensitively as whole words on title + summary.
# ----------------------------------------------------------------------------
MACRO_LEXICON = {
    "OIL": ["oil", "crude", "brent", "wti", "opec", "barrel", "petroleum",
            "gasoline", "refinery", "energy prices"],
    "GOLD": ["gold", "bullion", "precious metal", "safe haven", "spot gold"],
    "FED_RATES": ["federal reserve", "the fed", "interest rate", "rate hike",
                  "rate cut", "fomc", "powell", "monetary policy",
                  "basis points", "tightening", "easing"],
    "INFLATION": ["inflation", "cpi", "consumer price", "ppi", "deflation",
                  "cost of living", "price pressures"],
    "RECESSION": ["recession", "slowdown", "downturn", "contraction",
                  "soft landing", "hard landing", "gdp", "economic growth"],
    "GEOPOLITICS": ["war", "ukraine", "russia", "middle east", "israel",
                    "geopolitical", "sanctions", "conflict", "taiwan"],
    "TRADE_TARIFFS": ["tariff", "trade war", "import duty", "export ban",
                      "trade deal", "supply chain", "chips act", "export control"],
    "MARKET_RISK": ["volatility", "vix", "sell-off", "selloff", "market crash",
                    "correction", "bear market", "risk-off", "risk off"],
}

# Pre-compile whole-word regexes for speed.
MACRO_PATTERNS = {
    hub: [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in phrases]
    for hub, phrases in MACRO_LEXICON.items()
}
MACRO_HUBS = list(MACRO_LEXICON.keys())


# ============================================================================
# LOADER  -- normalises whatever JSON layout into a stream of articles
# ============================================================================

def _iter_raw_records(raw_dir):
    """Yield raw article dicts from any .json/.jsonl under raw_dir."""
    files = glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True) \
          + glob.glob(os.path.join(raw_dir, "**", "*.jsonl"), recursive=True)
    if not files:
        raise SystemExit(
            f"\n[!] No .json/.jsonl files found under:\n    {raw_dir}\n"
            f"    Edit RAW_DIR at the top of this script.\n")

    for fp in files:
        try:
            if fp.endswith(".jsonl"):
                with open(fp, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
                continue
            with open(fp, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as e:
            print(f"  [skip] {os.path.basename(fp)}: {e}")
            continue

        # Your slice layout: {"fetched_at":..., "response": {"feed": [...]}}
        # Also handles {"feed": [...]}, a bare list, or a single article.
        if isinstance(obj, dict):
            feed = None
            resp = obj.get("response")
            if isinstance(resp, dict) and isinstance(resp.get("feed"), list):
                feed = resp["feed"]                 # <- your nested case
            elif isinstance(obj.get("feed"), list):
                feed = obj["feed"]                  # flat fallback
            if feed is not None:
                for art in feed:
                    if isinstance(art, dict):
                        yield art
            else:
                # empty month / error slice with no feed -> skip silently
                continue
        elif isinstance(obj, list):
            for art in obj:
                if isinstance(art, dict):
                    yield art


def normalise_article(art):
    """
    Return (present_targets:set, macro_hubs:set) for one article,
    or None if it carries no target stock at all.
    """
    # --- tickers from AV's ticker_sentiment list ---
    present = set()
    for ts in art.get("ticker_sentiment", []) or []:
        sym = ts.get("ticker")
        if sym not in SECTOR_OF:
            continue
        try:
            rel = float(ts.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            rel = 0.0
        if rel >= RELEVANCE_MIN:
            present.add(sym)

    if not present:
        return None  # article doesn't touch our universe; ignore

    # --- macro hubs from phrase-matching title + summary ---
    text = " ".join([str(art.get("title", "")), str(art.get("summary", ""))])
    hubs = set()
    for hub, patterns in MACRO_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            hubs.add(hub)

    return present, hubs


# ============================================================================
# BUILD co-occurrence tallies
# ============================================================================

def build_tallies(raw_dir):
    article_count = 0
    used_count = 0
    stock_articles = Counter()                 # how many articles touch each stock
    stock_pair_direct = Counter()              # (a,b) -> # articles co-mentioning
    stock_hub = Counter()                      # (stock,hub) -> # articles
    hub_articles = Counter()

    for art in _iter_raw_records(raw_dir):
        article_count += 1
        norm = normalise_article(art)
        if norm is None:
            continue
        present, hubs = norm
        used_count += 1

        for s in present:
            stock_articles[s] += 1
        for a, b in combinations(sorted(present), 2):
            stock_pair_direct[(a, b)] += 1
        for s in present:
            for h in hubs:
                stock_hub[(s, h)] += 1
        for h in hubs:
            hub_articles[h] += 1

    return {
        "article_count": article_count,
        "used_count": used_count,
        "stock_articles": stock_articles,
        "stock_pair_direct": stock_pair_direct,
        "stock_hub": stock_hub,
        "hub_articles": hub_articles,
    }


# ============================================================================
# GRAPH + METRICS
# ============================================================================

def build_graph(tallies):
    G = nx.Graph()
    for t in TARGETS:
        G.add_node(t, kind="stock", sector=SECTOR_OF[t])
    for h in MACRO_HUBS:
        G.add_node(h, kind="macro")

    for (a, b), w in tallies["stock_pair_direct"].items():
        G.add_edge(a, b, weight=w, kind="direct")
    for (s, h), w in tallies["stock_hub"].items():
        if w > 0:
            G.add_edge(s, h, weight=w, kind="macro")
    return G


def shared_hub_count(tallies, a, b):
    """How many macro hubs both a and b attach to (indirect connectivity)."""
    ha = {h for (s, h) in tallies["stock_hub"] if s == a}
    hb = {h for (s, h) in tallies["stock_hub"] if s == b}
    return len(ha & hb)


def analyse(tallies, G):
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    stock_nodes = [n for n in TARGETS if n in G]
    sub = G.subgraph(stock_nodes + MACRO_HUBS)

    n_nodes = sub.number_of_nodes()
    n_edges = sub.number_of_edges()
    density = nx.density(sub)

    out("=" * 70)
    out("CO-OCCURRENCE CONNECTIVITY PROBE  --  RESULTS")
    out("=" * 70)
    out(f"Articles scanned                : {tallies['article_count']:,}")
    out(f"Articles touching your universe : {tallies['used_count']:,}")
    out(f"Stocks in universe              : {len(TARGETS)}")
    out(f"Macro hubs                      : {len(MACRO_HUBS)}")
    out("")
    out("--- GRAPH DESCRIPTIVE STATS (for the report) ---")
    out(f"Nodes (stocks + hubs)           : {n_nodes}")
    out(f"Edges                           : {n_edges}")
    out(f"Graph density                   : {density:.4f}")

    # connectivity over the whole graph (stocks + hubs)
    components = list(nx.connected_components(sub))
    components.sort(key=len, reverse=True)
    out(f"Connected components            : {len(components)}")
    biggest = components[0] if components else set()
    stocks_in_biggest = [s for s in stock_nodes if s in biggest]
    out(f"Stocks in largest component     : {len(stocks_in_biggest)} / {len(TARGETS)}")

    # isolated stocks (no edge to anything at all)
    isolated = [s for s in stock_nodes if sub.degree(s) == 0]
    out("")
    out("--- PER-STOCK CONNECTIVITY ---")
    rows = []
    for s in stock_nodes:
        direct_deg = sum(1 for nb in sub.neighbors(s) if nb in SECTOR_OF)
        hub_deg = sum(1 for nb in sub.neighbors(s) if nb in MACRO_HUBS)
        wdeg = sub.degree(s, weight="weight")
        rows.append({
            "ticker": s,
            "sector": SECTOR_OF[s],
            "articles": tallies["stock_articles"].get(s, 0),
            "direct_stock_links": direct_deg,
            "macro_hub_links": hub_deg,
            "weighted_degree": int(wdeg),
        })
    df = pd.DataFrame(rows).sort_values("weighted_degree")

    # verdict per stock
    def verdict(r):
        if r["articles"] == 0:
            return "FAIL: no articles"
        if r["direct_stock_links"] == 0 and r["macro_hub_links"] == 0:
            return "FAIL: isolated"
        if r["direct_stock_links"] <= 1 and r["macro_hub_links"] <= 1:
            return "WATCH: weakly linked"
        return "OK"
    df["verdict"] = df.apply(verdict, axis=1)

    # print the weakest 12 so problems surface first
    with pd.option_context("display.width", 200, "display.max_columns", None):
        out(df.head(12).to_string(index=False))

    out("")
    flagged = df[df["verdict"] != "OK"]
    if len(flagged) == 0:
        out("VERDICT: all 30 stocks are connected (direct and/or via macro hubs).")
    else:
        out(f"VERDICT: {len(flagged)} stock(s) need a look:")
        for _, r in flagged.iterrows():
            out(f"   - {r['ticker']:5s} ({r['sector']}): {r['verdict']}")
    if isolated:
        out(f"   fully isolated: {isolated}")

    return df, lines


# ============================================================================
# OUTPUTS
# ============================================================================

def write_outputs(tallies, G, df, summary_lines):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) per-node stats
    df.to_csv(os.path.join(OUTPUT_DIR, "node_stats.csv"), index=False)

    # 2) 30x30 direct co-mention matrix
    mat = pd.DataFrame(0, index=TARGETS, columns=TARGETS, dtype=int)
    for (a, b), w in tallies["stock_pair_direct"].items():
        mat.loc[a, b] = w
        mat.loc[b, a] = w
    mat.to_csv(os.path.join(OUTPUT_DIR, "stock_stock_comention_matrix.csv"))

    # 3) stock x macro-hub attachment
    hub_mat = pd.DataFrame(0, index=TARGETS, columns=MACRO_HUBS, dtype=int)
    for (s, h), w in tallies["stock_hub"].items():
        hub_mat.loc[s, h] = w
    hub_mat.to_csv(os.path.join(OUTPUT_DIR, "stock_macro_attachment.csv"))

    # 4) text summary
    with open(os.path.join(OUTPUT_DIR, "graph_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))

    # 5) heatmap of direct co-mention (optional)
    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(mat.values, cmap="viridis")
        ax.set_xticks(range(len(TARGETS)))
        ax.set_yticks(range(len(TARGETS)))
        ax.set_xticklabels(TARGETS, rotation=90, fontsize=7)
        ax.set_yticklabels(TARGETS, fontsize=7)
        ax.set_title("Direct co-mention counts (target stock pairs)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "comention_heatmap.png"), dpi=150)
        plt.close(fig)

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("  - node_stats.csv")
    print("  - stock_stock_comention_matrix.csv")
    print("  - stock_macro_attachment.csv")
    print("  - graph_summary.txt")
    if HAVE_PLT:
        print("  - comention_heatmap.png")


# ============================================================================
def main():
    print(f"Reading raw news from: {RAW_DIR}\n")
    tallies = build_tallies(RAW_DIR)
    G = build_graph(tallies)
    df, summary_lines = analyse(tallies, G)
    write_outputs(tallies, G, df, summary_lines)


if __name__ == "__main__":
    main()