"""
============================================================================
KNOWLEDGE GRAPH VISUALISATION
============================================================================
Draws ONE sliding-window co-occurrence graph from the real news, so you can
SEE what Config 3 is built on. Produces two figures:

  graph_backbone.png : the 30 target stocks (coloured by sector) + 8 macro
                       hubs (gold diamonds), force-directed so related nodes
                       pull together, edges weighted by co-occurrence.
                       --> the readable, report-quality figure.
  graph_full.png     : the same window with all context tickers faded in
                       behind it, to show the true scale/density.
                       --> the "this is the real graph" appendix figure.

Reuses the exact graph-construction logic from build_graph.py, so the
picture matches what the model actually used.

Run:
    python scripts/graph_viz.py
============================================================================
"""
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import build_graph as bg   # reuse loaders, macro lexicon, window-graph builder

# which window to draw (end date); the graph uses the prior WINDOW_DAYS of news
DRAW_DATE = "2024-10-15"
OUTDIR = os.path.join(bg.BASE_DIR, "data", "graph_viz")

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
SECTOR_LIST = sorted(set(SECTORS.values()))


def window_graph_for(date_str):
    arts = bg.load_articles(bg.RAW_DIR)
    d = pd.Timestamp(date_str).date()
    lo = d - timedelta(days=bg.WINDOW_DAYS)
    art_dates = np.array([a[0] for a in arts])
    hi_i = np.searchsorted(art_dates, d, side="left")
    lo_i = np.searchsorted(art_dates, lo, side="left")
    return bg.build_window_graph(arts[lo_i:hi_i]), d


def draw_backbone(G, d, palette):
    """Backbone = 30 stocks + 8 macro hubs only (induced subgraph)."""
    keep = [n for n in G.nodes if n in SECTORS or n in bg.MACRO_HUBS]
    B = G.subgraph(keep).copy()

    pos = nx.spring_layout(B, weight="weight", k=0.9, iterations=200, seed=7)

    stocks = [n for n in B.nodes if n in SECTORS]
    hubs = [n for n in B.nodes if n in bg.MACRO_HUBS]
    wdeg = dict(B.degree(weight="weight"))

    fig, ax = plt.subplots(figsize=(13, 10))

    # edges: width + alpha by weight
    weights = np.array([B[u][v]["weight"] for u, v in B.edges])
    if len(weights):
        wn = weights / weights.max()
        for (u, v), a in zip(B.edges, wn):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="gray", lw=0.3 + 3 * a, alpha=0.10 + 0.35 * a, zorder=1)

    # stock nodes by sector
    for s in SECTOR_LIST:
        ns = [n for n in stocks if SECTORS[n] == s]
        if not ns:
            continue
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                   s=[120 + 30 * wdeg.get(n, 0) for n in ns],
                   color=palette[s], edgecolors="white", linewidths=1.2,
                   label=s, zorder=3)
    # macro hubs as gold diamonds
    ax.scatter([pos[n][0] for n in hubs], [pos[n][1] for n in hubs],
               s=[300 + 25 * wdeg.get(n, 0) for n in hubs],
               color="#E8B500", marker="D", edgecolors="black", linewidths=1.2,
               label="Macro hub", zorder=4)

    for n in B.nodes:
        ax.annotate(n, pos[n], fontsize=8, fontweight="bold" if n in hubs else "normal",
                    ha="center", va="center", zorder=5,
                    xytext=(0, 11), textcoords="offset points")

    ax.set_title(f"News co-occurrence knowledge graph — backbone\n"
                 f"{bg.WINDOW_DAYS}-day window ending {d}", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, scatterpoints=1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "graph_backbone.png"), dpi=160)
    plt.close(fig)


def draw_full(G, d, palette):
    """Full graph with context tickers faded behind the backbone."""
    pos = nx.spring_layout(G, weight="weight", k=0.3,
                           iterations=80, seed=7)
    fig, ax = plt.subplots(figsize=(13, 11))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=0.3, alpha=0.08)

    ctx = [n for n in G.nodes if n not in SECTORS and n not in bg.MACRO_HUBS]
    nx.draw_networkx_nodes(G, pos, nodelist=ctx, node_size=12,
                           node_color="#cccccc", alpha=0.5, ax=ax)
    for s in SECTOR_LIST:
        ns = [n for n in G.nodes if SECTORS.get(n) == s]
        if ns:
            nx.draw_networkx_nodes(G, pos, nodelist=ns, node_size=130,
                                   node_color=[palette[s]] * len(ns),
                                   edgecolors="white", ax=ax)
    hubs = [n for n in G.nodes if n in bg.MACRO_HUBS]
    nx.draw_networkx_nodes(G, pos, nodelist=hubs, node_size=260,
                           node_color="#E8B500", node_shape="D",
                           edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes if n in SECTORS or n in hubs},
                            font_size=7, ax=ax)
    ax.set_title(f"Full news co-occurrence graph ({G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges) — window ending {d}", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "graph_full.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(SECTOR_LIST)))
    palette = {s: cmap[i] for i, s in enumerate(SECTOR_LIST)}

    G, d = window_graph_for(DRAW_DATE)
    print(f"Window graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    draw_backbone(G, d, palette)
    draw_full(G, d, palette)
    print(f"Figures written to: {OUTDIR}")
    print("  - graph_backbone.png  (the readable report figure)")
    print("  - graph_full.png      (full density, for an appendix)")


if __name__ == "__main__":
    main()