from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Project root is the parent of the scripts/ folder this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Immutable raw layer. We save exactly what the API returns, untouched, and
# build tidy tables FROM this later. The "alphavantage" subfolder leaves room
# for a second source (e.g. data/raw/news/<source>/) without disturbing this one.
RAW_NEWS_DIR = DATA_DIR / "raw" / "news" / "alphavantage"

# Where the probe writes its triage summary.
PROBE_OUTPUT = DATA_DIR / "probe" / "news_coverage_probe.csv"

# --------------------------------------------------------------------------
# Training / collection window
# --------------------------------------------------------------------------
# The full window we want to cover. Confirm the start is actually reachable
# with the coverage probe BEFORE trusting it.
WINDOW_START = "2022-01-01"
WINDOW_END = "2024-12-31"

# --------------------------------------------------------------------------
# Candidate stock universe (deliberately MORE than 30)
# --------------------------------------------------------------------------
# 4-5 high-news-traffic large caps per GICS sector, 9 sectors. The probe will
# tell us which clear the article threshold; we then trim to the final 30 while
# keeping sector balance. High news traffic is doubly important: it satisfies
# the article-volume rule AND ensures these names actually co-occur in the news,
# so they form real edges in the shared knowledge graph later.
CANDIDATES_BY_SECTOR = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AMD", "AVGO"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "Financials":             ["JPM", "BAC", "GS", "WFC", "MS"],
    "Health Care":            ["JNJ", "UNH", "PFE", "MRK", "LLY"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB"],
    "Industrials":            ["BA", "CAT", "GE", "UPS"],
    "Consumer Staples":       ["WMT", "KO", "PG", "PEP", "COST"],
    "Materials":              ["LIN", "FCX", "NEM"],
}

# Flat list of all candidate tickers.
CANDIDATE_TICKERS = [t for tickers in CANDIDATES_BY_SECTOR.values() for t in tickers]

# Lookup: ticker -> sector (handy for keeping balance when trimming to 30).
TICKER_SECTOR = {
    t: sector
    for sector, tickers in CANDIDATES_BY_SECTOR.items()
    for t in tickers
}

# Minimum articles for a stock to qualify (your sampling rule from Agrawal et al.).
MIN_ARTICLES = 100

# --------------------------------------------------------------------------
# API behaviour
# --------------------------------------------------------------------------
# Alpha Vantage caps each NEWS_SENTIMENT call at 1000 items.
AV_MAX_LIMIT = 1000

# Premium "Plan 75" = 75 requests/minute -> 60/75 = 0.8s between calls.
# We add a small safety buffer. Bump MIN_REQUEST_INTERVAL down if you upgrade.
MIN_REQUEST_INTERVAL = 0.85  # seconds between API calls

# Window size for the full collection sweep. Monthly is a safe default; the
# collector automatically splits a window further if it hits the 1000 cap.
COLLECTION_WINDOW = "monthly"  # "monthly" or "weekly"