import os
import time
import pandas as pd
import numpy as np


try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_DIR = os.path.join(BASE_DIR, "data", "prices")
OUT_FILE = os.path.join(BASE_DIR, "data", "config1_features.csv")

# Final locked 30
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD",
    "GOOGL", "META", "NFLX", "DIS",
    "AMZN", "TSLA", "MCD", "HD",
    "JPM", "BAC", "WFC", "MS",
    "JNJ", "PFE", "MRK", "LLY",
    "WMT", "KO", "PEP", "PG",
    "XOM", "CVX", "COP",
    "BA", "GE", "NEM",
]

# Official study window
START = "2022-01-01"
END = "2025-01-01"          # yfinance end is EXCLUSIVE, so this includes 2024-12-31

# Fetch a buffer BEFORE the start so the 20-day rolling windows are already
# "warm" on the first real trading days of Jan 2022 (otherwise early-2022 rows
# get dropped as warm-up and you lose data you actually want).
FETCH_START = "2021-11-01"
# Chronological split
TRAIN_END = "2023-12-31"    # Train: 2022-2023
VAL_END = "2024-06-30"      # Val:   2024 H1   -> Test: 2024 H2


# ============================================================================
# FEATURE ENGINEERING  (pure function -- unit-testable without yfinance)
# ============================================================================

def add_features_and_label(df):
    """
    df: single ticker's price frame, indexed by date (ascending), with at
        least columns Open, High, Low, Close, Volume (adjusted).
    Returns the frame with Config 1 features + label added.
    All rolling windows end at row t (use past+present only). Only `label`
    looks one day into the future.
    """
    df = df.sort_index().copy()

    c = df["Close"]
    ret1 = c.pct_change()                      # today's close-to-close return

    # --- returns / momentum (stationary) ---
    df["ret_1d"] = ret1
    df["ret_2d"] = c.pct_change(2)
    df["ret_5d"] = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_lag1"] = ret1.shift(1)             # yesterday's return
    df["ret_lag2"] = ret1.shift(2)
    df["ret_lag3"] = ret1.shift(3)

    # --- moving-average RELATIVES (ratios, not raw levels -> stationary) ---
    sma5 = c.rolling(5).mean()
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    df["close_to_sma5"] = c / sma5 - 1.0
    df["close_to_sma10"] = c / sma10 - 1.0
    df["close_to_sma20"] = c / sma20 - 1.0
    df["sma5_to_sma20"] = sma5 / sma20 - 1.0   # short vs long trend

    # --- volatility (rolling std of daily returns) ---
    df["vol_5"] = ret1.rolling(5).std()
    df["vol_10"] = ret1.rolling(10).std()
    df["vol_20"] = ret1.rolling(20).std()

    # --- intraday range & relative volume ---
    df["hl_range"] = (df["High"] - df["Low"]) / c
    df["vol_ratio_5"] = df["Volume"] / df["Volume"].rolling(5).mean()

    # --- THE LABEL: direction of NEXT day's close ---
    ret_next = c.shift(-1) / c - 1.0           # tomorrow's return
    df["ret_next"] = ret_next
    # (NaN > 0) is False in pandas, which would mislabel the final row as 0 --
    # so build the label then explicitly blank it where there is no future.
    df["label"] = (ret_next > 0).astype("Int64")
    df.loc[ret_next.isna(), "label"] = pd.NA
    # NOTE: neutral dead-zone variant (mark |ret_next|<eps as a third class /
    # drop) is a future robustness check -- not applied here.

    return df


FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d",
    "ret_lag1", "ret_lag2", "ret_lag3",
    "close_to_sma5", "close_to_sma10", "close_to_sma20", "sma5_to_sma20",
    "vol_5", "vol_10", "vol_20",
    "hl_range", "vol_ratio_5",
]


def assign_split(date):
    d = pd.Timestamp(date)
    if d <= pd.Timestamp(TRAIN_END):
        return "train"
    if d <= pd.Timestamp(VAL_END):
        return "val"
    return "test"


# ============================================================================
# DOWNLOAD
# ============================================================================

def download_one(ticker, retries=3):
    """Download adjusted OHLCV for one ticker, with light retry."""
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=FETCH_START, end=END, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("empty frame")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            return df
        except Exception as e:
            wait = 2 ** attempt
            print(f"   {ticker}: {e.__class__.__name__} ({e}); retry in {wait}s")
            time.sleep(wait)
    print(f"   {ticker}: FAILED after {retries} attempts")
    return None


def collect():
    if yf is None:
        raise SystemExit("yfinance not installed. Run: pip install yfinance")

    os.makedirs(PRICES_DIR, exist_ok=True)
    panels = []
    failed = []

    for i, ticker in enumerate(TICKERS, 1):
        print(f"[{i:2d}/{len(TICKERS)}] {ticker}")
        raw = download_one(ticker)
        if raw is None:
            failed.append(ticker)
            continue

        # save raw adjusted OHLCV (full buffer included, for transparency)
        raw.to_csv(os.path.join(PRICES_DIR, f"{ticker}.csv"))

        feat = add_features_and_label(raw)
        feat = feat.reset_index()
        feat.insert(0, "ticker", ticker)

        # trim to official window AFTER features (so Jan-2022 rows keep their
        # warm rolling windows from the fetched buffer)
        feat = feat[feat["date"] >= pd.Timestamp(START)]

        # drop warm-up rows (NaN features) and the final row (no next-day label)
        feat = feat.dropna(subset=FEATURE_COLS + ["label"]).copy()

        feat["split"] = feat["date"].apply(assign_split)
        panels.append(feat)

    if not panels:
        raise SystemExit("No data collected -- all tickers failed.")

    panel = pd.concat(panels, ignore_index=True)
    keep = ["ticker", "date"] + FEATURE_COLS + ["ret_next", "label", "split"]
    panel = panel[keep].sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["label"] = panel["label"].astype(int)
    panel.to_csv(OUT_FILE, index=False)

    summarise(panel, failed)


# ============================================================================
# SUMMARY
# ============================================================================

def summarise(panel, failed):
    print("\n" + "=" * 64)
    print("PRICE PANEL BUILT")
    print("=" * 64)
    print(f"Tickers collected : {panel['ticker'].nunique()} / {len(TICKERS)}")
    if failed:
        print(f"FAILED            : {failed}")
    print(f"Total rows        : {len(panel):,}")
    print(f"Date range        : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Features          : {len(FEATURE_COLS)}")

    print("\nClass balance (share of UP days):")
    overall = panel["label"].mean()
    print(f"  overall : {overall:.3f}")
    for sp in ["train", "val", "test"]:
        s = panel.loc[panel["split"] == sp, "label"]
        if len(s):
            print(f"  {sp:5s}   : {s.mean():.3f}   ({len(s):,} rows)")

    print("\nRows per ticker (min / median / max):")
    rpt = panel.groupby("ticker").size()
    print(f"  {rpt.min()} / {int(rpt.median())} / {rpt.max()}")
    thin = rpt[rpt < rpt.median() * 0.8]
    if len(thin):
        print(f"  thinner than usual: {dict(thin)}")

    print(f"\nWritten: {OUT_FILE}")
    print(f"Per-ticker raw OHLCV in: {PRICES_DIR}")


if __name__ == "__main__":
    collect()