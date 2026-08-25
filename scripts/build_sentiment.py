import os, glob, json, math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PRICE_FILE = os.path.join(BASE_DIR, "data", "config1_features.csv")
CACHE_FILE = os.path.join(BASE_DIR, "data", "finbert_cache.csv")   # resumable score cache
OUT_FILE = os.path.join(BASE_DIR, "data", "config2_features.csv")

TICKERS = [
    "AAPL","MSFT","NVDA","AMD","GOOGL","META","NFLX","DIS","AMZN","TSLA",
    "MCD","HD","JPM","BAC","WFC","MS","JNJ","PFE","MRK","LLY",
    "WMT","KO","PEP","PG","XOM","CVX","COP","BA","GE","NEM",
]
TARGET = set(TICKERS)

# sentiment aggregation knobs
HALF_LIFE_H = 48.0        # decay half-life (the meaningful knob)
WINDOW_H    = 100.0       # truncation of the faint tail
ANCHOR_HOUR_UTC = 21      # market-close anchor for each day (documented assumption)
STRICT_NO_SAMEDAY = False # True -> exclude all same-day news (anchor at start of day)
LAMBDA = math.log(2) / HALF_LIFE_H

# FinBERT batching (tuned for ~4.5 GB free on the RTX 4050)
BATCH_SIZE = 32
MAX_LEN = 256
CHECKPOINT_EVERY = 50     # flush cache every N batches


# ============================================================================
# STAGE 1 -- load + dedup
# ============================================================================
def _iter_articles(raw_dir):
    files = glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True)
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception:
            continue
        resp = obj.get("response") if isinstance(obj, dict) else None
        feed = resp.get("feed") if isinstance(resp, dict) else None
        if isinstance(feed, list):
            for art in feed:
                if isinstance(art, dict):
                    yield art


def parse_ts(s):
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%S")
    except Exception:
        return None


def load_and_dedup(raw_dir):
    """Return dict url -> {text, ts, tags:[(ticker, relevance, av_score)]}."""
    articles = {}
    seen_raw = 0
    for art in _iter_articles(raw_dir):
        seen_raw += 1
        # keep only articles that tag at least one of our 30
        tags = []
        for ts_ in art.get("ticker_sentiment", []) or []:
            tk = ts_.get("ticker")
            if tk in TARGET:
                try:
                    rel = float(ts_.get("relevance_score", 0.0))
                except (TypeError, ValueError):
                    rel = 0.0
                try:
                    avs = float(ts_.get("ticker_sentiment_score", 0.0))
                except (TypeError, ValueError):
                    avs = 0.0
                tags.append((tk, rel, avs))
        if not tags:
            continue

        url = art.get("url") or (str(art.get("title")) + str(art.get("time_published")))
        if url in articles:
            continue  # global dedup -- same story under multiple ticker slices

        ts = parse_ts(str(art.get("time_published", "")))
        if ts is None:
            continue
        text = (str(art.get("title", "")) + ". " + str(art.get("summary", ""))).strip()
        articles[url] = {"text": text, "ts": ts, "tags": tags}

    print(f"Stage 1: scanned {seen_raw:,} article records -> "
          f"{len(articles):,} unique articles touching the 30 tickers")
    return articles


# ============================================================================
# STAGE 2 -- FinBERT scoring (GPU, batched, resumable)
# ============================================================================
def load_cache(cache_file):
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        return dict(zip(df["url"], df["signed"]))
    return {}


def score_articles(articles, cache_file):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    done = load_cache(cache_file)
    todo = [(u, a["text"]) for u, a in articles.items() if u not in done]
    print(f"Stage 2: {len(done):,} already cached, {len(todo):,} to score")
    if not todo:
        return done

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: CUDA not available -- scoring on CPU (slow).")
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", use_safetensors=True)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()

    # resolve label indices robustly (don't hard-code order)
    lab2id = {k.lower(): v for k, v in model.config.label2id.items()}
    i_pos, i_neg = lab2id.get("positive", 0), lab2id.get("negative", 1)

    new_header = not os.path.exists(cache_file)
    fh = open(cache_file, "a", encoding="utf-8")
    if new_header:
        fh.write("url,signed\n")

    n_batches = math.ceil(len(todo) / BATCH_SIZE)
    with torch.no_grad():
        for b in range(n_batches):
            chunk = todo[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            urls = [u for u, _ in chunk]
            texts = [t for _, t in chunk]
            enc = tok(texts, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            signed = probs[:, i_pos] - probs[:, i_neg]
            for u, s in zip(urls, signed):
                done[u] = float(s)
                # escape commas in url just in case
                fh.write(f'"{u}",{s:.6f}\n')
            if (b + 1) % CHECKPOINT_EVERY == 0:
                fh.flush()
                print(f"   scored {min((b+1)*BATCH_SIZE, len(todo)):,}/{len(todo):,}")
    fh.close()
    print(f"Stage 2: done, {len(done):,} articles scored total")
    return done


# ============================================================================
# STAGE 3 -- fan out to per-ticker timestamped events
# ============================================================================
def build_events(articles, signed_by_url):
    """ticker -> arrays of (ts, finbert_signed, relevance, av_score), time-sorted."""
    rows = {t: [] for t in TICKERS}
    for url, a in articles.items():
        s = signed_by_url.get(url)
        if s is None:
            continue
        for (tk, rel, avs) in a["tags"]:
            rows[tk].append((a["ts"], s, rel, avs))
    events = {}
    for tk, lst in rows.items():
        lst.sort(key=lambda x: x[0])
        if lst:
            ts = np.array([x[0] for x in lst], dtype="datetime64[s]")
            fb = np.array([x[1] for x in lst], dtype=float)
            rl = np.array([x[2] for x in lst], dtype=float)
            av = np.array([x[3] for x in lst], dtype=float)
        else:
            ts = np.array([], dtype="datetime64[s]")
            fb = rl = av = np.array([], dtype=float)
        events[tk] = {"ts": ts, "fb": fb, "rl": rl, "av": av}
    return events


# ============================================================================
# STAGE 4 -- daily aggregation with decay + carry-forward, join to spine
# ============================================================================
def aggregate_ticker(ev, dates):
    """
    ev: dict of time-sorted arrays for one ticker.
    dates: sorted list of pandas Timestamps (that ticker's trading days).
    Returns DataFrame with one row per date.
    """
    ts = ev["ts"]
    out = []
    last_fb, last_av = 0.0, 0.0
    last_news_date = None

    for D in dates:
        if STRICT_NO_SAMEDAY:
            anchor = np.datetime64(pd.Timestamp(D.date())).astype("datetime64[s]")          # start of day D
        else:
            anchor = np.datetime64(pd.Timestamp(D.date()) + pd.Timedelta(hours=ANCHOR_HOUR_UTC)).astype("datetime64[s]")
        lo = anchor - np.timedelta64(int(WINDOW_H * 3600), "s")

        if ts.size:
            # window = (lo, anchor]
            hi_i = np.searchsorted(ts, anchor, side="right")
            lo_i = np.searchsorted(ts, lo, side="right")
            idx = slice(lo_i, hi_i)
            n = hi_i - lo_i
        else:
            n = 0

        if n > 0:
            sub_ts = ts[idx]
            hours_before = (anchor - sub_ts) / np.timedelta64(1, "s") / 3600.0
            decay = np.exp(-LAMBDA * hours_before)
            w = ev["rl"][idx] * decay
            wsum = w.sum()
            if wsum <= 0:
                fb_daily, av_daily = last_fb, last_av
                mass = 0.0
            else:
                fb_daily = float((w * ev["fb"][idx]).sum() / wsum)
                av_daily = float((w * ev["av"][idx]).sum() / wsum)
                mass = float(wsum)
            last_fb, last_av = fb_daily, av_daily
            last_news_date = D
            news_count = int(n)
        else:
            # quiet day: carry forward decayed prior value, flag via news_count=0
            fb_daily, av_daily, mass = last_fb, last_av, 0.0
            news_count = 0

        days_since = 0 if last_news_date is None else (D - last_news_date).days
        out.append((D, fb_daily, av_daily, news_count, mass, days_since))

    return pd.DataFrame(out, columns=[
        "date", "sent_finbert", "sent_av", "news_count_100h",
        "news_mass_decayed", "days_since_news"])


def main():
    print("Loading price spine:", PRICE_FILE)
    price = pd.read_csv(PRICE_FILE, parse_dates=["date"])

    articles = load_and_dedup(RAW_DIR)
    signed = score_articles(articles, CACHE_FILE)
    events = build_events(articles, signed)

    blocks = []
    for tk in TICKERS:
        dts = price.loc[price["ticker"] == tk, "date"].sort_values().tolist()
        if not dts:
            continue
        blk = aggregate_ticker(events[tk], dts)
        blk.insert(0, "ticker", tk)
        blocks.append(blk)
    sent = pd.concat(blocks, ignore_index=True)

    merged = price.merge(sent, on=["ticker", "date"], how="left")
    # any rows before a ticker's first-ever news: fill neutral + zero count
    for c, fill in [("sent_finbert", 0.0), ("sent_av", 0.0),
                    ("news_count_100h", 0), ("news_mass_decayed", 0.0),
                    ("days_since_news", 9999)]:
        merged[c] = merged[c].fillna(fill)

    merged.to_csv(OUT_FILE, index=False)
    summarise(merged, sent)


def summarise(merged, sent):
    print("\n" + "=" * 60)
    print("CONFIG 2 SENTIMENT BLOCK BUILT")
    print("=" * 60)
    print(f"Rows                : {len(merged):,}")
    quiet = (merged["news_count_100h"] == 0).mean()
    print(f"Quiet days (count=0): {quiet:.1%}")
    print(f"FinBERT sentiment   : mean {merged['sent_finbert'].mean():+.3f}  "
          f"sd {merged['sent_finbert'].std():.3f}")
    print(f"AV sentiment        : mean {merged['sent_av'].mean():+.3f}  "
          f"sd {merged['sent_av'].std():.3f}")
    # agreement between FinBERT and AV on days that actually had news
    live = merged[merged["news_count_100h"] > 0]
    if len(live) > 10:
        corr = live["sent_finbert"].corr(live["sent_av"])
        print(f"FinBERT vs AV corr  : {corr:+.3f}  (on {len(live):,} news days)")
    print(f"Median news_count   : {int(merged['news_count_100h'].median())}")
    print(f"\nWritten: {OUT_FILE}")


if __name__ == "__main__":
    main()