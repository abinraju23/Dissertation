from __future__ import annotations

import os
import sys
import json
import datetime as dt
from pathlib import Path

import config
from newsclient import AVNewsClient, RateLimited, AlphaVantageError


# --------------------------------------------------------------------------
# Window generation
# --------------------------------------------------------------------------
def month_windows(start: dt.datetime, end: dt.datetime):
    """Yield (slice_start, slice_end) month by month, inclusive of `end`."""
    cur = start
    while cur < end:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            nxt = cur.replace(month=cur.month + 1, day=1)
        slice_end = min(nxt, end)
        yield cur, slice_end
        cur = nxt


def week_windows(start: dt.datetime, end: dt.datetime):
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=7), end)
        yield cur, nxt
        cur = nxt


def windows(start: dt.datetime, end: dt.datetime):
    if config.COLLECTION_WINDOW == "weekly":
        yield from week_windows(start, end)
    else:
        yield from month_windows(start, end)


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def slice_path(ticker: str, start: dt.datetime, end: dt.datetime) -> Path:
    folder = config.RAW_NEWS_DIR / ticker
    name = f"{ticker}__{start:%Y%m%dT%H%M}__{end:%Y%m%dT%H%M}.json"
    return folder / name


def split_marker_path(ticker: str, start: dt.datetime, end: dt.datetime) -> Path:
    """Marker left when a window was bisected, so resume costs no extra calls."""
    folder = config.RAW_NEWS_DIR / ticker
    name = f"{ticker}__{start:%Y%m%dT%H%M}__{end:%Y%m%dT%H%M}.split"
    return folder / name


def split_point(start: dt.datetime, end: dt.datetime) -> dt.datetime:
    """Deterministic midpoint (day-aligned) so split/resume agree exactly."""
    mid = start + (end - start) / 2
    mid = mid.replace(hour=0, minute=0, second=0, microsecond=0)
    if mid <= start:
        mid = start + dt.timedelta(days=1)
    return mid


def save_slice(path: Path, ticker: str, start: dt.datetime,
               end: dt.datetime, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "source": "alphavantage",
        "ticker": ticker,
        "query": {
            "time_from": f"{start:%Y%m%dT%H%M}",
            "time_to": f"{end:%Y%m%dT%H%M}",
            "sort": "EARLIEST",
            "limit": config.AV_MAX_LIMIT,
        },
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "item_count": len(AVNewsClient.feed_items(data)),
        "response": data,
    }
    tmp = _tmp_for(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)
    # Atomic rename: the final file only ever appears in its complete form, so a
    # crash mid-write can't leave a corrupt slice that breaks a later resume.
    os.replace(tmp, path)


def _tmp_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


# --------------------------------------------------------------------------
# Fetch one slice, splitting if it hits the cap
# --------------------------------------------------------------------------
def fetch_slice(client: AVNewsClient, ticker: str,
                start: dt.datetime, end: dt.datetime, depth: int = 0) -> int:
    """Fetch one time slice; bisect on the 1000 cap. Returns items saved."""
    path = slice_path(ticker, start, end)
    marker = split_marker_path(ticker, start, end)
    indent = "    " + "  " * depth

    if path.exists():
        # Resume: already have this slice as a leaf. If the file is somehow
        # unreadable (e.g. a partial write from an older run), drop it and
        # re-fetch below rather than crashing the whole sweep.
        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f).get("item_count", 0)
        except (json.JSONDecodeError, OSError):
            print(f"{indent}WARNING: unreadable slice {path.name}, re-fetching")
            path.unlink(missing_ok=True)

    if marker.exists():
        # Resume: this window was split before -> recurse into its halves,
        # which are already on disk, so no API call is spent here.
        mid = split_point(start, end)
        return (fetch_slice(client, ticker, start, mid, depth + 1)
                + fetch_slice(client, ticker, mid, end, depth + 1))

    data = client.fetch_news(ticker, time_from=start, time_to=end,
                             sort="EARLIEST", limit=config.AV_MAX_LIMIT)
    items = AVNewsClient.feed_items(data)

    if len(items) >= config.AV_MAX_LIMIT and (end - start) > dt.timedelta(days=1):
        # Possibly truncated -> split this window in half by time and recurse.
        mid = split_point(start, end)
        print(f"{indent}{start:%Y-%m-%d}..{end:%Y-%m-%d} hit cap, splitting")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("split", encoding="utf-8")
        left = fetch_slice(client, ticker, start, mid, depth + 1)
        right = fetch_slice(client, ticker, mid, end, depth + 1)
        return left + right

    if len(items) >= config.AV_MAX_LIMIT:
        print(f"{indent}WARNING: {ticker} {start:%Y-%m-%d} single-day slice at "
              f"cap; some articles may be missed.")

    save_slice(path, ticker, start, end, data)
    return len(items)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def collect(tickers: list[str]) -> None:
    client = AVNewsClient()
    start = dt.datetime.strptime(config.WINDOW_START, "%Y-%m-%d")
    end = dt.datetime.strptime(config.WINDOW_END, "%Y-%m-%d")

    grand_total = 0
    for ticker in tickers:
        ticker_total = 0
        print(f"\n=== {ticker} ({config.TICKER_SECTOR.get(ticker, '?')}) ===")
        for ws, we in windows(start, end):
            try:
                n = fetch_slice(client, ticker, ws, we)
                ticker_total += n
                print(f"  {ws:%Y-%m} -> {n} items")
            except RateLimited as e:
                print(f"  RATE LIMITED on {ws:%Y-%m}: {e}")
                print("  Stopping. Re-run later; saved slices will be skipped.")
                return
            except AlphaVantageError as e:
                print(f"  API ERROR on {ws:%Y-%m}: {e} -- skipping slice")
        print(f"  {ticker} total: {ticker_total} items")
        grand_total += ticker_total

    print(f"\nDone. {grand_total} items across {len(tickers)} tickers.")
    print(f"Raw data under: {config.RAW_NEWS_DIR}")


if __name__ == "__main__":
    chosen = sys.argv[1:] if len(sys.argv) > 1 else config.CANDIDATE_TICKERS
    collect(chosen)