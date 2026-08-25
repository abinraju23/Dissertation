from __future__ import annotations

import csv
import datetime as dt

import config
from newsclient import AVNewsClient, RateLimited, AlphaVantageError


def parse_window() -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime.strptime(config.WINDOW_START, "%Y-%m-%d")
    end = dt.datetime.strptime(config.WINDOW_END, "%Y-%m-%d")
    return start, end


def earliest_date(items: list) -> str | None:
    """Pull the earliest time_published (YYYYMMDDTHHMMSS) from a feed."""
    stamps = [it.get("time_published") for it in items if it.get("time_published")]
    return min(stamps) if stamps else None


def main() -> None:
    client = AVNewsClient()
    start, end = parse_window()

    config.PROBE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"Probing {len(config.CANDIDATE_TICKERS)} candidates "
          f"over {config.WINDOW_START} to {config.WINDOW_END}\n")

    for ticker in config.CANDIDATE_TICKERS:
        sector = config.TICKER_SECTOR.get(ticker, "?")
        try:
            data = client.fetch_news(ticker, time_from=start, time_to=end,
                                     sort="EARLIEST", limit=config.AV_MAX_LIMIT)
            items = AVNewsClient.feed_items(data)
            count = len(items)
            earliest = earliest_date(items)
            hit_cap = count >= config.AV_MAX_LIMIT
            status = "OK"
        except RateLimited as e:
            count, earliest, hit_cap, status = 0, None, False, f"RATE_LIMIT: {e}"
        except AlphaVantageError as e:
            count, earliest, hit_cap, status = 0, None, False, f"API_ERROR: {e}"

        earliest_short = earliest[:8] if earliest else "-"
        flag = "" if (count >= config.MIN_ARTICLES or hit_cap) else "  <-- THIN"
        print(f"  {ticker:6s} {sector:24s} "
              f"items={count:4d}  cap={'Y' if hit_cap else 'n'}  "
              f"earliest={earliest_short}  {status}{flag}")

        rows.append({
            "ticker": ticker,
            "sector": sector,
            "items_in_call": count,
            "hit_1000_cap": hit_cap,
            "earliest_article": earliest_short,
            "status": status,
        })

    with config.PROBE_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved triage to {config.PROBE_OUTPUT}")
    print("Tickers flagged THIN are below the article bar and not capped -- "
          "review before locking the final 30.")


if __name__ == "__main__":
    main()