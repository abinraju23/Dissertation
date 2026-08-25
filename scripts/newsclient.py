from __future__ import annotations

import os
import time
import datetime as dt

import requests

import config


class AlphaVantageError(Exception):
    """A genuine API error (bad parameter, unknown function, etc.)."""


class RateLimited(Exception):
    """Alpha Vantage returned a rate-limit / informational note."""


def to_av_timestamp(d: dt.datetime) -> str:
    """Format a datetime as the YYYYMMDDTHHMM string the API expects."""
    return d.strftime("%Y%m%dT%H%M")


class AVNewsClient:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None,
                 min_interval: float = config.MIN_REQUEST_INTERVAL,
                 session: requests.Session | None = None):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key. Set the ALPHAVANTAGE_API_KEY environment variable."
            )
        self.min_interval = min_interval
        self._last_call = 0.0
        self.session = session or requests.Session()

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    def fetch_news(self, tickers: str,
                   time_from: dt.datetime | None = None,
                   time_to: dt.datetime | None = None,
                   sort: str = "EARLIEST",
                   limit: int = config.AV_MAX_LIMIT,
                   topics: str | None = None) -> dict:
        """
        Call NEWS_SENTIMENT once and return the parsed JSON body.

        `tickers` is a single ticker or a comma-separated string. We query one
        ticker at a time so each article is cleanly attributed and the per-ticker
        counts are unambiguous.
        """
        self._throttle()
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "tickers": tickers,
            "sort": sort,
            "limit": str(limit),
        }
        if time_from is not None:
            params["time_from"] = to_av_timestamp(time_from)
        if time_to is not None:
            params["time_to"] = to_av_timestamp(time_to)
        if topics is not None:
            params["topics"] = topics

        resp = self.session.get(self.BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Alpha Vantage signals problems inside a 200 response, not via status code.
        if "Information" in data:
            raise RateLimited(str(data["Information"]))
        if "Note" in data:
            raise RateLimited(str(data["Note"]))
        if "Error Message" in data:
            raise AlphaVantageError(str(data["Error Message"]))

        return data

    @staticmethod
    def feed_items(data: dict) -> list:
        """Return the article list from a response (empty list if none)."""
        return data.get("feed", []) or []