# Download stock returns from Yahoo Finance and earthquake data from USGS

import io
import logging
import urllib.request
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def download_stock_returns(
    tickers=("^FCHI", "^GSPC"),
    start="2000-01-01",
    end="2024-01-01"):
    # Download daily log-returns for given tickers
    tickers = list(tickers)
    logger.info("Downloading price data for: %s", tickers)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False)

    returns = {}
    for ticker in tickers:
        try:
            # yfinance uses multi-level columns for multiple tickers
            if len(tickers) == 1:
                close = raw["Close"]
            else:
                close = raw["Close"][ticker]
        except KeyError:
            logger.warning("Ticker %s not found in downloaded data.", ticker)
            continue

        log_ret = np.log(close / close.shift(1)).dropna()
        log_ret.name = ticker
        returns[ticker] = log_ret
        logger.info(
            "  %s: %d daily returns (%.4f .. %.4f)",
            ticker, len(log_ret), log_ret.min(), log_ret.max())

    return returns


def download_usgs_catalog(
    min_magnitude=2.0,
    start="2000-01-01",
    end="2024-01-01"):
    # Download USGS earthquake catalog, cache result to CSV
    import os

    cache_path = os.path.join(os.path.dirname(__file__), "usgs_catalog.csv")
    if os.path.exists(cache_path):
        logger.info("Loading USGS catalog from cache: %s", cache_path)
        df = pd.read_csv(cache_path, parse_dates=["time"])
        return df

    logger.info(
        "Downloading USGS catalog: M>=%.1f from %s to %s",
        min_magnitude, start, end)

    # split by year to avoid USGS row limit per request
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    chunks = []

    current = start_dt
    while current < end_dt:
        next_year = min(current + pd.DateOffset(years=1), end_dt)
        params = {
            "format": "csv",
            "starttime": current.strftime("%Y-%m-%d"),
            "endtime": next_year.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "orderby": "time-asc"}
        url = _USGS_BASE_URL + "?" + urlencode(params)
        logger.info("  Fetching %s → %s", current.date(), next_year.date())
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                content = resp.read().decode("utf-8")
            chunk = pd.read_csv(io.StringIO(content))
            if not chunk.empty:
                chunks.append(chunk)
        except Exception as exc:
            logger.warning("  Failed to fetch chunk: %s", exc)
        current = next_year

    if not chunks:
        raise RuntimeError("No data downloaded from USGS.")

    df_raw = pd.concat(chunks, ignore_index=True)

    # keep only needed columns, rename for consistency
    col_map = {}
    for col in df_raw.columns:
        cl = col.lower()
        if cl == "time":
            col_map[col] = "time"
        elif cl in ("lat", "latitude"):
            col_map[col] = "latitude"
        elif cl in ("lon", "longitude"):
            col_map[col] = "longitude"
        elif cl in ("mag", "magnitude"):
            col_map[col] = "magnitude"

    df = df_raw.rename(columns=col_map)[["time", "latitude", "longitude", "magnitude"]]
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["magnitude"]).reset_index(drop=True)

    # save to cache
    df.to_csv(cache_path, index=False)
    logger.info("Saved USGS catalog to %s (%d events)", cache_path, len(df))

    return df
