#!/usr/bin/env python3
"""
FAANG assignment script: download hourly data for previous five days,
normalize Close prices per ticker (start=1.0), and plot all on one chart.
"""
import argparse
import datetime as dt
import os
from pathlib import Path

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

FAANG = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]


def timestamp_str() -> str:
    now = dt.datetime.utcnow()
    return now.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_data(tickers: list[str] = FAANG) -> Path:
    """Download all hourly data for previous five days for given tickers.
    Save one combined CSV with columns including Ticker and Close.
    Returns the Path to the saved CSV.
    """
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=5)

    dfs = []
    for t in tickers:
        # Explicitly set auto_adjust=False to keep standard OHLCV columns
        df = yf.download(
            t,
            interval="1h",
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            continue
        df = df.reset_index()
        df["Ticker"] = t
        # Normalize column names and keep essential columns
        if "Date" in df.columns and "Datetime" not in df.columns:
            df.rename(columns={"Date": "Datetime"}, inplace=True)
        keep = ["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"]
        df = df[[c for c in keep if c in df.columns]]
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No data downloaded. Check tickers or network.")

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(["Datetime", "Ticker"], inplace=True)

    data_dir = ensure_dir("data")
    out_path = data_dir / f"{timestamp_str()}.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def plot_data() -> Path:
    """Open latest CSV in data/, normalize Close per ticker and plot overlay.
    Saves PNG into plots/ and returns the Path.
    """
    data_dir = ensure_dir("data")
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files in data/. Run download first.")
    latest = files[-1]
    df = pd.read_csv(latest, parse_dates=["Datetime"])  # combined file

    # Normalize Close per ticker: ensure numeric, divide by first non-NA value
    df = df.copy()
    df.sort_values(["Ticker", "Datetime"], inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    def _normalize(series: pd.Series) -> pd.Series:
        base = series.dropna().iloc[0] if not series.dropna().empty else pd.NA
        return series / base if pd.notna(base) and base != 0 else pd.NA
    df["NormClose"] = df.groupby("Ticker")["Close"].transform(_normalize)

    plt.figure(figsize=(10, 6))
    for t, g in df.groupby("Ticker"):
        g = g.dropna(subset=["NormClose"])  # ensure numeric series for plotting
        if not g.empty:
            plt.plot(g["Datetime"], g["NormClose"], label=t)
    plt.xlabel("Time")
    plt.ylabel("Normalized Close (start=1.0)")
    # Title uses the date of the latest CSV
    title_date = latest.stem.split("-")[0]
    plt.title(f"FAANG Normalized Close — {title_date}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    plots_dir = ensure_dir("plots")
    out_path = plots_dir / f"{timestamp_str()}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="FAANG data download and plot")
    parser.add_argument("action", choices=["download", "plot", "all"], nargs="?", default="all")
    args = parser.parse_args()

    if args.action in ("download", "all"):
        csv_path = get_data()
        print(f"Saved data: {csv_path}")
    if args.action in ("plot", "all"):
        png_path = plot_data()
        print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
