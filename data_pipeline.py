import os
import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

STOCKS = {
    "HAL": "HAL.NS",
    "BEL": "BEL.NS",
    "BDL": "BDL.NS",
    "MAZDOCK": "MAZDOCK.NS",
    "COCHINSHIP": "COCHINSHIP.NS",
    "GRSE": "GRSE.NS",
    "PARAS": "PARAS.NS"
}

MARKET_SYMBOLS = {
    "NIFTY": "^NSEI",
    "VIX": "^INDIAVIX"
}

START_DATE = "2010-01-01"
END_DATE = None

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_symbol(symbol, start_date=START_DATE):
    """
    Download daily OHLCV data from Yahoo Finance.
    """

    print(f"Downloading {symbol}...")

    data = yf.download(
        symbol,
        start=start_date,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for {symbol}"
        )

    # Handle Yahoo Finance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Standardize column names
    data.columns = [
        str(col).lower().replace(" ", "_")
        for col in data.columns
    ]

    # Required columns
    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: Missing columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"]
    ).dt.tz_localize(None)

    data = data[
        required
    ].copy()

    data = data.sort_values(
        "date"
    ).drop_duplicates(
        "date"
    )

    return data


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

stock_frames = []


for stock_name, ticker in STOCKS.items():

    stock = download_symbol(ticker)

    stock["stock"] = stock_name
    stock["ticker"] = ticker

    stock_frames.append(stock)


stocks_df = pd.concat(
    stock_frames,
    ignore_index=True
)


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

print("\nDownloading market data...")


nifty = download_symbol(
    MARKET_SYMBOLS["NIFTY"]
)

vix = download_symbol(
    MARKET_SYMBOLS["VIX"]
)


# ============================================================
# PREPARE NIFTY
# ============================================================

nifty = nifty[
    [
        "date",
        "close"
    ]
].rename(
    columns={
        "close": "nifty_close"
    }
)


# ============================================================
# PREPARE VIX
# ============================================================

vix = vix[
    [
        "date",
        "close"
    ]
].rename(
    columns={
        "close": "vix_close"
    }
)


# ============================================================
# MERGE MARKET DATA
# ============================================================

market_df = pd.merge(
    nifty,
    vix,
    on="date",
    how="outer"
).sort_values(
    "date"
)


# ============================================================
# MERGE MARKET DATA INTO STOCK DATA
# ============================================================

df = stocks_df.merge(
    market_df,
    on="date",
    how="left"
)


# ============================================================
# SORT FINAL DATASET
# ============================================================

df = df.sort_values(
    ["stock", "date"]
).reset_index(
    drop=True
)


# ============================================================
# BASIC CLEANING
# ============================================================

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "nifty_close",
    "vix_close"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"Total rows: {len(df):,}"
)

print(
    f"Stocks: {df['stock'].nunique()}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print("\nRows by stock:")

print(
    df.groupby("stock")
      .agg(
          rows=("date", "size"),
          start=("date", "min"),
          end=("date", "max")
      )
      .to_string()
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

missing = (
    df[
        numeric_columns
    ]
    .isna()
    .sum()
)

print(
    missing[
        missing > 0
    ].to_string()
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates = df.duplicated(
    subset=[
        "stock",
        "date"
    ]
).sum()

print(
    f"\nDuplicate stock/date rows: "
    f"{duplicates}"
)


# ============================================================
# SAVE CLEAN DATA
# ============================================================

output_file = os.path.join(
    DATA_DIR,
    "raw_market_data.csv"
)

df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("DATA PIPELINE COMPLETE")
print("=" * 70)

print(
    f"Saved dataset to: {output_file}"
)

print(
    f"Final shape: {df.shape}"
)