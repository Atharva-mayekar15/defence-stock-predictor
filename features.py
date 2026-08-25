import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/raw_market_data.csv"
OUTPUT_FILE = "data/features.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading raw market data...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)

df = df.sort_values(
    ["stock", "date"]
).reset_index(drop=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_rsi(series, period=14):
    """
    Calculate RSI using Wilder-style exponential smoothing.
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (
        100 / (1 + rs)
    )


def calculate_atr(group, period=14):

    previous_close = group["close"].shift(1)

    tr1 = (
        group["high"] -
        group["low"]
    )

    tr2 = (
        group["high"] -
        previous_close
    ).abs()

    tr3 = (
        group["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


# ============================================================
# STOCK FEATURE ENGINEERING
# ============================================================

print("Creating stock-level features...")


grouped = df.groupby(
    "stock",
    group_keys=False
)


# ------------------------------------------------------------
# Returns
# ------------------------------------------------------------

for days in [1, 3, 5, 10, 20, 60]:

    df[f"return_{days}d"] = grouped[
        "close"
    ].pct_change(
        days,
        fill_method=None
    )


# ------------------------------------------------------------
# Intraday behaviour
# ------------------------------------------------------------

df["intraday_return"] = (
    df["close"] /
    df["open"] -
    1
)

df["high_low_range"] = (
    df["high"] -
    df["low"]
) / df["close"]

df["close_position"] = (
    df["close"] -
    df["low"]
) / (
    df["high"] -
    df["low"]
).replace(0, np.nan)


# ------------------------------------------------------------
# Moving averages
# ------------------------------------------------------------

for period in [10, 20, 50, 100, 200]:

    df[f"ma_{period}"] = grouped[
        "close"
    ].transform(
        lambda x:
        x.rolling(
            period,
            min_periods=period
        ).mean()
    )


# ------------------------------------------------------------
# Price relative to moving averages
# ------------------------------------------------------------

for period in [10, 20, 50, 100, 200]:

    df[f"price_vs_ma{period}"] = (
        df["close"] /
        df[f"ma_{period}"] -
        1
    )


# ------------------------------------------------------------
# Moving-average relationships
# ------------------------------------------------------------

df["ma10_vs_ma50"] = (
    df["ma_10"] /
    df["ma_50"] -
    1
)

df["ma20_vs_ma50"] = (
    df["ma_20"] /
    df["ma_50"] -
    1
)

df["ma50_vs_ma200"] = (
    df["ma_50"] /
    df["ma_200"] -
    1
)


# ------------------------------------------------------------
# Moving-average slopes
# ------------------------------------------------------------

df["ma10_slope"] = grouped[
    "ma_10"
].pct_change(
    5,
    fill_method=None
)

df["ma20_slope"] = grouped[
    "ma_20"
].pct_change(
    10,
    fill_method=None
)

df["ma50_slope"] = grouped[
    "ma_50"
].pct_change(
    20,
    fill_method=None
)


# ------------------------------------------------------------
# Momentum / ROC
# ------------------------------------------------------------

for days in [5, 10, 20, 60]:

    df[f"roc_{days}"] = (
        df["close"] /
        grouped["close"].shift(days) -
        1
    )


# ------------------------------------------------------------
# Volatility
# ------------------------------------------------------------

for period in [5, 10, 20, 60]:

    df[f"volatility_{period}d"] = grouped[
        f"return_1d"
    ].transform(
        lambda x:
        x.rolling(
            period,
            min_periods=period
        ).std()
    )


# ------------------------------------------------------------
# Volume features
# ------------------------------------------------------------

for period in [5, 20, 60]:

    volume_ma = grouped[
        "volume"
    ].transform(
        lambda x:
        x.rolling(
            period,
            min_periods=period
        ).mean()
    )

    df[f"volume_ratio_{period}"] = (
        df["volume"] /
        volume_ma
    )

df["volume_change_1d"] = (
    grouped["volume"]
    .pct_change(
        1,
        fill_method=None
    )
)


# ============================================================
# RSI
# ============================================================

print("Calculating RSI...")

df["rsi_7"] = grouped[
    "close"
].transform(
    lambda x:
    calculate_rsi(x, 7)
)

df["rsi_14"] = grouped[
    "close"
].transform(
    lambda x:
    calculate_rsi(x, 14)
)

df["rsi_change"] = (
    df["rsi_14"] -
    grouped["rsi_14"].shift(5)
)


# ============================================================
# MACD
# ============================================================

print("Calculating MACD...")

df["ema_12"] = grouped[
    "close"
].transform(
    lambda x:
    x.ewm(
        span=12,
        adjust=False
    ).mean()
)

df["ema_26"] = grouped[
    "close"
].transform(
    lambda x:
    x.ewm(
        span=26,
        adjust=False
    ).mean()
)

df["macd"] = (
    df["ema_12"] -
    df["ema_26"]
)

df["macd_signal"] = grouped[
    "macd"
].transform(
    lambda x:
    x.ewm(
        span=9,
        adjust=False
    ).mean()
)

df["macd_histogram"] = (
    df["macd"] -
    df["macd_signal"]
)


# ============================================================
# BOLLINGER BANDS
# ============================================================

print("Calculating Bollinger Bands...")

bb_middle = df["ma_20"]

bb_std = grouped[
    "close"
].transform(
    lambda x:
    x.rolling(
        20,
        min_periods=20
    ).std()
)

bb_upper = (
    bb_middle +
    2 * bb_std
)

bb_lower = (
    bb_middle -
    2 * bb_std
)

df["bb_width"] = (
    bb_upper -
    bb_lower
) / bb_middle

df["bb_position"] = (
    df["close"] -
    bb_lower
) / (
    bb_upper -
    bb_lower
).replace(0, np.nan)


# ============================================================
# ATR
# ============================================================

print("Calculating ATR...")

df["atr_14"] = (
    df.groupby(
        "stock",
        group_keys=False
    )
    .apply(
        lambda group:
        calculate_atr(group, 14),
        include_groups=False
    )
    .reset_index(
        level=0,
        drop=True
    )
)

# Ensure correct alignment after groupby/apply
df["atr_percent"] = (
    df["atr_14"] /
    df["close"]
)


# ============================================================
# BREAKOUT / DRAWDOWN FEATURES
# ============================================================

rolling_high = grouped[
    "high"
].transform(
    lambda x:
    x.shift(1).rolling(
        20,
        min_periods=20
    ).max()
)

rolling_low = grouped[
    "low"
].transform(
    lambda x:
    x.shift(1).rolling(
        20,
        min_periods=20
    ).min()
)

df["breakout_20d"] = (
    df["close"] >
    rolling_high
).astype(int)

df["breakdown_20d"] = (
    df["close"] <
    rolling_low
).astype(int)


df["distance_from_20d_high"] = (
    df["close"] /
    rolling_high -
    1
)

df["distance_from_20d_low"] = (
    df["close"] /
    rolling_low -
    1
)


# ============================================================
# MARKET FEATURES
# ============================================================

print("Creating market-context features...")


# ------------------------------------------------------------
# Build a clean standalone market dataframe
# ------------------------------------------------------------

market = (
    df[
        [
            "date",
            "nifty_close",
            "vix_close"
        ]
    ]
    .drop_duplicates("date")
    .sort_values("date")
    .reset_index(drop=True)
)

# Forward-fill missing NIFTY and VIX values.
#
# Gaps in Yahoo Finance market data are data quality issues
# rather than genuine market closures.  Forward-filling with
# the previous available value is the standard treatment for
# daily market data: the last known level is the best estimate
# for a missing day.
#
# Without this, NaN cascades into all derived market features
# (vix_change_1d/5d/20d, nifty_return_Xd, etc.) for the
# affected dates, causing those rows to be silently dropped
# later by dropna — potentially removing ALL recent data if
# Yahoo Finance has a data gap in the last few months.

n_nifty_missing = market["nifty_close"].isna().sum()
n_vix_missing = market["vix_close"].isna().sum()

if n_nifty_missing > 0:
    print(
        f"  Forward-filling {n_nifty_missing} "
        f"missing NIFTY values."
    )
    market["nifty_close"] = market["nifty_close"].ffill()

if n_vix_missing > 0:
    print(
        f"  Forward-filling {n_vix_missing} "
        f"missing VIX values."
    )
    market["vix_close"] = market["vix_close"].ffill()


# ------------------------------------------------------------
# NIFTY returns
# ------------------------------------------------------------

for days in [1, 5, 10, 20, 60]:

    market[f"nifty_return_{days}d"] = (
        market["nifty_close"]
        .pct_change(
            days,
            fill_method=None
        )
    )


# ------------------------------------------------------------
# NIFTY moving averages
# ------------------------------------------------------------

for period in [20, 50, 200]:

    market[f"nifty_ma_{period}"] = (
        market["nifty_close"]
        .rolling(
            period,
            min_periods=period
        )
        .mean()
    )


# ------------------------------------------------------------
# NIFTY relative to moving averages
# ------------------------------------------------------------

market["nifty_vs_ma20"] = (
    market["nifty_close"] /
    market["nifty_ma_20"] -
    1
)

market["nifty_vs_ma50"] = (
    market["nifty_close"] /
    market["nifty_ma_50"] -
    1
)

market["nifty_vs_ma200"] = (
    market["nifty_close"] /
    market["nifty_ma_200"] -
    1
)


# ------------------------------------------------------------
# NIFTY volatility
# ------------------------------------------------------------

nifty_daily_return = (
    market["nifty_close"]
    .pct_change(
        1,
        fill_method=None
    )
)

market["nifty_volatility_20d"] = (
    nifty_daily_return
    .rolling(
        20,
        min_periods=20
    )
    .std()
)

market["nifty_volatility_60d"] = (
    nifty_daily_return
    .rolling(
        60,
        min_periods=60
    )
    .std()
)


# ------------------------------------------------------------
# VIX features
# ------------------------------------------------------------

market["vix_change_1d"] = (
    market["vix_close"] /
    market["vix_close"].shift(1) -
    1
)

market["vix_change_5d"] = (
    market["vix_close"] /
    market["vix_close"].shift(5) -
    1
)

market["vix_change_20d"] = (
    market["vix_close"] /
    market["vix_close"].shift(20) -
    1
)

market["vix_ma20"] = (
    market["vix_close"]
    .rolling(
        20,
        min_periods=20
    )
    .mean()
)

market["vix_vs_ma20"] = (
    market["vix_close"] /
    market["vix_ma20"] -
    1
)


# ------------------------------------------------------------
# Market regime
# ------------------------------------------------------------

market["market_bullish"] = (
    (
        market["nifty_close"] >
        market["nifty_ma_200"]
    ) &
    (
        market["nifty_ma_50"] >
        market["nifty_ma_200"]
    )
).astype(int)


market["market_above_ma50"] = (
    market["nifty_close"] >
    market["nifty_ma_50"]
).astype(int)


# ------------------------------------------------------------
# Remove temporary market columns
# ------------------------------------------------------------

market = market.drop(
    columns=[
        "nifty_ma_20",
        "nifty_ma_50",
        "nifty_ma_200",
        "vix_ma20"
    ]
)


# ------------------------------------------------------------
# Remove old market-feature columns from df
# ------------------------------------------------------------

market_feature_columns = [
    col
    for col in market.columns
    if col != "date"
]

df = df.drop(
    columns=[
        col
        for col in market_feature_columns
        if col in df.columns
    ],
    errors="ignore"
)


# ------------------------------------------------------------
# Merge clean market features by DATE
# ------------------------------------------------------------

df = df.merge(
    market,
    on="date",
    how="left"
)


# ============================================================
# RELATIVE STRENGTH VS NIFTY
# ============================================================

print("Calculating relative strength...")


for days in [5, 20, 60]:

    df[f"relative_return_{days}d"] = (
        df[f"return_{days}d"] -
        df[f"nifty_return_{days}d"]
    )

# ============================================================
# CLEAN TEMPORARY COLUMNS
# ============================================================

temporary_columns = [
    "ema_12",
    "ema_26",
    "nifty_ma_20",
    "nifty_ma_50",
    "nifty_ma_200",
    "vix_ma20"
]

df = df.drop(
    columns=[
        col
        for col in temporary_columns
        if col in df.columns
    ]
)


# ============================================================
# CLEAN INF VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


feature_columns = [
    col
    for col in df.columns
    if col not in [
        "date",
        "stock",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "nifty_close",
        "vix_close"
    ]
]


print(
    f"Feature count: {len(feature_columns)}"
)


# ============================================================
# FEATURE MISSING VALUES
# ============================================================

print("\nMissing values in features:")

missing_features = (
    df[feature_columns]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    missing_features[
        missing_features > 0
    ].to_string()
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("FEATURE DATASET SAVED")
print("=" * 70)

print(
    f"File: {OUTPUT_FILE}"
)

print(
    f"Shape: {df.shape}"
)