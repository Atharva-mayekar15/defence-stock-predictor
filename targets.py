import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/features.csv"
OUTPUT_FILE = "data/model_dataset.csv"

HORIZONS = {
    "1m": 21,
    "3m": 63,
    "6m": 126
}


# ============================================================
# LOAD FEATURES
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)

df = df.sort_values(
    ["stock", "date"]
).reset_index(drop=True)


# ============================================================
# CREATE FUTURE TARGETS
# ============================================================

print("Creating future-return targets...")


for name, days in HORIZONS.items():

    print(
        f"Creating {name} target ({days} trading days)..."
    )

    # --------------------------------------------------------
    # Future stock return (correctly grouped per stock)
    # --------------------------------------------------------

    future_stock_close = (
        df.groupby("stock")["close"]
        .shift(-days)
    )

    df[f"future_return_{name}"] = (
        future_stock_close /
        df["close"] -
        1
    )

    # --------------------------------------------------------
    # Future date (correctly grouped per stock)
    # --------------------------------------------------------

    df[f"future_date_{name}"] = (
        df.groupby("stock")["date"]
        .shift(-days)
    )

    # --------------------------------------------------------
    # Future NIFTY return
    #
    # CRITICAL FIX: compute on a standalone chronological
    # market series, then merge back by date.
    #
    # The previous code used df['nifty_close'].shift(-days)
    # on a df sorted by ['stock','date'].  At each stock
    # boundary, shift(-days) crossed into the next stock's
    # rows — producing fabricated NIFTY values from a
    # completely different point in time (e.g. -50% "returns"
    # because HAL's future NIFTY was pulling from MAZDOCK's
    # 2020 prices instead of 2026 prices).
    # --------------------------------------------------------

    market = (
        df[["date", "nifty_close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    market[f"nifty_future_close_{name}"] = (
        market["nifty_close"]
        .shift(-days)
    )

    market[f"nifty_future_return_{name}"] = (
        market[f"nifty_future_close_{name}"] /
        market["nifty_close"] -
        1
    )

    # Merge the correctly-computed NIFTY future return
    # back into the per-stock dataframe by date.
    df = df.merge(
        market[[
            "date",
            f"nifty_future_return_{name}"
        ]],
        on="date",
        how="left"
    )

    # --------------------------------------------------------
    # Relative future return
    #
    # Positive = stock outperformed NIFTY
    # Negative = stock underperformed NIFTY
    # --------------------------------------------------------

    df[f"relative_future_return_{name}"] = (
        df[f"future_return_{name}"] -
        df[f"nifty_future_return_{name}"]
    )


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("TARGET SUMMARY")
print("=" * 70)


for name in HORIZONS:

    stock_target = (
        f"future_return_{name}"
    )

    relative_target = (
        f"relative_future_return_{name}"
    )

    print(f"\n{name.upper()}:")

    print(
        f"Stock return target:"
        f" {stock_target}"
    )

    print(
        f"Relative return target:"
        f" {relative_target}"
    )

    print(
        f"Valid stock targets:"
        f" {df[stock_target].notna().sum():,}"
    )

    print(
        f"Valid relative targets:"
        f" {df[relative_target].notna().sum():,}"
    )

    if df[stock_target].notna().any():

        print(
            f"Stock return mean:"
            f" {df[stock_target].mean():.2%}"
        )

        print(
            f"Stock return median:"
            f" {df[stock_target].median():.2%}"
        )

        print(
            f"Relative return mean:"
            f" {df[relative_target].mean():.2%}"
        )

        print(
            f"Relative return median:"
            f" {df[relative_target].median():.2%}"
        )


# ============================================================
# CHECK FUTURE-DATE LOGIC
# ============================================================

print("\n" + "=" * 70)
print("TARGET DATE VALIDATION")
print("=" * 70)


sample = (
    df[
        [
            "stock",
            "date",
            "close",
            "future_date_1m",
            "future_return_1m",
            "relative_future_return_1m"
        ]
    ]
    .dropna(
        subset=["future_return_1m"]
    )
    .groupby("stock")
    .head(1)
)

print(
    sample.to_string(
        index=False
    )
)


# ============================================================
# CHECK TARGET LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("TARGET LEAKAGE CHECK")
print("=" * 70)

for name in HORIZONS:

    target_columns = [
        f"future_return_{name}",
        f"future_date_{name}",
        f"nifty_future_return_{name}",
        f"relative_future_return_{name}"
    ]

    print(
        f"{name}: "
        f"{target_columns}"
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
print("TARGET PIPELINE COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print(
    f"Final shape: {df.shape}"
)