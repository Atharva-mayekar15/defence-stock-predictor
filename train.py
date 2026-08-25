import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import spearmanr

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/model_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_END = "2023-12-31"
VALIDATION_END = "2025-12-31"

HORIZONS = {
    "1M": "future_return_1m",
    "3M": "future_return_3m",
    "6M": "future_return_6m"
}

RELATIVE_HORIZONS = {
    "1M": "relative_future_return_1m",
    "3M": "relative_future_return_3m",
    "6M": "relative_future_return_6m"
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading model dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "date",
        "future_date_1m",
        "future_date_3m",
        "future_date_6m"
    ]
)

df = df.sort_values(
    ["date", "stock"]
).reset_index(drop=True)

# Add one-hot encoded stock identities
stock_dummies = pd.get_dummies(df["stock"], prefix="stock", dtype=int)
df = pd.concat([df, stock_dummies], axis=1)


# ============================================================
# FEATURE SELECTION
# ============================================================

# Explicitly define columns that are NEVER allowed to become
# model features.

IDENTIFIER_COLUMNS = [
    "date",
    "stock",
    "ticker"
]

RAW_PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

MARKET_RAW_COLUMNS = [
    "nifty_close",
    "vix_close"
]

# Raw moving-average price levels encode the stock's absolute price,
# which is not comparable across stocks with very different levels
# (e.g. HAL ~4000 vs PARAS ~800).  The normalized equivalents
# (price_vs_ma10, ma10_vs_ma50, etc.) are already in the feature set.
# atr_14 is excluded in favour of atr_percent (already present).
RAW_NONSTATIONARY_COLUMNS = [
    "ma_10",
    "ma_20",
    "ma_50",
    "ma_100",
    "ma_200",
    "atr_14"
]

FUTURE_COLUMNS = [
    col
    for col in df.columns
    if (
        col.startswith("future_") or
        col.startswith("relative_future_") or
        col.startswith("nifty_future_")
    )
]

FEATURE_EXCLUSIONS = (
    IDENTIFIER_COLUMNS +
    RAW_PRICE_COLUMNS +
    MARKET_RAW_COLUMNS +
    RAW_NONSTATIONARY_COLUMNS +
    FUTURE_COLUMNS
)

FEATURES = [
    col
    for col in df.columns
    if col not in FEATURE_EXCLUSIONS
]


print("\n" + "=" * 70)
print("FEATURE SET")
print("=" * 70)

print(
    f"Number of features: {len(FEATURES)}"
)

for i, feature in enumerate(
    FEATURES,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# VERIFY NO TARGET LEAKAGE
# ============================================================

leakage_columns = [
    feature
    for feature in FEATURES
    if (
        "future" in feature.lower()
        or feature.startswith("relative_future")
    )
]

if leakage_columns:

    raise RuntimeError(
        "TARGET LEAKAGE DETECTED:\n"
        + "\n".join(leakage_columns)
    )

print(
    "\nLeakage check: PASSED"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def evaluate_predictions(
    y_true,
    predictions
):
    """
    Compute regression + directional metrics.
    Returns a dict of scalar metrics.
    """

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    direction = (
        np.sign(predictions) ==
        np.sign(y_true)
    ).mean()

    if np.std(predictions) > 0 and np.std(np.array(y_true)) > 0:
        correlation = np.corrcoef(
            y_true,
            predictions
        )[0, 1]
    else:
        correlation = float("nan")

    spearman = spearmanr(
        y_true,
        predictions,
        nan_policy="omit"
    ).statistic

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Direction": direction,
        "Correlation": correlation,
        "Spearman": spearman
    }


def evaluate_per_stock(
    data,
    mask,
    predictions,
    target
):
    """
    Break down directional accuracy and Spearman correlation
    by individual stock for the rows selected by `mask`.
    """

    subset = data.loc[mask].copy()
    subset["_pred"] = predictions

    rows = []

    for stock, group in subset.groupby("stock"):

        if len(group) < 5:
            continue

        direction = (
            np.sign(group["_pred"]) ==
            np.sign(group[target])
        ).mean()

        sp = spearmanr(
            group[target],
            group["_pred"],
            nan_policy="omit"
        ).statistic

        rows.append({
            "Stock": stock,
            "N": len(group),
            "Dir Acc": direction,
            "Spearman": sp
        })

    return pd.DataFrame(rows)


def print_metrics(
    name,
    metrics
):

    print(
        f"\n{name}"
    )

    print(
        f"MAE:                  "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"RMSE:                 "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"R²:                   "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"Directional Accuracy: "
        f"{metrics['Direction']:.2%}"
    )

    print(
        f"Correlation:          "
        f"{metrics['Correlation']:.4f}"
    )

    print(
        f"Spearman Rank Corr:   "
        f"{metrics['Spearman']:.4f}"
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []


for horizon, target in RELATIVE_HORIZONS.items():

    print("\n\n" + "=" * 90)
    print(
        f"RELATIVE RETURN TARGET: {horizon}"
    )
    print(
        f"Target: {target}"
    )
    print("=" * 90)


    # --------------------------------------------------------
    # Target-specific dataset
    # --------------------------------------------------------

    data = df[
        FEATURES + [
            "date",
            "stock",
            target
        ]
    ].copy()


    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=FEATURES + [target]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    train_mask = (
        data["date"] <= TRAIN_END
    )

    validation_mask = (
        (data["date"] > TRAIN_END) &
        (data["date"] <= VALIDATION_END)
    )

    test_mask = (
        data["date"] > VALIDATION_END
    )


    X_train = data.loc[
        train_mask,
        FEATURES
    ]

    y_train = data.loc[
        train_mask,
        target
    ]

    X_val = data.loc[
        validation_mask,
        FEATURES
    ]

    y_val = data.loc[
        validation_mask,
        target
    ]

    X_test = data.loc[
        test_mask,
        FEATURES
    ]

    y_test = data.loc[
        test_mask,
        target
    ]


    print(
        f"\nTraining samples:   {len(X_train):,}"
    )

    print(
        f"Validation samples: {len(X_val):,}"
    )

    print(
        f"Test samples:       {len(X_test):,}"
    )

    # Guard: skip this horizon if train or val is empty.
    if len(X_train) == 0 or len(X_val) == 0:
        print(
            f"\n[SKIP] {horizon}: insufficient "
            f"training or validation data."
        )
        continue

    # Note when test is empty (e.g. all recent rows have
    # NaN targets because there are not enough future
    # trading days in the dataset).
    has_test = len(X_test) > 0
    if not has_test:
        print(
            "\n[NOTE] No test samples available "
            "(recent rows lack future targets). "
            "Test metrics will be skipped."
        )


    # ========================================================
    # MODEL 1 — MEAN BASELINE
    # ========================================================

    mean_model = DummyRegressor(
        strategy="mean"
    )

    mean_model.fit(
        X_train,
        y_train
    )

    mean_val_pred = mean_model.predict(X_val)

    mean_val_metrics = evaluate_predictions(
        y_val,
        mean_val_pred
    )

    mean_test_metrics = (
        evaluate_predictions(
            y_test,
            mean_model.predict(X_test)
        )
        if has_test else None
    )

    print_metrics(
        "MEAN BASELINE — VALIDATION",
        mean_val_metrics
    )

    if mean_test_metrics:
        print_metrics(
            "MEAN BASELINE — TEST",
            mean_test_metrics
        )


    # ========================================================
    # MODEL 2 — RIDGE
    # ========================================================

    ridge_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(
                    alpha=10.0
                )
            )
        ]
    )


    ridge_model.fit(
        X_train,
        y_train
    )


    ridge_val_pred = ridge_model.predict(X_val)

    ridge_val_metrics = evaluate_predictions(
        y_val,
        ridge_val_pred
    )

    ridge_test_metrics = (
        evaluate_predictions(
            y_test,
            ridge_model.predict(X_test)
        )
        if has_test else None
    )

    print_metrics(
        "RIDGE — VALIDATION",
        ridge_val_metrics
    )

    if ridge_test_metrics:
        print_metrics(
            "RIDGE — TEST",
            ridge_test_metrics
        )


    # ========================================================
    # MODEL 3 — XGBOOST
    # ========================================================

    xgb_model = XGBRegressor(

        n_estimators=200,

        max_depth=2,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=20,

        reg_alpha=1.0,

        reg_lambda=5.0,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=-1
    )


    xgb_model.fit(
        X_train,
        y_train
    )


    xgb_val_pred = xgb_model.predict(X_val)

    xgb_val_metrics = evaluate_predictions(
        y_val,
        xgb_val_pred
    )

    xgb_test_metrics = (
        evaluate_predictions(
            y_test,
            xgb_model.predict(X_test)
        )
        if has_test else None
    )

    print_metrics(
        "XGBOOST — VALIDATION",
        xgb_val_metrics
    )

    if xgb_test_metrics:
        print_metrics(
            "XGBOOST — TEST",
            xgb_test_metrics
        )


    # ========================================================
    # PER-STOCK BREAKDOWN (XGBoost, Validation)
    # ========================================================

    stock_breakdown = evaluate_per_stock(
        data,
        validation_mask,
        xgb_val_pred,
        target
    )

    if not stock_breakdown.empty:

        print(
            f"\nXGBOOST — Per-Stock Validation ({horizon})"
        )

        print(
            stock_breakdown.to_string(
                index=False,
                formatters={
                    "Dir Acc": "{:.2%}".format,
                    "Spearman": "{:.4f}".format
                }
            )
        )


    # ========================================================
    # SAVE MODEL ARTIFACTS
    # ========================================================
    #
    # We save the XGBoost model (best performer), a fitted
    # StandardScaler for live inference, the feature list,
    # and key metadata so App.py can load everything without
    # re-running training.
    # ========================================================

    # Fit a scaler on the training data for inference use.
    inference_scaler = StandardScaler()
    inference_scaler.fit(X_train)

    model_path = os.path.join(
        MODEL_DIR,
        f"xgb_{horizon.lower()}.joblib"
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        f"scaler_{horizon.lower()}.joblib"
    )

    features_path = os.path.join(
        MODEL_DIR,
        f"features_{horizon.lower()}.json"
    )

    metadata_path = os.path.join(
        MODEL_DIR,
        f"metadata_{horizon.lower()}.json"
    )

    joblib.dump(xgb_model, model_path)
    joblib.dump(inference_scaler, scaler_path)

    with open(features_path, "w") as f:
        json.dump(FEATURES, f, indent=2)

    metadata = {
        "horizon": horizon,
        "target": target,
        "train_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "n_features": len(FEATURES),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "val_direction": float(xgb_val_metrics["Direction"]),
        "val_spearman": float(xgb_val_metrics["Spearman"]),
        "test_direction": (
            float(xgb_test_metrics["Direction"])
            if xgb_test_metrics else None
        ),
        "test_spearman": (
            float(xgb_test_metrics["Spearman"])
            if xgb_test_metrics else None
        )
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"\nSaved artifacts -> {MODEL_DIR}/"
        f"xgb_{horizon.lower()}.joblib"
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    models = {
        "Mean": (
            mean_val_metrics,
            mean_test_metrics
        ),

        "Ridge": (
            ridge_val_metrics,
            ridge_test_metrics
        ),

        "XGBoost": (
            xgb_val_metrics,
            xgb_test_metrics
        )
    }


    for model_name, (
        val_metrics,
        test_metrics
    ) in models.items():

        results.append({

            "Target": horizon,

            "Model": model_name,

            "Val MAE":
                val_metrics["MAE"],

            "Val RMSE":
                val_metrics["RMSE"],

            "Val R2":
                val_metrics["R2"],

            "Val Direction":
                val_metrics["Direction"],

            "Val Correlation":
                val_metrics["Correlation"],

            "Val Spearman":
                val_metrics["Spearman"],

            "Test MAE":
                test_metrics["MAE"],

            "Test RMSE":
                test_metrics["RMSE"],

            "Test R2":
                test_metrics["R2"],

            "Test Direction":
                test_metrics["Direction"],

            "Test Correlation":
                test_metrics["Correlation"],

            "Test Spearman":
                test_metrics["Spearman"]
        })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n\n" + "=" * 130)
print("ABSOLUTE RETURN MODEL COMPARISON")
print("=" * 130)


print(
    results_df.to_string(
        index=False,
        formatters={

            "Val MAE":
                "{:.4f}".format,

            "Val RMSE":
                "{:.4f}".format,

            "Val R2":
                "{:.4f}".format,

            "Val Direction":
                "{:.2%}".format,

            "Val Correlation":
                "{:.4f}".format,

            "Val Spearman":
                "{:.4f}".format,

            "Test MAE":
                "{:.4f}".format,

            "Test RMSE":
                "{:.4f}".format,

            "Test R2":
                "{:.4f}".format,

            "Test Direction":
                "{:.2%}".format,

            "Test Correlation":
                "{:.4f}".format,

            "Test Spearman":
                "{:.4f}".format
        }
    )
)


print("=" * 130)


# ============================================================
# BEST MODEL BY VALIDATION SPEARMAN
# ============================================================

best = results_df.loc[
    results_df["Val Spearman"].idxmax()
]


print("\n" + "=" * 80)
print("BEST MODEL BASED ON VALIDATION RANK CORRELATION")
print("=" * 80)


print(
    f"Target: "
    f"{best['Target']}"
)

print(
    f"Model: "
    f"{best['Model']}"
)

print(
    f"Validation Spearman: "
    f"{best['Val Spearman']:.4f}"
)

print(
    f"Unseen Test Spearman: "
    f"{best['Test Spearman']:.4f}"
)

print(
    f"Unseen Test Direction: "
    f"{best['Test Direction']:.2%}"
)

print("=" * 80)