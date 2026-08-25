"""
Defence Stock Predictor -- Streamlit Application

Loads trained XGBoost models from models/ and generates
1M / 3M / 6M return predictions for NSE defence stocks.
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

STOCKS = {
    "HAL":        "HAL.NS",
    "BEL":        "BEL.NS",
    "BDL":        "BDL.NS",
    "MAZDOCK":    "MAZDOCK.NS",
    "COCHINSHIP": "COCHINSHIP.NS",
    "GRSE":       "GRSE.NS",
    "PARAS":      "PARAS.NS",
}

MODEL_DIR     = "models"
HORIZONS      = ["1M", "3M", "6M"]
HISTORY_YEARS = 4


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Defence Stock Predictor",
    page_icon=":shield:",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.metric-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
    font-family: 'Rajdhani', sans-serif;
}
.metric-sub { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }

.pred-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 22px 16px;
    text-align: center;
}
.pred-horizon {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.pred-return {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'Rajdhani', sans-serif;
    line-height: 1.1;
}
.pred-direction { font-size: 0.82rem; margin-top: 6px; }
.pred-meta {
    font-size: 0.70rem;
    color: #8b949e;
    margin-top: 10px;
    border-top: 1px solid #30363d;
    padding-top: 8px;
}

.info-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #c9d1d9;
    line-height: 1.7;
}

.disclaimer {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #e3b341;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 16px;
}

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FEATURE COMPUTATION HELPERS
# (Mirrors features.py exactly so inference matches training)
# ============================================================

def _calculate_rsi(series, period=14):
    """Wilder-style RSI using exponential smoothing."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calculate_atr(group, period=14):
    """Average True Range using Wilder smoothing."""
    prev_close = group["close"].shift(1)
    tr = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - prev_close).abs(),
            (group["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def _download_symbol(ticker, start_date):
    """Download from Yahoo Finance; return tidy DataFrame or None."""
    end_date = datetime.now()
    start_date_obj = end_date - timedelta(days=500)
    
    # Add retry logic for cloud rate limits
    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start_date_obj, end=end_date, progress=False)
            if not raw.empty:
                break
        except Exception:
            pass
        time.sleep(1)
        
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    if "datetime" in raw.columns and "date" not in raw.columns:
        raw = raw.rename(columns={"datetime": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    return raw.sort_values("date").reset_index(drop=True)


# ============================================================
# DATA + FEATURE PIPELINE
# ============================================================

@st.cache_data(ttl=900)
def load_stock_features(ticker: str):
    """
    Download fresh stock + NIFTY + VIX data for `ticker`,
    compute all 64 model features using the same pipeline
    as features.py, and return the complete DataFrame.

    Returns (df, error_message).  error_message is None on success.
    """
    start_date = (
        pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS)
    ).strftime("%Y-%m-%d")

    # Stock data
    stock_raw = _download_symbol(ticker, start_date)
    if stock_raw is None:
        return None, f"Could not download data for {ticker}."
    required = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in stock_raw.columns for c in required):
        return None, f"Missing OHLCV columns for {ticker}."
    df = stock_raw[required].copy()
    df["stock"] = "STOCK"

    # NIFTY
    nifty_raw = _download_symbol("^NSEI", start_date)
    if nifty_raw is not None and "close" in nifty_raw.columns:
        df = df.merge(
            nifty_raw[["date", "close"]].rename(columns={"close": "nifty_close"}),
            on="date", how="left",
        )
    else:
        df["nifty_close"] = np.nan

    # VIX
    vix_raw = _download_symbol("^INDIAVIX", start_date)
    if vix_raw is not None and "close" in vix_raw.columns:
        df = df.merge(
            vix_raw[["date", "close"]].rename(columns={"close": "vix_close"}),
            on="date", how="left",
        )
    else:
        df["vix_close"] = np.nan

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df["nifty_close"] = df["nifty_close"].ffill()
    df["vix_close"]   = df["vix_close"].ffill()

    grouped = df.groupby("stock", group_keys=False)

    # Returns
    for days in [1, 3, 5, 10, 20, 60]:
        df[f"return_{days}d"] = grouped["close"].pct_change(days, fill_method=None)

    # Intraday
    df["intraday_return"] = df["close"] / df["open"] - 1
    df["high_low_range"]  = (df["high"] - df["low"]) / df["close"]
    df["close_position"]  = (
        (df["close"] - df["low"]) /
        (df["high"] - df["low"]).replace(0, np.nan)
    )

    # Moving averages
    for period in [10, 20, 50, 100, 200]:
        df[f"ma_{period}"] = grouped["close"].transform(
            lambda x, p=period: x.rolling(p, min_periods=p).mean()
        )
    for period in [10, 20, 50, 100, 200]:
        df[f"price_vs_ma{period}"] = df["close"] / df[f"ma_{period}"] - 1

    df["ma10_vs_ma50"]  = df["ma_10"]  / df["ma_50"]  - 1
    df["ma20_vs_ma50"]  = df["ma_20"]  / df["ma_50"]  - 1
    df["ma50_vs_ma200"] = df["ma_50"]  / df["ma_200"] - 1
    df["ma10_slope"]    = grouped["ma_10"].pct_change(5,  fill_method=None)
    df["ma20_slope"]    = grouped["ma_20"].pct_change(10, fill_method=None)
    df["ma50_slope"]    = grouped["ma_50"].pct_change(20, fill_method=None)

    # ROC & Volatility
    for days in [5, 10, 20, 60]:
        df[f"roc_{days}"] = df["close"] / grouped["close"].shift(days) - 1
    for period in [5, 10, 20, 60]:
        df[f"volatility_{period}d"] = grouped["return_1d"].transform(
            lambda x, p=period: x.rolling(p, min_periods=p).std()
        )

    # Volume
    for period in [5, 20, 60]:
        vol_ma = grouped["volume"].transform(
            lambda x, p=period: x.rolling(p, min_periods=p).mean()
        )
        df[f"volume_ratio_{period}"] = df["volume"] / vol_ma
    df["volume_change_1d"] = grouped["volume"].pct_change(1, fill_method=None)

    # RSI
    df["rsi_7"]      = grouped["close"].transform(lambda x: _calculate_rsi(x, 7))
    df["rsi_14"]     = grouped["close"].transform(lambda x: _calculate_rsi(x, 14))
    df["rsi_change"] = df["rsi_14"] - grouped["rsi_14"].shift(5)

    # MACD
    df["ema_12"]        = grouped["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df["ema_26"]        = grouped["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"]          = df["ema_12"] - df["ema_26"]
    df["macd_signal"]   = grouped["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_histogram"]= df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb_middle = df["ma_20"]
    bb_std    = grouped["close"].transform(lambda x: x.rolling(20, min_periods=20).std())
    bb_upper  = bb_middle + 2 * bb_std
    bb_lower  = bb_middle - 2 * bb_std
    df["bb_width"]    = (bb_upper - bb_lower) / bb_middle
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # ATR
    atr_parts = []
    for _, group in df.groupby("stock"):
        atr = _calculate_atr(group, 14)
        atr_parts.append(pd.DataFrame({"_idx": group.index, "atr_14": atr.values}))
    atr_df = pd.concat(atr_parts).set_index("_idx")
    df["atr_14"]      = atr_df["atr_14"]
    df["atr_percent"] = df["atr_14"] / df["close"]

    # Breakout
    rolling_high = grouped["high"].transform(lambda x: x.shift(1).rolling(20, min_periods=20).max())
    rolling_low  = grouped["low"].transform( lambda x: x.shift(1).rolling(20, min_periods=20).min())
    df["breakout_20d"]           = (df["close"] > rolling_high).astype(int)
    df["breakdown_20d"]          = (df["close"] < rolling_low ).astype(int)
    df["distance_from_20d_high"] = df["close"] / rolling_high - 1
    df["distance_from_20d_low"]  = df["close"] / rolling_low  - 1

    # Market features -- standalone chronological market df
    market = (
        df[["date", "nifty_close", "vix_close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    market["nifty_close"] = market["nifty_close"].ffill()
    market["vix_close"]   = market["vix_close"].ffill()

    for days in [1, 5, 10, 20, 60]:
        market[f"nifty_return_{days}d"] = market["nifty_close"].pct_change(days, fill_method=None)
    for period in [20, 50, 200]:
        market[f"nifty_ma_{period}"] = market["nifty_close"].rolling(period, min_periods=period).mean()

    market["nifty_vs_ma20"]  = market["nifty_close"] / market["nifty_ma_20"]  - 1
    market["nifty_vs_ma50"]  = market["nifty_close"] / market["nifty_ma_50"]  - 1
    market["nifty_vs_ma200"] = market["nifty_close"] / market["nifty_ma_200"] - 1

    _nret = market["nifty_close"].pct_change(1, fill_method=None)
    market["nifty_volatility_20d"] = _nret.rolling(20, min_periods=20).std()
    market["nifty_volatility_60d"] = _nret.rolling(60, min_periods=60).std()

    market["vix_change_1d"]  = market["vix_close"] / market["vix_close"].shift(1)  - 1
    market["vix_change_5d"]  = market["vix_close"] / market["vix_close"].shift(5)  - 1
    market["vix_change_20d"] = market["vix_close"] / market["vix_close"].shift(20) - 1
    market["vix_ma20"]       = market["vix_close"].rolling(20, min_periods=20).mean()
    market["vix_vs_ma20"]    = market["vix_close"] / market["vix_ma20"] - 1

    market["market_bullish"] = (
        (market["nifty_close"] > market["nifty_ma_200"]) &
        (market["nifty_ma_50"] > market["nifty_ma_200"])
    ).astype(int)
    market["market_above_ma50"] = (market["nifty_close"] > market["nifty_ma_50"]).astype(int)

    market = market.drop(columns=["nifty_ma_20", "nifty_ma_50", "nifty_ma_200", "vix_ma20"])

    market_feat_cols = [c for c in market.columns if c != "date"]
    df = df.drop(columns=[c for c in market_feat_cols if c in df.columns], errors="ignore")
    df = df.merge(market, on="date", how="left")

    # Relative returns
    for days in [5, 20, 60]:
        df[f"relative_return_{days}d"] = df[f"return_{days}d"] - df[f"nifty_return_{days}d"]

    df = df.drop(columns=["ema_12", "ema_26", "atr_14"], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """Load all saved XGBoost models, feature lists, and metadata."""
    loaded = {}
    for h in HORIZONS:
        h_lower = h.lower()
        paths = {
            "model":    os.path.join(MODEL_DIR, f"xgb_{h_lower}.joblib"),
            "features": os.path.join(MODEL_DIR, f"features_{h_lower}.json"),
            "metadata": os.path.join(MODEL_DIR, f"metadata_{h_lower}.json"),
        }
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            return None, (
                f"Missing {missing} for horizon {h}. "
                "Run python train.py first."
            )
        model = joblib.load(paths["model"])
        with open(paths["features"]) as f:
            features = json.load(f)
        with open(paths["metadata"]) as f:
            metadata = json.load(f)
        loaded[h] = {"model": model, "features": features, "metadata": metadata}
    return loaded, None


# ============================================================
# PREDICTION HELPER
# ============================================================

def predict_horizon(df, models_dict, horizon):
    """Return (predicted_return_float, error_string)."""
    feature_list = models_dict[horizon]["features"]
    model        = models_dict[horizon]["model"]
    
    # Ensure one-hot encoded stock features exist (set missing to 0)
    for f in feature_list:
        if f not in df.columns:
            if f.startswith("stock_"):
                df[f] = 1 if f == f"stock_{ticker}" else 0
            else:
                df[f] = 0

    valid_rows   = df[feature_list].dropna()
    if valid_rows.empty:
        return None, "Not enough data to compute all features."
    X = df.loc[[valid_rows.index[-1]], feature_list]
    return float(model.predict(X)[0]), None


# ============================================================
# DIRECTION LABEL
# ============================================================

def direction_info(pred_return):
    if pred_return >  0.08: return "Strong Upside Signal",   "#3fb950"
    if pred_return >  0.03: return "Moderate Upside Signal", "#56d364"
    if pred_return > -0.03: return "Neutral / Uncertain",    "#e3b341"
    if pred_return > -0.08: return "Moderate Downside Risk", "#f85149"
    return                         "Strong Downside Risk",   "#ff7b72"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## NSE Defence Stocks")
    st.markdown("---")
    stock_name = st.selectbox("Select a stock", list(STOCKS.keys()), index=0)
    ticker = STOCKS[stock_name]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Algorithm:** XGBoost Regressor")
    st.markdown("**Features:** 64 technical + market indicators")
    st.markdown("**Trained:** 2010 to Dec 2023")
    st.markdown("**Validated:** 2024 to Dec 2025")
    st.markdown("**Horizons:** 1M / 3M / 6M relative forward returns (vs Nifty)")
    st.markdown("---")
    st.markdown(
        "<div class='disclaimer'>"
        "<b>Not financial advice.</b><br>"
        "Predictions are probabilistic estimates for educational purposes only."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("# Defence Stock Predictor")
st.markdown(f"### {stock_name}  &nbsp; `{ticker}`")
st.markdown("---")


# ============================================================
# LOAD MODELS
# ============================================================

models_dict, model_err = load_models()
if model_err:
    st.error(f"Model error: {model_err}")
    st.info("Run `python train.py` in the project directory to generate model files.")
    st.stop()


# ============================================================
# LOAD + COMPUTE FEATURES
# ============================================================

with st.spinner(f"Downloading {stock_name} + NIFTY + VIX and computing features..."):
    df, data_err = load_stock_features(ticker)

if data_err:
    st.error(f"Data error: {data_err}")
    st.stop()


# ============================================================
# LATEST VALUES FOR DISPLAY
# ============================================================

valid_df = df.dropna(subset=["close"])
if valid_df.empty:
    st.error("No valid rows after feature computation.")
    st.stop()

latest     = valid_df.iloc[-1]
last_close = float(latest["close"])
last_date  = latest["date"]
prev_close = float(valid_df.iloc[-2]["close"]) if len(valid_df) > 1 else last_close
daily_chg  = (last_close / prev_close - 1) if prev_close else 0.0

def _fmt(val, fmt=".1f", fallback="N/A"):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return fallback
        return f"{float(val):{fmt}}"
    except Exception:
        return fallback

rsi_val  = latest.get("rsi_14")
vix_val  = latest.get("vix_close")
bb_pos   = latest.get("bb_position")

chg_color = "#3fb950" if daily_chg >= 0 else "#f85149"
chg_sign  = "+" if daily_chg >= 0 else ""

rsi_f     = _fmt(rsi_val)
rsi_color = "#f85149" if rsi_val and float(rsi_val) > 70 else (
            "#3fb950" if rsi_val and float(rsi_val) < 30 else "#e6edf3")
rsi_label = ("Overbought" if rsi_val and float(rsi_val) > 70 else
             "Oversold"   if rsi_val and float(rsi_val) < 30 else "Neutral")

vix_f     = _fmt(vix_val)
vix_color = "#f85149" if vix_val and float(vix_val) > 20 else "#3fb950"

bb_f      = _fmt(bb_pos, ".2f")
bb_color  = ("#f85149" if bb_pos and float(bb_pos) > 0.8 else
             "#3fb950" if bb_pos and float(bb_pos) < 0.2 else "#e6edf3")


# ============================================================
# KEY METRICS ROW
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Close</div>
        <div class="metric-value">Rs {last_close:,.2f}</div>
        <div class="metric-sub" style="color:{chg_color}">
            {chg_sign}{daily_chg:.2%} today &nbsp; {last_date.strftime('%d %b %Y')}
        </div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RSI (14)</div>
        <div class="metric-value" style="color:{rsi_color}">{rsi_f}</div>
        <div class="metric-sub" style="color:{rsi_color}">{rsi_label}</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">India VIX</div>
        <div class="metric-value" style="color:{vix_color}">{vix_f}</div>
        <div class="metric-sub">Market fear gauge</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Bollinger Position</div>
        <div class="metric-value" style="color:{bb_color}">{bb_f}</div>
        <div class="metric-sub">0 = lower band &nbsp; 1 = upper band</div>
    </div>""", unsafe_allow_html=True)


# ============================================================
# PREDICTION CARDS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Model Predictions")
st.caption(
    "Estimated forward returns based on current technical state. "
    "The model was trained to rank relative performance, not to predict exact returns."
)

H_LABELS = {
    "1M": "1 Month (~21 trading days)",
    "3M": "3 Months (~63 trading days)",
    "6M": "6 Months (~126 trading days)",
}

p1, p2, p3 = st.columns(3)
for col, horizon in zip([p1, p2, p3], HORIZONS):
    pred, pred_err = predict_horizon(df, models_dict, horizon)
    meta    = models_dict[horizon]["metadata"]
    val_d   = meta.get("val_direction")
    tst_d   = meta.get("test_direction")
    val_sp  = meta.get("val_spearman", float("nan"))
    val_d_s = f"{val_d:.1%}" if val_d is not None else "N/A"
    tst_d_s = f"{tst_d:.1%}" if tst_d is not None else "N/A"

    with col:
        if pred_err or pred is None:
            st.warning(f"{horizon}: {pred_err}")
        else:
            dir_text, dir_color = direction_info(pred)
            sign = "+" if pred > 0 else ""
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-horizon">{H_LABELS[horizon]}</div>
                <div class="pred-return" style="color:{dir_color}">{sign}{pred:.1%}</div>
                <div class="pred-direction" style="color:{dir_color}">{dir_text}</div>
                <div class="pred-meta">
                    Val dir. acc: {val_d_s} &nbsp;|&nbsp;
                    Test dir. acc: {tst_d_s}<br>
                    Val Spearman rho: {val_sp:.3f}
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📖 How to read these numbers (Glossary)"):
    st.markdown("""
    **Understanding the Target:**
    * **Relative Return (vs Nifty):** The model doesn't just guess if the stock goes up. It guesses if the stock will *beat* the overall market (Nifty 50). If the model predicts +5%, it means the stock is expected to outperform the Nifty by 5%. 
    
    **Understanding Model Metrics:**
    * **Directional Accuracy:** The percentage of times the model correctly guessed whether the stock would beat or lag the market on unseen data. In the stock market, a coin flip is 50%. A model that consistently scores **55% to 60%** on unseen future data is considered a highly viable, professional-grade signal. (Note: Tutorials claiming 85% accuracy almost always suffer from data leakage).
    * **Spearman Rank Correlation (rho):** This measures how well the model *ranks* stocks from best to worst. A score of **1.0** is perfect. In quantitative finance, a rank correlation between **0.05 and 0.15** on unseen test data is considered highly profitable for a portfolio.
    
    **Understanding Technical Indicators:**
    * **RSI (14):** Relative Strength Index measures momentum. Under 30 is considered "Oversold" (potentially cheap), and over 70 is "Overbought" (potentially expensive).
    * **Bollinger Position:** Shows where the price is relative to its normal volatility range. `0` means the price is sitting at its lower band (historically cheap), and `1` means it is at its upper band (historically expensive).
    * **India VIX:** The "fear gauge" of the broader Indian market. A high VIX (> 20) means high volatility and uncertainty. A low VIX means a calm market.
    """)


# ============================================================
# CHART
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"### {stock_name} -- Price, Volume & RSI")

chart_df = df.tail(120).copy()

BG   = "#0d1117"; GRID = "#21262d"; TEXT = "#8b949e"
GRN  = "#3fb950"; RED  = "#f85149"; BLUE = "#58a6ff"
YELL = "#e3b341"; PURP = "#bc8cff"; ORNG = "#ffa657"

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.25, 0.20],
    vertical_spacing=0.04,
    subplot_titles=("Price & Moving Averages", "Volume", "RSI (14)"),
)

fig.add_trace(go.Scatter(
    x=chart_df["date"], y=chart_df["close"],
    name="Close", line=dict(color=BLUE, width=2),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
), row=1, col=1)

for period, color, dash in [(10, GRN, "dot"), (50, YELL, "dash"), (200, ORNG, "longdash")]:
    col_n = f"ma_{period}"
    if col_n in chart_df and not chart_df[col_n].isna().all():
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df[col_n],
            name=f"MA {period}", line=dict(color=color, width=1.3, dash=dash),
        ), row=1, col=1)

bar_colors = [GRN if r >= 0 else RED for r in chart_df["close"].pct_change().fillna(0)]
fig.add_trace(go.Bar(
    x=chart_df["date"], y=chart_df["volume"],
    name="Volume", marker_color=bar_colors, opacity=0.7,
), row=2, col=1)

if "rsi_14" in chart_df:
    fig.add_trace(go.Scatter(
        x=chart_df["date"], y=chart_df["rsi_14"],
        name="RSI 14", line=dict(color=PURP, width=1.6),
    ), row=3, col=1)
    for yv, c, d in [(70, RED, "dot"), (30, GRN, "dot"), (50, TEXT, "dash")]:
        fig.add_hline(y=yv, line_dash=d, line_color=c, opacity=0.4, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.04, row=3, col=1, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GRN, opacity=0.04, row=3, col=1, line_width=0)

fig.update_layout(
    height=640, paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Inter"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified", xaxis_rangeslider_visible=False,
)
for row in [1, 2, 3]:
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, row=row, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, row=row, col=1)
fig.update_yaxes(title_text="Price (Rs)", row=1, col=1)
fig.update_yaxes(title_text="Volume",     row=2, col=1)
fig.update_yaxes(title_text="RSI",        row=3, col=1, range=[0, 100])

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL CONTEXT
# ============================================================

st.markdown("---")
st.markdown("### Model Performance Context")
st.caption("Out-of-sample metrics on data the model never saw during training.")

c1, c2, c3 = st.columns(3)
for col, horizon in zip([c1, c2, c3], HORIZONS):
    meta  = models_dict[horizon]["metadata"]
    val_d = meta.get("val_direction")
    tst_d = meta.get("test_direction")
    val_s = meta.get("val_spearman", float("nan"))
    tst_s = meta.get("test_spearman")

    with col:
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-label">{horizon} XGBoost Model</div>
            Train rows: {meta['n_train']:,}<br>
            Val rows: {meta['n_val']:,}<br>
            Test rows: {meta['n_test']:,}<br>
            <br>
            <b>Validation (2024-2025)</b><br>
            Directional accuracy: {f"{val_d:.1%}" if val_d else "N/A"}<br>
            Spearman rho: {val_s:.3f}<br>
            <br>
            <b>Test (2026)</b><br>
            Directional accuracy: {f"{tst_d:.1%}" if tst_d else "N/A"}<br>
            Spearman rho: {f"{tst_s:.3f}" if tst_s else "N/A"}
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<b>Disclaimer:</b>
This application is for <b>educational and portfolio demonstration purposes only</b>.
Model predictions are statistical estimates based on historical price patterns.
They do not constitute financial advice. The model cannot anticipate news,
earnings surprises, regulatory changes, or geopolitical events.
Always conduct independent research before making investment decisions.
</div>
""", unsafe_allow_html=True)
