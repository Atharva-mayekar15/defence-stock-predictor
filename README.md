# NSE Defence Stock Predictor 🚀

A production-grade machine learning pipeline and interactive web application designed to predict the relative forward returns (alpha) of Indian Defence Sector stocks. 

**Live Demo:** [Check out the Live Dashboard here!](https://defence-stock-predictor.streamlit.app/) *(Note: If you just deployed, update this link with your actual Streamlit URL!)*

## 📌 Project Overview
Stock return prediction is notoriously difficult and heavily prone to "data leakage" in beginner projects. This project was built with professional quant-finance rigor:
1. **Target Isolation:** Instead of predicting absolute returns (which are easily skewed by broader market rallies/crashes), the model is trained to predict **Relative Returns** (Stock Return minus Nifty 50 Return).
2. **Leakage Prevention:** Features and targets are generated with strict chronological boundaries. The train/test split is purely chronological (Train: 2010-2023, Test: 2026), ensuring the model never looks into the future.
3. **Heavy Regularization:** The XGBoost models utilize shallow trees (`max_depth=2`) and heavy L1/L2 regularization to prevent overfitting on noisy financial data.

## 📊 Model Performance
In quantitative finance, a Spearman rank correlation above `0.10` on unseen future data is considered a highly viable signal. On our strictly unseen 2026 test set (1-Month Horizon):

* **Directional Accuracy:** 56.92% (Correctly predicts if the stock will beat or lag the Nifty).
* **Spearman Rank Correlation:** 0.105 (Strong ability to rank stocks from best to worst).
* **Top Performers:** Mid-cap stocks like GRSE achieved validation Spearman correlations as high as `0.545`.

## 🛠️ Tech Stack & Pipeline
* **Data Ingestion:** `yfinance`, Pandas
* **Feature Engineering (64 Features):** Moving Averages, RSI, MACD, Bollinger Bands, ATR, Nifty/VIX relative indicators.
* **Modeling:** `xgboost`, `scikit-learn`
* **App & Visualization:** `streamlit`, `plotly`

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/defence-stock-predictor.git
   cd defence-stock-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run App.py
   ```

## 📂 Project Structure
* `features.py`: Computes 64 technical and market-context indicators.
* `targets.py`: Generates the relative forward return targets for 1M, 3M, and 6M horizons.
* `train.py`: Handles chronological splitting, one-hot encoding, and trains the XGBoost models.
* `App.py`: The main Streamlit dashboard that downloads live data, computes features on the fly, and runs inference.
* `models/`: Contains the serialized XGBoost `.joblib` weights.

---
*Disclaimer: This project is for educational and portfolio purposes only. It does not constitute financial advice.*
