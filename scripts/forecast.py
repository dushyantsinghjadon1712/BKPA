"""
forecast.py
-----------
Phase 2, Core Feature: Price Forecasting Model

Built with scikit-learn (GradientBoostingRegressor on lag + rolling + calendar
features) so it runs anywhere with no extra installs. If you have statsmodels
or prophet installed locally, forecast_arima() below is provided as a drop-in
alternative -- swap it in wherever forecast_price() is called.

Approach:
  For a given (crop, market), build a per-record series ordered by date,
  engineer lag-1/2/3 prices, a rolling mean, and calendar features (month,
  day-of-year), train/test split by TIME (not random, since this is a time
  series), then forecast forward using iterative (recursive) prediction:
  each future prediction becomes the next lag feature.

Run standalone for a demo: python3 scripts/forecast.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

CLEAN_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/mandi_prices_clean.csv"


def load_series(df: pd.DataFrame, crop: str, market: str) -> pd.DataFrame:
    s = df[(df["Commodity"].str.lower() == crop.lower()) & (df["Market"].str.lower() == market.lower())]
    s = s.sort_values("Arrival_Date")[["Arrival_Date", "Modal_Price"]].reset_index(drop=True)
    return s


def build_features(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["lag_1"] = s["Modal_Price"].shift(1)
    s["lag_2"] = s["Modal_Price"].shift(2)
    s["lag_3"] = s["Modal_Price"].shift(3)
    s["rolling_mean_5"] = s["Modal_Price"].shift(1).rolling(5, min_periods=1).mean()
    s["rolling_std_5"] = s["Modal_Price"].shift(1).rolling(5, min_periods=1).std()
    s["month"] = s["Arrival_Date"].dt.month
    s["day_of_year"] = s["Arrival_Date"].dt.dayofyear
    s["days_since_start"] = (s["Arrival_Date"] - s["Arrival_Date"].min()).dt.days
    return s.dropna().reset_index(drop=True)


FEATURE_COLS = ["lag_1", "lag_2", "lag_3", "rolling_mean_5", "rolling_std_5",
                "month", "day_of_year", "days_since_start"]


def train_and_evaluate(s_feat: pd.DataFrame, test_frac: float = 0.15):
    """Time-ordered train/test split (never shuffle time series data)."""
    n_test = max(5, int(len(s_feat) * test_frac))
    train, test = s_feat.iloc[:-n_test], s_feat.iloc[-n_test:]

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[FEATURE_COLS], train["Modal_Price"])

    preds = model.predict(test[FEATURE_COLS])
    mae = mean_absolute_error(test["Modal_Price"], preds)
    mape = mean_absolute_percentage_error(test["Modal_Price"], preds) * 100

    return model, {"MAE": round(mae, 1), "MAPE_pct": round(mape, 2), "test_size": n_test}


def forecast_forward(model, s_feat: pd.DataFrame, n_periods: int = 12, freq_days: int = 7) -> pd.DataFrame:
    """
    Recursive forecast: predict the next point, feed it back in as a lag,
    repeat. freq_days approximates typical reporting gaps in this market's data
    (Agmarknet arrivals aren't perfectly daily, so this is a rough calendar step).
    """
    history = s_feat[["Arrival_Date", "Modal_Price"]].copy()
    last_date = history["Arrival_Date"].max()
    start_date = history["Arrival_Date"].min()

    future_rows = []
    recent_prices = history["Modal_Price"].tolist()

    for i in range(1, n_periods + 1):
        next_date = last_date + pd.Timedelta(days=freq_days * i)
        lag_1, lag_2, lag_3 = recent_prices[-1], recent_prices[-2], recent_prices[-3]
        rolling_mean_5 = np.mean(recent_prices[-5:])
        rolling_std_5 = np.std(recent_prices[-5:])
        month = next_date.month
        day_of_year = next_date.dayofyear
        days_since_start = (next_date - start_date).days

        x = pd.DataFrame([{
            "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
            "rolling_mean_5": rolling_mean_5, "rolling_std_5": rolling_std_5,
            "month": month, "day_of_year": day_of_year, "days_since_start": days_since_start,
        }])
        pred = model.predict(x[FEATURE_COLS])[0]
        recent_prices.append(pred)
        future_rows.append({"Arrival_Date": next_date, "Forecast_Price": round(pred, 0)})

    return pd.DataFrame(future_rows)


def forecast_price(df: pd.DataFrame, crop: str, market: str, n_periods: int = 12):
    """Main entry point: returns (metrics dict, forecast dataframe) or (None, None) if too little data."""
    s = load_series(df, crop, market)
    s_feat = build_features(s)
    if len(s_feat) < 30:
        return None, None
    model, metrics = train_and_evaluate(s_feat)
    forecast_df = forecast_forward(model, s_feat, n_periods=n_periods)
    return metrics, forecast_df


# ---------------------------------------------------------------------------
# OPTIONAL: classical time-series alternative (requires `pip install statsmodels`
# or `pip install prophet` locally -- not available in this sandbox).
# Swap this in for forecast_price() if you prefer ARIMA over gradient boosting.
# ---------------------------------------------------------------------------
def forecast_arima(df: pd.DataFrame, crop: str, market: str, n_periods: int = 12):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("Run `pip install statsmodels` locally to use forecast_arima().")

    s = load_series(df, crop, market)
    ts = s.set_index("Arrival_Date")["Modal_Price"].asfreq("D").interpolate()
    model = ARIMA(ts, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=n_periods)
    return forecast


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH, parse_dates=["Arrival_Date"])
    print("=== Forecasting Wheat @ Kota ===")
    metrics, forecast_df = forecast_price(df, "Wheat", "Kota", n_periods=8)
    print("Model quality (on held-out recent data):", metrics)
    print(forecast_df)
