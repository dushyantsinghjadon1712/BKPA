
import sys
import os
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from analysis import load_clean, best_mandis_for_crop, price_trend, volatility_ranking, crop_recommendation
from forecast import forecast_price
from crop_yield_model import load_and_merge, train_and_evaluate, recommend_crops_for_state
from rajasthan_bridge import combined_rajasthan_recommendation

st.set_page_config(page_title="Bharat Krishi Price Advisor", layout="wide")

st.title("🌾 Bharat Krishi Price Advisor (BKPA)")
st.caption("Phase 1 + Phase 2 — Mandi Price Intelligence, Forecasting & Crop Yield Prediction")

df = load_clean()


@st.cache_resource
def get_yield_model():
    yield_df = load_and_merge()
    pipe, metrics = train_and_evaluate(yield_df)
    return yield_df, pipe, metrics


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📍 Best Mandi Finder", "📈 Price Trend", "⚠️ Price Volatility",
     "🌱 Crop Recommendation (combined: real price + proxy yield)", "🔮 Price Forecast",
     "🌍 National Crop Yield Predictor"]
)

with tab1:
    st.subheader("Find the best mandi (market) for your crop")
    crop = st.selectbox("Select Crop", sorted(df["Commodity"].unique()), key="t1_crop")
    top_n = st.slider("Number of mandis to show", 3, 10, 5)
    result = best_mandis_for_crop(df, crop, top_n)
    st.dataframe(result, use_container_width=True)
    st.bar_chart(result.set_index("Market")["Avg_Modal_Price"])

with tab2:
    st.subheader("Price trend over time")
    crop2 = st.selectbox("Select Crop", sorted(df["Commodity"].unique()), key="t2_crop")
    markets = ["All markets"] + sorted(df[df["Commodity"] == crop2]["Market"].unique().tolist())
    market_choice = st.selectbox("Select Market (optional)", markets)
    market_arg = None if market_choice == "All markets" else market_choice
    trend = price_trend(df, crop2, market_arg)
    st.line_chart(trend.set_index("Arrival_Date")["Modal_Price"])
    st.dataframe(trend, use_container_width=True)

with tab3:
    st.subheader("Which crops are risky right now?")
    vol = volatility_ranking(df)
    st.dataframe(vol, use_container_width=True)
    st.bar_chart(vol.set_index("Commodity")["Volatility_StdDev_Pct"])
    st.caption("HIGH = avoid unless you can absorb price swings. LOW/MEDIUM = more stable income.")

with tab4:
    st.subheader("Which crop should I grow? (real Rajasthan price + proxy-based national yield model)")
    season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"], key="t4_season")
    top_n4 = st.slider("Number of crops to show", 3, 10, 8)

    yield_df, yield_pipe, yield_metrics = get_yield_model()
    combined = combined_rajasthan_recommendation(df, yield_df, yield_pipe, season=season, top_n=top_n4)

    if combined.empty:
        st.info("No overlapping crops found for this season.")
    else:
        st.dataframe(combined, use_container_width=True)

    st.info(
        "**Avg_Rajasthan_Price** and **Volatility_StdDev_Pct** come from your real mandi price data. "
        "**Proxy_Relative_Yield_Performance** comes from the national yield model (R²≈0.963), fed with a "
        "soil/weather profile *averaged from Gujarat, Madhya Pradesh & Haryana* since Rajasthan isn't in the "
        "soil/weather dataset — treat that column as an approximate signal, not ground truth for Rajasthan. "
        "**Combined_Score** blends price + yield performance and penalizes volatility."
    )

with tab5:
    st.subheader("Forecast future mandi prices")
    crop5 = st.selectbox("Select Crop", sorted(df["Commodity"].unique()), key="t5_crop")
    markets5 = sorted(df[df["Commodity"] == crop5]["Market"].unique().tolist())
    market5 = st.selectbox("Select Market", markets5, key="t5_market")
    n_periods = st.slider("Weeks to forecast ahead", 4, 26, 12)

    metrics, forecast_df = forecast_price(df, crop5, market5, n_periods=n_periods)
    if metrics is None:
        st.warning("Not enough history for this crop/market combination to forecast reliably (need 30+ records).")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Model error (MAE)", f"₹{metrics['MAE']}/quintal")
        col2.metric("Model error (MAPE)", f"{metrics['MAPE_pct']}%")
        st.line_chart(forecast_df.set_index("Arrival_Date")["Forecast_Price"])
        st.dataframe(forecast_df, use_container_width=True)
        st.caption(
            "Forecast is a recursive gradient-boosting model trained on lag + rolling + "
            "calendar features, evaluated on a held-out recent slice of real history. "
            "MAPE shown is the model's typical % error on that held-out data — treat forecasts "
            "as directional guidance, not guaranteed prices."
        )

with tab6:
    st.subheader("Predict crop yield & get recommendations (national model)")
    st.warning(
        "Trained on real yield/soil/weather data for 30 Indian states — "
        "**Rajasthan is not among them**, so this cannot yet recommend crops "
        "for Rajasthan or connect to the mandi price tabs above. It's a "
        "standalone national model until Rajasthan-specific soil/weather data is added."
    )
    yield_df, yield_pipe, yield_metrics = get_yield_model()

    col1, col2 = st.columns(2)
    col1.metric("Model MAE", yield_metrics["MAE"])
    col2.metric("Model R²", yield_metrics["R2"])

    state6 = st.selectbox("Select State", sorted(yield_df["state"].unique()))
    season6 = st.selectbox("Select Season", ["Any"] + sorted(yield_df["season"].unique()))
    top_n6 = st.slider("Number of crops to recommend", 3, 10, 5)

    rec6 = recommend_crops_for_state(yield_pipe, yield_df, state6, season6, top_n6)
    if rec6.empty:
        st.info("No historical records for this state/season combination.")
    else:
        st.dataframe(rec6, use_container_width=True)
        st.caption(
            "Relative_Performance = predicted yield ÷ that crop's historical average yield "
            "across all states — lets crops with very different units (e.g. coconut counted "
            "in nuts vs wheat in tonnes/hectare) be compared fairly."
        )

st.divider()
st.caption("Data: real Agmarknet-style mandi price records (Rajasthan) — 2001-2026, 14 markets, 112 commodities. National crop yield model: 30 states, 55 crops, 1997-2020.")
