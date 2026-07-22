"""
main.py
-------
Phase 3: FastAPI backend for BKPA.

Design choice (per project decision): models are loaded / trained ONCE at
startup and kept in memory (app.state), not retrained per request. This
matches how a real production service would run — training on every request
would make the API unusably slow and non-deterministic between calls.

Run:
    cd bkpa
    pip install -r requirements.txt
    uvicorn api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/ for the web app, or http://127.0.0.1:8000/docs
for interactive API docs (FastAPI auto-generates these from the code below).
"""

import sys
import os
import math
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from analysis import load_clean, best_mandis_for_crop, price_trend, volatility_ranking
from forecast import forecast_price
from crop_yield_model import load_and_merge, train_and_evaluate, recommend_crops_for_state
from rajasthan_bridge import combined_rajasthan_recommendation

WEBAPP_DIR = os.path.join(os.path.dirname(__file__), "..", "webapp")


def clean_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records: dates -> ISO strings, NaN -> None."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    # numpy scalar types (e.g. np.float64) aren't natively JSON-serializable
    for r in records:
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                r[k] = v.item()
            elif isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- load everything once at startup ---
    print("Loading mandi price data...")
    app.state.mandi_df = load_clean()

    print("Training national crop yield model (one-time)...")
    app.state.yield_df = load_and_merge()
    app.state.yield_pipe, app.state.yield_metrics = train_and_evaluate(app.state.yield_df)

    print("BKPA API ready.")
    yield
    # (no teardown needed — in-memory only)


app = FastAPI(
    title="Bharat Krishi Price Advisor (BKPA) API",
    description="Mandi price intelligence, forecasting, and crop yield prediction for Indian farmers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten before a real public deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    """Lists of valid values for building frontend dropdowns."""
    df = app.state.mandi_df
    yield_df = app.state.yield_df
    return {
        "crops": sorted(df["Commodity"].unique().tolist()),
        "markets": sorted(df["Market"].unique().tolist()),
        "seasons": sorted(df["Season"].unique().tolist()),
        "national_states": sorted(yield_df["state"].unique().tolist()),
        "yield_model_metrics": app.state.yield_metrics,
    }


@app.get("/api/best-mandi")
def best_mandi(crop: str = Query(...), top_n: int = Query(5, ge=1, le=20)):
    df = app.state.mandi_df
    if crop not in df["Commodity"].unique():
        raise HTTPException(404, f"Unknown crop: {crop}")
    result = best_mandis_for_crop(df, crop, top_n)
    return clean_records(result)


@app.get("/api/price-trend")
def price_trend_endpoint(crop: str = Query(...), market: str | None = Query(None)):
    df = app.state.mandi_df
    if crop not in df["Commodity"].unique():
        raise HTTPException(404, f"Unknown crop: {crop}")
    result = price_trend(df, crop, market)
    return clean_records(result)


@app.get("/api/volatility")
def volatility():
    df = app.state.mandi_df
    result = volatility_ranking(df)
    return clean_records(result)


@app.get("/api/crop-recommendation")
def crop_recommendation_endpoint(season: str = Query("Kharif"), top_n: int = Query(8, ge=1, le=20)):
    """Combined real-price + proxy-yield recommendation for Rajasthan (see README for methodology)."""
    result = combined_rajasthan_recommendation(
        app.state.mandi_df, app.state.yield_df, app.state.yield_pipe, season=season, top_n=top_n
    )
    return clean_records(result)


@app.get("/api/forecast")
def forecast_endpoint(
    crop: str = Query(...),
    market: str = Query(...),
    periods: int = Query(12, ge=1, le=52),
):
    df = app.state.mandi_df
    if crop not in df["Commodity"].unique():
        raise HTTPException(404, f"Unknown crop: {crop}")
    if market not in df[df["Commodity"] == crop]["Market"].unique():
        raise HTTPException(404, f"No data for {crop} at market {market}")

    metrics, forecast_df = forecast_price(df, crop, market, n_periods=periods)
    if metrics is None:
        raise HTTPException(422, "Not enough history for this crop/market to forecast (need 30+ records).")
    return {"metrics": metrics, "forecast": clean_records(forecast_df)}


@app.get("/api/national-yield")
def national_yield(
    state: str = Query(...),
    season: str = Query("Any"),
    top_n: int = Query(5, ge=1, le=20),
):
    yield_df = app.state.yield_df
    if state not in yield_df["state"].unique():
        raise HTTPException(404, f"Unknown state: {state}")
    result = recommend_crops_for_state(app.state.yield_pipe, yield_df, state, season, top_n)
    if result.empty:
        raise HTTPException(404, "No historical records for this state/season combination.")
    return clean_records(result)


# --- serve the web app frontend ---
app.mount("/", StaticFiles(directory=WEBAPP_DIR, html=True), name="webapp")
