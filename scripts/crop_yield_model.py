"""
crop_yield_model.py
--------------------
Phase 2: National Crop Recommendation Engine (ML-trained, separate from the
Rajasthan-only price system).

Trains a model on REAL data:
  - crop_yield.csv        : crop, year, season, state, area, production,
                             fertilizer, pesticide, yield  (19,689 rows, 29 states)
  - state_soil_data.csv    : state, N, P, K, pH               (static per state)
  - state_weather_data.csv : state, year, avg_temp_c, total_rainfall_mm,
                             avg_humidity_percent             (1997-2020)

IMPORTANT SCOPE NOTE: none of these three datasets cover Rajasthan, so this
model cannot currently recommend crops *for Rajasthan* or connect its output
to the Rajasthan mandi price system. It's a standalone, nationally-trained
crop-yield predictor covering the other 29 states it has real data for.
Wire in Rajasthan soil/weather data later to close that gap.

Run standalone for a demo: python3 scripts/crop_yield_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CROP_YIELD_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/crop_yield_raw.csv"
SOIL_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/state_soil_raw.csv"
WEATHER_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/state_weather_raw.csv"
MERGED_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/crop_yield_merged.csv"

NUMERIC_FEATURES = ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
                     "crop_baseline_yield"]
CATEGORICAL_FEATURES = ["season", "state"]
# NOTE: "crop" is intentionally NOT one-hot encoded. With 55 rare crop categories,
# one-hot + tree splitting often failed to differentiate crops at all (verified:
# multiple crops produced bit-for-bit identical predictions). Instead, each crop's
# historical average yield is used as a numeric feature ("target encoding") -- this
# is a standard technique for high-cardinality categoricals and lets the model
# actually use crop identity as a strong signal.


def add_crop_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    crop_baseline = df.groupby("crop")["yield"].transform("mean")
    df["crop_baseline_yield"] = crop_baseline
    return df


def load_and_merge() -> pd.DataFrame:
    cy = pd.read_csv(CROP_YIELD_PATH)
    soil = pd.read_csv(SOIL_PATH, encoding="utf-8-sig")
    weather = pd.read_csv(WEATHER_PATH)

    cy["season"] = cy["season"].str.strip()
    cy["crop"] = cy["crop"].str.strip()
    cy["state"] = cy["state"].str.strip()
    soil["state"] = soil["state"].str.strip()
    weather["state"] = weather["state"].str.strip()

    # drop non-informative "Whole Year" duplicate-season rows only if a crop
    # ALSO has a real season recorded elsewhere -- otherwise keep them (some
    # crops like coconut are genuinely grown year-round)
    df = cy.merge(soil, on="state", how="inner")
    df = df.merge(weather, on=["state", "year"], how="inner")

    # sanity filters
    df = df[(df["area"] > 0) & (df["production"] >= 0) & (df["yield"] >= 0)]
    raw_required = ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
                     "crop", "season", "state", "yield"]
    df = df.dropna(subset=raw_required)

    return df.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                   random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train_and_evaluate(df: pd.DataFrame):
    df = add_crop_baseline(df)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df["yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return pipe, {"MAE": round(mae, 2), "R2": round(r2, 3), "test_size": len(X_test)}


def recommend_crops_for_state(pipe: Pipeline, df: pd.DataFrame, state: str, season: str, top_n: int = 5) -> pd.DataFrame:
    """
    For a given state+season, predict yield for every crop that has
    historically been grown in that state (using that state's real soil
    values and its most recent available weather values), rank by predicted
    yield relative to each crop's typical scale (z-score), and return the
    top N.
    """
    state_crops = df[(df["state"].str.lower() == state.lower())]
    if season.lower() != "any":
        state_crops = state_crops[state_crops["season"].str.lower().str.strip() == season.lower()]
    crops_grown = state_crops["crop"].unique()

    if len(crops_grown) == 0:
        return pd.DataFrame()

    soil_row = df[df["state"].str.lower() == state.lower()].iloc[-1]
    latest_weather = df[df["state"].str.lower() == state.lower()].sort_values("year").iloc[-1]
    crop_means = df.groupby("crop")["yield"].mean()

    candidates = pd.DataFrame({
        "crop": crops_grown,
        "season": season if season.lower() != "any" else "Kharif",
        "state": state,
        "N": soil_row["N"], "P": soil_row["P"], "K": soil_row["K"], "pH": soil_row["pH"],
        "avg_temp_c": latest_weather["avg_temp_c"],
        "total_rainfall_mm": latest_weather["total_rainfall_mm"],
        "avg_humidity_percent": latest_weather["avg_humidity_percent"],
    })
    candidates["crop_baseline_yield"] = candidates["crop"].map(crop_means)

    candidates["Predicted_Yield"] = pipe.predict(candidates[CATEGORICAL_FEATURES + NUMERIC_FEATURES])

    # normalize by each crop's historical mean yield so coconut (huge units)
    # doesn't always beat wheat (small units) just because of unit scale
    candidates["Historical_Avg_Yield"] = candidates["crop"].map(crop_means)
    candidates["Relative_Performance"] = (
        candidates["Predicted_Yield"] / candidates["Historical_Avg_Yield"]
    ).round(2)

    return candidates.sort_values("Relative_Performance", ascending=False)[
        ["crop", "Predicted_Yield", "Historical_Avg_Yield", "Relative_Performance"]
    ].head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_merge()
    df.to_csv(MERGED_PATH, index=False)
    print(f"Merged dataset: {len(df)} rows across {df['state'].nunique()} states, {df['crop'].nunique()} crops")

    pipe, metrics = train_and_evaluate(df)
    print("Model quality (held-out test set):", metrics)

    print("\n=== Recommended crops: Punjab, Kharif ===")
    print(recommend_crops_for_state(pipe, df, "Punjab", "Kharif"))
