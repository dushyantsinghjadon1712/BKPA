"""
rajasthan_bridge.py
--------------------
Connects the two previously-separate pipelines:

  1. Rajasthan mandi price data  (analysis.py, real, Rajasthan-only)
  2. National crop yield model   (crop_yield_model.py, real, 30 states, NO Rajasthan)

Rajasthan has no soil/weather rows in the national dataset, so this module
builds an APPROXIMATE proxy profile by averaging the soil and most-recent
weather values of Rajasthan's real-data neighbor states: Gujarat, Madhya
Pradesh, and Haryana. Every output row is labeled as a proxy-based estimate,
never presented as ground truth for Rajasthan.

It also maps mandi commodity names (e.g. "Bengal Gram(Gram)(Whole)") to the
national dataset's crop names (e.g. "Gram") so the two real datasets can be
joined on an actual shared concept, not just a shared column name.

Run standalone: python3 scripts/rajasthan_bridge.py
"""

import pandas as pd

from analysis import load_clean, best_mandis_for_crop, volatility_ranking
from crop_yield_model import load_and_merge, train_and_evaluate, CATEGORICAL_FEATURES, NUMERIC_FEATURES

PROXY_NEIGHBOR_STATES = ["Gujarat", "Madhya Pradesh", "Haryana"]

# Mandi commodity name -> national crop_yield dataset crop name.
# Built by manually matching the two real datasets' vocabularies (see README).
MANDI_TO_YIELD_CROP = {
    "Arhar (Tur/Red Gram)(Whole)": "Arhar/Tur",
    "Arhar Dal(Tur Dal)": "Arhar/Tur",
    "Red Gram": "Arhar/Tur",
    "Bajra(Pearl Millet/Cumbu)": "Bajra",
    "Banana": "Banana",
    "Barley (Jau)": "Barley",
    "Barley(Jau)": "Barley",
    "Bengal Gram(Gram)(Whole)": "Gram",
    "Gram Raw(Chholia)": "Gram",
    "Black Gram (Urd Beans)(Whole)": "Urad",
    "Black Gram(Urd Beans)(Whole)": "Urad",
    "Castor Seed": "Castor seed",
    "Coconut Seed": "Coconut",
    "Coriander(Leaves)": "Coriander",
    "Corriander Seed": "Coriander",
    "Cowpea (Lobia/Karamani)": "Cowpea(Lobia)",
    "Cowpea(Lobia/Karamani)": "Cowpea(Lobia)",
    "Garlic": "Garlic",
    "Ginger(Dry)": "Ginger",
    "Ginger(Green)": "Ginger",
    "Green Gram (Moong)(Whole)": "Moong(Green Gram)",
    "Green Gram(Moong)(Whole)": "Moong(Green Gram)",
    "Groundnut": "Groundnut",
    "Groundnut Pods (Raw)": "Groundnut",
    "Guar": "Guar seed",
    "Guar Seed(Cluster Beans Seed)": "Guar seed",
    "Jowar(Sorghum)": "Jowar",
    "Lentil (Masur)(Whole)": "Masoor",
    "Lentil(Masur)(Whole)": "Masoor",
    "Masur Dal": "Masoor",
    "Linseed": "Linseed",
    "Maize": "Maize",
    "Mustard": "Rapeseed &Mustard",
    "Niger Seed (Ramtil)": "Niger seed",
    "Onion": "Onion",
    "Paddy(Basmati)": "Rice",
    "Paddy(Common)": "Rice",
    "Paddy(Dhan)(Common)": "Rice",
    "Potato": "Potato",
    "Sesamum(Sesame,Gingelly,Til)": "Sesamum",
    "Soyabean": "Soyabean",
    "Sunflower": "Sunflower",
    "Sweet Potato": "Sweet potato",
    "Turmeric (Raw)": "Turmeric",
    "Wheat": "Wheat",
}


def build_rajasthan_proxy_profile(yield_df: pd.DataFrame) -> dict:
    """Average soil (static) + most-recent-year weather across the 3 proxy states."""
    neighbor_rows = yield_df[yield_df["state"].isin(PROXY_NEIGHBOR_STATES)]

    soil_avg = neighbor_rows[["N", "P", "K", "pH"]].mean().to_dict()

    latest_per_state = (
        neighbor_rows.sort_values("year")
        .groupby("state")
        .tail(1)[["avg_temp_c", "total_rainfall_mm", "avg_humidity_percent"]]
    )
    weather_avg = latest_per_state.mean().to_dict()

    return {**soil_avg, **weather_avg}


def combined_rajasthan_recommendation(
    mandi_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    yield_pipe,
    season: str = "Kharif",
    top_n: int = 8,
) -> pd.DataFrame:
    """
    For each mandi commodity that has a known mapping to the national yield
    dataset's crop names, predict yield using the Rajasthan neighbor-state
    proxy profile, then join with REAL Rajasthan price + volatility stats.
    Returns one row per mandi commodity with both signals side by side.
    """
    profile = build_rajasthan_proxy_profile(yield_df)
    crop_means = yield_df.groupby("crop")["yield"].mean()

    rows = []
    for mandi_crop, yield_crop in MANDI_TO_YIELD_CROP.items():
        if mandi_crop not in mandi_df["Commodity"].unique():
            continue
        if yield_crop not in yield_df["crop"].unique():
            continue

        x = pd.DataFrame([{
            "season": season, "state": "Rajasthan",  # state is one-hot; unseen category is fine (handle_unknown="ignore")
            "crop_baseline_yield": crop_means[yield_crop],
            **profile,
        }])
        predicted_yield = yield_pipe.predict(x[CATEGORICAL_FEATURES + NUMERIC_FEATURES])[0]
        relative_perf = round(predicted_yield / crop_means[yield_crop], 2)

        price_rows = mandi_df[mandi_df["Commodity"] == mandi_crop]
        avg_price = round(price_rows["Modal_Price"].mean(), 0)
        volatility = price_rows["Price_Change_Pct"].std()

        rows.append({
            "Mandi_Commodity": mandi_crop,
            "Matched_National_Crop": yield_crop,
            "Avg_Rajasthan_Price": avg_price,
            "Volatility_StdDev_Pct": round(volatility, 2) if pd.notna(volatility) else None,
            "Proxy_Relative_Yield_Performance": relative_perf,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # combined score: normalize price and yield performance, penalize volatility
    result["Price_Norm"] = result["Avg_Rajasthan_Price"] / result["Avg_Rajasthan_Price"].max()
    result["Yield_Norm"] = result["Proxy_Relative_Yield_Performance"] / result["Proxy_Relative_Yield_Performance"].max()
    vol_max = result["Volatility_StdDev_Pct"].max() or 1
    result["Combined_Score"] = (
        0.4 * result["Price_Norm"]
        + 0.4 * result["Yield_Norm"]
        - 0.2 * (result["Volatility_StdDev_Pct"].fillna(0) / vol_max)
    )

    result = result.drop(columns=["Price_Norm", "Yield_Norm"])
    return result.sort_values("Combined_Score", ascending=False).head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    mandi_df = load_clean()
    yield_df = load_and_merge()
    yield_pipe, metrics = train_and_evaluate(yield_df)
    print("Yield model quality:", metrics)

    print("\n=== Combined Rajasthan Recommendation (Kharif, proxy-based) ===")
    print(combined_rajasthan_recommendation(mandi_df, yield_df, yield_pipe, season="Kharif"))
