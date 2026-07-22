"""
analysis.py
-----------
Phase 1, Step 3: Price Analysis
Reusable functions for the dashboard and for a quick CLI report.
"""

import pandas as pd

CLEAN_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/mandi_prices_clean.csv"


def load_clean(path: str = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Arrival_Date"])
    return df


def best_mandis_for_crop(df: pd.DataFrame, crop: str, top_n: int = 5) -> pd.DataFrame:
    """Average modal price per market for a crop, sorted best (highest) first."""
    subset = df[df["Commodity"].str.lower() == crop.lower()]
    result = (
        subset.groupby(["Market", "State"])["Modal_Price"]
        .mean()
        .round(0)
        .sort_values(ascending=False)
        .reset_index()
        .head(top_n)
    )
    result.columns = ["Market", "State", "Avg_Modal_Price"]
    return result


def price_trend(df: pd.DataFrame, crop: str, market: str = None) -> pd.DataFrame:
    """Monthly average price trend for a crop, optionally filtered to one market."""
    subset = df[df["Commodity"].str.lower() == crop.lower()]
    if market:
        subset = subset[subset["Market"].str.lower() == market.lower()]
    trend = (
        subset.groupby([subset["Arrival_Date"].dt.to_period("M")])["Modal_Price"]
        .mean()
        .round(0)
        .reset_index()
    )
    trend["Arrival_Date"] = trend["Arrival_Date"].astype(str)
    return trend


def volatility_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank crops by price volatility (std dev of % daily price change)."""
    result = (
        df.groupby("Commodity")["Price_Change_Pct"]
        .std()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )
    result.columns = ["Commodity", "Volatility_StdDev_Pct"]

    def label(v):
        if v > 8:
            return "HIGH"
        elif v > 3:
            return "MEDIUM"
        return "LOW"

    result["Risk_Level"] = result["Volatility_StdDev_Pct"].apply(label)
    return result


def crop_recommendation(df: pd.DataFrame, state: str, season: str, top_n: int = 3) -> pd.DataFrame:
    """
    Simple profitability-proxy recommendation: rank crops grown in a
    state+season by average modal price and low volatility.
    (Placeholder for the Phase-2 ML model — same interface, swappable.)
    """
    subset = df[(df["State"].str.lower() == state.lower()) & (df["Season"].str.lower() == season.lower())]
    stats = subset.groupby("Commodity").agg(
        Avg_Price=("Modal_Price", "mean"),
        Volatility=("Price_Change_Pct", "std"),
    ).reset_index()
    stats["Avg_Price"] = stats["Avg_Price"].round(0)
    stats["Volatility"] = stats["Volatility"].round(2)
    # profitability score: reward higher price, penalize volatility
    stats["Score"] = (stats["Avg_Price"] / stats["Avg_Price"].max()) - (
        stats["Volatility"].fillna(0) / (stats["Volatility"].max() or 1) * 0.3
    )
    return stats.sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    df = load_clean()
    print("=== Best Mandis for Wheat ===")
    print(best_mandis_for_crop(df, "Wheat"))
    print("\n=== Volatility Ranking ===")
    print(volatility_ranking(df))
    print("\n=== Crop Recommendation: Rajasthan, Kharif ===")
    print(crop_recommendation(df, "Rajasthan", "Kharif"))
