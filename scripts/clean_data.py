"""
clean_data.py
-------------
Phase 1, Step 2 & 4: Data Cleaning + Feature Engineering

Handles the REAL Agmarknet export schema:
State, District, Market, Commodity, Variety, Grade, Arrival_Date (DD/MM/YYYY),
Min_Price, Max_Price, Modal_Price, Commodity_Code

Input : data/mandi_prices_raw.csv
Output: data/mandi_prices_clean.csv

Run: python3 scripts/clean_data.py
"""

import pandas as pd
import numpy as np

RAW_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/mandi_prices_raw.csv"
CLEAN_PATH = "C:/Users/dushy/OneDrive/Desktop/BKPA/bkpa/data/mandi_prices_clean.csv"

KHARIF_MONTHS = {6, 7, 8, 9, 10}
RABI_MONTHS = {11, 12, 1, 2, 3}


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    required = ["Commodity", "Market", "Modal_Price", "Arrival_Date"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    for col in ["Commodity", "Market", "State", "District", "Variety", "Grade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], format="%d/%m/%Y", errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["Arrival_Date"])
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with unparseable dates")

    dedup_cols = [c for c in ["Market", "Commodity", "Variety", "Grade", "Arrival_Date"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols)

    df = df[(df["Min_Price"] > 0) & (df["Max_Price"] > 0) & (df["Modal_Price"] > 0)]
    df = df[(df["Min_Price"] <= df["Modal_Price"]) & (df["Modal_Price"] <= df["Max_Price"])]

    return df.reset_index(drop=True)


def derive_season(month: int) -> str:
    if month in KHARIF_MONTHS:
        return "Kharif"
    elif month in RABI_MONTHS:
        return "Rabi"
    return "Zaid"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Commodity", "Market", "Arrival_Date"]).reset_index(drop=True)

    df["Month"] = df["Arrival_Date"].dt.month
    df["Year"] = df["Arrival_Date"].dt.year
    df["Season"] = df["Month"].apply(derive_season)

    df["Price_Change_Pct"] = (
        df.groupby(["Commodity", "Market"])["Modal_Price"].pct_change() * 100
    )

    df["Rolling_Avg_Price"] = (
        df.groupby(["Commodity", "Market"])["Modal_Price"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )

    df["Volatility_Score"] = (
        df.groupby(["Commodity", "Market"])["Price_Change_Pct"]
        .transform(lambda s: s.rolling(window=10, min_periods=3).std())
    )

    return df


if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    df = engineer_features(df)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned {len(df)} rows -> {CLEAN_PATH}")
    print(f"Date range: {df['Arrival_Date'].min().date()} to {df['Arrival_Date'].max().date()}")
    print(f"Commodities: {df['Commodity'].nunique()}  Markets: {df['Market'].nunique()}")
