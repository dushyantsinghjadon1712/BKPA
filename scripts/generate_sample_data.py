"""
generate_sample_data.py
------------------------
Creates a synthetic mandi price dataset shaped exactly like Agmarknet exports,
so the rest of the Phase 1 pipeline (cleaning, analysis, dashboard) can be
built and tested right now, before you plug in the real government CSV.

Real data source (download manually, no auth needed for bulk reports):
https://agmarknet.gov.in/  ->  Price and Arrival Reports

Once you have the real file, just replace data/mandi_prices_raw.csv with it.
The column names below match Agmarknet's export headers, so no code changes
should be needed in clean_data.py.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

CROPS = ["Wheat", "Bajra", "Moong", "Groundnut", "Tomato", "Onion", "Mustard", "Gram"]

# state -> list of (district, market) pairs
STATE_MARKETS = {
    "Rajasthan": [
        ("Jaipur", "Kota Mandi"),
        ("Jaipur", "Bharatpur Mandi"),
        ("Alwar", "Alwar Mandi"),
        ("Bikaner", "Bikaner Mandi"),
    ],
    "Madhya Pradesh": [
        ("Indore", "Indore Mandi"),
        ("Bhopal", "Bhopal Mandi"),
    ],
    "Punjab": [
        ("Ludhiana", "Ludhiana Mandi"),
        ("Amritsar", "Amritsar Mandi"),
    ],
}

BASE_PRICE = {
    "Wheat": 2350, "Bajra": 1900, "Moong": 6800, "Groundnut": 5600,
    "Tomato": 1400, "Onion": 1600, "Mustard": 5100, "Gram": 5300,
}

VOLATILITY = {  # relative daily noise; tomato/onion are famously volatile
    "Wheat": 0.02, "Bajra": 0.02, "Moong": 0.03, "Groundnut": 0.03,
    "Tomato": 0.18, "Onion": 0.15, "Mustard": 0.03, "Gram": 0.03,
}

rows = []
start_date = datetime(2023, 1, 1)
num_days = 540  # ~18 months of history

for day_offset in range(0, num_days, 3):  # arrivals reported every ~3 days
    date = start_date + timedelta(days=day_offset)
    month = date.month
    season = "Kharif" if month in [6, 7, 8, 9, 10] else "Rabi"

    for crop in CROPS:
        for state, markets in STATE_MARKETS.items():
            for district, market in markets:
                base = BASE_PRICE[crop]
                vol = VOLATILITY[crop]

                # seasonal drift + market-specific offset + random noise
                seasonal = 1 + 0.05 * np.sin(2 * np.pi * month / 12)
                market_offset = np.random.uniform(-0.04, 0.04)
                noise = np.random.normal(0, vol)

                modal_price = round(base * seasonal * (1 + market_offset + noise), 0)
                min_price = round(modal_price * np.random.uniform(0.92, 0.97), 0)
                max_price = round(modal_price * np.random.uniform(1.03, 1.08), 0)
                arrival_qty = round(np.random.uniform(50, 900), 1)  # quintals

                rows.append({
                    "State": state,
                    "District": district,
                    "Market": market,
                    "Commodity": crop,
                    "Season": season,
                    "Arrival_Date": date.strftime("%Y-%m-%d"),
                    "Min_Price": min_price,
                    "Max_Price": max_price,
                    "Modal_Price": modal_price,
                    "Arrival_Qty_Quintal": arrival_qty,
                })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/bkpa/data/mandi_prices_raw.csv", index=False)
print(f"Generated {len(df)} rows -> data/mandi_prices_raw.csv")
print(df.head())
