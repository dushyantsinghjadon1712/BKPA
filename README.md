# Bharat Krishi Price Advisor (BKPA)

AI + data analytics platform to help Indian farmers/landlords decide which
crop to grow, which mandi gives the best price, how volatile a crop's price
is, and what prices are likely to look like in the coming weeks — built on
real Indian agriculture data across three phases, now combined into one
running system.

---

## 1. Data — what's real, what's a proxy, and why

| Dataset | Coverage | Source | Role |
|---|---|---|---|
| `mandi_prices_raw.csv` | Rajasthan, 14 markets, 112 commodities, 2001–2026 | your Agmarknet-style export | Phase 1 & 2 pricing + forecasting |
| `crop_yield_raw.csv` | 30 states (**not Rajasthan**), 55 crops, 1997–2020 | your upload | Phase 2 national yield model |
| `state_soil_raw.csv` | 30 states (**not Rajasthan**) | your upload | soil inputs (N, P, K, pH) |
| `state_weather_raw.csv` | 30 states (**not Rajasthan**), 1997–2020 | your upload | weather inputs |

**The core scope limit, stated plainly:** none of your soil/yield/weather
data covers Rajasthan, while 100% of your price data is Rajasthan-only.
There is no real row anywhere saying "Rajasthan + this soil + this rainfall
→ this yield." Rather than fake that row, `rajasthan_bridge.py` builds an
explicit, labeled **proxy**: it averages the soil and most-recent weather
values of Rajasthan's real neighbor states — **Gujarat, Madhya Pradesh,
Haryana** — and feeds that into the national yield model. Every output
column derived this way is named and documented as a proxy, never presented
as Rajasthan ground truth.

---

## 2. Project layout

```
bkpa/
├── data/
│   ├── mandi_prices_raw.csv / _clean.csv       # Phase 1 — real Rajasthan mandi prices
│   ├── crop_yield_raw.csv                      # Phase 2 — real national yield data
│   ├── state_soil_raw.csv                      # Phase 2 — real national soil data
│   ├── state_weather_raw.csv                   # Phase 2 — real national weather data
│   └── crop_yield_merged.csv                   # Phase 2 — joined + cleaned output
├── scripts/
│   ├── clean_data.py            # Phase 1: cleaning + feature engineering (mandi prices)
│   ├── analysis.py              # Phase 1: best mandi, trend, volatility
│   ├── forecast.py              # Phase 2: price forecasting (gradient boosting)
│   ├── crop_yield_model.py      # Phase 2: national crop yield model (random forest)
│   ├── rajasthan_bridge.py      # Phase 2→3: links price data + yield model via proxy
│   └── generate_sample_data.py  # (legacy) synthetic data generator, unused now real data exists
├── dashboard/
│   └── app.py                   # Streamlit dashboard — 6 tabs, good for quick local exploration
├── api/
│   └── main.py                  # Phase 3: FastAPI backend — loads/trains models ONCE at startup
├── webapp/
│   ├── index.html / style.css / app.js   # Phase 3: farmer-facing web app (vanilla JS, no build step)
└── requirements.txt
```

**One thing to know about how this was built**: the FastAPI/Streamlit code was
written and its *logic* was fully validated against your real data by running
every underlying function directly and checking outputs (all confirmed
working — see metrics below). The web server itself (`uvicorn`) could not be
started in the sandbox this was built in (no network access to install it),
so run the actual `uvicorn` command locally as your first smoke test.

---

## 3. How to run the whole thing

```bash
cd bkpa
pip install -r requirements.txt
```

**Option A — FastAPI + web app (the real Phase 3 deliverable):**
```bash
uvicorn api.main:app --reload --port 8000
```
Open **http://127.0.0.1:8000/** for the farmer-facing web app, or
**http://127.0.0.1:8000/docs** for interactive API docs.

**Option B — Streamlit dashboard (faster to poke around locally):**
```bash
streamlit run dashboard/app.py
```

Both read the same underlying scripts/data — they are two front ends on one
system, not two separate projects.

---

## 4. What each phase actually delivers (with real, tested numbers)

### Phase 1 — Mandi Price Intelligence
- Cleaned 241,485 real price records (from 241,521 raw rows) across 14
  Rajasthan markets and 112 commodities
- Best Mandi Finder, Price Trend, Volatility Ranking — all tested against
  real data (e.g. Khatauli came out best for wheat at ₹2,480/quintal avg;
  Plum and Cherry are the most volatile commodities)

### Phase 2 — Forecasting + National Yield Model
- **Price forecasting** (`forecast.py`): gradient boosting on lag + rolling +
  calendar features, time-ordered train/test split. Wheat @ Kota:
  **MAE ≈ ₹276/quintal, MAPE ≈ 10.5%** on 946 held-out real records.
- **National crop yield model** (`crop_yield_model.py`): random forest
  trained on real yield/soil/weather data, 30 states, 55 crops.
  **MAE ≈ 11.5, R² ≈ 0.962** on held-out real data. Crop identity is encoded
  via each crop's historical average yield (not raw one-hot) — an earlier
  version one-hot-encoded 55 rare crop categories directly and several crops
  collapsed to identical predictions; this was caught, diagnosed, and fixed.
- **Rajasthan bridge** (`rajasthan_bridge.py`): joins real Rajasthan price +
  volatility with the yield model's proxy-based prediction via a ~40-crop
  name mapping between the two datasets' vocabularies (e.g.
  `"Bengal Gram(Gram)(Whole)"` ↔ `"Gram"`). Ranks by a blended
  `Combined_Score` (40% price, 40% yield, −20% volatility penalty).

### Phase 3 — Full Platform
- **FastAPI backend** (`api/main.py`): all models loaded/trained once at
  startup (not per-request), exposed as REST endpoints: `/api/best-mandi`,
  `/api/price-trend`, `/api/volatility`, `/api/crop-recommendation`,
  `/api/forecast`, `/api/national-yield`, `/api/meta`. Every endpoint's
  underlying logic was run against real data and confirmed to produce valid
  JSON output.
- **Web app** (`webapp/`): vanilla HTML/CSS/JS (no build step, lightweight to
  deploy), 6 tabs mirroring the dashboard, calling the API via `fetch()`.

---

## 5. Known limitations (stated honestly, not buried)

1. **Rajasthan crop recommendation is proxy-based, not ground truth** — see
   section 1. If you can source real Rajasthan soil/rainfall data later,
   swap it into `state_soil_raw.csv` / `state_weather_raw.csv` and
   `rajasthan_bridge.py`'s proxy logic becomes unnecessary.
2. **Forecasting model** uses scikit-learn gradient boosting, not
   Prophet/ARIMA, because those libraries couldn't be installed in the
   sandbox this was built in (no network). `forecast_arima()` in
   `forecast.py` is a ready-to-use alternative once you `pip install
   statsmodels` locally.
3. **The FastAPI server itself hasn't been run live** in this build
   environment — the logic underneath every endpoint has been, but treat
   your first local `uvicorn` run as the real integration test, not a
   formality.
4. **CORS is wide open** (`allow_origins=["*"]`) for local development
   convenience — tighten this before any public deployment.

---

## 6. Resume / hackathon pitch

> BKPA is an end-to-end agricultural intelligence platform combining a real
> 241K-row Rajasthan mandi price dataset with a nationally-trained crop
> yield model (30 states, R² ≈ 0.96), bridged through an explicitly-labeled
> proxy methodology, and served via a FastAPI backend + web app — with every
> number in this README verified against real data, not simulated.
