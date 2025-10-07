# NYC Citi Bike 2022 — Weather & Demand Dashboard 🚲

**Goal:** understand **where/when** Citi Bike NYC faces **inventory stress** and turn it into **actionable ops recommendations**.  
**Deliverable:** an interactive, multi-page **Streamlit** dashboard plus the **notebooks** and **data prep** used to build it.

**Live app:** https://citi-bike-2022-weather-ciup73f7hxc9iub5xacdch.streamlit.app/  
**Repository:** https://github.com/moeinmmm70/citi-bike-2022-weather

---

## 📦 What’s Inside

- **Data & Notebooks**: scripts/notebooks to clean, merge, and reduce data for deployment.  
- **Streamlit App**: multi-page dashboard with seasonality, station popularity, trip-flow map, and time-of-day heatmap.  
- **Deployment**: configuration to run on Streamlit Community Cloud using a **≤ 25 MB** sample.

---

## 📁 Repository Structure

> Click through to explore folders/files in this repo.

- [`app/`](app/)  
  - [`st_dashboard_Part_2.py`](app/st_dashboard_Part_2.py) — Main Streamlit app (multi-page)  
  - [`requirements.txt`](app/requirements.txt) — App dependencies for local/dev/Cloud
- [`data/`](data/)  
  - [`raw/`](data/raw/) — Raw trip & weather data *(git-ignored; placeholder only)*  
  - [`processed/`](data/processed/)  
    - [`reduced_citibike_2022.csv`](data/processed/reduced_citibike_2022.csv) — **≤25MB** sample used by the app
- [`notebooks/`](notebooks/) — Jupyter notebooks for download, cleaning, merging, sampling  
- [`reports/`](reports/)  
  - [`map/`](reports/map/)  
    - [`citibike_trip_flows_2022.html`](reports/map/citibike_trip_flows_2022.html)  
    - [`NYC_Bike_Trips_Aggregated.html`](reports/map/NYC_Bike_Trips_Aggregated.html) — Kepler.gl map exports (optional)
- [`.gitignore`](.gitignore) — excludes large data and local artifacts  
- [`README.md`](README.md) — you are here

> 💡 **Note:** Large raw datasets are intentionally **excluded** from version control. Only the **processed, reduced sample** needed for the app is tracked.

---

## 🗃️ Data

- **Citi Bike trips (2022)**: trip-level records (e.g., `started_at`, station names).  
- **Daily weather for NYC (2022)**: aggregated to daily average temperature.

The app reads a **processed sample** at:

- [`data/processed/`](data/processed/reduced_citibike_2022.csv)


**Columns expected by the app** (any extras are fine):

- `date` (daily date)  
- **Either** `bike_rides_daily` (daily aggregated count) **or** raw trip rows with `started_at`  
- Optional: `avg_temp_c` (daily average temperature in °C)  
- Optional: `season` (Winter/Spring/Summer/Autumn); if missing, the app infers it from `date`

---

## ✂️ Creating the ≤ 25 MB Sample

If you’re working from trip-level data, create a light sample and keep only the columns you need. Example:

```python
import pandas as pd
import numpy as np

# Load your full dataset (trip-level)
df = pd.read_csv("data/raw/citibike_2022_full.csv", low_memory=False)

# (Optional) parse dates for later grouping
df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")

# Keep only columns you actually need for the app
cols = [
    "started_at", "start_station_name",  # trip-level charts
    # if you already computed daily aggregates and weather, include those:
    "date", "bike_rides_daily", "avg_temp_c", "season"
]
df = df[[c for c in cols if c in df.columns]]

# Make a reproducible downsample (adjust frac until the CSV is <25MB)
np.random.seed(32)
sample = df.sample(frac=0.08, replace=False, random_state=32)  # ~8% as a starting point

# Save to processed/
sample.to_csv("data/processed/reduced_citibike_2022.csv", index=False)
```

- If **`bike_rides_daily`** is missing, the app automatically computes daily ride counts based on the **`started_at`** timestamp.  
- If **`season`** is missing, the app derives it from the **month** as follows:  
  - **Winter:** December–February (12–2)  
  - **Spring:** March–May (3–5)  
  - **Summer:** June–August (6–8)  
  - **Autumn:** September–November (9–11)

## 💻 Run Locally

### 1️⃣ Clone & create a virtual environment
```bash
git clone https://github.com/moeinmmm70/citi-bike-2022-weather.git
cd citi-bike-2022-weather
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 2️⃣ Install requirements
```bash
pip install -r app/requirements.txt
```

### 3️⃣ Ensure the sample data exists
Place the file:
```bash
data/processed/reduced_citibike_2022.csv  (≤ 25 MB)
```
### 4️⃣ Map HTML
Export your Kepler.gl map to either:
```bash
reports/map/citibike_trip_flows_2022.html
# or
reports/map/NYC_Bike_Trips_Aggregated.html
```

### 5️⃣ Run the app
```bash
streamlit run app/st_dashboard_Part_2.py
```

## 🧭 App Pages & Decisions

### 🏁 Intro
Explains what the dashboard covers, the sample scope, and how to navigate.

---

### 🌤️ Weather vs Bike Usage (dual-axis line)
**Shows:** strong seasonality — warm months (≈ May–Oct) correlate with higher ride volumes.  
**Decision:** scale dock stock and rebalancing windows on warm days and during warm months.

---

### 🚉 Most Popular Stations (bar + season filter + KPIs)
**Shows:** demand concentrates at a handful of hubs (waterfront, Midtown, commute nodes).  
**Decision:** prioritize dock capacity and proactive rebalancing at hotspots — especially in summer and during commute peaks.

---

### 🗺️ Interactive Trip Flows Map (Kepler.gl HTML)
**Shows:** corridors and loops connecting the waterfront and CBD; high-volume OD pairs.  
**Decision:** align truck loop routes with those corridors and stage vehicles near repeated high-flow endpoints.

---

### ⏰ Extra: Weekday × Hour Heatmap
**Shows:** AM/PM weekday peaks (commutes) and weekend midday leisure.  
**Decision:** pre-load commute hubs before **7–9 AM** and **5–7 PM**; shift some rebalancing to late evening.

---

## 🚀 Recommendations

- **Scale hotspot capacity** (use portable docks if needed)  
- **Predictive stocking** by weather + weekday → target ≥ 85 % fill before AM peak, ≥ 70 % before PM peak  
- **Corridor-aligned logistics** (loop routes + staging)  
- **Rider incentives** to return to under-stocked docks  

**KPIs:**  
- Dock-out rate < 5 % (peaks)  
- Complaints ↓ 30 % MoM  
- Truck miles per rebalanced bike ↓ 15 %  
- On-time dock readiness ≥ 90 %

## 🌐 Deploy on Streamlit Community Cloud

1. Push your repo with the **`app/`** code, **processed sample CSV (≤ 25 MB)**, and any **map HTMLs**.  
2. In **Streamlit Cloud**, point to:  
- [`app/`](app/st_dashboard_Part_2.py)
3. Ensure **`app/requirements.txt`** includes all required packages.  
4. If you change data or theme, use **“Clear cache” → “Restart”**.

> ⚙️ **Tip:** If builds stall on heavy wheels, keep dependencies lean and pre-export map HTMLs instead of rendering maps server-side.

## 🔁 Reproducibility & Notes

- The sample generation uses a fixed seed (**32**) for reproducibility.  

- The app is **resilient**:  
  - If **`bike_rides_daily`** is missing → it aggregates from **`started_at`**.  
  - If **`season`** is missing → it’s computed from **month**.
  - Temperature is optional.

- Charts use accessibility-friendly colors (**`plotly_white`** theme + colorblind-safe palette).

## 🧭 Roadmap / Next Steps

- Add **per-station inventory** (if available) to tighten operational recommendations.  
- Integrate **events/holidays** and **precipitation/wind** to explain residual variance.  
- Develop and deploy a **next-day stocking model** with station-level targets.  
- Implement a **CI workflow** to lint/test notebooks and the app on pull requests (PRs).

## 📝 License & Credit

- **Data:** Public sources (Citi Bike trip data & NYC weather); respect each source’s terms.  
- **Code:** MIT License *(or your preferred license)*.  
- **Author:** **Moein Mellat** — built for a strategy stakeholder audience, with a bias toward actionable operational decisions.

## ⚡ Quick Start (TL;DR)

```bash
git clone https://github.com/moeinmmm70/citi-bike-2022-weather.git
cd citi-bike-2022-weather
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt

# Make sure this file exists and is <25 MB:
# data/processed/reduced_citibike_2022.csv

streamlit run app/st_dashboard_Part_2.py
```

> 💡 **Tip:**  
> If something breaks, it’s usually one of the following:  
> - Python version mismatch  
> - Missing sample CSV  
> - Missing Kepler HTML  
>  
> Fix those and you’re back in business. ✅
