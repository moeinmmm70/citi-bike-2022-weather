import streamlit as st
import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="NYC Bike Strategy Dashboard", layout="wide")
st.title("NYC Bike Strategy Dashboard")
st.markdown("Interactive analysis of **popular stations** and **daily rides vs temperature**.")

# ---- Paths ----
ROOT = Path(r"C:\Users\moein\anaconda3\citi-bike-2022-weather")
PROC_DIR = ROOT / "data" / "processed"
top20_path   = PROC_DIR / "top20_stations.csv"
dataset_path = PROC_DIR / "citibike_2022_trips_with_weather.csv"

# ---- Caching loader ----
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# ---- Load ----
top20 = load_csv(top20_path)
df    = load_csv(dataset_path)

DATE_COL = "date"
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
# normalize temperature name → avg_temp_c
for k in ["avg_temp_c", "temperature", "temp_avg"]:
    if k in df.columns:
        df = df.rename(columns={k: "avg_temp_c"})

# ---- daily aggregate ----
if "bike_rides_daily" in df.columns:
    daily = df[[DATE_COL, "bike_rides_daily"]].dropna().drop_duplicates()
else:
    daily = df.groupby(DATE_COL, as_index=False).size().rename(columns={"size": "bike_rides_daily"})

if "avg_temp_c" in df.columns:
    tmp = (df.groupby(DATE_COL, as_index=False)["avg_temp_c"].mean()
           if len(df) > len(daily) else
           df[[DATE_COL, "avg_temp_c"]].dropna().drop_duplicates())
    daily = daily.merge(tmp, on=DATE_COL, how="left")

daily = daily.sort_values(DATE_COL)

# ---- simple human formatter (avoids numerize) ----
def human(n):
    try:
        n = float(n)
    except Exception:
        return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}P"

# ---- KPIs ----
total_rides   = int(daily["bike_rides_daily"].sum())
avg_daily     = float(daily["bike_rides_daily"].mean())
avg_temp_c    = float(daily["avg_temp_c"].mean()) if "avg_temp_c" in daily.columns else None

c1, c2, c3 = st.columns(3)
c1.metric("Total Rides (sum)", human(total_rides))
c2.metric("Avg Daily Rides",   human(round(avg_daily)))
c3.metric("Avg Temp (°C)", f"{avg_temp_c} °C" if avg_temp_c is not None else "—")

st.markdown("---")

# ---- Bar: Top 20 Stations ----
station_col = "start_station_name" if "start_station_name" in top20.columns else top20.columns[0]
ycol = "trips" if "trips" in top20.columns else top20.columns[1]
fig_bar = go.Figure(go.Bar(
    x=top20[station_col], y=top20[ycol],
    marker={"color": top20[ycol], "colorscale": "Spectral"},
    hovertemplate="<b>%{x}</b><br>Trips: %{y:,}<extra></extra>"
))
fig_bar.update_layout(title="Top 20 Most Popular Stations — NYC (2022)",
                      xaxis_title="Start Station", yaxis_title="Trips",
                      height=520, margin=dict(l=40, r=20, t=60, b=120))
fig_bar.update_xaxes(tickangle=40, tickfont=dict(size=10))
st.plotly_chart(fig_bar, use_container_width=True)

# ---- Dual-axis line: rides vs temp ----
fig_line = make_subplots(specs=[[{"secondary_y": True}]])
fig_line.add_trace(go.Scatter(x=daily[DATE_COL], y=daily["bike_rides_daily"],
                              mode="lines", name="Daily Bike Rides"),
                   secondary_y=False)
if "avg_temp_c" in daily.columns:
    fig_line.add_trace(go.Scatter(x=daily[DATE_COL], y=daily["avg_temp_c"],
                                  mode="lines", name="Daily Temperature (°C)"),
                       secondary_y=True)
fig_line.update_layout(title="Daily Bike Rides vs Daily Temperature — NYC (2022)",
                       height=520, hovermode="x unified")
fig_line.update_layout(colorway=["#6a040f", "#457b9d", "#7570B3"])

fig_line.update_xaxes(rangeslider_visible=True)
fig_line.update_yaxes(title_text="Bike Rides (count)", secondary_y=False)
if "avg_temp_c" in daily.columns:
    fig_line.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
st.plotly_chart(fig_line, use_container_width=True)