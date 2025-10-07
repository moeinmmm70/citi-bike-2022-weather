# app/st_dashboard_Part_2.py
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from streamlit.components.v1 import html as st_html

# --- Page + Plotly theme ---
st.set_page_config(page_title="NYC Citi Bike — Strategy Dashboard", page_icon="🚲", layout="wide")

# ---- Global font injection (Google Fonts) ----
st_html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root { --app-font: 'Source Sans 3', system-ui, -apple-system, 'Segoe UI', Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif; }

/* Blanket override with high specificity */
html, body, .stApp, .block-container, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], [data-testid="stSidebar"] *,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[class^="css-"], [class*=" css-"] {
  font-family: var(--app-font) !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--app-font) !important;
  font-weight: 700 !important;
  letter-spacing: .2px;
}

/* Metrics / tabs / inputs */
[data-testid="stMetricValue"], [data-baseweb="tab"], [role="tab"],
input, textarea, button, select {
  font-family: var(--app-font) !important;
  font-variant-numeric: tabular-nums;
}
</style>
""", height=0)

# Match Plotly figures to the app font
pio.templates.default = "plotly_white"
try:
    pio.templates["plotly_white"].layout.font.family = \
        "Source Sans 3, system-ui, -apple-system, 'Segoe UI', Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif"
except Exception:
    pass

# --- Paths ---
DATA_PATH = Path("data/processed/reduced_citibike_2022.csv")   # <=25MB sample
MAP_HTMLS = [
    Path("reports/map/citibike_trip_flows_2022.html"),
    Path("reports/map/NYC_Bike_Trips_Aggregated.html"),
]

# --- Colors (defined once, not between if/elif) ---
RIDES_COLOR = "#0072B2"   # colorblind-safe blue
TEMP_COLOR  = "#D55E00"   # vermillion

# ---- Small helpers (to avoid extra deps) ----
def kfmt(x):
    """Human-readable number formatting (e.g., 12.3K)."""
    try:
        x = float(x)
    except Exception:
        return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(x) < 1000:
            return f"{x:,.0f}{unit}" if unit == "" else f"{x:.1f}{unit}"
        x /= 1000.0
    return f"{x:.1f}T"

@st.cache_data
def load_data(path: Path, _sig: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # normalize/parse
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "started_at" in df.columns:
        df["date"] = pd.to_datetime(df["started_at"], errors="coerce").dt.floor("D")
    # season if missing
    if "season" not in df.columns and "date" in df.columns:
        def season_from_month(m):
            if m in (12, 1, 2):  return "Winter"
            if m in (3, 4, 5):   return "Spring"
            if m in (6, 7, 8):   return "Summer"
            return "Autumn"
        df["season"] = df["date"].dt.month.map(season_from_month)
    return df

def ensure_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure we have daily metrics for line chart."""
    daily = None
    if {"date","bike_rides_daily"}.issubset(df.columns):
        daily = df[["date","bike_rides_daily"]].dropna().drop_duplicates()
    elif "date" in df.columns:
        # trip-level sample: approximate rides per day by counting rows
        daily = df.groupby("date", as_index=False).agg(bike_rides_daily=("date","size"))
    if daily is not None:
        # attach temperature if any column resembles it
        for cand in ["avg_temp_c","avgTemp","avg_temp","temperature_c"]:
            if cand in df.columns:
                temp = (df.groupby("date", as_index=False)[cand]
                          .mean()
                          .rename(columns={cand:"avg_temp_c"}))
                daily = daily.merge(temp, on="date", how="left")
                break
    return daily

def kpi(value, label):
    st.metric(label=label, value=kfmt(value) if pd.notnull(value) else "—")

# ---- Sidebar Navigation ----
page = st.sidebar.selectbox(
    "📑 Select an aspect of the analysis",
    ["Intro","Weather vs Bike Usage","Most Popular Stations","Interactive Trip Flows Map","Extra: Weekday × Hour Heatmap","Recommendations"]
)

# ---- Load data ----
if not DATA_PATH.exists():
    st.error(f"Data file not found: {DATA_PATH}. Please create the ≤25MB sample first.")
    st.stop()

df = load_data(DATA_PATH, DATA_PATH.stat().st_mtime)

st.sidebar.success("Data loaded")
st.sidebar.write(f"Rows: {len(df):,}")
st.sidebar.info("Using a reduced sample (≤25 MB) for deployment. Figures reflect the sample, not full-year totals.")

# ---- Pages ----

# 1) Intro
if page == "Intro":
    st.title("NYC Citi Bike — Strategy Dashboard")
    st.markdown("### ")
    hero_path = Path("reports/cover_bike.webp")
    if hero_path.exists():
        st.image(hero_path.as_posix(), use_column_width=True, caption="Citi Bike NYC. Photo © citibikenyc.com", output_format="auto")
    st.markdown("""
**Purpose** — 🔎 Pinpoint **where/when** Citi Bike NYC faces **inventory stress** and what to do about it.

**You’ll see**
1. 🌤️ **Weather vs. Usage** — seasonality & demand swings  
2. 🚉 **Popular Stations** — hotspots to prioritize  
3. 🗺️ **Trip Flow Map** — corridors for efficient rebalancing  
4. ⏰ **Weekday × Hour Heatmap** — temporal load patterns  
5. 🚀 **Recommendations** — concrete, ops-ready actions

**Scope** — 📦 Reduced sample of trips + daily weather (≤25 MB) to enable deployment.  
**Tip** — Use the sidebar to switch pages and filter seasons.
""")

# 2) Weather vs Bike Usage (dual-axis)
elif page == "Weather vs Bike Usage":
    st.header("🌤️ Daily Bike Rides vs Temperature (NYC)")
    daily = ensure_daily(df)
    if daily is None or daily.empty:
        st.warning("Daily metrics are not available in the sample. Provide 'bike_rides_daily' or raw trip rows.")
    else:
        daily = daily.sort_values("date")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # left axis: rides
        fig.add_trace(
            go.Scatter(
                x=daily["date"], y=daily["bike_rides_daily"],
                mode="lines", name="Daily Bike Rides",
                line=dict(width=2, color=RIDES_COLOR)
            ),
            secondary_y=False
        )
        # right axis: temp (if available)
        if "avg_temp_c" in daily.columns and daily["avg_temp_c"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=daily["date"], y=daily["avg_temp_c"],
                    mode="lines", name="Daily Temperature (°C)",
                    line=dict(width=2, dash="dot", color=TEMP_COLOR)
                ),
                secondary_y=True
            )
            fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
        else:
            st.info("No temperature column found in sample; showing rides only.")

        # hovers & layout
        fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Rides: %{y:,}",
                          selector=dict(name="Daily Bike Rides"))
        fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Temp: %{y:.1f} °C",
                          selector=dict(name="Daily Temperature (°C)"))
        fig.update_layout(
            title="Daily Bike Rides vs Temperature — NYC (Sample)",
            height=520,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        fig.update_xaxes(rangeslider_visible=True, showgrid=False)
        fig.update_yaxes(title_text="Bike rides (count)", secondary_y=False, showgrid=True, gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ")
        st.markdown("""
**🔎 Takeaway** — Usage **peaks May–Oct**, dips in winter—clear **seasonality**. Warmer days often align with **higher ride volumes**.

**✅ Action** — Scale **dock stock & rebalancing windows** during warm months and on forecasted warm days.

**🧠 Note** — This shows **association**, not causation; account for events/holidays.
""")

# 3) Most Popular Stations (with season filter + KPI)
elif page == "Most Popular Stations":
    st.header("🚉 Top Start Stations — with Season Filter")
    # season filter
    if "season" in df.columns:
        with st.sidebar:
            options = sorted(df["season"].dropna().unique().tolist())
            season_filter = st.multiselect("Select season(s)", options=options, default=options)
        df1 = df.query("season in @season_filter") if season_filter else df.copy()
    else:
        st.info("No 'season' column; showing all data.")
        df1 = df.copy()

    # KPI — total rides (trip-level) or sum of daily rides
    if "ride_id" in df1.columns:
        total = float(len(df1))
    elif "bike_rides_daily" in df1.columns:
        total = float(df1["bike_rides_daily"].sum())
    else:
        total = np.nan

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi(total, "Total rides (filtered)")
    with c2:
        kpi(df1['start_station_name'].nunique() if "start_station_name" in df1 else "—", "Active start stations")
    with c3:
        kpi(len(df1['date'].unique()) if "date" in df1 else "—", "Days covered")

    # guard for empty after filter
    if df1.empty:
        st.info("No rows match the current season filter.")
        st.stop()

    # Bar chart
    if "start_station_name" in df1.columns:
        tmp = df1.assign(value=1).groupby("start_station_name", as_index=False)["value"].sum()
        top20 = tmp.nlargest(20, "value")
        fig = go.Figure(
            go.Bar(
                x=top20["start_station_name"],
                y=top20["value"],
                marker=dict(color="#0E76A8")
            )
        )
        # labels and hover
        fig.update_traces(text=top20["value"].map("{:,}".format),
                          textposition="outside", cliponaxis=False,
                          hovertemplate="<b>%{x}</b><br>Trips: %{y:,}<extra></extra>")
        # layout and axes
        fig.update_layout(
            title="Top 20 Most Popular Start Stations (Filtered)",
            xaxis_title="Start station",
            yaxis_title="Trips (count)",
            height=600,
            uniformtext_minsize=10, uniformtext_mode="hide",
            margin=dict(t=80),
            legend=dict(orientation="h", y=1.1)
        )
        fig.update_xaxes(tickangle=45, automargin=True, showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ")
        st.markdown("""
**🔎 Takeaway** — Demand concentrates at a **handful of hubs** (waterfront, Midtown, commute nodes).

**✅ Action** — Prioritize **dock capacity** and **proactive rebalancing** at these stations—especially in **summer** and **commute peaks**.
""")

    else:
        st.warning("Column 'start_station_name' not available in the sample.")

# 4) Kepler.gl Map (HTML embed)
elif page == "Interactive Trip Flows Map":
    st.header("🗺️ Interactive Map — Aggregated Trip Flows")
    path_to_html = next((p for p in MAP_HTMLS if p.exists()), None)

    if not path_to_html:
        st.error(
            "Kepler.gl HTML not found. Export your flow map to either:\n"
            "• reports/map/citibike_trip_flows_2022.html\n"
            "• reports/map/NYC_Bike_Trips_Aggregated.html"
        )
    else:
        try:
            with open(path_to_html, "r", encoding="utf-8") as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=900, scrolling=True)

            st.markdown("### ")
            st.markdown("""
**How to use this map**
- 🧭 **Zoom & pan** to explore hot corridors.  
- 🌊 Waterfront ↔ CBD **loops** often mix commute + leisure.

**✅ Action** — Align **truck loops** with these corridors and **stage vehicles** near repeated high-flow endpoints.
""")

        except Exception as e:
            st.error(f"Failed to load map HTML: {e}")

# 5) Extra chart (Weekday × Hour Heatmap)
elif page == "Extra: Weekday × Hour Heatmap":
    st.header("⏰ Temporal Load — Weekday × Start Hour")
    if "started_at" in df.columns:
        dt = pd.to_datetime(df["started_at"], errors="coerce")
        dfx = pd.DataFrame({
            "weekday": dt.dt.day_name(),
            "hour": dt.dt.hour
        }).dropna()
        piv = dfx.pivot_table(index="weekday", columns="hour", aggfunc=len, fill_value=0)
        # order weekdays
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        piv = piv.reindex(order)
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=piv.values,
            x=piv.columns.astype(str),
            y=piv.index,
            coloraxis="coloraxis"
        ))
        fig.update_traces(hovertemplate="Weekday: %{y}<br>Hour: %{x}<br>Starts: %{z:,}<extra></extra>")
        fig.update_xaxes(title_text="Start hour (0–23)", showgrid=False)
        fig.update_yaxes(title_text="Weekday", showgrid=True, gridcolor="rgba(0,0,0,0.06)")
        fig.update_layout(
            title="Starts by Weekday × Hour (Count)",
            height=600,
            coloraxis=dict(colorscale="Viridis"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ")
        st.markdown("""
**🔎 Takeaway** — **AM/PM weekday peaks** = commutes; **weekend midday** = leisure.

**✅ Action** — Pre-load commute hubs before **7–9 AM** and **5–7 PM**. Shift some rebalancing to **late evening** to prep for morning demand.
""")

    else:
        st.info("No 'started_at' column in sample. For this chart, keep a small set of raw trips in the ≤25MB CSV.")

# 6) Recommendations
elif page == "Recommendations":
    st.header("🚀 Conclusion & Recommendations")
    st.markdown("### ")
    st.markdown("""
### Recommendations (4–8 weeks)

1) **Scale hotspot capacity**  
   - 🧱 Portable/temporary docks where feasible.  
   - 🎯 Target **≥85% fill at open (AM)** and **≥70% before PM peak** at top-20 stations.

2) **Predictive stocking: weather + weekday**  
   - 📈 Simple regression/rules for **next-day dock targets** by station.  
   - 🌡️ Escalate stocking when **forecast highs ≥ 22 °C**.

3) **Corridor-aligned rebalancing**  
   - 🚚 Stage trucks at **repeated high-flow endpoints**; run **loop routes**.

4) **Rider incentives**  
   - 🎟️ Credits for returns to **under-stocked docks** during commute windows.

**KPIs**  
- ⛔ **Dock-out rate** < 5% at top-20 stations during peaks  
- 📉 **Empty/Full dock complaints** ↓ 30% MoM  
- 🛣️ **Truck miles per rebalanced bike** ↓ 15%  
- ⏱️ **On-time dock readiness** ≥ 90% (before AM peak)
""")
    st.markdown("> **Next** — 🧪 Pilot at the top 10 stations for 2 weeks; compare KPIs before/after.")
    st.caption("🧱 Limitations: sample reduced for deployment; no per-dock inventory; events/holidays not modeled.")

    st.markdown("### ")
    st.markdown("### ")
    st.video("https://www.youtube.com/watch?v=vm37IuX7UPQ")
















