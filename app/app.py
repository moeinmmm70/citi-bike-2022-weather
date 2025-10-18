# app/st_dashboard_Part_2.py
from pathlib import Path
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ────────────────────────────── Page/Theming ──────────────────────────────
st.set_page_config(page_title="NYC Citi Bike — Strategy Dashboard", page_icon="🚲", layout="wide")
pio.templates.default = "plotly_white"

# ────────────────────────────── Paths/Constants ───────────────────────────
DATA_PATH = Path("data/processed/reduced_citibike_2022.csv")   # ≤25MB sample
MAP_HTMLS = [
    Path("reports/map/citibike_trip_flows_2022.html"),
    Path("reports/map/NYC_Bike_Trips_Aggregated.html"),
]
cover_path = Path("reports/cover_bike.webp")

RIDES_COLOR = "#1f77b4"   # readable blue
TEMP_COLOR  = "#d62728"   # readable red

# ────────────────────────────── Helpers ───────────────────────────────────
def kfmt(x):
    try:
        x = float(x)
    except Exception:
        return "—"
    units = ["", "K", "M", "B", "T"]
    for u in units:
        if abs(x) < 1000 or u == units[-1]:
            return f"{x:,.0f}{u}" if u == "" else f"{x:.1f}{u}"
        x /= 1000.0

def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.dropna(), b.dropna()
    j = a.index.intersection(b.index)
    if len(j) < 3:
        return None
    c = np.corrcoef(a.loc[j], b.loc[j])[0,1]
    return float(c)

@st.cache_data
def load_data(path: Path, _sig: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Normalize/parse timestamps → date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "started_at" in df.columns:
        df["date"] = pd.to_datetime(df["started_at"], errors="coerce").dt.floor("D")
    # Station names to category (performance)
    for col in ["start_station_name", "end_station_name", "member_casual", "usertype"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # Season if missing
    if "season" not in df.columns and "date" in df.columns:
        def season_from_month(m):
            if m in (12, 1, 2):  return "Winter"
            if m in (3, 4, 5):   return "Spring"
            if m in (6, 7, 8):   return "Summer"
            return "Autumn"
        df["season"] = df["date"].dt.month.map(season_from_month).astype("category")
    return df

def ensure_daily(df: pd.DataFrame) -> pd.DataFrame | None:
    """Guarantee a daily table with columns: date, bike_rides_daily, optional avg_temp_c."""
    if {"date", "bike_rides_daily"}.issubset(df.columns):
        daily = df[["date", "bike_rides_daily"]].dropna().drop_duplicates()
    elif "date" in df.columns:
        # Trip-level sample: approximate daily rides by counting rows
        daily = df.groupby("date", as_index=False).agg(bike_rides_daily=("date", "size"))
    else:
        return None

    # Attach a temperature column if any reasonable candidate exists
    for cand in ["avg_temp_c", "avgTemp", "avg_temp", "temperature_c"]:
        if cand in df.columns:
            temp = (df.groupby("date", as_index=False)[cand]
                      .mean()
                      .rename(columns={cand: "avg_temp_c"}))
            daily = daily.merge(temp, on="date", how="left")
            break
    return daily.sort_values("date")

def apply_filters(df: pd.DataFrame,
                  daterange: tuple[pd.Timestamp, pd.Timestamp] | None,
                  seasons: list[str] | None,
                  usertype: str | None,
                  temp_range: tuple[float, float] | None) -> pd.DataFrame:
    out = df.copy()
    if daterange and "date" in out.columns:
        out = out[(out["date"] >= daterange[0]) & (out["date"] <= daterange[1])]
    if seasons and "season" in out.columns:
        out = out[out["season"].isin(seasons)]
    # User type column can be "member_casual" (modern) or "usertype" (classic)
    if usertype and usertype != "All":
        if "member_casual" in out.columns:
            out = out[out["member_casual"].astype(str).str.lower() == usertype.lower()]
        elif "usertype" in out.columns:
            out = out[out["usertype"].astype(str).str.lower() == usertype.lower()]
    # Temperature filter only meaningful on daily-level, but allow rough trip-level filter too
    if temp_range and any(col in out.columns for col in ["avg_temp_c", "avgTemp", "avg_temp", "temperature_c"]):
        tempcol = "avg_temp_c" if "avg_temp_c" in out.columns else \
                  "avgTemp" if "avgTemp" in out.columns else \
                  "avg_temp" if "avg_temp" in out.columns else "temperature_c"
        out = out[(out[tempcol] >= temp_range[0]) & (out[tempcol] <= temp_range[1])]
    return out

def linear_fit(x: pd.Series, y: pd.Series):
    """Return slope, intercept for y ~ a*x + b and yhat func. Requires ≥2 points."""
    valid = (~x.isna()) & (~y.isna())
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None, None, lambda z: np.nan
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b), (lambda z: a * np.asarray(z) + b)

def kpi(value, label, delta=None):
    st.metric(label=label, value=kfmt(value) if pd.notnull(value) else "—",
              delta=(f"{delta:+.1f}%" if isinstance(delta, (int, float)) else None))

# ────────────────────────────── Sidebar / Data ─────────────────────────────
st.sidebar.header("⚙️ Controls")

if not DATA_PATH.exists():
    st.sidebar.error(f"Missing data: {DATA_PATH}")
    st.error("Data file not found. Create the ≤25MB sample CSV at data/processed/reduced_citibike_2022.csv.")
    st.stop()

df = load_data(DATA_PATH, DATA_PATH.stat().st_mtime)

# Filters (built from available columns)
date_min = pd.to_datetime(df["date"].min()) if "date" in df.columns else None
date_max = pd.to_datetime(df["date"].max()) if "date" in df.columns else None
date_range = None
if date_min is not None:
    date_range = st.sidebar.date_input("Date range", value=(date_min, date_max))

seasons_all = ["Winter", "Spring", "Summer", "Autumn"]
seasons = None
if "season" in df.columns:
    seasons = st.sidebar.multiselect("Season(s)", seasons_all, default=seasons_all)

usertype = None
if "member_casual" in df.columns or "usertype" in df.columns:
    usertype = st.sidebar.selectbox("User type", ["All", "member", "casual"])

temp_range = None
for tc in ["avg_temp_c", "avgTemp", "avg_temp", "temperature_c"]:
    if tc in df.columns:
        tmin, tmax = float(np.nanmin(df[tc])), float(np.nanmax(df[tc]))
        temp_range = st.sidebar.slider("Temperature filter (°C)", tmin, tmax, (tmin, tmax))
        break

st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "📑 Analysis page",
    [
        "Intro",
        "Weather vs Bike Usage",
        "Station Popularity",
        "Pareto: Share of Rides",
        "Trip Flows Map",
        "Weekday × Hour Heatmap",
        "What-if: Temp → Rides",
        "Recommendations",
    ],
)

# One filtered frame to rule them all
df_f = apply_filters(
    df,
    (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])) if date_range else None,
    seasons,
    usertype,
    temp_range,
)

# Daily table (post-filter) for many charts/KPIs
daily_all = ensure_daily(df)
daily_f   = ensure_daily(df_f)

# ────────────────────────────── KPI Band (top of all pages) ───────────────
with st.container():
    c1, c2, c3, c4, c5 = st.columns(5)
    # Total rides in current filter
    total_rides = len(df_f) if "bike_rides_daily" not in df_f.columns else int(df_f["bike_rides_daily"].sum())
    # Avg rides/day
    avg_day = None
    if daily_f is not None and not daily_f.empty:
        avg_day = float(daily_f["bike_rides_daily"].mean())
    # Corr temp↔rides
    corr_tr = None
    if daily_f is not None and "avg_temp_c" in daily_f.columns:
        corr_tr = safe_corr(daily_f.set_index("date")["bike_rides_daily"], daily_f.set_index("date")["avg_temp_c"])
    # Peak season (by total rides)
    peak_szn = None
    if "season" in df_f.columns and len(df_f) > 0:
        peak_szn = df_f.groupby("season").size().sort_values(ascending=False).index[0]
    # WoW delta (weekly total)
    wow = None
    if daily_f is not None and not daily_f.empty:
        wk = daily_f.set_index("date")["bike_rides_daily"].resample("W").sum()
        if len(wk) >= 2 and wk.iloc[-2] != 0:
            wow = (wk.iloc[-1] - wk.iloc[-2]) / wk.iloc[-2] * 100

    c1.metric("Total rides (filtered)", kfmt(total_rides))
    c2.metric("Avg rides/day", kfmt(avg_day) if avg_day is not None else "—")
    c3.metric("Temp ↔ rides corr", f"{corr_tr:+.2f}" if corr_tr is not None else "—")
    c4.metric("Peak season", peak_szn if peak_szn else "—")
    kpi(wk.iloc[-1] if daily_f is not None and len(daily_f) else None, "Weekly rides", wow)

st.markdown("---")

# ────────────────────────────── Pages ──────────────────────────────────────
if page == "Intro":
    st.title("NYC Citi Bike — Strategy Dashboard")
    if cover_path.exists():
       st.image(str(cover_path), use_container_width=True, caption="NYC Citi Bike — 2022 Season Snapshot")
    else:
       st.warning("Cover image not found at reports/cover_bike.webp")
    st.caption("Purpose: pinpoint **where/when** inventory pressure emerges and what to do about it.")
    st.markdown(
        "- 🌤️ **Weather vs Usage** — seasonality & demand swings\n"
        "- 🚉 **Station Popularity** — hotspots to prioritize\n"
        "- 📈 **Pareto** — concentration of demand\n"
        "- 🗺️ **Trip Flows** — likely rebalancing corridors\n"
        "- ⏰ **Weekday × Hour** — temporal load patterns\n"
        "- 🧪 **What-if** — quick temperature sensitivity"
    )
    st.info("Use the sidebar filters; all charts follow your selections.")

elif page == "Weather vs Bike Usage":
    st.header("🌤️ Daily Bike Rides vs Temperature")
    if daily_f is None or daily_f.empty:
        st.warning("Daily metrics aren’t available. Provide `bike_rides_daily` or raw trips with `date`.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Rides
        fig.add_trace(
            go.Scatter(
                x=daily_f["date"], y=daily_f["bike_rides_daily"],
                mode="lines", name="Daily Bike Rides", line=dict(color=RIDES_COLOR, width=2)
            ),
            secondary_y=False
        )
        # Temperature (if present)
        if "avg_temp_c" in daily_f.columns and daily_f["avg_temp_c"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=daily_f["date"], y=daily_f["avg_temp_c"],
                    mode="lines", name="Avg Temperature (°C)", line=dict(color=TEMP_COLOR, width=2, dash="dot")
                ),
                secondary_y=True
            )
            fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

        fig.update_layout(title="Daily Bike Rides vs Temperature — NYC (2022)", hovermode="x unified", height=520)
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_yaxes(title_text="Bike Rides (count)", secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)

        # Scatter + simple OLS fit (no heavy deps)
        if "avg_temp_c" in daily_f.columns and daily_f["avg_temp_c"].notna().any():
            st.subheader("Temperature sensitivity (scatter)")
            x = daily_f["avg_temp_c"]
            y = daily_f["bike_rides_daily"]
            a, b, yhat = linear_fit(x, y)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Days"))
            if a is not None:
                xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                fig2.add_trace(go.Scatter(x=xs, y=yhat(xs), mode="lines", name=f"Fit: rides ≈ {a:.1f}·temp + {b:.0f}"))
            fig2.update_layout(height=480, xaxis_title="Avg Temp (°C)", yaxis_title="Bike Rides")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**So what?** Warm days lift demand; shoulder seasons show the steepest slope. Staff rebalancing accordingly.")

elif page == "Station Popularity":
    st.header("🚉 Most Popular Start Stations")
    if "start_station_name" not in df_f.columns:
        st.warning("`start_station_name` not found in sample.")
    else:
        topN = st.slider("Top N stations", 10, 100, 20, 5)
        s = (df_f.assign(n=1)
                  .groupby("start_station_name")["n"].sum()
                  .sort_values(ascending=False)
                  .head(topN)
                  .reset_index())
        fig = go.Figure(go.Bar(x=s["start_station_name"], y=s["n"]))
        fig.update_layout(
            height=550,
            title=f"Top {topN} start stations — rides count",
            xaxis_title="Start Station", yaxis_title="Rides (count)",
            margin=dict(l=20, r=20, t=60, b=120)
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download top stations (CSV)", s.to_csv(index=False).encode("utf-8"), "top_stations.csv", "text/csv")

elif page == "Pareto: Share of Rides":
    st.header("📈 Pareto Curve — Demand Concentration")
    if "start_station_name" not in df_f.columns:
        st.warning("`start_station_name` not found in sample.")
    else:
        counts = (df_f.assign(n=1)
                        .groupby("start_station_name")["n"].sum()
                        .sort_values(ascending=False))
        cum = (counts.cumsum() / counts.sum()).reset_index()
        cum["rank"] = np.arange(1, len(cum) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum["rank"], y=cum["n"], mode="lines", name="Cumulative share"))
        # 80% line
        fig.add_hline(y=0.80, line_dash="dot")
        # rank at 80%
        idx80 = int(np.searchsorted(cum["n"].values, 0.8))
        if 0 < idx80 < len(cum):
            fig.add_vline(x=cum.loc[idx80, "rank"], line_dash="dot")
            fig.add_annotation(x=cum.loc[idx80, "rank"], y=0.82, showarrow=False,
                               text=f"Top ~{int(cum.loc[idx80,'rank']):,} stations ≈ 80% of rides")
        fig.update_layout(height=520, xaxis_title="Stations (ranked)", yaxis_title="Cumulative share of rides",
                          title="How concentrated is demand?")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Action:** prioritize inventory and maintenance on the head of the curve; treat the tail as on-demand.")

elif page == "Trip Flows Map":
    st.header("🗺️ Trip Flows (Kepler.gl)")
    path_to_html = None
    for p in MAP_HTMLS:
        if p.exists():
            path_to_html = p
            break
    if not path_to_html:
        st.info("Kepler.gl HTML not found.\n\nPlace one of:\n"
                "- `reports/map/citibike_trip_flows_2022.html`\n"
                "- `reports/map/NYC_Bike_Trips_Aggregated.html`")
    else:
        try:
            html_data = Path(path_to_html).read_text(encoding="utf-8")
            st.components.v1.html(html_data, height=900, scrolling=True)
        except Exception as e:
            st.error(f"Failed to load map HTML: {e}")

    # Optional Sankey (if end stations present)
    if {"start_station_name", "end_station_name"}.issubset(df_f.columns):
        st.subheader("Top Origin → Destination flows (Sankey)")
        flows = (df_f.groupby(["start_station_name", "end_station_name"])
                      .size().reset_index(name="count")
                      .sort_values("count", ascending=False).head(20))
        labels = pd.Index(pd.concat([flows["start_station_name"], flows["end_station_name"]]).unique())
        lab2id = {v: i for i, v in enumerate(labels)}
        src = flows["start_station_name"].map(lab2id)
        tgt = flows["end_station_name"].map(lab2id)
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels.tolist()),
            link=dict(source=src, target=tgt, value=flows["count"])
        )])
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Weekday × Hour Heatmap":
    st.header("⏰ Temporal Load — Weekday × Start Hour")
    if "started_at" not in df_f.columns:
        st.info("Need `started_at` timestamps in the sample to build this view.")
    else:
        dt = pd.to_datetime(df_f["started_at"], errors="coerce")
        dfx = pd.DataFrame({"weekday": dt.dt.weekday, "hour": dt.dt.hour})
        grid = (dfx.groupby(["weekday", "hour"]).size().rename("rides").reset_index())
        # Pivot to a 7×24 grid
        mat = grid.pivot(index="weekday", columns="hour", values="rides").reindex(index=range(0,7), columns=range(0,24)).fillna(0)
        fig = px.imshow(mat, aspect="auto", origin="lower",
                        labels=dict(color="Rides"),
                        x=list(range(0,24)), y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        fig.update_layout(height=580, title="Heatmap of rides by weekday/hour")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Tip:** identify commute peaks vs weekend leisure hours for targeted rebalancing.")

elif page == "What-if: Temp → Rides":
    st.header("🧪 What-if: Temperature impact on rides")
    if daily_all is None or daily_all.empty or "avg_temp_c" not in daily_all.columns:
        st.info("Requires a daily table with `avg_temp_c`.")
    else:
        # Fit on ALL daily data to keep model stable; apply current filters for context
        x = daily_all["avg_temp_c"]
        y = daily_all["bike_rides_daily"]
        a, b, yhat = linear_fit(x, y)
        tmin, tmax = float(np.nanmin(x)), float(np.nanmax(x))
        t = st.slider("Forecast average temperature (°C)", tmin, tmax, float(np.nanmedian(x)), 0.5)
        pred = yhat([t])[0] if a is not None else np.nan

        col1, col2, col3 = st.columns(3)
        col1.metric("Model", f"rides ≈ {a:.1f}·temp + {b:.0f}" if a is not None else "—")
        col2.metric("Temp (°C)", f"{t:.1f}")
        col3.metric("Predicted rides", kfmt(pred))

        # Show fitted relationship on filtered context
        if daily_f is not None and not daily_f.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_f["avg_temp_c"], y=daily_f["bike_rides_daily"],
                                     mode="markers", name="Filtered days"))
            xs = np.linspace(tmin, tmax, 100)
            fig.add_trace(go.Scatter(x=xs, y=yhat(xs), mode="lines", name="Fitted (all data)"))
            fig.update_layout(height=520, xaxis_title="Avg Temp (°C)", yaxis_title="Bike Rides",
                              title="Observed (filtered) vs fitted line (all days)")
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Lightweight linear fit shown for demonstrative planning; enrich later with rain, wind, weekday effects.")

elif page == "Recommendations":
    st.header("🚀 Recommendations (ops-ready)")
    st.markdown(
        "- **Staff for heat:** lift afternoon staging on warm days; pre-position near waterfront leisure loops.\n"
        "- **Target the head:** top stations (Pareto head) deserve higher baseline inventory and faster swap cycles.\n"
        "- **Commute windows:** weekday 7–9 and 17–19 spikes — schedule rebalancing trucks just before peaks.\n"
        "- **Corridors:** align truck loops with persistent OD corridors; stage at endpoints to reduce empty miles.\n"
        "- **Monitor anomalies:** calendar-level outliers (events/weather alerts) should trigger temporary boosts."
    )

# ────────────────────────────── Footer ─────────────────────────────────────
st.markdown("---")
st.caption("Built for stakeholder decisions. Data: Citi Bike (2022) + reduced daily weather sample.")
