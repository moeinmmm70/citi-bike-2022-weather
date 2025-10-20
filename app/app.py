# app/st_dashboard_Part_2.py
from pathlib import Path
import numpy as np
import pandas as pd
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

RIDES_COLOR = "#1f77b4"
TEMP_COLOR  = "#d62728"

# Legend / label mapping
MEMBER_LABELS = {
    "member": "Member 🧑‍💼",
    "casual": "Casual 🚲",
}
MEMBER_LEGEND_TITLE = "Member Type"

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

def shorten_name(s: str, max_len: int = 22) -> str:
    if not isinstance(s, str):
        return str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def friendly_axis(fig, x=None, y=None, title=None, colorbar=None):
    if x: fig.update_xaxes(title_text=x)
    if y: fig.update_yaxes(title_text=y)
    if title: fig.update_layout(title=title)
    if colorbar and hasattr(fig, "data"):
        for tr in fig.data:
            if hasattr(tr, "colorbar") and tr.colorbar:
                tr.colorbar.title = colorbar

def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.dropna(), b.dropna()
    j = a.index.intersection(b.index)
    if len(j) < 3:
        return None
    c = np.corrcoef(a.loc[j], b.loc[j])[0,1]
    return float(c)

def linear_fit(x: pd.Series, y: pd.Series):
    valid = (~x.isna()) & (~y.isna())
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None, None, lambda z: np.nan
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b), (lambda z: a * np.asarray(z) + b)

def show_cover(cover_path: Path):
    if not cover_path.exists():
        st.warning("Cover image not found at reports/cover_bike.webp")
        return
    try:
        st.image(str(cover_path), use_container_width=True,
                 caption="🚲 Exploring one year of bike sharing in New York City. Photo © citibikenyc.com")
    except TypeError:
        st.image(str(cover_path), use_column_width=True,
                 caption="🚲 Exploring one year of bike sharing in New York City. Photo © citibikenyc.com")

# UI helpers (Intro hero + KPI cards)
def kpi_card(title: str, value: str, sub: str = "", icon: str = "📊"):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{icon} {title}</div>
            <div class="kpi-value" title="{value}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_hero_panel():
    st.markdown(
        """
        <style>
        .hero-panel {
            background: linear-gradient(180deg, rgba(18,22,28,0.95) 0%, rgba(18,22,28,0.86) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 22px 24px;
            box-shadow: 0 10px 26px rgba(0,0,0,0.38);
            text-align: center;
        }
        .hero-title {
            color: #f8fafc;
            font-size: clamp(1.4rem, 1.2rem + 1.6vw, 2.3rem);
            font-weight: 800;
            letter-spacing: .2px;
            margin: 2px 0 6px 0;
        }
        .hero-sub {
            color: #cbd5e1;
            font-size: clamp(.85rem, .8rem + .3vw, 1.0rem);
            margin: 0;
        }
        .kpi-card {
            background: linear-gradient(180deg, rgba(25,31,40,0.80) 0%, rgba(16,21,29,0.86) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 16px 18px;
            box-shadow: 0 10px 26px rgba(0,0,0,0.36);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            min-height: 190px;
            display: flex; flex-direction: column; justify-content: space-between;
        }
        .kpi-title {
            font-size: .95rem;
            color: #cbd5e1;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: clamp(1.25rem, 1.0rem + 1.2vw, 2.0rem);
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.08;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .kpi-sub {
            font-size: .90rem;
            color: #94a3b8;
            margin-top: 6px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .element-container img { border-radius: 16px; }
        </style>
        <div class="hero-panel">
            <h1 class="hero-title">NYC Citi Bike — Strategy Dashboard</h1>
            <p class="hero-sub">Seasonality • Weather–demand correlation • Station intelligence • Time patterns</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────── Data loading & features ────────────────────
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

@st.cache_data
def load_data(path: Path, _sig: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

# ── Enrich with full daily weather if available
wx_path = Path("data/processed/nyc_weather_2022_daily_full.csv")
if wx_path.exists():
    wx = pd.read_csv(wx_path, parse_dates=["date"])
    keep_cols = [c for c in wx.columns if c in [
        "date","avg_temp_c","tmin_c","tmax_c",
        "precip_mm","snow_mm","snow_depth_mm",
        "wind_mps","wind_kph","gust_mps","gust_kph",
        "wet_day","precip_bin","wind_bin"
    ] or c.startswith("wt")]
    wx = wx[keep_cols].copy()
    if "date" in df.columns:
        df = df.merge(wx, on="date", how="left")
        
    # Parse timestamps
    for col in ["date", "started_at", "ended_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure date exists
    if "date" not in df.columns and "started_at" in df.columns:
        df["date"] = df["started_at"].dt.floor("D")

    # Season if missing
    if "season" not in df.columns and "date" in df.columns:
        def season_from_month(m):
            if m in (12,1,2): return "Winter"
            if m in (3,4,5):  return "Spring"
            if m in (6,7,8):  return "Summer"
            return "Autumn"
        df["season"] = df["date"].dt.month.map(season_from_month).astype("category")

    # Trip metrics
    if {"started_at","ended_at"}.issubset(df.columns):
        dur = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
        df["duration_min"] = dur.clip(lower=0)

    if {"start_lat","start_lng","end_lat","end_lng"}.issubset(df.columns):
        df["distance_km"] = _haversine_km(df["start_lat"], df["start_lng"],
                                          df["end_lat"],   df["end_lng"]).astype(float)
    if "duration_min" in df.columns and "distance_km" in df.columns:
        df["speed_kmh"] = (df["distance_km"] / (df["duration_min"]/60)).replace([np.inf,-np.inf], np.nan).clip(upper=60)

    # Temporal fields
    if "started_at" in df.columns:
        ts = df["started_at"]
        df["hour"]    = ts.dt.hour
        df["weekday"] = ts.dt.weekday
        df["month"]   = ts.dt.to_period("M").dt.to_timestamp()

    # Categories for perf
    for c in ["start_station_name","end_station_name","member_type","rideable_type","season"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Pretty legend text for member_type
    if "member_type" in df.columns:
        df["member_type_display"] = (
            df["member_type"].astype(str)
              .map(MEMBER_LABELS)
              .fillna(df["member_type"].astype(str).str.title())
        ).astype("category")

    return df

def ensure_daily(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty or "date" not in df.columns:
        return None

    # Base: rides per day
    daily = df.groupby("date", as_index=False).size().rename(columns={"size":"bike_rides_daily"})

    # Attach weather fields (use first/mean sensibly; daily NOAA is already per-day)
    attach = {}
    if "avg_temp_c"   in df.columns: attach["avg_temp_c"]   = "mean"
    if "tmin_c"       in df.columns: attach["tmin_c"]       = "mean"
    if "tmax_c"       in df.columns: attach["tmax_c"]       = "mean"
    if "precip_mm"    in df.columns: attach["precip_mm"]    = "mean"  # same value; mean avoids duplication
    if "snow_mm"      in df.columns: attach["snow_mm"]      = "mean"
    if "wind_kph"     in df.columns: attach["wind_kph"]     = "mean"
    if "gust_kph"     in df.columns: attach["gust_kph"]     = "mean"
    if "wet_day"      in df.columns: attach["wet_day"]      = "max"   # boolean-ish
    if "precip_bin"   in df.columns: attach["precip_bin"]   = lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
    if "wind_bin"     in df.columns: attach["wind_bin"]     = lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan

    if attach:
        w = df.groupby("date", as_index=False).agg(attach)
        daily = daily.merge(w, on="date", how="left")

    # Season (mode)
    if "season" in df.columns:
        s = df.groupby("date")["season"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan).reset_index()
        daily = daily.merge(s, on="date", how="left")

    return daily.sort_values("date")


def apply_filters(df: pd.DataFrame,
                  daterange: tuple[pd.Timestamp, pd.Timestamp] | None,
                  seasons: list[str] | None,
                  usertype: str | None,
                  temp_range: tuple[float, float] | None,
                  hour_range: tuple[int, int] | None = None,
                  weekdays: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()

    if daterange and "date" in out.columns:
        out = out[(out["date"] >= pd.to_datetime(daterange[0])) & (out["date"] <= pd.to_datetime(daterange[1]))]

    if seasons and "season" in out.columns:
        out = out[out["season"].isin(seasons)]

    if usertype and usertype != "All" and "member_type" in out.columns:
        out = out[out["member_type"].astype(str) == usertype]

    if temp_range and "avg_temp_c" in out.columns:
        out = out[(out["avg_temp_c"] >= temp_range[0]) & (out["avg_temp_c"] <= temp_range[1])]

    if hour_range and "hour" in out.columns:
        lo, hi = hour_range
        out = out[(out["hour"] >= lo) & (out["hour"] <= hi)]

    if weekdays and "weekday" in out.columns:
        name_to_idx = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        idxs = [name_to_idx[w] for w in weekdays if w in name_to_idx]
        out = out[out["weekday"].isin(idxs)]

    return out

def compute_core_kpis(df_f: pd.DataFrame, daily_f: pd.DataFrame | None):
    total_rides = len(df_f)
    avg_day = float(daily_f["bike_rides_daily"].mean()) if daily_f is not None and not daily_f.empty else None
    corr_tr = safe_corr(daily_f.set_index("date")["bike_rides_daily"], daily_f.set_index("date")["avg_temp_c"]) \
              if daily_f is not None and "avg_temp_c" in daily_f.columns else None
    return dict(total_rides=total_rides, avg_day=avg_day, corr_tr=corr_tr)

# Robust plotting helpers
def quantile_bounds(s: pd.Series, lo=0.01, hi=0.995):
    s = pd.to_numeric(s, errors="coerce")
    ql, qh = s.quantile(lo), s.quantile(hi)
    return float(ql), float(qh)

def inlier_mask(df: pd.DataFrame, col: str, lo=0.01, hi=0.995):
    if col not in df.columns:
        return pd.Series([True]*len(df), index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    ql, qh = quantile_bounds(s, lo, hi)
    return (s >= ql) & (s <= qh)

# ────────────────────────────── Sidebar / Data ─────────────────────────────
st.sidebar.header("⚙️ Controls")

if not DATA_PATH.exists():
    st.sidebar.error(f"Missing data: {DATA_PATH}")
    st.error("Data file not found. Create the ≤25MB sample CSV at data/processed/reduced_citibike_2022.csv.")
    st.stop()

df = load_data(DATA_PATH, DATA_PATH.stat().st_mtime)

if st.sidebar.button("🔄 Reload data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Date range
date_min = pd.to_datetime(df["date"].min()) if "date" in df.columns else None
date_max = pd.to_datetime(df["date"].max()) if "date" in df.columns else None
date_range = st.sidebar.date_input("Date range", value=(date_min, date_max)) if date_min is not None else None

# Season
seasons_all = ["Winter","Spring","Summer","Autumn"]
seasons = st.sidebar.multiselect("Season(s)", seasons_all, default=seasons_all) if "season" in df.columns else None

# Member type (pretty labels; raw value for filtering)
usertype = None
if "member_type" in df.columns:
    raw_opts = ["All"] + sorted(df["member_type"].astype(str).unique().tolist())
    usertype = st.sidebar.selectbox(
        "User type",
        raw_opts,
        format_func=lambda v: "All" if v == "All" else MEMBER_LABELS.get(v, str(v).title())
    )

# Temperature filter (optional)
temp_range = None
if "avg_temp_c" in df.columns:
    tmin, tmax = float(np.nanmin(df["avg_temp_c"])), float(np.nanmax(df["avg_temp_c"]))
    temp_range = st.sidebar.slider("Temperature filter (°C)", tmin, tmax, (tmin, tmax))

# --- Time filters ---
hour_range = None
if "hour" in df.columns:
    hour_range = st.sidebar.slider("Hour of day", 0, 23, (6, 22))

weekday_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
weekdays = None
if "weekday" in df.columns:
    weekdays = st.sidebar.multiselect("Weekday(s)", weekday_names, default=weekday_names)

st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "📑 Analysis page",
    [
        "Intro",
        "Weather vs Bike Usage",
        "Trip Metrics (Duration • Distance • Speed)",     # NEW
        "Member vs Casual Profiles",                      # NEW
        "OD Flows (Sankey) & Matrix",                     # NEW
        "Station Popularity",
        "Station Imbalance (In vs Out)",                  # NEW
        "Pareto: Share of Rides",
        "Weekday × Hour Heatmap",
        "Recommendations",
    ],
)

# Filtered data
df_f = apply_filters(
    df,
    (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])) if date_range else None,
    seasons,
    usertype,
    temp_range,
    hour_range=hour_range,
    weekdays=weekdays,
)

daily_all = ensure_daily(df)
daily_f   = ensure_daily(df_f)

# ────────────────────────────── Pages ──────────────────────────────────────
if page == "Intro":
    render_hero_panel()
    show_cover(cover_path)
    st.caption("⚙️ Powered by NYC Citi Bike data • 365 days • Interactive visuals")

    KPIs = compute_core_kpis(df_f, daily_f)

    # Simple weather uplift
    weather_uplift_pct = None
    if daily_f is not None and not daily_f.empty and "avg_temp_c" in daily_f.columns:
        comfy = daily_f.loc[(daily_f["avg_temp_c"] >= 15) & (daily_f["avg_temp_c"] <= 25), "bike_rides_daily"].mean()
        extreme = daily_f.loc[(daily_f["avg_temp_c"] < 5) | (daily_f["avg_temp_c"] > 30), "bike_rides_daily"].mean()
        if pd.notnull(comfy) and pd.notnull(extreme) and extreme not in (0, np.nan):
            weather_uplift_pct = (comfy - extreme) / extreme * 100.0
    weather_str = f"{weather_uplift_pct:+.0f}%" if weather_uplift_pct is not None else "—"

    # Peak Season text
    peak_value, peak_sub = "—", ""
    if "season" in df_f.columns and daily_f is not None and not daily_f.empty:
        tmp = daily_f.copy()
        if "season" not in tmp.columns:
            s_map = df_f.groupby("date")["season"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan).reset_index()
            tmp = tmp.merge(s_map, on="date", how="left")
        m = tmp.groupby("season")["bike_rides_daily"].mean().sort_values(ascending=False)
        if len(m):
            peak_value = f"{m.index[0]}"
            peak_sub   = f"{kfmt(m.iloc[0])} avg trips"

    cA, cB, cC, cD, cE = st.columns(5)
    with cA: kpi_card("Total Trips", kfmt(KPIs["total_rides"]), "Across all stations", "🧮")
    with cB: kpi_card("Daily Average", kfmt(KPIs["avg_day"]) if KPIs["avg_day"] is not None else "—", "Trips per day", "📅")
    with cC: kpi_card("Temp Impact", f"{KPIs['corr_tr']:+.3f}" if KPIs["corr_tr"] is not None else "—", "Correlation coeff.", "🌡️")
    with cD: kpi_card("Weather Uplift", weather_str, "Comfy (15–25°C) vs extreme", "🌦️")
    with cE: kpi_card("Peak Season", peak_value, peak_sub, "🏆")

    st.markdown("### What you’ll find here")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Advanced KPIs**\n\nTotals, Avg/day, Temp↔Rides correlation.")
    c2.info("**Weather Deep-Dive**\n\nScatter + fit, temperature bands.")
    c3.info("**Station Intelligence**\n\nTop stations, Pareto, OD flows (Sankey/Matrix).")
    c4.info("**Time Patterns**\n\nWeekday×Hour heatmap, seasonal/monthly swings.")
    st.caption("Use the sidebar filters; every view updates live.")

elif page == "Weather vs Bike Usage":
    st.header("🌤️ Daily bike rides vs weather")

    if daily_f is None or daily_f.empty:
        st.warning("Daily metrics aren’t available. Provide trip rows with `date` to aggregate.")
    else:
        # Controls
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            roll_win = st.selectbox("Rolling window", ["Off","7d","14d","30d"], index=1)
        with c2:
            show_precip = st.checkbox("Show precipitation bars (mm)", value=("precip_mm" in daily_f.columns))
        with c3:
            show_wind = st.checkbox("Show wind line (kph)", value=("wind_kph" in daily_f.columns))
        with c4:
            color_scatter_by = st.selectbox("Scatter color", ["None","wet_day","precip_bin","wind_bin"], index=1)

        d = daily_f.sort_values("date").copy()

        # Rolling smoother
        if roll_win != "Off":
            n = int(roll_win.replace("d",""))
            for col in ["bike_rides_daily","avg_temp_c","wind_kph"]:
                if col in d.columns:
                    d[f"{col}_roll"] = d[col].rolling(n, min_periods=max(2, n//2), center=True).mean()

        # Build time-series figure with optional precip/wind
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # rides line
        y_rides = d["bike_rides_daily"]
        if roll_win != "Off":
            y_rides = d.get("bike_rides_daily_roll", d["bike_rides_daily"])
        fig.add_trace(
            go.Scatter(
                x=d["date"], y=y_rides, mode="lines", name="Daily bike rides",
                line=dict(color=RIDES_COLOR, width=2)
            ),
            secondary_y=False
        )

        # temperature line (secondary)
        if "avg_temp_c" in d.columns and d["avg_temp_c"].notna().any():
            y_temp = d["avg_temp_c"]
            if roll_win != "Off":
                y_temp = d.get("avg_temp_c_roll", d["avg_temp_c"])
            fig.add_trace(
                go.Scatter(
                    x=d["date"], y=y_temp, mode="lines",
                    name="Average temperature (°C)", line=dict(color=TEMP_COLOR, width=2, dash="dot")
                ),
                secondary_y=True
            )

        # wind line (secondary, faint)
        if show_wind and "wind_kph" in d.columns and d["wind_kph"].notna().any():
            y_wind = d["wind_kph"]
            if roll_win != "Off":
                y_wind = d.get("wind_kph_roll", d["wind_kph"])
            fig.add_trace(
                go.Scatter(
                    x=d["date"], y=y_wind, mode="lines", name="Avg wind (kph)",
                    line=dict(width=1), opacity=0.5
                ),
                secondary_y=True
            )

        # precip bars (primary y, but small)
        if show_precip and "precip_mm" in d.columns and d["precip_mm"].notna().any():
            fig.add_trace(
                go.Bar(
                    x=d["date"], y=d["precip_mm"], name="Precipitation (mm)",
                    marker_color="rgba(100,100,120,0.35)", yaxis="y1", opacity=0.4
                ),
                secondary_y=False
            )

        fig.update_layout(hovermode="x unified", barmode="overlay", height=560)
        fig.update_yaxes(title_text="Bike rides (count)", secondary_y=False)
        y2_title = "Temperature (°C)" + (" + Wind (kph)" if show_wind and "wind_kph" in d.columns else "")
        fig.update_yaxes(title_text=y2_title, secondary_y=True)
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(title="Daily rides vs temperature, precipitation, and wind — NYC (2022)")

        st.plotly_chart(fig, use_container_width=True)

        # ===== Scatter: rides vs temperature with color options =====
        st.subheader("Sensitivity scatter")
        scatter_df = d.dropna(subset=["avg_temp_c", "bike_rides_daily"]).copy()
        color_col = None if color_scatter_by == "None" else color_scatter_by
        labels = {
            "avg_temp_c": "Average temperature (°C)",
            "bike_rides_daily": "Bike rides (count)",
            "wet_day": "Wet day",
            "precip_bin": "Precipitation",
            "wind_bin": "Wind",
        }
        fig2 = px.scatter(
            scatter_df, x="avg_temp_c", y="bike_rides_daily", color=color_col,
            labels=labels, opacity=0.85, trendline="ols"
        )
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)

        # ===== Distribution: rides by precip_bin (or wet/dry fallback) =====
        st.subheader("Distribution by rainfall")
        if "precip_bin" in d.columns and d["precip_bin"].notna().any():
            fig3 = px.box(
                d, x="precip_bin", y="bike_rides_daily",
                labels={"precip_bin":"Precipitation", "bike_rides_daily":"Bike rides per day"},
                category_orders={"precip_bin":["Low","Medium","High"]}
            )
            st.plotly_chart(fig3, use_container_width=True, height=420)
        elif "wet_day" in d.columns:
            fig3 = px.box(
                d, x=d["wet_day"].map({0:"Dry",1:"Wet"}), y="bike_rides_daily",
                labels={"x":"Day type", "bike_rides_daily":"Bike rides per day"}
            )
            st.plotly_chart(fig3, use_container_width=True, height=420)

        # ===== Quick deltas (plain text) =====
        kcols = st.columns(3)
        with kcols[0]:
            if "precip_bin" in d.columns and d["precip_bin"].notna().any():
                lo = d.loc[d["precip_bin"]=="Low","bike_rides_daily"].mean()
                hi = d.loc[d["precip_bin"]=="High","bike_rides_daily"].mean()
                if pd.notnull(lo) and pd.notnull(hi) and lo>0:
                    st.metric("High rain vs Low", f"{(hi-lo)/lo*100:+.0f}%")
        with kcols[1]:
            if "wet_day" in d.columns:
                dry = d.loc[d["wet_day"]==0,"bike_rides_daily"].mean()
                wet = d.loc[d["wet_day"]==1,"bike_rides_daily"].mean()
                if pd.notnull(dry) and pd.notnull(wet) and dry>0:
                    st.metric("Wet vs Dry", f"{(wet-dry)/dry*100:+.0f}%")
        with kcols[2]:
            if "wind_kph" in d.columns:
                breezy = d.loc[d["wind_kph"]>=20,"bike_rides_daily"].mean()
                calm   = d.loc[d["wind_kph"]<10,"bike_rides_daily"].mean()
                if pd.notnull(calm) and pd.notnull(breezy) and calm>0:
                    st.metric("Windy (≥20) vs Calm (<10)", f"{(breezy-calm)/calm*100:+.0f}%")

        st.caption("Notes: Precipitation = mm/day (NOAA), Wind = daily avg kph. Rolling overlay helps see seasonal structure and weather dips.")

elif page == "Trip Metrics (Duration • Distance • Speed)":
    st.header("🚴 Trip metrics (robust view)")

    need = {"duration_min","distance_km","speed_kmh"}
    if not need.issubset(df_f.columns):
        st.info("Need duration, distance, and speed (engineered in load_data).")
    else:
        # Controls
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            robust = st.checkbox("Robust clipping (99.5%)", value=True, help="Hide extreme outliers that crush the axes.")
        with c2:
            log_duration = st.checkbox("Log X: Duration", value=False)
        with c3:
            log_distance = st.checkbox("Log X: Distance", value=False)
        with c4:
            log_speed = st.checkbox("Log X: Speed", value=False)

        # Inlier masks + physical bounds
        m_dur = (inlier_mask(df_f, "duration_min", hi=0.995) if robust else pd.Series(True, index=df_f.index)) & \
                df_f["duration_min"].between(0.5, 240, inclusive="both")
        m_dst = (inlier_mask(df_f, "distance_km", hi=0.995)  if robust else pd.Series(True, index=df_f.index)) & \
                df_f["distance_km"].between(0.01, 30, inclusive="both")
        m_spd = (inlier_mask(df_f, "speed_kmh",   hi=0.995)  if robust else pd.Series(True, index=df_f.index)) & \
                df_f["speed_kmh"].between(0.5, 60, inclusive="both")

        clipped_dur = int((~m_dur).sum()); clipped_dst = int((~m_dst).sum()); clipped_spd = int((~m_spd).sum())

        cA, cB, cC = st.columns(3)
        with cA:
            d = df_f.loc[m_dur, "duration_min"]
            ql, qh = d.quantile([0.01, 0.995]).tolist()
            fig = px.histogram(d, x="duration_min", nbins=60,
                               labels={"duration_min":"Duration (min)"},
                               log_x=log_duration,
                               range_x=[ql, qh] if robust and not log_duration else None)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Clipped rows (duration): {clipped_dur:,}")

        with cB:
            d = df_f.loc[m_dst, "distance_km"]
            ql, qh = d.quantile([0.01, 0.995]).tolist()
            fig = px.histogram(d, x="distance_km", nbins=60,
                               labels={"distance_km":"Distance (km)"},
                               log_x=log_distance,
                               range_x=[ql, qh] if robust and not log_distance else None)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Clipped rows (distance): {clipped_dst:,}")

        with cC:
            d = df_f.loc[m_spd, "speed_kmh"]
            ql, qh = d.quantile([0.01, 0.995]).tolist()
            fig = px.histogram(d, x="speed_kmh", nbins=60,
                               labels={"speed_kmh":"Speed (km/h)"},
                               log_x=log_speed,
                               range_x=[ql, qh] if robust and not log_speed else None)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Clipped rows (speed): {clipped_spd:,}")

        # Scatter: distance vs duration (by member type)
        st.subheader("Distance vs duration (by member type)")
        color_col = "member_type_display" if "member_type_display" in df_f.columns else None

        inliers = df_f[m_dst & m_dur].copy()
        outliers = df_f[~(m_dst & m_dur)].copy()

        nmax = 30000
        if len(inliers) > nmax:
            inliers = inliers.sample(n=nmax, random_state=3)

        fig2 = px.scatter(
            inliers, x="distance_km", y="duration_min", color=color_col,
            labels={"distance_km":"Distance (km)", "duration_min":"Duration (min)", "member_type_display": MEMBER_LEGEND_TITLE},
            opacity=0.9
        )
        if len(outliers):
            fig2.add_trace(go.Scatter(
                x=outliers["distance_km"], y=outliers["duration_min"],
                mode="markers", name="Outliers", opacity=0.15, marker=dict(size=6)
            ))
        xql, xqh = inliers["distance_km"].quantile([0.01, 0.995]).tolist()
        yql, yqh = inliers["duration_min"].quantile([0.01, 0.995]).tolist()
        fig2.update_xaxes(range=[xql, xqh])
        fig2.update_yaxes(range=[yql, yqh])
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)

        # Optional: Speed vs temperature (robust)
        if "avg_temp_c" in df_f.columns:
            st.subheader("Speed vs temperature")
            m_temp = df_f["avg_temp_c"].between(df_f["avg_temp_c"].quantile(0.01),
                                                df_f["avg_temp_c"].quantile(0.995))
            in2 = df_f[m_spd & m_temp]
            if len(in2) > nmax:
                in2 = in2.sample(n=nmax, random_state=4)
            fig3 = px.scatter(
                in2, x="avg_temp_c", y="speed_kmh", color=color_col,
                labels={"avg_temp_c":"Avg temperature (°C)", "speed_kmh":"Speed (km/h)", "member_type_display": MEMBER_LEGEND_TITLE},
                opacity=0.85
            )
            fig3.update_layout(height=520)
            st.plotly_chart(fig3, use_container_width=True)

        st.caption("Robust view clips only for plotting. All rows remain available for other pages/exports.")

elif page == "Member vs Casual Profiles":
    st.header("👥 Member vs Casual riding patterns")
    if "member_type_display" not in df_f.columns or "hour" not in df_f.columns:
        st.info("Need `member_type` and `started_at` (engineered hour).")
    else:
        st.subheader("Hourly profile")
        g = (df_f.groupby(["member_type_display","hour"]).size().rename("rides").reset_index())
        fig = px.line(
            g, x="hour", y="rides", color="member_type_display",
            labels={"hour":"Hour of day", "rides":"Rides", "member_type_display": MEMBER_LEGEND_TITLE}
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Weekday profile")
        g2 = (df_f.groupby(["member_type_display","weekday"]).size().rename("rides").reset_index())
        g2["weekday_name"] = g2["weekday"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
        fig2 = px.line(
            g2, x="weekday_name", y="rides", color="member_type_display",
            labels={"weekday_name":"Weekday","rides":"Rides","member_type_display": MEMBER_LEGEND_TITLE}
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

        if "rideable_type" in df_f.columns and "season" in df_f.columns:
            st.subheader("Rideable type mix by season")
            g3 = (df_f.groupby(["season","member_type_display","rideable_type"]).size().rename("rides").reset_index())
            fig3 = px.bar(
                g3, x="season", y="rides", color="rideable_type", barmode="relative",
                facet_col="member_type_display", facet_col_wrap=2,
                labels={"rides":"Rides","season":"Season","rideable_type":"Bike type","member_type_display": MEMBER_LEGEND_TITLE}
            )
            fig3.update_layout(height=600)
            st.plotly_chart(fig3, use_container_width=True)

elif page == "OD Flows (Sankey) & Matrix":
    st.header("🔀 Origin → Destination flows")
    needed = {"start_station_name","end_station_name"}
    if not needed.issubset(df_f.columns):
        st.info("Need start/end station names.")
    else:
        topN = st.slider("Top N flows", 10, 200, 50, 10)
        flows = (df_f.groupby(["start_station_name","end_station_name"])
                      .size().rename("rides").reset_index()
                      .sort_values("rides", ascending=False)
                      .head(topN))
        st.caption(f"Showing top {len(flows)} OD pairs by rides.")

        # Sankey
        labs = pd.Index(pd.unique(flows[["start_station_name","end_station_name"]].values.ravel()))
        lab_to_i = {s:i for i,s in enumerate(labs)}
        src = flows["start_station_name"].map(lab_to_i)
        tgt = flows["end_station_name"].map(lab_to_i)
        fig = go.Figure(go.Sankey(
            node=dict(label=labs.astype(str).tolist(), pad=6, thickness=12),
            link=dict(source=src, target=tgt, value=flows["rides"].astype(int))
        ))
        fig.update_layout(height=650, title="Top OD flows (Sankey)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("OD Matrix (interactive)")
        keep_starts = flows["start_station_name"].unique().tolist()
        keep_ends   = flows["end_station_name"].unique().tolist()
        small = df_f[df_f["start_station_name"].isin(keep_starts) & df_f["end_station_name"].isin(keep_ends)]
        mat = (small.groupby(["start_station_name","end_station_name"])
                     .size().rename("rides").reset_index()
                     .pivot(index="start_station_name", columns="end_station_name", values="rides").fillna(0))
        figm = px.imshow(mat, aspect="auto", origin="lower",
                         labels=dict(color="Rides"))
        figm.update_layout(height=700)
        st.plotly_chart(figm, use_container_width=True)

elif page == "Station Popularity":
    st.header("🚉 Most popular start stations")
    if "start_station_name" not in df_f.columns:
        st.warning("`start_station_name` not found in sample.")
    else:
        topN = st.slider("Top N stations", 10, 100, 20, 5)
        s = (df_f.assign(n=1)
                  .groupby("start_station_name")["n"].sum()
                  .sort_values(ascending=False)
                  .head(topN)
                  .reset_index())
        s["label"] = s["start_station_name"].astype(str).map(lambda z: shorten_name(z, 28))

        fig = go.Figure(go.Bar(
            x=s["label"], y=s["n"], text=s["n"], textposition="outside",
            hovertext=s["start_station_name"],
            hovertemplate="<b>%{hovertext}</b><br>Rides: %{y:,}<extra></extra>"
        ))
        fig.update_layout(
            height=600,
            title=f"Top {topN} start stations — rides",
            xaxis_title="Station",
            yaxis_title="Rides (count)",
            margin=dict(l=20, r=20, t=60, b=100)
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download top stations (CSV)",
            s[["start_station_name","n"]].rename(columns={"n":"rides"}).to_csv(index=False).encode("utf-8"),
            "top_stations.csv",
            "text/csv"
        )

elif page == "Station Imbalance (In vs Out)":
    st.header("⚖️ Station imbalance (arrivals − departures)")
    need = {"start_station_name","end_station_name"}
    if not need.issubset(df_f.columns):
        st.info("Need start/end station names.")
    else:
        dep = df_f.groupby("start_station_name").size().rename("out").reset_index()
        arr = df_f.groupby("end_station_name").size().rename("in").reset_index()
        s = dep.merge(arr, left_on="start_station_name", right_on="end_station_name", how="outer")
        s["station"] = s["start_station_name"].fillna(s["end_station_name"])
        s = s.drop(columns=["start_station_name","end_station_name"])
        s["in"]  = s["in"].fillna(0).astype(int)
        s["out"] = s["out"].fillna(0).astype(int)
        s["imbalance"] = s["in"] - s["out"]

        topK = st.slider("Show top ±K stations", 5, 40, 15)
        biggest = pd.concat([
            s.sort_values("imbalance", ascending=False).head(topK),
            s.sort_values("imbalance", ascending=True).head(topK)
        ])
        biggest["label"] = biggest["station"].astype(str).str.slice(0,28)

        fig = go.Figure(go.Bar(x=biggest["label"], y=biggest["imbalance"],
                               marker_color=np.where(biggest["imbalance"]>=0, "#2ca02c", "#d62728")))
        fig.update_layout(height=560, title="Stations with largest net IN (green) / OUT (red)")
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        if {"start_lat","start_lng"}.issubset(df_f.columns):
            import pydeck as pdk

            st.subheader("Map — stations sized by net IN/OUT")

            # approximate station coords from starts
            coords = (df_f.groupby("start_station_name")[["start_lat","start_lng"]]
                          .median().rename(columns={"start_lat":"lat","start_lng":"lon"}))

            m = biggest.join(coords, on="station", how="left").dropna(subset=["lat","lon"]).copy()

            if m.empty:
                st.info("No stations to display for the current filters.")
            else:
                # color: green (IN) / red (OUT) — build a list per row (no broadcasting issues)
                m["color"] = m["imbalance"].ge(0).map({
                    True:  [34, 197, 94, 200],   # green-ish
                    False: [220, 38, 38, 200],   # red-ish
                }).astype(object)  # keep as list objects for pydeck

                # radius scale: 60 m base + 35 * sqrt(|Δ|); tweak scale if needed
                scale = st.slider("Map bubble scale", 10, 50, 15)
                m["radius"] = (60 + scale * np.sqrt(m["imbalance"].abs().clip(lower=1))).astype(float)
                tooltip = {
                    "html": "<b>{station}</b><br>IN: {in}<br>OUT: {out}<br>&Delta;: {imbalance}",
                    "style": {"backgroundColor": "rgba(17,17,17,0.85)", "color": "white"}
                }

                # center on data (fallback to Manhattan)
                center_lat = float(m["lat"].median())
                center_lon = float(m["lon"].median())
                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=0, bearing=0)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=m.reset_index(drop=True),
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                )

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/dark-v11",
                    tooltip=tooltip,
                )

                st.pydeck_chart(deck)
                st.caption("Tip: combine with the **Hour of day** and **Weekday(s)** filters in the sidebar to compare AM vs PM redistribution.")

elif page == "Pareto: Share of Rides":
    st.header("📈 Pareto curve — demand concentration")
    if "start_station_name" not in df_f.columns:
        st.warning("`start_station_name` not found in sample.")
    else:
        counts = (df_f.assign(n=1)
                        .groupby("start_station_name")["n"].sum()
                        .sort_values(ascending=False))
        cum = (counts.cumsum() / counts.sum()).reset_index()
        cum["rank"] = np.arange(1, len(cum) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum["rank"], y=cum["n"], mode="lines", name="Cumulative share",
                                 hovertemplate="Rank: %{x}<br>Cumulative share: %{y:.1%}<extra></extra>"))
        fig.add_hline(y=0.80, line_dash="dot")
        idx80 = int(np.searchsorted(cum["n"].values, 0.8))
        if 0 < idx80 < len(cum):
            fig.add_vline(x=cum.loc[idx80, "rank"], line_dash="dot")
            fig.add_annotation(x=cum.loc[idx80, "rank"], y=0.82, showarrow=False,
                               text=f"Top ~{int(cum.loc[idx80,'rank']):,} stations ≈ 80% of rides")
        fig.update_layout(height=520)
        friendly_axis(fig, x="Stations (ranked)", y="Cumulative share of rides", title="Demand concentration (Pareto)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Action:** prioritize inventory and maintenance on the head of the curve; treat the tail as on-demand.")

elif page == "Weekday × Hour Heatmap":
    st.header("⏰ Temporal load — weekday × start hour")
    if "started_at" not in df_f.columns:
        st.info("Need `started_at` timestamps in the sample to build this view.")
    else:
        dt = pd.to_datetime(df_f["started_at"], errors="coerce")
        dfx = pd.DataFrame({"weekday": dt.dt.weekday, "hour": dt.dt.hour})
        grid = (dfx.groupby(["weekday", "hour"]).size().rename("rides").reset_index())
        mat = grid.pivot(index="weekday", columns="hour", values="rides").reindex(index=range(0,7), columns=range(0,24)).fillna(0)
        fig = px.imshow(
            mat, aspect="auto", origin="lower",
            x=list(range(0,24)), y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            labels=dict(color="Rides")
        )
        friendly_axis(fig, x="Hour of day", y="Day of week", title="Rides by weekday and hour", colorbar="Rides")
        fig.update_layout(height=580)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Tip:** identify commute peaks vs weekend leisure hours for targeted rebalancing.")

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

# ────────────────────────────── Footer ─────────────────────────────────────
st.markdown("---")
st.caption("Built for stakeholder decisions. Data: Citi Bike (2022) + reduced daily weather sample.")
