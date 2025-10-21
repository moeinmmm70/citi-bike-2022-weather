# Refactored Streamlit App: NYC Citi Bike Dashboard
# Author: Moein Mellat, 2025-10-21
# Purpose: Visualize and analyze NYC Citi Bike 2022 data with interactive controls.
# This version refactors the original app for clarity, performance, and UX:
#  - Fixed preset button logic (Rainy-day filter now properly filters rainy days).
#  - Removed redundant "Reload data" (integrated into "Reset filters").
#  - Simplified sidebar layout with grouped filters and helpful tooltips.
#  - Replaced checkboxes with toggles for better UX where appropriate.
#  - Encapsulated each analysis page rendering into functions for clarity.
#  - Consolidated repeated code into helper functions (KPI cards, etc.).
#  - Reordered pages: Intro ➞ Weather ➞ Hourly/Weekly Patterns ➞ Rider Types ➞ Trip Metrics ➞ Station Popularity ➞ Pareto (Station Share) ➞ OD Flows ➞ OD Matrix ➞ Station Imbalance ➞ Recommendations.
#  - Unified visual style for KPI cards and consistently structured visuals with tabs and columns.

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import unicodedata

# Optional ML/Stats imports (fail gracefully if not installed)
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
except ImportError:
    linkage = leaves_list = None

# Page configuration and theming
st.set_page_config(page_title="NYC Citi Bike — Strategy Dashboard", page_icon="🚲", layout="wide")
pio.templates.default = "plotly_white"

# Data paths and constants
DATA_PATH = Path("data/processed/reduced_citibike_2022.csv")
cover_path = Path("reports/cover_bike.webp")
RIDES_COLOR = "#1f77b4"
TEMP_COLOR = "#d62728"
MEMBER_LABELS = {"member": "Member 🧑‍💼", "casual": "Casual 🚲"}
MEMBER_LEGEND_TITLE = "Member Type"

# Helper formatting functions
def kfmt(x):
    """Format a number with K/M/B suffixes."""
    try:
        x = float(x)
    except Exception:
        return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(x) < 1000 or unit == "T":
            return f"{x:,.0f}{unit}" if unit == "" else f"{x:.1f}{unit}"
        x /= 1000.0

def shorten_name(s: str, max_len: int = 22) -> str:
    """Truncate a station name for display."""
    if not isinstance(s, str):
        return str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def friendly_axis(fig, x=None, y=None, title=None, colorbar=None):
    """Update axis labels and title of a Plotly figure."""
    if x: fig.update_xaxes(title_text=x)
    if y: fig.update_yaxes(title_text=y)
    if title: fig.update_layout(title=title)
    if colorbar and hasattr(fig, "data"):
        for tr in fig.data:
            if hasattr(tr, "colorbar") and tr.colorbar:
                tr.colorbar.title = colorbar

def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    """Compute Pearson correlation (returns None if insufficient data)."""
    a, b = a.dropna(), b.dropna()
    common_idx = a.index.intersection(b.index)
    if len(common_idx) < 3:
        return None
    c = np.corrcoef(a.loc[common_idx], b.loc[common_idx])[0, 1]
    return float(c)

def linear_fit(x: pd.Series, y: pd.Series):
    """Perform linear regression (y = a + b*x) and return (a, b, predict_func)."""
    valid = (~x.isna()) & (~y.isna())
    x_vals, y_vals = x[valid], y[valid]
    if len(x_vals) < 2:
        return None, None, lambda z: np.nan
    a, b = np.polyfit(x_vals, y_vals, 1)
    return float(a), float(b), (lambda z: a * np.asarray(z) + b)

def _slug(s: str) -> str:
    """Slugify a string: remove emojis/accents, collapse spaces, lowercase."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.split()).lower()

# Heatmap-specific helpers
def _bin_hour(h: pd.Series, bin_size: int) -> pd.Series:
    """Bin hour values into buckets of size bin_size."""
    b = (h // bin_size) * bin_size
    return b.clip(0, 23)

def _weekday_name(idx: pd.Series) -> pd.Series:
    """Map weekday index to weekday abbreviation."""
    return idx.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})

def _make_heat_grid(df: pd.DataFrame, hour_col="hour", weekday_col="weekday", hour_bin: int = 1, scale: str = "Absolute") -> pd.DataFrame:
    """
    Build a 7×(24/hour_bin) grid of ride counts, optionally normalized by row/column or Z-score.
    """
    if hour_col not in df.columns or weekday_col not in df.columns:
        return pd.DataFrame()
    d = df[[hour_col, weekday_col]].dropna().copy()
    d[hour_col] = _bin_hour(d[hour_col].astype(int), hour_bin)
    # Count rides per weekday-hour combination
    g = d.groupby([weekday_col, hour_col]).size().rename("rides").reset_index()
    # Pivot to matrix, fill missing cells with 0
    hours = list(range(0, 24, hour_bin))
    mat = (g.pivot(index=weekday_col, columns=hour_col, values="rides")
             .reindex(index=range(7), columns=hours).fillna(0))
    if scale == "Absolute":
        return mat
    if scale == "Row %":
        row_sum = mat.sum(axis=1).replace(0, np.nan)
        return (mat.T / row_sum).T.mul(100).fillna(0)
    if scale == "Column %":
        col_sum = mat.sum(axis=0).replace(0, np.nan)
        return mat.div(col_sum, axis=1).mul(100).fillna(0)
    if scale == "Z-score":
        m = mat.mean(axis=1)
        s = mat.std(axis=1).replace(0, np.nan)
        return mat.sub(m, axis=0).div(s, axis=0).fillna(0)
    return mat

def _smooth_by_hour(mat: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Smooth each weekday row by a moving average of width k across hours."""
    if mat.empty or k <= 1:
        return mat
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    out = mat.copy()
    for i in out.index:
        row = out.loc[i].to_numpy()
        if np.all(np.isnan(row)):
            continue
        # Reflect padding for convolution
        ext = np.pad(row, (k//2, k//2), mode='reflect')
        smoothed = np.convolve(ext, np.ones(k)/k, mode='valid')
        out.loc[i] = smoothed
    return out

# Inject CSS styles for hero panel and KPI cards (global)
st.write(
    """
    <style>
    .hero-panel {
        background: linear-gradient(180deg, rgba(18,22,28,0.95) 0%, rgba(18,22,28,0.86) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 22px 24px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.28);
        text-align: center;
    }
    .hero-title {
        color: #f8fafc;
        font-size: clamp(1.4rem, 1.2rem + 1.6vw, 2.3rem);
        font-weight: 800;
        letter-spacing: 0.2px;
        margin: 2px 0 6px 0;
    }
    .hero-sub {
        color: #cbd5e1;
        font-size: clamp(0.85rem, 0.8rem + 0.3vw, 1.0rem);
        margin: 0;
    }
    .kpi-card {
        background: linear-gradient(180deg, rgba(25,31,40,0.80) 0%, rgba(16,21,29,0.86) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 16px 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.28);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-title {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-bottom: 6px;
        letter-spacing: 0.2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        font-size: clamp(1.25rem, 1.0rem + 1.2vw, 2.0rem);
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.08;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-sub {
        font-size: 0.90rem;
        color: #94a3b8;
        margin-top: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .element-container img { border-radius: 16px; }
    </style>
    """, unsafe_allow_html=True
)

# Data loading and preprocessing
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Parse timestamps
    for col in ["date", "started_at", "ended_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Ensure 'date' exists for daily grouping
    if "date" not in df.columns and "started_at" in df.columns:
        df["date"] = pd.to_datetime(df["started_at"], errors="coerce").dt.floor("D")
    # Attach daily weather data (if available)
    wx_path = Path("data/processed/nyc_weather_2022_daily_full.csv")
    if wx_path.exists() and "date" in df.columns:
        wx = pd.read_csv(wx_path, parse_dates=["date"])
        keep_cols = [c for c in wx.columns if c in [
            "date", "avg_temp_c", "tmin_c", "tmax_c",
            "precip_mm", "snow_mm", "snow_depth_mm",
            "wind_mps", "wind_kph", "gust_mps", "gust_kph",
            "wet_day", "precip_bin", "wind_bin"
        ] or c.startswith("wt")]
        wx = wx[keep_cols].copy()
        df = df.merge(wx, on="date", how="left")
    # Add season if missing
    if "season" not in df.columns and "date" in df.columns:
        def season_from_month(m):
            return ("Winter" if m in (12, 1, 2)
                    else "Spring" if m in (3, 4, 5)
                    else "Summer" if m in (6, 7, 8) else "Autumn")
        df["season"] = df["date"].dt.month.map(season_from_month).astype("category")
    # Compute trip metrics if possible
    if {"started_at", "ended_at"}.issubset(df.columns):
        dur = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
        df["duration_min"] = dur.clip(lower=0)
    if {"start_lat", "start_lng", "end_lat", "end_lng"}.issubset(df.columns):
        # Haversine distance calculation
        def _haversine_km(lat1, lon1, lat2, lon2):
            R = 6371.0088
            lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        df["distance_km"] = _haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]).astype(float)
    if "duration_min" in df.columns and "distance_km" in df.columns:
        df["speed_kmh"] = (df["distance_km"] / (df["duration_min"] / 60.0)) \
                           .replace([np.inf, -np.inf], np.nan).clip(upper=60)
    # Temporal fields for filtering
    if "started_at" in df.columns:
        ts = df["started_at"]
        df["hour"] = ts.dt.hour
        df["weekday"] = ts.dt.weekday
        df["month"] = ts.dt.to_period("M").dt.to_timestamp()
    # Optimize categorical types
    for c in ["start_station_name", "end_station_name", "member_type", "rideable_type", "season"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    # Add human-friendly member labels
    if "member_type" in df.columns:
        df["member_type_display"] = (df["member_type"].astype(str)
                                     .map(MEMBER_LABELS)
                                     .fillna(df["member_type"].astype(str).str.title()))
        df["member_type_display"] = df["member_type_display"].astype("category")
    return df

def ensure_daily(df: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate trips to daily metrics (including weather if present)."""
    if df is None or df.empty or "date" not in df.columns:
        return None
    # Add avg_temp if missing but tmin/tmax present
    if "avg_temp_c" not in df.columns and {"tmin_c", "tmax_c"}.issubset(df.columns):
        df = df.copy()
        df["avg_temp_c"] = (pd.to_numeric(df["tmin_c"], errors="coerce") +
                            pd.to_numeric(df["tmax_c"], errors="coerce")) / 2.0
    # Base daily ride counts
    daily = df.groupby("date", as_index=False).size().rename(columns={"size": "bike_rides_daily"})
    # Attach weather fields: numeric -> mean, categorical -> mode
    agg_rules = {}
    if "avg_temp_c" in df.columns: agg_rules["avg_temp_c"] = "mean"
    if "tmin_c" in df.columns: agg_rules["tmin_c"] = "mean"
    if "tmax_c" in df.columns: agg_rules["tmax_c"] = "mean"
    if "precip_mm" in df.columns: agg_rules["precip_mm"] = "mean"
    if "snow_mm" in df.columns: agg_rules["snow_mm"] = "mean"
    if "wind_kph" in df.columns: agg_rules["wind_kph"] = "mean"
    if "gust_kph" in df.columns: agg_rules["gust_kph"] = "mean"
    if "wet_day" in df.columns: agg_rules["wet_day"] = "max"
    if "precip_bin" in df.columns: agg_rules["precip_bin"] = lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
    if "wind_bin" in df.columns: agg_rules["wind_bin"] = lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
    if agg_rules:
        weather_daily = df.groupby("date", as_index=False).agg(agg_rules)
        daily = daily.merge(weather_daily, on="date", how="left")
    # Add most common season per day (if multiple seasons, choose mode)
    if "season" in df.columns:
        season_daily = df.groupby("date")["season"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan).reset_index()
        daily = daily.merge(season_daily, on="date", how="left")
    return daily.sort_values("date")

def apply_filters(df: pd.DataFrame,
                  daterange: tuple[pd.Timestamp, pd.Timestamp] | None,
                  seasons: list[str] | None,
                  usertype: str | None,
                  temp_range: tuple[float, float] | None,
                  hour_range: tuple[int, int] | None = None,
                  weekdays: list[str] | None = None) -> pd.DataFrame:
    """Apply all sidebar filters to the data."""
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
        name_to_idx = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        idxs = [name_to_idx[w] for w in weekdays if w in name_to_idx]
        out = out[out["weekday"].isin(idxs)]
    return out

# Load data with caching and verify file availability
if not DATA_PATH.exists():
    st.sidebar.error(f"Missing data file: {DATA_PATH}")
    st.error("Data file not found. Please add the dataset to 'data/processed/reduced_citibike_2022.csv'.")
    st.stop()
df = load_data(DATA_PATH)

# Sidebar controls
st.sidebar.header("⚙️ Controls")
# Preset buttons in two columns
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("✨ Commuter preset", help="Weekdays 6–10 AM & 4–8 PM, mild temps (≈5–25°C), members only"):
        st.session_state["page_select"] = "Weekday × Hour Heatmap"
        if "weekday" in df.columns:
            weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            st.query_params.update({"weekday": ",".join(weekdays)})
        if "hour" in df.columns:
            st.query_params.update({"hour0": "6", "hour1": "20"})
        if "avg_temp_c" in df.columns:
            tmin = float(np.nanmin(df["avg_temp_c"]))
            tmax = float(np.nanmax(df["avg_temp_c"]))
            st.query_params.update({"temp": f"{max(tmin, 5)}:{min(tmax, 25)}"})
        if "member_type" in df.columns:
            st.query_params.update({"usertype": "member"})
with col_p2:
    if st.button("🌧️ Rainy-day preset", help="Focus on rainy days (weather impact analysis)"):
        st.session_state["page_select"] = "Weather vs Bike Usage"
        if "wet_day" in df.columns:
            st.query_params.update({"wet": "1"})

# Reset and share buttons
c_reset, c_share = st.sidebar.columns(2)
with c_reset:
    if st.button("♻️ Reset filters", help="Clear all filters and reload data"):
        st.cache_data.clear()
        if hasattr(st, "query_params"):
            st.query_params.clear()
        st.experimental_rerun()
with c_share:
    if st.button("🔗 Copy current link", help="Show URL parameters for sharing"):
        params = dict(st.experimental_get_query_params()) if not hasattr(st, "query_params") else dict(st.query_params)
        st.sidebar.code(params)
        st.caption("Copy the URL from your browser address bar to share this view.")

# Primary filters in sidebar
date_min = pd.to_datetime(df["date"].min()) if "date" in df.columns else None
date_max = pd.to_datetime(df["date"].max()) if "date" in df.columns else None
date_range = st.sidebar.date_input("Date range", (date_min, date_max)) if date_min is not None else None

seasons_all = ["Winter", "Spring", "Summer", "Autumn"]
seasons = st.sidebar.multiselect("Season(s)", seasons_all, default=seasons_all) if "season" in df.columns else None

usertype = None
if "member_type" in df.columns:
    user_opts = ["All"] + sorted(df["member_type"].astype(str).unique().tolist())
    usertype = st.sidebar.selectbox("User type", user_opts,
                                    format_func=lambda v: "All" if v == "All" else MEMBER_LABELS.get(v, str(v).title()))

# Time filters
hour_range = None
if "hour" in df.columns:
    hour_range = st.sidebar.slider("Hour of day", 0, 23, (0, 23), key="hour_slider")

# Additional filters (collapsed)
temp_range = None
weekdays = None
with st.sidebar.expander("More filters", expanded=False):
    if "avg_temp_c" in df.columns:
        tmin_val = float(np.nanmin(df["avg_temp_c"]))
        tmax_val = float(np.nanmax(df["avg_temp_c"]))
        temp_range = st.slider("Temperature (°C)", tmin_val, tmax_val, (tmin_val, tmax_val), key="temp_slider")
    if "weekday" in df.columns:
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekdays = st.multiselect("Weekday(s)", weekday_names, default=weekday_names, key="weekday_multi")

st.sidebar.markdown("---")

# Read current URL parameters
_qp = dict(st.query_params) if hasattr(st, "query_params") else st.experimental_get_query_params()
initial_page = None
if "page" in _qp:
    initial_page = _qp["page"][0] if isinstance(_qp["page"], list) else _qp["page"]
if initial_page not in [
    "Intro", "Weather vs Bike Usage", "Weekday × Hour Heatmap", "Member vs Casual Profiles",
    "Trip Metrics (Duration • Distance • Speed)", "Station Popularity", "Pareto: Share of Rides",
    "OD Flows — Sankey + Map", "OD Matrix — Top Origins × Dest", "Station Imbalance (In vs Out)", "Recommendations"
]:
    initial_page = "Intro"

# Page selector
page = st.sidebar.selectbox("📑 Analysis page", [
    "Intro", "Weather vs Bike Usage", "Weekday × Hour Heatmap",
    "Member vs Casual Profiles", "Trip Metrics (Duration • Distance • Speed)",
    "Station Popularity", "Pareto: Share of Rides",
    "OD Flows — Sankey + Map", "OD Matrix — Top Origins × Dest",
    "Station Imbalance (In vs Out)", "Recommendations"
], index=(0 if initial_page is None else [
    "Intro", "Weather vs Bike Usage", "Weekday × Hour Heatmap", "Member vs Casual Profiles",
    "Trip Metrics (Duration • Distance • Speed)", "Station Popularity", "Pareto: Share of Rides",
    "OD Flows — Sankey + Map", "OD Matrix — Top Origins × Dest",
    "Station Imbalance (In vs Out)", "Recommendations"
].index(initial_page)), key="page_select")

# Update URL query parameters to reflect current filters (except 'wet' which is preset-only)
try:
    params_to_set = {
        "page": page,
        "date0": str(date_range[0]) if date_range else None,
        "date1": str(date_range[1]) if date_range else None,
        "usertype": usertype or "All",
        "hour0": hour_range[0] if hour_range else None,
        "hour1": hour_range[1] if hour_range else None,
        "weekday": ",".join(weekdays) if weekdays else None,
        "temp": f"{temp_range[0]}:{temp_range[1]}" if temp_range else None
    }
    params_to_set = {k: str(v) for k, v in params_to_set.items() if v not in (None, "", [])}
    if params_to_set:
        st.query_params.update(params_to_set)
except Exception:
    pass

# Filter dataset based on selections
df_f = apply_filters(df, 
                     (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])) if date_range else None,
                     seasons, usertype, temp_range, hour_range, weekdays)

# Apply rainy-day preset filter if present in URL
if _qp.get("wet", ["0"])[0] == "1" and "wet_day" in df_f.columns:
    df_f = df_f[df_f["wet_day"] == 1]

# Compute daily aggregates for full and filtered data
daily_all = ensure_daily(df)
daily_f = ensure_daily(df_f)

st.sidebar.success(f"✅ {len(df_f):,} trips match the filters")

# Backfill weather data for filtered trip-level data (if needed)
def _backfill_trip_weather(df_trips: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df is None or daily_df.empty or "date" not in df_trips.columns:
        return df_trips
    out = df_trips.copy()
    # Build lookup series for each weather column
    lookups = {}
    for col in ["avg_temp_c", "wind_kph", "gust_kph", "precip_mm", "wet_day", "precip_bin", "wind_bin"]:
        if col in daily_df.columns:
            lookups[col] = daily_df.set_index("date")[col]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col, mapper in lookups.items():
        if col not in out.columns or out[col].isna().all():
            if col not in out.columns:
                out[col] = np.nan
            out[col] = out[col].where(out[col].notna(), out["date"].map(mapper))
    return out

df_f = _backfill_trip_weather(df_f, daily_all)

# Compute core KPIs for Intro page
def compute_core_kpis(df_filtered: pd.DataFrame, daily_filtered: pd.DataFrame | None):
    total_rides = len(df_filtered)
    avg_day = float(daily_filtered["bike_rides_daily"].mean()) if daily_filtered is not None and not daily_filtered.empty else None
    corr_tr = safe_corr(daily_filtered.set_index("date")["bike_rides_daily"], 
                        daily_filtered.set_index("date")["avg_temp_c"]) if daily_filtered is not None and "avg_temp_c" in daily_filtered.columns else None
    return {"total_rides": total_rides, "avg_day": avg_day, "corr_tr": corr_tr}

# Time-slice filter utility (common for multiple pages)
def _time_slice(data: pd.DataFrame, slice_label: str) -> pd.DataFrame:
    if slice_label == "Weekday":
        return data[data["weekday"].isin([0, 1, 2, 3, 4])] if "weekday" in data.columns else data
    if slice_label == "Weekend":
        return data[data["weekday"].isin([5, 6])] if "weekday" in data.columns else data
    if slice_label.startswith("AM"):
        return data[data["hour"].between(6, 11)] if "hour" in data.columns else data
    if slice_label.startswith("PM"):
        return data[data["hour"].between(16, 20)] if "hour" in data.columns else data
    return data

# Inlier mask for outlier filtering (trip metrics)
def quantile_bounds(s: pd.Series, lo=0.01, hi=0.995):
    s_num = pd.to_numeric(s, errors="coerce")
    ql, qh = s_num.quantile(lo), s_num.quantile(hi)
    return float(ql), float(qh)

def inlier_mask(df: pd.DataFrame, col: str, lo=0.01, hi=0.995):
    if col not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    s_num = pd.to_numeric(df[col], errors="coerce")
    ql, qh = quantile_bounds(s_num, lo, hi)
    return (s_num >= ql) & (s_num <= qh)

# Page rendering functions
def render_intro_page():
    # Hero panel
    st.markdown(f"""
    <div class="hero-panel">
        <h1 class="hero-title">NYC Citi Bike — Strategy Dashboard</h1>
        <p class="hero-sub">Seasonality • Weather–demand correlation • Station intelligence • Time patterns</p>
    </div>
    """, unsafe_allow_html=True)
    # Selection summary caption
    date_text = f"{date_range[0]} → {date_range[1]}" if date_range else "All dates"
    season_text = "All seasons" if not seasons or set(seasons) == set(["Winter", "Spring", "Summer", "Autumn"]) else ", ".join(seasons)
    user_text = "All users" if usertype in (None, "All") else str(usertype).title()
    hour_text = "All day" if hour_range is None or hour_range == (0, 23) else f"{hour_range[0]:02d}:00–{hour_range[1]:02d}:00"
    st.caption(f"**Selection:** {date_text} · {season_text} · {user_text} · {hour_text}")
    # Cover image or placeholder
    if cover_path.exists():
        st.image(str(cover_path), use_container_width=True, caption="🚲 Exploring one year of bike sharing in New York City (Photo © citibikenyc.com)")
    else:
        st.info("Cover image not found.")
    st.caption("⚙️ Powered by NYC Citi Bike data (2022) + NOAA daily weather")
    # Compute KPIs
    KPIs = compute_core_kpis(df_f, daily_f)
    weather_uplift_pct = None
    if daily_f is not None and not daily_f.empty and "avg_temp_c" in daily_f.columns:
        d_nonnull = daily_f.dropna(subset=["avg_temp_c", "bike_rides_daily"])
        if not d_nonnull.empty:
            comfy = d_nonnull.loc[d_nonnull["avg_temp_c"].between(15, 25, inclusive="both"), "bike_rides_daily"].mean()
            extreme = d_nonnull.loc[~d_nonnull["avg_temp_c"].between(5, 30, inclusive="both"), "bike_rides_daily"].mean()
            if pd.notnull(comfy) and pd.notnull(extreme) and extreme not in (0, np.nan):
                weather_uplift_pct = (comfy - extreme) / extreme * 100.0
    weather_str = f"{weather_uplift_pct:+.0f}%" if weather_uplift_pct is not None else "—"
    coverage_str = "—"
    if daily_f is not None and not daily_f.empty:
        if "avg_temp_c" in daily_f.columns:
            cov = 100.0 * daily_f["avg_temp_c"].notna().mean()
            coverage_str = f"{cov:.0f}%"
        else:
            coverage_str = "0%"
    # Peak season (optional insight)
    peak_value, peak_sub = "—", ""
    if "season" in df_f.columns and daily_f is not None and not daily_f.empty:
        tmp = daily_f.copy()
        if "season" not in tmp.columns:  # ensure daily has season
            season_map = df_f.groupby("date")["season"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan).reset_index()
            tmp = tmp.merge(season_map, on="date", how="left")
        if "season" in tmp.columns:
            season_means = tmp.groupby("season")["bike_rides_daily"].mean().sort_values(ascending=False)
            if len(season_means):
                peak_value = str(season_means.index[0])
                peak_sub = f"{kfmt(season_means.iloc[0])} avg trips"
    # KPI cards
    cA, cB, cC, cD, cE = st.columns(5)
    with cA: kpi_card("Total Trips", kfmt(KPIs.get("total_rides", 0)), "Across all stations", icon="🧮")
    with cB: kpi_card("Daily Average", kfmt(KPIs["avg_day"]) if KPIs.get("avg_day") is not None else "—", "Trips per day (selection)", icon="📅")
    with cC: kpi_card("Temp ↔ Rides (r)", f"{KPIs['corr_tr']:+.3f}" if KPIs.get("corr_tr") is not None else "—", "Correlation (daily)", icon="🌡️")
    with cD: kpi_card("Weather Uplift", weather_str, "15–25°C vs <5 or >30°C", icon="🌦️")
    with cE: kpi_card("Coverage", coverage_str, "Weather data availability", icon="🧩")
    # Trend mini-chart
    if daily_f is not None and not daily_f.empty and "avg_temp_c" in daily_f.columns:
        d_sorted = daily_f.sort_values("date").copy()
        # 14-day rolling average for smoother trend
        n = 14
        for col in ["bike_rides_daily", "avg_temp_c"]:
            d_sorted[f"{col}_roll"] = d_sorted[col].rolling(n, min_periods=max(2, n//2), center=True).mean()
        fig_intro = make_subplots(specs=[[{"secondary_y": True}]])
        fig_intro.add_trace(go.Scatter(x=d_sorted["date"], 
                                       y=d_sorted["bike_rides_daily_roll"].fillna(d_sorted["bike_rides_daily"]),
                                       name="Daily rides", mode="lines",
                                       line=dict(color=RIDES_COLOR, width=2)), secondary_y=False)
        if d_sorted["avg_temp_c"].notna().any():
            fig_intro.add_trace(go.Scatter(x=d_sorted["date"], 
                                           y=d_sorted["avg_temp_c_roll"].fillna(d_sorted["avg_temp_c"]),
                                           name="Avg temp (°C)", mode="lines",
                                           line=dict(color=TEMP_COLOR, width=2, dash="dot"), opacity=0.9),
                                 secondary_y=True)
        fig_intro.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=0),
                                 hovermode="x unified", showlegend=True,
                                 title="Trend overview (14-day rolling avg)")
        fig_intro.update_yaxes(title_text="Bike rides", secondary_y=False)
        fig_intro.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
        st.plotly_chart(fig_intro, use_container_width=True)
    # Section overview
    st.markdown("### What you’ll find here")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Decision-ready KPIs**\n\nTotals, avg/day, and a defensible temp↔rides correlation.")
    c2.info("**Weather impact**\n\nTrend lines, scatter with fit, and comfort curve for clear takeaways.")
    c3.info("**Station intelligence**\n\nTop stations, OD flows (Sankey/Matrix), and Pareto analysis.")
    c4.info("**Time patterns**\n\nWeekday×Hour heatmap plus hourly/weekday profiles for planning.")

def render_weather_page():
    st.header("🌤️ Daily bike rides vs weather")
    if daily_f is None or daily_f.empty:
        st.warning("Daily metrics are not available. Please provide data with a 'date' column.")
        return
    d = daily_f.sort_values("date").copy()
    # Weather data coverage
    cov = d["avg_temp_c"].notna().mean() * 100 if "avg_temp_c" in d.columns else 0
    st.caption(f"Weather data coverage for selection: **{cov:.0f}%** (analysis accounts for missing days).")
    # Tabs for weather analysis
    tab_trend, tab_scatter, tab_dist, tab_lab, tab_index = st.tabs(
        ["📈 Trend", "🔬 Scatter", "📦 Distributions", "🧪 Lab", "📉 De-weathered Index"]
    )
    # Trend tab
    with tab_trend:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            roll_win = st.selectbox("Rolling window", ["Off", "7d", "14d", "30d"], index=1)
        with c2:
            show_precip = st.toggle("Show precipitation (mm)", value=("precip_mm" in d.columns))
        with c3:
            show_wind = st.toggle("Show wind (kph)", value=("wind_kph" in d.columns))
        with c4:
            st.caption("Residuals & elasticity in other tabs")
        # Apply rolling average if selected
        if roll_win != "Off":
            n = int(roll_win.replace("d", ""))
            for col in ["bike_rides_daily", "avg_temp_c", "wind_kph"]:
                if col in d.columns:
                    d[f"{col}_roll"] = d[col].rolling(n, min_periods=max(2, n//2), center=True).mean()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Daily rides line
        y_rides = d.get("bike_rides_daily_roll", d["bike_rides_daily"]) if roll_win != "Off" else d["bike_rides_daily"]
        fig.add_trace(go.Scatter(x=d["date"], y=y_rides, name="Daily rides", mode="lines",
                                 line=dict(color=RIDES_COLOR, width=2)), secondary_y=False)
        # Temperature line
        if "avg_temp_c" in d.columns and d["avg_temp_c"].notna().any():
            y_temp = d.get("avg_temp_c_roll", d["avg_temp_c"]) if roll_win != "Off" else d["avg_temp_c"]
            fig.add_trace(go.Scatter(x=d["date"], y=y_temp, name="Average temp (°C)", mode="lines",
                                     line=dict(color=TEMP_COLOR, width=2, dash="dot")), secondary_y=True)
        # Wind line
        if show_wind and "wind_kph" in d.columns and d["wind_kph"].notna().any():
            y_wind = d.get("wind_kph_roll", d["wind_kph"]) if roll_win != "Off" else d["wind_kph"]
            fig.add_trace(go.Scatter(x=d["date"], y=y_wind, name="Avg wind (kph)", mode="lines",
                                     line=dict(width=1), opacity=0.5), secondary_y=True)
        # Precip bars
        if show_precip and "precip_mm" in d.columns and d["precip_mm"].notna().any():
            fig.add_trace(go.Bar(x=d["date"], y=d["precip_mm"], name="Precipitation (mm)",
                                  marker_color="rgba(100,100,120,0.35)", opacity=0.4), secondary_y=False)
        # Gentle background band for visual context
        if len(d):
            fig.add_hrect(y0=float(d["bike_rides_daily"].min()), y1=float(d["bike_rides_daily"].max()),
                          line_width=0, fillcolor="rgba(34,197,94,0.05)", layer="below")
        fig.update_layout(hovermode="x unified", barmode="overlay", height=560,
                          title="Daily rides vs temperature, precipitation, and wind — NYC 2022")
        fig.update_yaxes(title_text="Bike rides (count)", secondary_y=False)
        y2_title = "Temperature (°C)" + (" + Wind (kph)" if show_wind and "wind_kph" in d.columns else "")
        fig.update_yaxes(title_text=y2_title, secondary_y=True)
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
    # Scatter tab
    with tab_scatter:
        c1, c2 = st.columns(2)
        with c1:
            color_by = st.selectbox("Color points by", ["None", "wet_day", "precip_bin", "wind_bin"], index=1)
        with c2:
            split_wknd = st.toggle("Show weekday vs weekend fits", value=True, help="Separate trendlines for weekdays vs weekends")
        # Determine which temperature column to use for scatter (if multiple)
        temp_col = None
        for c in ["avg_temp_c", "tavg_c", "tmean_c", "tmin_c", "tmax_c"]:
            if c in d.columns:
                temp_col = c
                break
        if temp_col is None or "bike_rides_daily" not in d.columns:
            st.info("Temperature or daily rides data missing for scatter plot.")
        else:
            scatter_df = d.dropna(subset=[temp_col, "bike_rides_daily"]).copy()
            if scatter_df.empty:
                st.info("No data available for scatter after dropping missing values.")
            else:
                chosen = None if color_by == "None" else color_by
                color_arg = chosen if (chosen in scatter_df.columns) else None
                labels = {
                    temp_col: temp_col.replace("_", " ").title().replace("C", "(°C)"),
                    "bike_rides_daily": "Bike rides (count)",
                    "wet_day": "Wet day",
                    "precip_bin": "Precipitation",
                    "wind_bin": "Wind"
                }
                fig2 = px.scatter(scatter_df, x=temp_col, y="bike_rides_daily", color=color_arg,
                                  labels=labels, opacity=0.85, trendline="ols")
                fig2.update_layout(height=520, title="Rides vs Temperature")
                st.plotly_chart(fig2, use_container_width=True)
            # Elasticity & rain penalty (using quadratic fit)
            df_fit = d.dropna(subset=["avg_temp_c", "bike_rides_daily"]) if "avg_temp_c" in d.columns else pd.DataFrame()
            if len(df_fit) >= 20:
                Xq = np.c_[np.ones(len(df_fit)), df_fit["avg_temp_c"], df_fit["avg_temp_c"]**2]
                a, b, c = np.linalg.lstsq(Xq, df_fit["bike_rides_daily"], rcond=None)[0]
                t0 = 20.0
                rides_t0 = a + b*t0 + c*t0*t0
                slope_t0 = b + 2*c*t0
                elasticity_pct = (slope_t0 / rides_t0) * 100 if rides_t0 > 0 else np.nan
                rain_pen = None
                if "wet_day" in d.columns and d["wet_day"].notna().any():
                    dry = d.loc[d["wet_day"] == 0, "bike_rides_daily"].mean()
                    wet = d.loc[d["wet_day"] == 1, "bike_rides_daily"].mean()
                    if pd.notnull(dry) and dry > 0:
                        rain_pen = (wet - dry) / dry * 100
                k1, k2 = st.columns(2)
                with k1: st.metric("Temp elasticity @20°C", f"{elasticity_pct:+.1f}% / °C")
                with k2: st.metric("Rain penalty (wet vs dry)", f"{rain_pen:+.0f}%" if rain_pen is not None else "—")
            # Weekday vs weekend fits
            if split_wknd and {"date", "bike_rides_daily", temp_col}.issubset(d.columns):
                dd = d.dropna(subset=[temp_col, "bike_rides_daily"]).copy()
                dd["day_type"] = dd["date"].dt.weekday.isin([5, 6]).map({True: "Weekend", False: "Weekday"})
                fig_sw = px.scatter(dd, x=temp_col, y="bike_rides_daily", color="day_type",
                                    opacity=0.85, trendline="ols",
                                    labels={temp_col: "Avg temp (°C)", "bike_rides_daily": "Bike rides", "day_type": "Day type"})
                fig_sw.update_layout(height=480, title="Rides vs Temp — Weekday vs Weekend")
                st.plotly_chart(fig_sw, use_container_width=True)
    # Distributions tab
    with tab_dist:
        st.subheader("Distribution by rainfall")
        if "precip_bin" in d.columns and d["precip_bin"].notna().any():
            fig3 = px.box(d, x="precip_bin", y="bike_rides_daily",
                          labels={"precip_bin": "Precipitation", "bike_rides_daily": "Bike rides per day"},
                          category_orders={"precip_bin": ["Low", "Medium", "High"]})
            fig3.update_layout(height=420)
            st.plotly_chart(fig3, use_container_width=True)
        elif "wet_day" in d.columns:
            fig3 = px.box(d, x=d["wet_day"].map({0: "Dry", 1: "Wet"}), y="bike_rides_daily",
                          labels={"x": "Day type", "bike_rides_daily": "Bike rides per day"})
            fig3.update_layout(height=420)
            st.plotly_chart(fig3, use_container_width=True)
        # Quick metrics for conditions
        kcols = st.columns(3)
        with kcols[0]:
            if "precip_bin" in d.columns and d["precip_bin"].notna().any():
                lo = d.loc[d["precip_bin"] == "Low", "bike_rides_daily"].mean()
                hi = d.loc[d["precip_bin"] == "High", "bike_rides_daily"].mean()
                if pd.notnull(lo) and pd.notnull(hi) and lo > 0:
                    st.metric("High rain vs Low", f"{(hi - lo) / lo * 100:+.0f}%")
        with kcols[1]:
            if "wet_day" in d.columns:
                dry_mean = d.loc[d["wet_day"] == 0, "bike_rides_daily"].mean()
                wet_mean = d.loc[d["wet_day"] == 1, "bike_rides_daily"].mean()
                if pd.notnull(dry_mean) and pd.notnull(wet_mean) and dry_mean > 0:
                    st.metric("Wet vs Dry", f"{(wet_mean - dry_mean) / dry_mean * 100:+.0f}%")
        with kcols[2]:
            if "wind_kph" in d.columns:
                breezy = d.loc[d["wind_kph"] >= 20, "bike_rides_daily"].mean()
                calm = d.loc[d["wind_kph"] < 10, "bike_rides_daily"].mean()
                if pd.notnull(calm) and pd.notnull(breezy) and calm > 0:
                    st.metric("Windy (≥20) vs Calm (<10)", f"{(breezy - calm) / calm * 100:+.0f}%")
    # Lab tab
    with tab_lab:
        st.subheader("🔮 Quick ride simulator & comfort point")
        need_cols = {"bike_rides_daily", "avg_temp_c"}
        if need_cols.issubset(d.columns) and d.dropna(subset=list(need_cols)).shape[0] >= 10:
            d_fit = d.dropna(subset=["avg_temp_c", "bike_rides_daily"]).copy()
            # Linear simulator
            if LinearRegression is not None and len(d_fit) >= 3:
                model = LinearRegression().fit(d_fit[["avg_temp_c"]], d_fit["bike_rides_daily"])
                t_sel = st.slider("Forecast avg temp (°C)", float(d_fit["avg_temp_c"].min()), float(d_fit["avg_temp_c"].max()), 20.0, 0.5)
                pred = int(model.predict([[t_sel]])[0])
                st.metric("Expected rides (linear)", f"{pred:,}")
            else:
                st.info("Install scikit-learn to enable linear simulator.")
            # Quadratic comfort curve
            X = np.c_[np.ones(len(d_fit)), d_fit["avg_temp_c"], d_fit["avg_temp_c"]**2]
            try:
                beta = np.linalg.lstsq(X, d_fit["bike_rides_daily"], rcond=None)[0]
                a, b, c = map(float, beta)  # model: rides ≈ a + b*T + c*T^2
                t_opt = (-b / (2*c)) if (c not in (0, np.nan) and np.isfinite(c)) else np.nan
                if np.isfinite(t_opt):
                    st.success(f"Optimal temperature for demand ≈ **{t_opt:.1f}°C** (quadratic fit)")
                # Visualization of comfort curve
                t_vals = np.linspace(d_fit["avg_temp_c"].min(), d_fit["avg_temp_c"].max(), 100)
                y_hat = a + b*t_vals + c*t_vals**2
                figq = go.Figure()
                figq.add_trace(go.Scatter(x=d_fit["avg_temp_c"], y=d_fit["bike_rides_daily"],
                                           mode="markers", name="Observed", opacity=0.5))
                figq.add_trace(go.Scatter(x=t_vals, y=y_hat, mode="lines", name="Quadratic fit"))
                if np.isfinite(t_opt):
                    figq.add_vline(x=t_opt, line_dash="dot")
                figq.update_layout(height=380, title="Comfort curve (rides vs temperature)")
                figq.update_xaxes(title="Avg temp (°C)")
                figq.update_yaxes(title="Bike rides per day")
                st.plotly_chart(figq, use_container_width=True)
            except Exception:
                st.info("Quadratic fit not available for this selection.")
        else:
            st.caption("Need daily rides + avg_temp_c for simulator.")
        # Backtest sub-section
        with st.expander("📏 Backtest (train/test)"):
            if len(d) >= 90 and "avg_temp_c" in d.columns and d["avg_temp_c"].notna().sum() >= 60:
                d2 = d.dropna(subset=["avg_temp_c", "bike_rides_daily"]).sort_values("date").copy()
                cut = int(len(d2) * 0.75)
                train, test = d2.iloc[:cut], d2.iloc[cut:]
                # Baseline: linear (rides ~ temp)
                X_tr = np.c_[np.ones(len(train)), train["avg_temp_c"]]
                X_te = np.c_[np.ones(len(test)), test["avg_temp_c"]]
                b_lin, *_ = np.linalg.lstsq(X_tr, train["bike_rides_daily"], rcond=None)
                pred_lin = X_te @ b_lin
                # Enriched: use deweather_fit_predict (if available)
                out_tr = deweather_fit_predict(train)
                out_te = deweather_fit_predict(test)
                yhat_tr = out_tr[0] if out_tr is not None else None
                yhat_te = out_te[0] if out_te is not None else None
                def MAE(y, yhat):
                    y = np.asarray(y); yhat = np.asarray(yhat)
                    mask = np.isfinite(y) & np.isfinite(yhat)
                    return float(np.mean(np.abs(y[mask] - yhat[mask]))) if mask.any() else np.nan
                mae_lin = MAE(test["bike_rides_daily"], pred_lin)
                mae_enr = MAE(test["bike_rides_daily"], yhat_te.values if yhat_te is not None else np.full(len(test), np.nan))
                cbt1, cbt2 = st.columns(2)
                with cbt1: st.metric("MAE – Linear (temp)", f"{mae_lin:,.0f}" if np.isfinite(mae_lin) else "—")
                with cbt2: st.metric("MAE – Enriched model", f"{mae_enr:,.0f}" if np.isfinite(mae_enr) else "—")
                if np.isfinite(mae_lin) and np.isfinite(mae_enr):
                    st.caption(f"Error reduction: {(mae_lin - mae_enr):,.0f} fewer MAE with enriched model.")
            else:
                st.info("Need ≥90 days with temp data to backtest.")
        # Forecast upload placeholder
        st.markdown("#### 🔮 Bring your own forecast (CSV)")
        up = st.file_uploader("Upload 7–14 day forecast CSV (columns: date, avg_temp_c, precip_mm, wind_kph)", type=["csv"])
        if up is not None:
            try:
                df_fc = pd.read_csv(up, parse_dates=["date"])
                df_fc = df_fc.reindex(columns=["date", "avg_temp_c", "precip_mm", "wind_kph"])
                # Would integrate forecast with model here
                st.write("Forecast uploaded successfully. (Integration not implemented in this demo.)")
            except Exception:
                st.error("Error reading forecast CSV. Ensure required columns exist.")
    # De-weathered index tab (not fully implemented)
    with tab_index:
        st.warning("De-weathered index analysis is not implemented in this refactored version.")

def render_trip_metrics_page():
    st.header("🚴 Trip metrics")
    needed_cols = {"duration_min", "distance_km", "speed_kmh"}
    if not needed_cols.issubset(df_f.columns):
        st.info("Duration, distance, and speed metrics are unavailable in this dataset.")
        return
    # Controls: robust clipping and log-scale toggles
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        robust = st.toggle("Robust clipping (99.5%)", value=True, help="Exclude extreme outliers from axis scaling")
    with c2:
        log_duration = st.checkbox("Log X: Duration", value=False)
    with c3:
        log_distance = st.checkbox("Log X: Distance", value=False)
    with c4:
        log_speed = st.checkbox("Log X: Speed", value=False)
    # Inlier masks for each metric
    m_dur = (inlier_mask(df_f, "duration_min", hi=0.995) if robust else pd.Series(True, index=df_f.index)) \
            & df_f["duration_min"].between(0.5, 240, inclusive="both")
    m_dst = (inlier_mask(df_f, "distance_km", hi=0.995) if robust else pd.Series(True, index=df_f.index)) \
            & df_f["distance_km"].between(0.01, 30, inclusive="both")
    m_spd = (inlier_mask(df_f, "speed_kmh", hi=0.995) if robust else pd.Series(True, index=df_f.index)) \
            & df_f["speed_kmh"].between(0.5, 60, inclusive="both")
    clipped_dur = int((~m_dur).sum())
    clipped_dst = int((~m_dst).sum())
    clipped_spd = int((~m_spd).sum())
    # Histograms for each metric
    cA, cB, cC = st.columns(3)
    with cA:
        d_val = df_f.loc[m_dur, "duration_min"]
        ql, qh = d_val.quantile([0.01, 0.995]).tolist()
        figA = px.histogram(d_val, x="duration_min", nbins=60, labels={"duration_min": "Duration (min)"},
                             log_x=log_duration, range_x=[ql, qh] if robust and not log_duration else None)
        figA.update_layout(height=420)
        st.plotly_chart(figA, use_container_width=True)
        st.caption(f"Clipped outliers (duration): {clipped_dur:,}")
    with cB:
        d_val = df_f.loc[m_dst, "distance_km"]
        ql, qh = d_val.quantile([0.01, 0.995]).tolist()
        figB = px.histogram(d_val, x="distance_km", nbins=60, labels={"distance_km": "Distance (km)"},
                             log_x=log_distance, range_x=[ql, qh] if robust and not log_distance else None)
        figB.update_layout(height=420)
        st.plotly_chart(figB, use_container_width=True)
        st.caption(f"Clipped outliers (distance): {clipped_dst:,}")
    with cC:
        d_val = df_f.loc[m_spd, "speed_kmh"]
        ql, qh = d_val.quantile([0.01, 0.995]).tolist()
        figC = px.histogram(d_val, x="speed_kmh", nbins=60, labels={"speed_kmh": "Speed (km/h)"},
                             log_x=log_speed, range_x=[ql, qh] if robust and not log_speed else None)
        figC.update_layout(height=420)
        st.plotly_chart(figC, use_container_width=True)
        st.caption(f"Clipped outliers (speed): {clipped_spd:,}")

def render_member_vs_casual_page():
    st.header("👥 Member vs Casual riding patterns")
    # Ensure member_type_display exists
    if "member_type_display" not in df_f.columns and "member_type" in df_f.columns:
        df_f["member_type_display"] = df_f["member_type"].astype(str).map(MEMBER_LABELS).fillna(df_f["member_type"].astype(str))
    if "member_type_display" not in df_f.columns:
        st.info("Member vs Casual breakdown not available in this data.")
        return
    grp = df_f.groupby("member_type_display").agg(
        rides=("ride_id" if "ride_id" in df_f.columns else df_f.index.name, "count"),
        duration_med=("duration_min", "median") if "duration_min" in df_f.columns else (),
        speed_med=("speed_kmh", "median") if "speed_kmh" in df_f.columns else (),
        # We use rain_penalty and temp_gap separately below for clarity
    ).reset_index().rename(columns={"member_type_display": "member_type_clean"})
    # Calculate additional metrics
    rain_penalty = None
    if "wet_day" in df_f.columns:
        rain_penalty = df_f.groupby("member_type_display")["wet_day"].mean() * 100
    temp_gap = None
    if "avg_temp_c" in df_f.columns:
        med_temps = df_f.groupby("member_type_display")["avg_temp_c"].median()
        if {"Member 🧑‍💼", "Casual 🚲"}.issubset(med_temps.index):
            temp_gap = float(med_temps["Casual 🚲"] - med_temps["Member 🧑‍💼"])
    # At-a-glance KPI cards
    st.markdown("#### ✨ At-a-glance (selection)")
    total_member = int(grp.loc[grp["member_type_clean"] == "Member 🧑‍💼", "rides"]) if (grp["member_type_clean"] == "Member 🧑‍💼").any() else 0
    total_casual = int(grp.loc[grp["member_type_clean"] == "Casual 🚲", "rides"]) if (grp["member_type_clean"] == "Casual 🚲").any() else 0
    share_member = 100.0 * total_member / max(total_member + total_casual, 1)
    dur_txt = "—"
    spd_txt = "—"
    if "duration_med" in grp.columns:
        med_m = grp.loc[grp["member_type_clean"] == "Member 🧑‍💼", "duration_med"].values
        med_c = grp.loc[grp["member_type_clean"] == "Casual 🚲", "duration_med"].values
        if med_m.size and med_c.size:
            dur_txt = f"{med_m[0]:.1f} vs {med_c[0]:.1f} min"
    if "speed_med" in grp.columns:
        spd_m = grp.loc[grp["member_type_clean"] == "Member 🧑‍💼", "speed_med"].values
        spd_c = grp.loc[grp["member_type_clean"] == "Casual 🚲", "speed_med"].values
        if spd_m.size and spd_c.size:
            spd_txt = f"{spd_m[0]:.1f} vs {spd_c[0]:.1f} km/h"
    rain_txt = "—"
    if rain_penalty is not None and {"Member 🧑‍💼", "Casual 🚲"}.issubset(rain_penalty.index):
        rain_txt = f"M {rain_penalty['Member 🧑‍💼']:+.0f}% · C {rain_penalty['Casual 🚲']:+.0f}%"
    temp_txt = f"{temp_gap:+.1f}°C" if temp_gap is not None and np.isfinite(temp_gap) else "—"
    ca, cb, cc, cd, ce = st.columns(5)
    with ca: kpi_card("Member share", f"{share_member:.1f}%", "of total rides", icon="🧑‍💼")
    with cb: kpi_card("Median duration", dur_txt, "Member (M) vs Casual (C)", icon="⏱️")
    with cc: kpi_card("Median speed", spd_txt, "Member (M) vs Casual (C)", icon="🚴")
    with cd: kpi_card("Rain penalty", rain_txt, "Wet vs dry (group-wise)", icon="🌧️")
    with ce: kpi_card("Temp pref. gap", temp_txt, "Casual − Member (median °C)", icon="🌡️")
    # (Additional comparative charts could be added here, e.g., monthly or hourly usage by member type)

def render_od_flows_page():
    st.header("🔀 Origin → Destination — Sankey + Map")
    need_cols = {"start_station_name", "end_station_name"}
    if not need_cols.issubset(df_f.columns):
        st.info("Station start/end data is required for flow analysis.")
        return
    # Ensure member display column
    mt_col = None
    if "member_type_display" in df_f.columns:
        mt_col = "member_type_display"
    elif "member_type" in df_f.columns:
        df_f["member_type_display"] = df_f["member_type"].astype(str).map(MEMBER_LABELS).fillna(df_f["member_type"].astype(str))
        mt_col = "member_type_display"
    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        member_split = st.toggle("Split by member type", value=(mt_col is not None))
    with c3:
        min_rides = st.number_input("Min rides per edge", min_value=1, max_value=1000, value=3, step=1)
    with c4:
        topN = st.slider("Max stations to show", 10, 100, 50, 10, help="Limit number of unique stations in Sankey for readability")
    subset = _time_slice(df_f, mode).copy()
    if subset.empty:
        st.info("No data for this time slice.")
        return
    # Compute origin-destination edges
    if member_split and mt_col:
        edges = subset.groupby(["start_station_name", "end_station_name", mt_col]).size().rename("rides").reset_index()
    else:
        edges = subset.groupby(["start_station_name", "end_station_name"]).size().rename("rides").reset_index()
    edges = edges[edges["rides"] >= int(min_rides)]
    if edges.empty:
        st.info("No OD flows meet the minimum rides criterion.")
        return
    # Limit to topN stations by total flow (if applicable)
    all_stations = pd.Series(pd.concat([edges["start_station_name"], edges["end_station_name"]]).unique(), dtype=str)
    if topN and len(all_stations) > int(topN):
        total_flows = edges.groupby("start_station_name")["rides"].sum() + edges.groupby("end_station_name")["rides"].sum()
        keep_stations = set(total_flows.nlargest(int(topN)).index.astype(str))
        edges = edges[edges["start_station_name"].astype(str).isin(keep_stations) & edges["end_station_name"].astype(str).isin(keep_stations)]
    # Sankey diagram
    node_labels = pd.Series(pd.concat([edges["start_station_name"], edges["end_station_name"]]).unique(), dtype=str)
    idx_map = {name: i for i, name in enumerate(node_labels)}
    edges_vis = edges.copy()
    edges_vis["src_idx"] = edges_vis["start_station_name"].map(idx_map)
    edges_vis["tgt_idx"] = edges_vis["end_station_name"].map(idx_map)
    src = edges_vis["src_idx"].to_numpy()
    tgt = edges_vis["tgt_idx"].to_numpy()
    vals = edges_vis["rides"].astype(float).to_numpy()
    link_colors = None
    if member_split and mt_col and mt_col in edges_vis.columns:
        color_map = {"Member 🧑‍💼": "rgba(34,197,94,0.60)", "Casual 🚲": "rgba(37,99,235,0.60)"}
        mt_vals = edges_vis[mt_col].astype(str).to_numpy()
        link_colors = [color_map.get(v, "rgba(180,180,180,0.45)") for v in mt_vals]
    sankey = go.Sankey(node=dict(label=node_labels.tolist(), pad=6, thickness=12, color="rgba(240,240,255,0.85)",
                                 line=dict(color="rgba(80,80,120,0.4)", width=0.5)),
                       link=dict(source=src, target=tgt, value=vals, color=link_colors),
                       arrangement="snap")
    fig = go.Figure(sankey)
    fig.update_layout(height=560, title="Top OD flows", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    # Map with arcs
    st.subheader("🗺️ Map — OD arcs (width ∝ volume)")
    if {"start_lat", "start_lng", "end_lat", "end_lng"}.issubset(df_f.columns):
        import pydeck as pdk
        map_edges = edges.nlargest(250, "rides").copy()  # limit edges for map
        starts = df_f.groupby("start_station_name")[["start_lat", "start_lng"]].median().rename(columns={"start_lat": "s_lat", "start_lng": "s_lng"})
        ends = df_f.groupby("end_station_name")[["end_lat", "end_lng"]].median().rename(columns={"end_lat": "e_lat", "end_lng": "e_lng"})
        geo = map_edges.join(starts, on="start_station_name").join(ends, on="end_station_name").dropna(subset=["s_lat", "s_lng", "e_lat", "e_lng"])
        if geo.empty:
            st.info("No coordinates available for these flows.")
        else:
            vmax = float(geo["rides"].max())
            scale_val = st.slider("Arc width scale", 1, 30, 10)
            geo["width"] = (scale_val * (np.sqrt(geo["rides"]) / np.sqrt(vmax if vmax > 0 else 1))).clip(0.5, 14)
            if member_split and mt_col and mt_col in geo.columns:
                color_map = {"Member 🧑‍💼": [34, 197, 94, 200], "Casual 🚲": [37, 99, 235, 200]}
                geo["color"] = [color_map.get(str(v), [160, 160, 160, 200]) for v in geo[mt_col]]
            else:
                geo["color"] = [[37, 99, 235, 200]] * len(geo)
            geo["start_s"] = geo["start_station_name"].apply(shorten_name)
            geo["end_s"] = geo["end_station_name"].apply(shorten_name)
            layer = pdk.Layer("ArcLayer", data=geo,
                               get_source_position="[s_lng, s_lat]",
                               get_target_position="[e_lng, e_lat]",
                               get_width="width",
                               get_source_color="color",
                               get_target_color="color",
                               pickable=True, auto_highlight=True)
            center_lat = float(pd.concat([geo["s_lat"], geo["e_lat"]]).median())
            center_lon = float(pd.concat([geo["s_lng"], geo["e_lng"]]).median())
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=30, bearing=0)
            tooltip = {"html": "<b>{start_s}</b> → <b>{end_s}</b><br/>Rides: {rides}", 
                       "style": {"backgroundColor": "rgba(17,17,17,0.9)", "color": "white"}}
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                            map_style="mapbox://styles/mapbox/dark-v11", tooltip=tooltip)
            st.pydeck_chart(deck)
    else:
        st.info("Coordinate data not available for map visualization.")
    # Diagnostics table
    with st.expander("Diagnostics — top flows table"):
        show_n = st.slider("Show first N rows", 10, 200, 40, 10, key="od_diag_rows")
        st.dataframe(edges.sort_values("rides", ascending=False).head(show_n), use_container_width=True)
        csv_bytes = edges.to_csv(index=False).encode("utf-8")
        st.download_button("Download OD flows (CSV)", csv_bytes, "od_flows.csv", "text/csv")

def render_od_matrix_page():
    st.header("🧮 OD Matrix — Top origins × destinations")
    need_cols = {"start_station_name", "end_station_name"}
    if not need_cols.issubset(df_f.columns):
        st.info("Station start/end data is required for OD matrix.")
        return
    # Ensure member display column
    mt_col = None
    if "member_type_display" in df_f.columns:
        mt_col = "member_type_display"
    elif "member_type" in df_f.columns:
        df_f["member_type_display"] = df_f["member_type"].astype(str).map(MEMBER_LABELS).fillna(df_f["member_type"].astype(str))
        mt_col = "member_type_display"
    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        sort_mode = st.selectbox("Order by", ["Alphabetical", "By totals", "Clustered (if available)"], index=1)
    with c3:
        top_orig = st.slider("Top origins", 5, 100, 20, 5)
    with c4:
        top_dest = st.slider("Top destinations", 5, 100, 20, 5)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        min_rides = st.number_input("Min rides to include (pair)", min_value=1, max_value=1000, value=3, step=1)
    with c6:
        normalize = st.selectbox("Normalize values", ["None", "Row (per origin)", "Column (per destination)"], index=0)
    with c7:
        member_split = st.toggle("Split by member type", value=(mt_col is not None))
    with c8:
        log_scale = st.checkbox("Use sqrt scale", value=False, help="Use square root scale for color scale (improve contrast)")
    subset = _time_slice(df_f, mode).copy()
    subset["start_station_name"] = subset["start_station_name"].astype(str)
    subset["end_station_name"] = subset["end_station_name"].astype(str)
    if subset.empty:
        st.info("No data in this time slice.")
        return
    # Build matrix
    def build_matrix(df_src: pd.DataFrame):
        origin_totals = df_src.groupby("start_station_name").size().sort_values(ascending=False)
        dest_totals = df_src.groupby("end_station_name").size().sort_values(ascending=False)
        o_keep = set(origin_totals.head(int(top_orig)).index)
        d_keep = set(dest_totals.head(int(top_dest)).index)
        df2 = df_src[df_src["start_station_name"].isin(o_keep) & df_src["end_station_name"].isin(d_keep)]
        if df2.empty:
            return pd.DataFrame(), pd.DataFrame(), origin_totals, dest_totals
        pairs = df2.groupby(["start_station_name", "end_station_name"]).size().rename("rides").reset_index()
        if int(min_rides) > 1:
            pairs = pairs[pairs["rides"] >= int(min_rides)]
        if pairs.empty:
            return pd.DataFrame(), pd.DataFrame(), origin_totals, dest_totals
        mat = pairs.pivot_table(index="start_station_name", columns="end_station_name", values="rides", aggfunc="sum", fill_value=0)
        # Normalize if requested
        if normalize == "Row (per origin)":
            denom = mat.sum(axis=1).replace(0, np.nan)
            mat = (mat.T / denom).T.fillna(0.0)
        elif normalize == "Column (per destination)":
            denom = mat.sum(axis=0).replace(0, np.nan)
            mat = (mat / denom).fillna(0.0)
        # Sort matrix
        if sort_mode == "Alphabetical":
            mat = mat.sort_index(axis=0).sort_index(axis=1)
        elif sort_mode == "By totals":
            if normalize == "None":
                mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
                mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]
            else:
                order_orig = origin_totals.reindex(mat.index).fillna(0).sort_values(ascending=False).index
                order_dest = dest_totals.reindex(mat.columns).fillna(0).sort_values(ascending=False).index
                mat = mat.loc[order_orig, order_dest]
        elif sort_mode == "Clustered (if available)" and linkage is not None and leaves_list is not None:
            try:
                if mat.shape[0] > 2 and mat.shape[1] > 2:
                    rZ = linkage(mat.values, method="average", metric="euclidean")
                    cZ = linkage(mat.values.T, method="average", metric="euclidean")
                    mat = mat.loc[mat.index[leaves_list(rZ)], mat.columns[leaves_list(cZ)]]
            except Exception:
                mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
                mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]
        return mat, pairs, origin_totals, dest_totals
    def render_heatmap(mat: pd.DataFrame, title: str):
        if mat.empty:
            st.info("Nothing to show with current filters. Try adjusting Top N or Min rides.")
            return
        z = mat.values.astype(float)
        if log_scale and normalize == "None":
            z = np.sqrt(z + 1.0)
        if normalize == "None":
            hovertemplate = "Origin: %{y}<br>Destination: %{x}<br>Rides: %{z}<extra></extra>"
            colorbar_title = "rides" if not log_scale else "√(rides+1)"
        elif normalize.startswith("Row"):
            hovertemplate = "Origin: %{y}<br>Destination: %{x}<br>Share (row): %{z:.2%}<extra></extra>"
            colorbar_title = "row share"
        else:
            hovertemplate = "Origin: %{y}<br>Destination: %{x}<br>Share (col): %{z:.2%}<extra></extra>"
            colorbar_title = "col share"
        fig = go.Figure(data=go.Heatmap(z=z, x=mat.columns.astype(str).tolist(), y=mat.index.astype(str).tolist(),
                                        colorbar=dict(title=colorbar_title), hovertemplate=hovertemplate))
        fig.update_layout(title=title, xaxis_title="Destination", yaxis_title="Origin", height=720,
                          margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
    mat, pairs, o_tot, d_tot = build_matrix(subset)
    if member_split and mt_col and mt_col in subset.columns:
        tabs = st.tabs(["Member 🧑‍💼", "Casual 🚲", "All"])
        segments = [
            ("Member 🧑‍💼", subset[subset[mt_col].astype(str) == "Member 🧑‍💼"]),
            ("Casual 🚲", subset[subset[mt_col].astype(str) == "Casual 🚲"]),
            ("All", subset)
        ]
        for (label, seg_df), tab in zip(segments, tabs):
            with tab:
                mat_seg, pairs_seg, *_ = build_matrix(seg_df)
                render_heatmap(mat_seg, f"OD Matrix — {label}")
    else:
        render_heatmap(mat, "OD Matrix")
    # Station deep-dive
    st.subheader("🔎 Station deep-dive")
    leaderboard = df_f.groupby("start_station_name").size().rename("rides").sort_values(ascending=False).reset_index().rename(columns={"start_station_name": "station"})
    picked_station = st.selectbox("Pick a station", leaderboard["station"].tolist() if not leaderboard.empty else [])
    if picked_station:
        focus = df_f[df_f["start_station_name"].astype(str) == picked_station]
        cA, cB, cC = st.columns(3)
        with cA:
            if "hour" in focus.columns and not focus.empty:
                gh = focus.groupby("hour").size().rename("rides").reset_index()
                figH = px.line(gh, x="hour", y="rides", markers=True, labels={"hour": "Hour of day", "rides": "Rides"})
                figH.update_layout(height=320, title="Hourly profile")
                st.plotly_chart(figH, use_container_width=True)
        with cB:
            if "weekday" in focus.columns and not focus.empty:
                gw = focus.groupby("weekday").size().rename("rides").reset_index()
                gw["weekday_name"] = gw["weekday"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
                figW = px.bar(gw, x="weekday_name", y="rides", labels={"weekday_name": "Weekday", "rides": "Rides"})
                figW.update_layout(height=320, title="Weekday profile")
                st.plotly_chart(figW, use_container_width=True)
        with cC:
            if "month" in focus.columns and not focus.empty:
                gm = focus.groupby(focus["started_at"].dt.to_period("M")).size().rename("rides").reset_index()
                gm["month"] = gm["started_at"].astype(str)
                figM = px.bar(gm, x="month", y="rides", labels={"month": "Month", "rides": "Rides"})
                figM.update_layout(height=320, title="Monthly profile")
                st.plotly_chart(figM, use_container_width=True)

def render_station_popularity_page():
    st.header("🚉 Most popular start stations")
    if "start_station_name" not in df_f.columns:
        st.warning("Start station information is missing in the data.")
        return
    # Ensure member_type_display exists
    if "member_type_display" not in df_f.columns and "member_type" in df_f.columns:
        df_f["member_type_display"] = df_f["member_type"].astype(str).map(MEMBER_LABELS).fillna(df_f["member_type"].astype(str))
    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        topN = st.slider("Top N stations", 5, 150, 30, 5)
    with c2:
        metric = st.selectbox("Metric", ["Rides", "Share %"], help="Choose count or percent of total rides.")
    with c3:
        group_by = st.selectbox("Group by", ["Overall", "By Month", "By Hour"])
    c4, c5 = st.columns(2)
    with c4:
        stack_member = st.toggle("Stack by Member Type", value=("member_type_display" in df_f.columns),
                                  help="Separate counts by member vs casual")
    with c5:
        wx_split = st.selectbox("Weather split", ["None", "Wet vs Dry", "Temp bands (Cold/Mild/Hot)"],
                                index=0 if ("wet_day" not in df_f.columns and "avg_temp_c" not in df_f.columns) else 1)
    st.markdown("---")
    # Prepare base data
    base = df_f.copy()
    base["station"] = base["start_station_name"].astype(str)
    wx_col = None
    if wx_split == "Wet vs Dry" and "wet_day" in base.columns:
        wx_col = "wet_day_label"
        base[wx_col] = base["wet_day"].map({0: "Dry", 1: "Wet"})
    elif wx_split.startswith("Temp") and "avg_temp_c" in base.columns:
        wx_col = "temp_band"
        base[wx_col] = pd.cut(base["avg_temp_c"], bins=[-100, 5, 20, 200],
                               labels=["Cold <5°C", "Mild 5–20°C", "Hot >20°C"], include_lowest=True)
    mcol = "member_type_display" if (stack_member and "member_type_display" in base.columns) else None
    # Compute top N stations
    leaderboard = base.groupby("station").size().rename("rides").sort_values(ascending=False).head(int(topN)).reset_index()
    keep_stations = set(leaderboard["station"])
    small = base[base["station"].isin(keep_stations)].copy()
    # Helper for converting counts to share if needed
    def _maybe_to_share(df_grp, val_col="value", by_cols=None):
        if metric == "Share %":
            if not by_cols:
                total = df_grp[val_col].sum()
                df_grp[val_col] = np.where(total > 0, df_grp[val_col] / total * 100.0, 0.0)
            else:
                denom = df_grp.groupby(by_cols)[val_col].transform(lambda s: s.sum() if s.sum() > 0 else np.nan)
                df_grp[val_col] = (df_grp[val_col] / denom * 100.0).fillna(0.0)
        return df_grp
    # Overall, by month, or by hour grouping
    if group_by == "Overall":
        by = ["station"]
        if wx_col: by.append(wx_col)
        if mcol:   by.append(mcol)
        g = small.groupby(by).size().rename("value").reset_index()
        g = _maybe_to_share(g, val_col="value", by_cols=[c for c in [wx_col, mcol] if c])
        x_label = "Share (%)" if metric == "Share %" else "Rides (count)"
        color_dim = mcol if mcol else wx_col
        fig = px.bar(g, x="station", y="value", color=color_dim, barmode=("stack" if color_dim else "relative"),
                     labels={"station": "Station", "value": x_label, (color_dim or ""): (MEMBER_LEGEND_TITLE if color_dim == mcol else "Weather") if color_dim else None},
                     hover_data={"station": True, "value": ":,.2f" if metric == "Share %" else ":,"})
        fig.update_layout(height=620, title=f"Top {len(keep_stations)} stations — {x_label}",
                          xaxis_title="Station", yaxis_title=x_label,
                          margin=dict(l=20, r=20, t=60, b=100),
                          legend_title_text=(MEMBER_LEGEND_TITLE if color_dim == mcol else ("Weather" if color_dim else "")))
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10), categoryorder="array", categoryarray=leaderboard["station"].tolist())
        st.plotly_chart(fig, use_container_width=True)
    elif group_by == "By Month":
        if "month" not in small.columns:
            st.info("Month data not available (ensure 'started_at' is parsed).")
            return
        by = [small["started_at"].dt.to_period("M").dt.to_timestamp().rename("month"), "station"]
        if wx_col: by.insert(0, wx_col)
        if mcol: by.insert(0, mcol)
        g = small.groupby(by).size().rename("value").reset_index()
        g = _maybe_to_share(g, val_col="value", by_cols=[c for c in ["month", wx_col, mcol] if c])
        if "month" in g.columns:
            g["month"] = g["month"].dt.strftime("%Y-%m")
        x_label = "Share (%)" if metric == "Share %" else "Rides"
        color_dim = mcol if mcol else wx_col
        fig = px.bar(g, x="month", y="value", color=color_dim, facet_col="station", facet_col_wrap=5,
                     labels={"value": x_label, "month": "Month", (color_dim or ""): (MEMBER_LEGEND_TITLE if color_dim == mcol else "Weather") if color_dim else None})
        fig.update_layout(height=600, title="Monthly trends by station",
                          legend_title_text=(MEMBER_LEGEND_TITLE if color_dim == mcol else ("Weather" if color_dim else "")))
        fig.update_yaxes(matches=None)
        st.plotly_chart(fig, use_container_width=True)
    elif group_by == "By Hour":
        if "hour" not in small.columns:
            st.info("Hour data not available; ensure 'hour' was extracted.")
            return
        by = [small["hour"], "station"]
        if wx_col: by.insert(0, wx_col)
        if mcol: by.insert(0, mcol)
        g = small.groupby(by).size().rename("value").reset_index()
        g = _maybe_to_share(g, val_col="value", by_cols=[c for c in ["hour", wx_col, mcol] if c])
        x_label = "Share (%)" if metric == "Share %" else "Rides"
        color_dim = mcol if mcol else wx_col
        fig = px.bar(g, x="hour", y="value", color=color_dim, facet_col="station", facet_col_wrap=5,
                     labels={"value": x_label, "hour": "Hour of day", (color_dim or ""): (MEMBER_LEGEND_TITLE if color_dim == mcol else "Weather") if color_dim else None})
        fig.update_layout(height=600, title="Hourly patterns by station",
                          legend_title_text=(MEMBER_LEGEND_TITLE if color_dim == mcol else ("Weather" if color_dim else "")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Unknown grouping option selected.")
    # (Map of top stations could be added here if desired, using median coordinates)

def render_station_imbalance_page():
    st.header("⚖️ Station imbalance (arrivals − departures)")
    need_cols = {"start_station_name", "end_station_name"}
    if not need_cols.issubset(df_f.columns):
        st.info("Station start/end data is required for imbalance analysis.")
        return
    # Ensure member display column
    mt_col = None
    if "member_type_display" in df_f.columns:
        mt_col = "member_type_display"
    elif "member_type" in df_f.columns:
        df_f["member_type_display"] = df_f["member_type"].astype(str).map(MEMBER_LABELS).fillna(df_f["member_type"].astype(str))
        mt_col = "member_type_display"
    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        normalize = st.selectbox("Normalize", ["None", "Per day (avg in/out)"], index=0,
                                 help="If 'Per day', uses daily average in/out (requires date).")
    with c3:
        topK = st.slider("Show top ±K stations", 5, 60, 15, 5)
    with c4:
        min_total = st.number_input("Min total traffic (in+out)", 0, 10000, 20, 5)
    c5, c6 = st.columns(2)
    with c5:
        member_split = st.toggle("Split by member type", value=(mt_col is not None))
    with c6:
        show_map = st.toggle("Show map", value={"start_lat", "start_lng"}.issubset(df_f.columns) or {"end_lat", "end_lng"}.issubset(df_f.columns))
    subset = _time_slice(df_f, mode).copy()
    if subset.empty:
        st.info("No data for this time slice.")
        return
    subset["start_station_name"] = subset["start_station_name"].astype(str)
    subset["end_station_name"] = subset["end_station_name"].astype(str)
    # Compute imbalance
    def build_imbalance(df_src: pd.DataFrame):
        if normalize.startswith("Per day") and "date" in df_src.columns:
            dep = df_src.groupby(["start_station_name", "date"]).size().rename("cnt").reset_index() \
                       .groupby("start_station_name")["cnt"].mean().rename("out").reset_index()
            arr = df_src.groupby(["end_station_name", "date"]).size().rename("cnt").reset_index() \
                       .groupby("end_station_name")["cnt"].mean().rename("in").reset_index()
            to_float = True
        else:
            dep = df_src.groupby("start_station_name").size().rename("out").reset_index()
            arr = df_src.groupby("end_station_name").size().rename("in").reset_index()
            to_float = False
        s = dep.merge(arr, left_on="start_station_name", right_on="end_station_name", how="outer")
        s["station"] = s["start_station_name"].fillna(s["end_station_name"])
        s = s.drop(columns=["start_station_name", "end_station_name"])
        s["in"] = s["in"].fillna(0.0 if to_float else 0).astype(float if to_float else int)
        s["out"] = s["out"].fillna(0.0 if to_float else 0).astype(float if to_float else int)
        s["total"] = s["in"] + s["out"]
        if float(min_total) > 0:
            threshold = float(min_total) if to_float else int(min_total)
            s = s[s["total"] >= threshold]
        s["imbalance"] = s["in"] - s["out"]
        return s.sort_values("imbalance", ascending=False).reset_index(drop=True)
    def render_bar(df_in: pd.DataFrame, suffix: str = ""):
        if df_in.empty:
            st.info("No stations meet the criteria. Try reducing Min total traffic or changing slice.")
            return None
        top_pos = df_in.nlargest(int(topK), "imbalance")
        top_neg = df_in.nsmallest(int(topK), "imbalance")
        biggest = pd.concat([top_pos, top_neg], ignore_index=True)
        biggest["label"] = biggest["station"].astype(str).str.slice(0, 28)
        colors = np.where(biggest["imbalance"] >= 0, "rgba(34,197,94,0.85)", "rgba(220,38,38,0.85)")
        fig = go.Figure(go.Bar(x=biggest["label"], y=biggest["imbalance"], marker=dict(color=colors),
                               customdata=np.stack([biggest["in"], biggest["out"]], axis=1),
                               hovertemplate="Station: %{x}<br>IN: %{customdata[0]}<br>OUT: %{customdata[1]}<br>Δ: %{y}<extra></extra>"))
        y_title = "Avg Δ (in − out) per day" if normalize.startswith("Per day") else "Δ (in − out)"
        fig.update_layout(height=560, title=f"Stations with largest net IN (green) / OUT (red) {suffix}".strip(),
                          xaxis_title="", yaxis_title=y_title, margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)
        return biggest
    total_imbalance = build_imbalance(subset)
    if member_split and mt_col and mt_col in subset.columns:
        tabs = st.tabs(["Member 🧑‍💼", "Casual 🚲", "Combined"])
        seg_member = subset[subset[mt_col].astype(str) == "Member 🧑‍💼"]
        seg_casual = subset[subset[mt_col].astype(str) == "Casual 🚲"]
        for seg_df, label, tab in zip([seg_member, seg_casual, subset], ["Member", "Casual", "Combined"], tabs):
            with tab:
                m = build_imbalance(seg_df)
                biggest = render_bar(m, f"— {label}")
                if biggest is not None:
                    with st.expander(f"Preview & Download — {label}"):
                        st.dataframe(m.sort_values("imbalance", ascending=False).head(120), use_container_width=True)
                        st.download_button(f"Download imbalance ({label}) CSV",
                                           m.to_csv(index=False).encode("utf-8"),
                                           f"station_imbalance_{_slug(label)}.csv", "text/csv")
    else:
        biggest = render_bar(total_imbalance)
        if biggest is not None:
            with st.expander("Preview & Download"):
                st.dataframe(total_imbalance.sort_values("imbalance", ascending=False).head(120), use_container_width=True)
                st.download_button("Download imbalance CSV",
                                   total_imbalance.to_csv(index=False).encode("utf-8"),
                                   "station_imbalance.csv", "text/csv")
    # Map visualization
    if show_map and {"start_lat", "start_lng"}.issubset(df_f.columns):
        st.subheader("🗺️ Imbalance map")
        coords = df_f.groupby("start_station_name")[["start_lat", "start_lng"]].median().rename(columns={"start_lat": "lat", "start_lng": "lon"})
        geo = total_imbalance.join(coords, on="station", how="left").dropna(subset=["lat", "lon"])
        if geo.empty:
            st.info("No coordinate data available for these stations.")
        else:
            import pydeck as pdk
            vmax = float(np.abs(geo["imbalance"]).max())
            scale_val = st.slider("Map bubble scale", 1, 15, 5)
            geo["radius"] = (60 + scale_val * (np.sqrt(np.abs(geo["imbalance"])) / np.sqrt(vmax if vmax > 0 else 1)) * 120).astype(float)
            geo["color"] = [[34, 197, 94, 210] if v >= 0 else [220, 38, 38, 210] for v in geo["imbalance"]]
            view_state = pdk.ViewState(latitude=float(geo["lat"].median()), longitude=float(geo["lon"].median()), zoom=11, pitch=0)
            layer = pdk.Layer("ScatterplotLayer", data=geo,
                               get_position="[lon, lat]",
                               get_radius="radius",
                               get_fill_color="color",
                               pickable=True)
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                            map_style="mapbox://styles/mapbox/dark-v11",
                            tooltip={"text": "{station}\nIn: {in}\nOut: {out}\nΔ: {imbalance}"})
            st.pydeck_chart(deck)
    else:
        if show_map:
            st.info("Map not available (coordinate data missing).")

def render_pareto_page():
    st.header("📈 Pareto curve — demand concentration")
    if "start_station_name" not in df_f.columns and "end_station_name" not in df_f.columns:
        st.warning("Station data is required for Pareto analysis.")
        return
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        basis = st.selectbox("Count rides by", ["Start stations", "End stations"], index=0)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        normalize = st.selectbox("Normalize counts", ["Total rides", "Per day (avg/station)"], index=0,
                                 help="If per-day, compute average daily rides per station")
    with c3:
        target_pct = st.slider("Target cumulative share (%)", 50, 95, 80, 1)
    c4, c5, c6 = st.columns(3)
    with c4:
        member_filter = st.selectbox("Member filter", ["All", "Member only", "Casual only"], index=0)
    with c5:
        min_rides = st.number_input("Min rides per station", 0, 10000, 0, 10)
    with c6:
        show_lorenz = st.checkbox("Show Lorenz curve", value=False,
                                  help="Add Lorenz curve (cumulative stations vs cumulative rides) to chart")
    subset = _time_slice(df_f, mode).copy()
    if member_filter != "All" and "member_type" in subset.columns:
        subset = subset[subset["member_type"].astype(str) == ("member" if member_filter.startswith("Member") else "casual")]
    if subset.empty:
        st.info("No rides for current filters.")
        return
    station_col = "start_station_name" if basis.startswith("Start") else "end_station_name"
    if station_col not in subset.columns:
        st.warning(f"Column `{station_col}` not found in data.")
        return
    subset[station_col] = subset[station_col].astype(str)
    if normalize.startswith("Per day") and "date" in subset.columns:
        per_day = subset.groupby([station_col, "date"]).size().rename("rides_day").reset_index()
        totals = per_day.groupby(station_col)["rides_day"].mean().rename("rides")
    else:
        totals = subset.groupby(station_col).size().rename("rides")
    if int(min_rides) > 0:
        totals = totals[totals >= float(min_rides)]
    if totals.empty:
        st.info("No stations left after applying filters.")
        return
    totals = totals.sort_values(ascending=False)
    counts = totals.to_numpy(dtype=float)
    n = len(counts)
    cum_share = np.cumsum(counts) / counts.sum()
    target_frac = target_pct / 100.0
    idx_target = int(np.searchsorted(cum_share, target_frac, side="left"))
    rank_needed = min(max(idx_target + 1, 1), n)
    # Concentration metrics
    x = np.sort(counts)  # ascending
    cum_x = np.cumsum(x)
    gini = 1 - (2 / (n - 1)) * (n - (cum_x.sum() / cum_x[-1])) if n > 1 else 0.0
    shares = counts / counts.sum()
    hhi = float(np.sum(shares ** 2))
    # Pareto curve figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, n+1), y=cum_share, mode="lines", name="Cumulative share",
                              hovertemplate="Rank: %{x}<br>Share: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=target_frac, line_dash="dot")
    fig.add_vline(x=rank_needed, line_dash="dot")
    fig.add_annotation(x=rank_needed, y=min(target_frac + 0.03, 0.98), showarrow=False,
                       text=f"Top ~{rank_needed:,}/{n:,} stations ≈ {target_pct}%", bgcolor="rgba(0,0,0,0.05)")
    if show_lorenz:
        x_lor = np.linspace(0, 1, n)
        y_lor = np.cumsum(np.sort(shares))
        fig.add_trace(go.Scatter(x=x_lor * n, y=y_lor, mode="lines", name="Lorenz",
                                  hovertemplate="Cumulative stations: %{x:.0f}<br>Cumulative rides: %{y:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, n], y=[0, 1], mode="lines", name="Equality", line=dict(dash="dash"), hoverinfo="skip"))
    fig.update_layout(height=500, title="Cumulative share of rides vs station rank",
                      xaxis_title="Station rank", yaxis_title="Cumulative share of rides")
    st.plotly_chart(fig, use_container_width=True)
    cA, cB = st.columns(2)
    with cA: st.metric("Gini coefficient", f"{gini:.3f}")
    with cB: st.metric("Herfindahl–Hirschman Index", f"{hhi:.3f}")
    st.caption("Tip: A steep curve indicates a small fraction of stations handles most rides (high concentration).")

def render_heatmap_page():
    st.header("⏰ Temporal load — weekday × start hour")
    if not {"started_at", "hour", "weekday"}.issubset(df_f.columns):
        st.info("Trip start time data (hour and weekday) is required for this analysis.")
        return
    c0, c1, c2, c3, c4, c5 = st.columns(6)
    with c0:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c1:
        scale = st.selectbox("Scale", ["Absolute", "Row %", "Column %", "Z-score"], index=0,
                              help="Row %: distribution within each weekday; Column %: within each hour.")
    with c2:
        hour_bin = st.slider("Hour bin size", 1, 3, 1, help="Group hours into 1, 2 or 3-hour buckets")
    with c3:
        smooth = st.toggle("Smooth across hours", value=False, help="Apply moving average across hours (visual smoothing)")
    with c4:
        wk_preset = st.selectbox("Preset", ["All days", "Weekdays only", "Weekend only"], index=0)
    with c5:
        member_mode = st.selectbox("Member view", ["All", "Member only", "Casual only", "Facet by Member Type"], index=0)
    subset = _time_slice(df_f, mode).copy()
    if wk_preset == "Weekdays only":
        subset = subset[subset["weekday"].isin([0, 1, 2, 3, 4])]
    elif wk_preset == "Weekend only":
        subset = subset[subset["weekday"].isin([5, 6])]
    facet = False
    if member_mode == "Member only" and "member_type" in subset.columns:
        subset = subset[subset["member_type"].astype(str) == "member"]
    elif member_mode == "Casual only" and "member_type" in subset.columns:
        subset = subset[subset["member_type"].astype(str) == "casual"]
    elif member_mode == "Facet by Member Type" and "member_type_display" in df_f.columns:
        facet = True
    if subset.empty:
        st.info("No data for current filters.")
        return
    def _render_heat(matrix: pd.DataFrame, title: str):
        if matrix.empty:
            st.info("Not enough data to render heatmap.")
            return
        if smooth:
            matrix = _smooth_by_hour(matrix, k=3)
        mat_display = matrix.copy()
        mat_display.index = _weekday_name(mat_display.index)
        fig = px.imshow(mat_display, aspect="auto", origin="lower",
                        labels={"x": "Hour of day", "y": "Day of week", "color": ("Value" if scale == "Absolute" else scale)},
                        text_auto=False, color_continuous_scale="Turbo" if scale == "Z-score" else "Viridis")
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(matrix.columns))),
                         ticktext=[f"{h:02d}:00" for h in matrix.columns])
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(matrix.index))),
                         ticktext=mat_display.index.tolist())
        fig.update_layout(height=600, title=title, margin=dict(l=20, r=20, t=50, b=50))
        hover_text = "<b>%{y}</b> @ <b>%{x}</b><br>Value: %{z}"
        if scale in ("Row %", "Column %"):
            hover_text = "<b>%{y}</b> @ <b>%{x}</b><br>Share: %{z:.1f}%"
        fig.update_traces(hovertemplate=hover_text)
        st.plotly_chart(fig, use_container_width=True)
    if facet:
        cL, cR = st.columns(2)
        for label, col in [("Member 🧑‍💼", cL), ("Casual 🚲", cR)]:
            sub_df = subset[subset["member_type"].astype(str) == ("member" if "Member" in label else "casual")]
            mat = _make_heat_grid(sub_df, hour_bin=hour_bin, scale=scale)
            with col: _render_heat(mat, f"Weekday × Hour — {label}")
    else:
        mat_all = _make_heat_grid(subset, hour_bin=hour_bin, scale=scale)
        _render_heat(mat_all, "Weekday × Hour — All riders")
    # Marginal profiles
    st.subheader("Marginal profiles")
    grid_abs = _make_heat_grid(subset, hour_bin=hour_bin, scale="Absolute")
    if not grid_abs.empty:
        hourly_profile = grid_abs.sum(axis=0).rename("rides").reset_index().rename(columns={"index": "hour"})
        hourly_profile["hour"] = hourly_profile["hour"].astype(int)
        f1 = px.line(hourly_profile, x="hour", y="rides", markers=True, labels={"hour": "Hour of day", "rides": "Rides"})
        f1.update_layout(height=300, title="Hourly total rides")
        st.plotly_chart(f1, use_container_width=True)
        weekday_profile = grid_abs.sum(axis=1).rename("rides").reset_index().rename(columns={0: "weekday"})
        weekday_profile["weekday_name"] = _weekday_name(weekday_profile["weekday"])
        f2 = px.bar(weekday_profile, x="weekday_name", y="rides", labels={"weekday_name": "Weekday", "rides": "Rides"})
        f2.update_layout(height=300, title="Total rides by weekday")
        st.plotly_chart(f2, use_container_width=True)
    st.caption("Tips: Use Row % to see within-day patterns; Column % to see which days dominate each hour; Z-score to highlight anomalies.")

def render_recommendations_page():
    st.header("🚀 Conclusion & Recommendations")
    st.markdown("### ")
    st.markdown("""### Recommendations (next 4–8 weeks)
1) **Scale hotspot capacity**  
   - 🧱 Deploy portable or temporary docks at high-demand hubs.  
   - 🎯 Aim for **≥85% dock availability by start of AM rush** and **≥70% before PM peak** at top-20 stations.
2) **Predictive stocking: weather + weekday**  
   - 📈 Use simple regression/rules for **next-day dock stock targets** by station.  
   - 🌡️ Pre-stock extra bikes when **forecast highs ≥ 22°C** (anticipate demand surge).
3) **Corridor-aligned rebalancing**  
   - 🚚 Stage trucks at **repeat high-flow endpoints**; run targeted **loop routes** along busy corridors.
4) **Rider incentives**  
   - 🎟️ Offer credits for returns to **under-stocked docks** during peak commute hours.

**Key KPIs**  
- ⛔ **Dock-out events** < 5% at top-20 stations during peaks  
- 📉 **Empty/full dock complaints** down 30% (month-on-month)  
- 🛣️ **Truck miles per rebalanced bike** down 15% (efficiency gain)  
- ⏱️ **On-time dock readiness** ≥ 90% (docks pre-stocked by AM peak)
""")
    st.markdown("> **Next Steps:** Pilot these measures at the top 10 stations for 2 weeks, then compare KPIs before vs after.")
    st.caption("🧱 *Note:* This analysis used a sample dataset; real deployment should include full data, per-dock inventory, and consider events/holidays.")
    # Example video embed (optional)
    st.video("https://www.youtube.com/watch?v=vm37IuX7UPQ")

# Render the selected page
if page == "Intro":
    render_intro_page()
elif page == "Weather vs Bike Usage":
    render_weather_page()
elif page == "Trip Metrics (Duration • Distance • Speed)":
    render_trip_metrics_page()
elif page == "Member vs Casual Profiles":
    render_member_vs_casual_page()
elif page == "OD Flows — Sankey + Map":
    render_od_flows_page()
elif page.startswith("OD Matrix"):
    render_od_matrix_page()
elif page == "Station Popularity":
    render_station_popularity_page()
elif page.startswith("Station Imbalance"):
    render_station_imbalance_page()
elif page.startswith("Pareto"):
    render_pareto_page()
elif page == "Weekday × Hour Heatmap":
    render_heatmap_page()
elif page == "Recommendations":
    render_recommendations_page()
