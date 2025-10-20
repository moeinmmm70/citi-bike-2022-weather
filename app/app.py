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
def _bin_hour(h: pd.Series, bin_size: int) -> pd.Series:
    # Bin 0-23 into 1/2/3-hr buckets
    b = (h // bin_size) * bin_size
    return b.clip(0, 23)

def _weekday_name(idx: pd.Series) -> pd.Series:
    return idx.map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})

def _make_heat_grid(df: pd.DataFrame,
                    hour_col="hour",
                    weekday_col="weekday",
                    hour_bin: int = 1,
                    scale: str = "Absolute") -> pd.DataFrame:
    """
    Returns a 7 x (24/hr_bin) grid with chosen scaling.
    scale ∈ {"Absolute", "Row %", "Column %", "Z-score"}.
    """
    if hour_col not in df.columns or weekday_col not in df.columns:
        return pd.DataFrame()

    d = df[[hour_col, weekday_col]].dropna().copy()
    d[hour_col] = _bin_hour(d[hour_col].astype(int), hour_bin)
    # count rides
    g = d.groupby([weekday_col, hour_col]).size().rename("rides").reset_index()
    # build full matrix (fill missing cells with 0)
    hours = list(range(0, 24, hour_bin))
    mat = (g.pivot(index=weekday_col, columns=hour_col, values="rides")
             .reindex(index=range(0,7), columns=hours)
             .fillna(0))

    if scale == "Absolute":
        return mat

    if scale == "Row %":
        # Normalize by weekday (row)
        row_sum = mat.sum(axis=1).replace(0, np.nan)
        return (mat.div(row_sum, axis=0) * 100).fillna(0)

    if scale == "Column %":
        # Normalize by hour (column)
        col_sum = mat.sum(axis=0).replace(0, np.nan)
        return (mat.div(col_sum, axis=1) * 100).fillna(0)

    if scale == "Z-score":
        # Center/scale per row (weekday)
        m = mat.mean(axis=1)
        s = mat.std(axis=1).replace(0, np.nan)
        return ((mat.sub(m, axis=0)).div(s, axis=0)).fillna(0)

    return mat

def _smooth_by_hour(mat: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Simple 1D moving-average smoothing across hours per weekday (window k, odd)."""
    if mat.empty or k <= 1:
        return mat
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    out = mat.copy()
    for i in out.index:
        row = out.loc[i].values
        s = pd.Series(row).rolling(k, center=True, min_periods=max(1, k//2)).mean().to_numpy()
        out.loc[i] = s
    return out

def _add_peak_annotation(fig, mat: pd.DataFrame, title_suffix=""):
    # find max cell; annotate
    if mat.empty:
        return fig
    idx = np.unravel_index(np.nanargmax(mat.values), mat.shape)
    r, c = idx[0], idx[1]
    wk = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][r]
    hr = mat.columns[c]
    val = mat.iloc[r, c]
    fig.add_annotation(
        x=c, y=r, text=f"Peak: {wk} {hr:02d}:00<br>{val:,.0f}" if np.isfinite(val) else "Peak",
        showarrow=True, arrowhead=2, ax=40, ay=-40, bgcolor="rgba(0,0,0,0.6)", font=dict(color="white", size=11)
    )
    if title_suffix:
        fig.update_layout(title=fig.layout.title.text + title_suffix)
    return fig

def _time_slice(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "hour" not in df.columns or "weekday" not in df.columns:
        return df
    if mode == "AM (06–11)":   return df[(df["hour"] >= 6)  & (df["hour"] <= 11)]
    if mode == "PM (16–20)":   return df[(df["hour"] >= 16) & (df["hour"] <= 20)]
    if mode == "Weekend":      return df[df["weekday"].isin([5,6])]
    if mode == "Weekday":      return df[df["weekday"].isin([0,1,2,3,4])]
    return df

def _build_od_edges(df: pd.DataFrame,
                    per_origin: bool,
                    topk: int,
                    min_rides: int,
                    drop_self_loops: bool,
                    member_split: bool) -> pd.DataFrame:
    # Guard
    need = {"start_station_name", "end_station_name"}
    if not need.issubset(df.columns) or df.empty:
        return pd.DataFrame(columns=["start_station_name","end_station_name","rides"])

    gb_cols = ["start_station_name", "end_station_name"]
    if member_split and "member_type_display" in df.columns:
        gb_cols.append("member_type_display")

    # Count rides
    g = (df.dropna(subset=["start_station_name","end_station_name"])
           .groupby(gb_cols).size().rename("rides").reset_index())

    # Optional medians if available
    if "distance_km" in df.columns:
        med_dist = df.groupby(gb_cols)["distance_km"].median().reset_index(name="med_distance_km")
        g = g.merge(med_dist, on=gb_cols, how="left")
    if "duration_min" in df.columns:
        med_dur = df.groupby(gb_cols)["duration_min"].median().reset_index(name="med_duration_min")
        g = g.merge(med_dur, on=gb_cols, how="left")

    # Drop self-loops (cast to str to be safe)
    if drop_self_loops and not g.empty:
        s = g["start_station_name"].astype(str)
        e = g["end_station_name"].astype(str)
        g = g[s != e]

    # Threshold
    g = g[g["rides"] >= int(min_rides)]
    if g.empty:
        return g

    # Top-k selection
    if per_origin:
        by = ["start_station_name"]
        if "member_type_display" in g.columns and member_split:
            by.append("member_type_display")
        g = (g.sort_values("rides", ascending=False)
               .groupby(by, as_index=False)
               .head(int(topk)))
    else:
        g = g.sort_values("rides", ascending=False).head(int(topk))

    return g.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def _cached_edges(df: pd.DataFrame,
                  per_origin: bool,
                  topk: int,
                  min_rides: int,
                  drop_self_loops: bool,
                  member_split: bool) -> pd.DataFrame:
    try:
        return _build_od_edges(df, per_origin, topk, min_rides, drop_self_loops, member_split)
    except Exception as e:
        st.warning(f"Edge build failed: {e}")
        return pd.DataFrame(columns=["start_station_name","end_station_name","rides"])

def _matrix_from_edges(edges: pd.DataFrame, member_split: bool) -> pd.DataFrame:
    if edges.empty: return pd.DataFrame()
    base = edges.copy()
    if member_split and "member_type_display" in base.columns:
        base = base.groupby(["start_station_name","end_station_name"], as_index=False)["rides"].sum()
    mat = (base.pivot(index="start_station_name", columns="end_station_name", values="rides").fillna(0))
    # order for readability
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index,
                  mat.sum(axis=0).sort_values(ascending=False).index]
    return mat

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

    # ── Parse timestamps first
    for col in ["date", "started_at", "ended_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure 'date' exists
    if "date" not in df.columns and "started_at" in df.columns:
        df["date"] = pd.to_datetime(df["started_at"], errors="coerce").dt.floor("D")

    # ── Enrich with full daily weather if available (merge on 'date')
    wx_path = Path("data/processed/nyc_weather_2022_daily_full.csv")
    if wx_path.exists() and "date" in df.columns:
        wx = pd.read_csv(wx_path, parse_dates=["date"])
        keep_cols = [c for c in wx.columns if c in [
            "date","avg_temp_c","tmin_c","tmax_c",
            "precip_mm","snow_mm","snow_depth_mm",
            "wind_mps","wind_kph","gust_mps","gust_kph",
            "wet_day","precip_bin","wind_bin"
        ] or c.startswith("wt")]
        wx = wx[keep_cols].copy()
        df = df.merge(wx, on="date", how="left")

    # ── Season if missing
    if "season" not in df.columns and "date" in df.columns:
        def season_from_month(m):
            if m in (12, 1, 2):  return "Winter"
            if m in (3, 4, 5):   return "Spring"
            if m in (6, 7, 8):   return "Summer"
            return "Autumn"
        df["season"] = df["date"].dt.month.map(season_from_month).astype("category")

    # ── Trip metrics
    if {"started_at","ended_at"}.issubset(df.columns):
        dur = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
        df["duration_min"] = dur.clip(lower=0)

    if {"start_lat","start_lng","end_lat","end_lng"}.issubset(df.columns):
        df["distance_km"] = _haversine_km(
            df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
        ).astype(float)

    if "duration_min" in df.columns and "distance_km" in df.columns:
        df["speed_kmh"] = (
            df["distance_km"] / (df["duration_min"] / 60.0)
        ).replace([np.inf, -np.inf], np.nan).clip(upper=60)

    # ── Temporal fields
    if "started_at" in df.columns:
        ts = df["started_at"]
        df["hour"]    = ts.dt.hour
        df["weekday"] = ts.dt.weekday
        df["month"]   = ts.dt.to_period("M").dt.to_timestamp()

    # ── Categories for perf
    for c in ["start_station_name","end_station_name","member_type","rideable_type","season"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # ── Pretty legend text for member_type
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

    # --- Ensure we have a temp column available at the row level
    if "avg_temp_c" not in df.columns and {"tmin_c", "tmax_c"}.issubset(df.columns):
        with pd.option_context("mode.use_inf_as_na", True):
            df = df.copy()
            df["avg_temp_c"] = (pd.to_numeric(df["tmin_c"], errors="coerce") + 
                                pd.to_numeric(df["tmax_c"], errors="coerce")) / 2.0

    # Base: rides per day
    daily = df.groupby("date", as_index=False).size().rename(columns={"size":"bike_rides_daily"})

    # Attach weather fields (use first/mean sensibly; daily NOAA is already per-day)
    attach = {}
    if "avg_temp_c"   in df.columns: attach["avg_temp_c"]   = "mean"
    if "tmin_c"       in df.columns: attach["tmin_c"]       = "mean"
    if "tmax_c"       in df.columns: attach["tmax_c"]       = "mean"
    if "precip_mm"    in df.columns: attach["precip_mm"]    = "mean"
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

# Sidebar reload button (works on old/new Streamlit)
if st.sidebar.button("🔄 Reload data"):
    st.cache_data.clear()
    if hasattr(st, "rerun"):
        st.rerun()
    else:
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
        "Trip Metrics (Duration • Distance • Speed)",     
        "Member vs Casual Profiles",                      
        "OD Flows — Sankey + Map",
        "OD Matrix — Top Origins × Dest",
        "Station Popularity",
        "Station Imbalance (In vs Out)",                  
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

# ---- backfill trip-level weather from daily ----
def _backfill_trip_weather(df_trips: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df is None or daily_df.empty or "date" not in df_trips.columns:
        return df_trips

    out = df_trips.copy()

    # build daily lookups only for columns that exist
    lookups = {}
    for col in ["avg_temp_c", "wind_kph", "gust_kph", "precip_mm", "wet_day", "precip_bin", "wind_bin"]:
        if col in daily_df.columns:
            lookups[col] = daily_df.set_index("date")[col]

    # ensure both are datetime (day-level) for mapping
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # daily_df['date'] is already datetime from ensure_daily()

    for col, mapper in lookups.items():
        if col not in out.columns or out[col].notna().sum() == 0:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = out[col].where(out[col].notna(), out["date"].map(mapper))

    return out

# apply backfill once
df_f = _backfill_trip_weather(df_f, daily_all)

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
            n = int(roll_win.replace("d", ""))
            for col in ["bike_rides_daily", "avg_temp_c", "wind_kph"]:
                if col in d.columns:
                    d[f"{col}_roll"] = d[col].rolling(n, min_periods=max(2, n // 2), center=True).mean()

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
                    marker_color="rgba(100,100,120,0.35)", opacity=0.4
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

        # pick a temperature column that exists
        temp_col = None
        for c in ["avg_temp_c", "tavg_c", "tmean_c", "tmin_c", "tmax_c"]:
            if c in d.columns:
                temp_col = c
                break

        if temp_col is None or "bike_rides_daily" not in d.columns:
            st.info("Temperature or daily rides missing for scatter. Check weather merge.")
        else:
            scatter_df = d.dropna(subset=[temp_col, "bike_rides_daily"]).copy()
            if scatter_df.empty:
                st.info("No rows available for scatter after dropping missing values.")
            else:
                # Only use color if the column exists
                chosen = None if color_scatter_by == "None" else color_scatter_by
                color_arg = chosen if (chosen in scatter_df.columns) else None

                labels = {
                    temp_col: temp_col.replace("_", " ").title().replace("C", "(°C)"),
                    "bike_rides_daily": "Bike rides (count)",
                    "wet_day": "Wet day",
                    "precip_bin": "Precipitation",
                    "wind_bin": "Wind",
                }
                fig2 = px.scatter(
                    scatter_df, x=temp_col, y="bike_rides_daily", color=color_arg,
                    labels=labels, opacity=0.85, trendline="ols"
                )
                fig2.update_layout(height=520)
                st.plotly_chart(fig2, use_container_width=True)

        # ===== Distribution: rides by precip_bin (or wet/dry fallback) =====
        st.subheader("Distribution by rainfall")
        if "precip_bin" in d.columns and d["precip_bin"].notna().any():
            fig3 = px.box(
                d, x="precip_bin", y="bike_rides_daily",
                labels={"precip_bin": "Precipitation", "bike_rides_daily": "Bike rides per day"},
                category_orders={"precip_bin": ["Low", "Medium", "High"]}
            )
            fig3.update_layout(height=420)
            st.plotly_chart(fig3, use_container_width=True)
        elif "wet_day" in d.columns:
            fig3 = px.box(
                d, x=d["wet_day"].map({0: "Dry", 1: "Wet"}), y="bike_rides_daily",
                labels={"x": "Day type", "bike_rides_daily": "Bike rides per day"}
            )
            fig3.update_layout(height=420)
            st.plotly_chart(fig3, use_container_width=True)

        # ===== Quick deltas (plain text) =====
        kcols = st.columns(3)
        with kcols[0]:
            if "precip_bin" in d.columns and d["precip_bin"].notna().any():
                lo = d.loc[d["precip_bin"] == "Low", "bike_rides_daily"].mean()
                hi = d.loc[d["precip_bin"] == "High", "bike_rides_daily"].mean()
                if pd.notnull(lo) and pd.notnull(hi) and lo > 0:
                    st.metric("High rain vs Low", f"{(hi - lo) / lo * 100:+.0f}%")
        with kcols[1]:
            if "wet_day" in d.columns:
                dry = d.loc[d["wet_day"] == 0, "bike_rides_daily"].mean()
                wet = d.loc[d["wet_day"] == 1, "bike_rides_daily"].mean()
                if pd.notnull(dry) and pd.notnull(wet) and dry > 0:
                    st.metric("Wet vs Dry", f"{(wet - dry) / dry * 100:+.0f}%")
        with kcols[2]:
            if "wind_kph" in d.columns:
                breezy = d.loc[d["wind_kph"] >= 20, "bike_rides_daily"].mean()
                calm   = d.loc[d["wind_kph"] < 10,  "bike_rides_daily"].mean()
                if pd.notnull(calm) and pd.notnull(breezy) and calm > 0:
                    st.metric("Windy (≥20) vs Calm (<10)", f"{(breezy - calm) / calm * 100:+.0f}%")

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

        # ===== Histograms (robust) =====
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

        # ===== Scatter: distance vs duration (by member type) =====
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

        # ===== Weather relationships (new) =====
        def _add_fit_line(fig_, xvals, yvals, name):
            # robust: require at least 3 valid points and 2 unique x's
            x = pd.to_numeric(xvals, errors="coerce")
            y = pd.to_numeric(yvals, errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() >= 3 and x[ok].nunique() >= 2:
                a, b = np.polyfit(x[ok], y[ok], 1)
                xs = np.linspace(x[ok].min(), x[ok].max(), 100)
                ys = a * xs + b
                fig_.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, line=dict(dash="dash")))
            return fig_

        st.subheader("Weather relationships")
        c1, c2 = st.columns(2)

        temp_ok = "avg_temp_c" in df_f.columns and df_f["avg_temp_c"].notna().any()
        wind_ok = "wind_kph"   in df_f.columns and df_f["wind_kph"].notna().any()

        # Speed vs temperature
        with c1:
            if temp_ok:
                dat = df_f[m_spd & df_f["avg_temp_c"].notna()]
                if len(dat) > nmax: dat = dat.sample(n=nmax, random_state=4)
                figt = px.scatter(
                    dat, x="avg_temp_c", y="speed_kmh",
                    color=color_col, opacity=0.7,
                    labels={"avg_temp_c":"Avg temperature (°C)", "speed_kmh":"Speed (km/h)", "member_type_display": MEMBER_LEGEND_TITLE}
                )
                figt = _add_fit_line(figt, dat["avg_temp_c"], dat["speed_kmh"], "Linear fit")
                figt.update_layout(height=480, title="Speed vs Temperature")
                st.plotly_chart(figt, use_container_width=True)
            else:
                st.info("No temperature column available for this view.")

        # Speed vs wind
        with c2:
            if wind_ok:
                dat = df_f[m_spd & df_f["wind_kph"].notna()]
                if len(dat) > nmax: dat = dat.sample(n=nmax, random_state=5)
                figw = px.scatter(
                    dat, x="wind_kph", y="speed_kmh",
                    color=color_col, opacity=0.7,
                    labels={"wind_kph":"Wind (kph)", "speed_kmh":"Speed (km/h)", "member_type_display": MEMBER_LEGEND_TITLE}
                )
                figw = _add_fit_line(figw, dat["wind_kph"], dat["speed_kmh"], "Linear fit")
                figw.update_layout(height=480, title="Speed vs Wind")
                st.plotly_chart(figw, use_container_width=True)
            else:
                st.info("No wind column available for this view.")

        # Distance/Duration vs Temperature (optional, helps tell comfort story)
        c3, c4 = st.columns(2)
        with c3:
            if temp_ok:
                dat = df_f[m_dur & df_f["avg_temp_c"].notna()]
                if len(dat) > nmax: dat = dat.sample(n=nmax, random_state=6)
                figdt = px.scatter(
                    dat, x="avg_temp_c", y="duration_min",
                    color=color_col, opacity=0.6,
                    labels={"avg_temp_c":"Avg temperature (°C)", "duration_min":"Duration (min)", "member_type_display": MEMBER_LEGEND_TITLE}
                )
                figdt = _add_fit_line(figdt, dat["avg_temp_c"], dat["duration_min"], "Linear fit")
                figdt.update_layout(height=420, title="Duration vs Temperature")
                st.plotly_chart(figdt, use_container_width=True)

        with c4:
            if temp_ok:
                dat = df_f[m_dst & df_f["avg_temp_c"].notna()]
                if len(dat) > nmax: dat = dat.sample(n=nmax, random_state=7)
                figDxT = px.scatter(
                    dat, x="avg_temp_c", y="distance_km",
                    color=color_col, opacity=0.6,
                    labels={"avg_temp_c":"Avg temperature (°C)", "distance_km":"Distance (km)", "member_type_display": MEMBER_LEGEND_TITLE}
                )
                figDxT = _add_fit_line(figDxT, dat["avg_temp_c"], dat["distance_km"], "Linear fit")
                figDxT.update_layout(height=420, title="Distance vs Temperature")
                st.plotly_chart(figDxT, use_container_width=True)

        # ===== Rain/Wet impact on duration & speed =====
        st.subheader("Rain impact on trip characteristics")
        has_precip_bin = ("precip_bin" in df_f.columns) and df_f["precip_bin"].notna().any()
        has_wet_flag   = ("wet_day" in df_f.columns)

        if has_precip_bin or has_wet_flag:
            cc1, cc2 = st.columns(2)

            # Duration boxes
            with cc1:
                if has_precip_bin:
                    figpb = px.box(
                        df_f[m_dur], x="precip_bin", y="duration_min",
                        category_orders={"precip_bin":["Low","Medium","High"]},
                        labels={"precip_bin":"Precipitation", "duration_min":"Duration (min)"}
                    )
                    figpb.update_layout(height=420, title="Duration by Precipitation")
                    st.plotly_chart(figpb, use_container_width=True)
                elif has_wet_flag:
                    figwd = px.box(
                        df_f[m_dur].assign(day_type=lambda x: x["wet_day"].map({0:"Dry",1:"Wet"})),
                        x="day_type", y="duration_min",
                        labels={"day_type":"Day type", "duration_min":"Duration (min)"}
                    )
                    figwd.update_layout(height=420, title="Duration: Wet vs Dry")
                    st.plotly_chart(figwd, use_container_width=True)

            # Speed boxes
            with cc2:
                if has_precip_bin:
                    figpbs = px.box(
                        df_f[m_spd], x="precip_bin", y="speed_kmh",
                        category_orders={"precip_bin":["Low","Medium","High"]},
                        labels={"precip_bin":"Precipitation", "speed_kmh":"Speed (km/h)"}
                    )
                    figpbs.update_layout(height=420, title="Speed by Precipitation")
                    st.plotly_chart(figpbs, use_container_width=True)
                elif has_wet_flag:
                    figwds = px.box(
                        df_f[m_spd].assign(day_type=lambda x: x["wet_day"].map({0:"Dry",1:"Wet"})),
                        x="day_type", y="speed_kmh",
                        labels={"day_type":"Day type", "speed_kmh":"Speed (km/h)"}
                    )
                    figwds.update_layout(height=420, title="Speed: Wet vs Dry")
                    st.plotly_chart(figwds, use_container_width=True)
        else:
            st.info("No precipitation flags available to show rain impact.")

        # ===== Quick weather deltas (KPIs) =====
        k1, k2, k3, k4 = st.columns(4)
        # Wet vs Dry (speed)
        with k1:
            if has_wet_flag and df_f["wet_day"].notna().any():
                dry_spd = df_f.loc[m_spd & (df_f["wet_day"]==0), "speed_kmh"].mean()
                wet_spd = df_f.loc[m_spd & (df_f["wet_day"]==1), "speed_kmh"].mean()
                if pd.notnull(dry_spd) and pd.notnull(wet_spd) and dry_spd>0:
                    st.metric("Speed: Wet vs Dry", f"{(wet_spd-dry_spd)/dry_spd*100:+.1f}%")

        # Windy vs Calm (speed)
        with k2:
            if wind_ok:
                calm_spd   = df_f.loc[m_spd & (df_f["wind_kph"]<10),  "speed_kmh"].mean()
                windy_spd  = df_f.loc[m_spd & (df_f["wind_kph"]>=20), "speed_kmh"].mean()
                if pd.notnull(calm_spd) and pd.notnull(windy_spd) and calm_spd>0:
                    st.metric("Speed: Windy (≥20) vs Calm (<10)", f"{(windy_spd-calm_spd)/calm_spd*100:+.1f}%")

        # Comfy vs Extreme temps (speed)
        with k3:
            if temp_ok:
                comfy = df_f.loc[m_spd & df_f["avg_temp_c"].between(15,25), "speed_kmh"].mean()
                extreme = df_f.loc[m_spd & (~df_f["avg_temp_c"].between(5,30)), "speed_kmh"].mean()
                if pd.notnull(comfy) and pd.notnull(extreme):
                    st.metric("Speed: Comfy (15–25°C) vs Extreme", f"{(comfy-extreme)/comfy*100:+.1f}%")

        # Rain effect on duration (optional)
        with k4:
            if has_precip_bin:
                low_dur  = df_f.loc[m_dur & (df_f["precip_bin"]=="Low"), "duration_min"].mean()
                high_dur = df_f.loc[m_dur & (df_f["precip_bin"]=="High"), "duration_min"].mean()
                if pd.notnull(low_dur) and pd.notnull(high_dur) and low_dur>0:
                    st.metric("Duration: High rain vs Low", f"{(high_dur-low_dur)/low_dur*100:+.1f}%")

        st.caption("Robust view clips only for plotting. All rows remain available for other pages/exports.")

elif page == "Member vs Casual Profiles":
    st.header("👥 Member vs Casual riding patterns")

    if "member_type_display" not in df_f.columns or "hour" not in df_f.columns:
        st.info("Need `member_type` and `started_at` (engineered hour).")
    else:
        # ——— Base behavioural profiles ———
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

        # ——— Weather-aware mix: Rain ———
        st.subheader("Weather mix by user type — rain")
        has_precip_bin = ("precip_bin" in df_f.columns) and df_f["precip_bin"].notna().any()
        has_wet_flag   = ("wet_day" in df_f.columns)

        cwx1, cwx2 = st.columns(2)

        with cwx1:
            if has_precip_bin:
                g3 = (df_f.dropna(subset=["precip_bin"])
                          .groupby(["member_type_display","precip_bin"])
                          .size().rename("rides").reset_index())
                g3["precip_bin"] = pd.Categorical(g3["precip_bin"], ["Low","Medium","High"], ordered=True)
                fig3 = px.bar(
                    g3, x="precip_bin", y="rides", color="member_type_display", barmode="group",
                    labels={"precip_bin":"Precipitation", "rides":"Rides", "member_type_display": MEMBER_LEGEND_TITLE}
                )
                fig3.update_layout(height=420, title="Ride volume by precipitation bin")
                st.plotly_chart(fig3, use_container_width=True)
            elif has_wet_flag:
                g3b = (df_f.assign(day_type=lambda x: x["wet_day"].map({0:"Dry",1:"Wet"}))
                            .groupby(["member_type_display","day_type"])
                            .size().rename("rides").reset_index())
                fig3b = px.bar(
                    g3b, x="day_type", y="rides", color="member_type_display", barmode="group",
                    labels={"day_type":"Day type", "rides":"Rides", "member_type_display": MEMBER_LEGEND_TITLE}
                )
                fig3b.update_layout(height=420, title="Ride volume: Wet vs Dry")
                st.plotly_chart(fig3b, use_container_width=True)
            else:
                st.info("No rain columns (`precip_bin` or `wet_day`) available.")

        # ——— Temperature distribution by user type ———
        with cwx2:
            if "avg_temp_c" in df_f.columns and df_f["avg_temp_c"].notna().any():
                vdat = df_f.dropna(subset=["avg_temp_c"])
                figv = px.violin(
                    vdat, x="member_type_display", y="avg_temp_c", box=True, points=False,
                    labels={"member_type_display": MEMBER_LEGEND_TITLE, "avg_temp_c":"Avg temp during rides (°C)"}
                )
                figv.update_layout(height=420, title="Where each group rides by temperature")
                st.plotly_chart(figv, use_container_width=True)
            else:
                st.info("Temperature not available to plot distributions.")

        # ——— Wind effect ———
        st.subheader("Wind effect by user type")
        if "wind_bin" in df_f.columns and df_f["wind_bin"].notna().any():
            g4 = (df_f.dropna(subset=["wind_bin"])
                      .groupby(["member_type_display","wind_bin"])
                      .size().rename("rides").reset_index())
            # keep a sensible order if present
            order_wind = ["Calm","Breeze","Windy","Very Windy"]
            present = [x for x in order_wind if x in g4["wind_bin"].unique().tolist()]
            if present:
                g4["wind_bin"] = pd.Categorical(g4["wind_bin"], present, ordered=True)
            fig4 = px.bar(
                g4, x="wind_bin", y="rides", color="member_type_display", barmode="group",
                labels={"wind_bin":"Wind", "rides":"Rides", "member_type_display": MEMBER_LEGEND_TITLE}
            )
            fig4.update_layout(height=420)
            st.plotly_chart(fig4, use_container_width=True)
        elif "wind_kph" in df_f.columns and df_f["wind_kph"].notna().any():
            # fallback: bin wind_kph
            bins = [-1, 10, 20, 30, 200]
            labels_w = ["<10","10–20","20–30","30+"]
            g4b = (df_f.assign(wind_bin=lambda x: pd.cut(x["wind_kph"], bins, labels=labels_w, include_lowest=True))
                        .groupby(["member_type_display","wind_bin"])
                        .size().rename("rides").reset_index())
            fig4b = px.bar(
                g4b, x="wind_bin", y="rides", color="member_type_display", barmode="group",
                labels={"wind_bin":"Wind (kph)", "rides":"Rides", "member_type_display": MEMBER_LEGEND_TITLE}
            )
            fig4b.update_layout(height=420)
            st.plotly_chart(fig4b, use_container_width=True)
        else:
            st.info("No wind columns available.")

        # ——— Performance vs temperature (median speed & duration by temp band) ———
        st.subheader("Performance vs temperature")
        if {"avg_temp_c","speed_kmh","duration_min"}.issubset(df_f.columns) and df_f["avg_temp_c"].notna().any():
            # Bin temperature (you can tweak bands)
            tbins = [-20,-5,0,5,10,15,20,25,30,35,50]
            tlabs = ["<-5","-5–0","0–5","5–10","10–15","15–20","20–25","25–30","30–35",">35"]
            dat = df_f.dropna(subset=["avg_temp_c"]).copy()
            dat["temp_band"] = pd.cut(dat["avg_temp_c"], tbins, labels=tlabs, include_lowest=True)

            # Median speed by band & user type
            gs = (dat.groupby(["member_type_display","temp_band"])["speed_kmh"]
                      .median().reset_index().dropna(subset=["temp_band"]))
            figS = px.line(
                gs, x="temp_band", y="speed_kmh", color="member_type_display", markers=True,
                labels={"temp_band":"Temperature band (°C)", "speed_kmh":"Median speed (km/h)", "member_type_display": MEMBER_LEGEND_TITLE}
            )
            figS.update_layout(height=420)
            st.plotly_chart(figS, use_container_width=True)

            # Median duration by band & user type
            gd = (dat.groupby(["member_type_display","temp_band"])["duration_min"]
                      .median().reset_index().dropna(subset=["temp_band"]))
            figD = px.line(
                gd, x="temp_band", y="duration_min", color="member_type_display", markers=True,
                labels={"temp_band":"Temperature band (°C)", "duration_min":"Median duration (min)", "member_type_display": MEMBER_LEGEND_TITLE}
            )
            figD.update_layout(height=420)
            st.plotly_chart(figD, use_container_width=True)
        else:
            st.info("Need avg_temp_c, duration_min, and speed_kmh to chart performance vs temperature.")

        # ——— Optional: bike type mix by season (kept from earlier) ———
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

# ───────────────────────── OD Flows (Sankey) & Matrix ─────────────────────────
elif page == "OD Flows — Sankey + Map":
    st.header("🔀 Origin → Destination — Sankey + Map")

    need = {"start_station_name", "end_station_name"}
    if not need.issubset(df_f.columns):
        st.info("Need start/end station names.")
        st.stop()

    # ── Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        per_origin = st.checkbox("Top-k per origin", value=True)
    with c3:
        topk = st.slider("Top-k edges", 10, 200, 40, 10)  # lighter default
    with c4:
        member_split = st.checkbox("Split by Member Type", value=("member_type_display" in df_f.columns))

    c5, c6, c7 = st.columns(3)
    with c5:
        min_rides = st.number_input("Min rides per edge", min_value=1, max_value=1000, value=2, step=1)
    with c6:
        drop_loops = st.checkbox("Exclude self-loops", value=True)
    with c7:
        render_now = st.checkbox("Render visualizations", value=False, help="Tick to build Sankey and Map")

    # Apply time slice & build edges
    sub = _time_slice(df_f, mode)
    edges = _cached_edges(sub, per_origin, topk, min_rides, drop_loops, member_split)
    if edges is None or not isinstance(edges, pd.DataFrame):
        edges = pd.DataFrame(columns=["start_station_name", "end_station_name", "rides"])

    # Empty case → compute precise suggestion with the SAME rules
    if edges.empty:
        gb_cols = ["start_station_name", "end_station_name"]
        if member_split and "member_type_display" in sub.columns:
            gb_cols.append("member_type_display")

        g0 = sub.groupby(gb_cols).size().rename("rides").reset_index()

        if drop_loops and not g0.empty:
            s = g0["start_station_name"].astype(str)
            e = g0["end_station_name"].astype(str)
            g0 = g0[s != e]

        if g0.empty:
            st.info("No OD pairs in the current slice (check date/hour/weekday filters).")
            st.stop()

        counts = np.sort(g0["rides"].to_numpy())[::-1]
        max_edge = int(counts[0])
        kth = int(counts[min(topk - 1, len(counts) - 1)])
        suggested = max(1, min(kth, max_edge))

        st.info(
            f"No OD edges for current filters. "
            f"Try **Min rides per edge ≤ {suggested}** "
            f"(max edge has {max_edge} rides; the {topk}-th heaviest has {kth})."
        )

        with st.expander("Preview top OD pairs (before min-rides cut)"):
            preview = g0.sort_values("rides", ascending=False).head(min(25, len(g0)))
            st.dataframe(preview, use_container_width=True)

        st.stop()

    # Guard: only render heavy visuals on demand
    if not render_now:
        st.success(f"{len(edges):,} edges match. Tick **Render visualizations** to draw Sankey + Map.")
        st.stop()

    # ===== Sankey (hard caps) =====
    st.subheader("Sankey — top flows (capped)")
    MAX_LINKS = 350
    MAX_NODES = 110

    edges_vis = edges.nlargest(MAX_LINKS, "rides").copy()
    node_labels = pd.Index(pd.unique(edges_vis[["start_station_name", "end_station_name"]].values.ravel()))
    if len(node_labels) > MAX_NODES:
        deg = pd.concat(
            [
                edges_vis.groupby("start_station_name")["rides"].sum(),
                edges_vis.groupby("end_station_name")["rides"].sum(),
            ],
            axis=1,
        ).fillna(0).sum(axis=1).sort_values(ascending=False)
        keep = set(deg.head(MAX_NODES).index)
        edges_vis = edges_vis[
            edges_vis["start_station_name"].isin(keep) & edges_vis["end_station_name"].isin(keep)
        ]
        node_labels = pd.Index(sorted(keep))
        st.info(f"Limited to {len(node_labels)} nodes / {len(edges_vis)} links for performance.")

    if edges_vis.empty:
        st.info("Nothing to render after caps; relax filters or lower caps.")
    else:
        idx = pd.Series(range(len(node_labels)), index=node_labels)
        src = edges_vis["start_station_name"].map(idx)
        tgt = edges_vis["end_station_name"].map(idx)

        link_colors = None
        if member_split and "member_type_display" in edges_vis.columns:
            cmap = {"Member 🧑‍💼": "rgba(34,197,94,0.6)", "Casual 🚲": "rgba(37,99,235,0.6)"}
            link_colors = (
                edges_vis["member_type_display"].astype("object").map(cmap).fillna("rgba(160,160,160,0.45)").tolist()
            )

        sankey = go.Sankey(
            node=dict(label=node_labels.astype(str).tolist(), pad=6, thickness=12),
            link=dict(source=src, target=tgt, value=edges_vis["rides"].astype(float), color=link_colors),
        )
        fig = go.Figure(sankey)
        fig.update_layout(height=540, title="Top OD flows (capped)")
        st.plotly_chart(fig, use_container_width=True)

    # ===== Map (capped & lightweight) =====
    if {"start_lat", "start_lng", "end_lat", "end_lng"}.issubset(df_f.columns):
        st.subheader("🗺️ Map — OD arcs (width ∝ volume)")
        import pydeck as pdk

        map_edges = edges.nlargest(250, "rides").copy()  # hard cap for GPU
        starts = (
            df_f.groupby("start_station_name")[["start_lat", "start_lng"]]
            .median()
            .rename(columns={"start_lat": "s_lat", "start_lng": "s_lng"})
        )
        ends = (
            df_f.groupby("end_station_name")[["end_lat", "end_lng"]]
            .median()
            .rename(columns={"end_lat": "e_lat", "end_lng": "e_lng"})
        )

        geo = (
            map_edges.join(starts, on="start_station_name")
            .join(ends, on="end_station_name")
            .dropna(subset=["s_lat", "s_lng", "e_lat", "e_lng"])
            .copy()
        )

        if geo.empty:
            st.info("No coordinates available for these edges.")
        else:
            vmax = float(geo["rides"].max())
            w_scale = st.slider("Arc width scale", 1, 25, 8)
            geo["width"] = (w_scale * (np.sqrt(geo["rides"]) / np.sqrt(vmax if vmax > 0 else 1))).clip(0.5, 12)

            if member_split and "member_type_display" in geo.columns:
                geo["color"] = geo["member_type_display"].astype("object").map(
                    {"Member 🧑‍💼": [34, 197, 94, 200], "Casual 🚲": [37, 99, 235, 200]}
                )
                geo["color"] = geo["color"].apply(lambda v: v if isinstance(v, list) else [160, 160, 160, 180])
            else:
                geo["color"] = [[37, 99, 235, 200]] * len(geo)

            layer = pdk.Layer(
                "ArcLayer",
                data=geo,
                get_source_position="[s_lng, s_lat]",
                get_target_position="[e_lng, e_lat]",
                get_width="width",
                get_source_color="color",
                get_target_color="color",
                pickable=False,
                auto_highlight=False,
            )

            center_lat = float(pd.concat([geo["s_lat"], geo["e_lat"]]).median())
            center_lon = float(pd.concat([geo["s_lng"], geo["e_lng"]]).median())
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=30)

            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v11")
            st.pydeck_chart(deck)
    else:
        st.info("Trip coordinates not available for map.")

# ───────────────────────── OD Matrix — Top Origins × Destinations ─────────────────────────
elif page == "OD Matrix — Top Origins × Dest":
    st.header("📊 OD Matrix — Top origins × destinations")

    need = {"start_station_name", "end_station_name"}
    if not need.issubset(df_f.columns):
        st.info("Need start/end station names.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        per_origin = st.checkbox("Top-k per origin", value=True)
    with c3:
        topk = st.slider("Top-k edges", 10, 300, 80, 10)  # matrix tolerates a bit more
    with c4:
        member_split = st.checkbox("Split by Member Type", value=("member_type_display" in df_f.columns))

    c5, c6 = st.columns(2)
    with c5:
        min_rides = st.number_input("Min rides per edge", 1, 1000, 5, 1)
    with c6:
        log_matrix = st.checkbox("Log color", value=True)

    sub = _time_slice(df_f, mode)
    edges = _cached_edges(sub, per_origin, topk, min_rides, True, member_split)
    if edges is None or not isinstance(edges, pd.DataFrame):
        edges = pd.DataFrame(columns=["start_station_name", "end_station_name", "rides"])

    if edges.empty:
        gb_cols = ["start_station_name", "end_station_name"]
        if member_split and "member_type_display" in sub.columns:
            gb_cols.append("member_type_display")

        counts = sub.groupby(gb_cols).size()
        total_pairs = int(counts.shape[0])
        if total_pairs == 0:
            st.info("No OD pairs in the current data slice (check date/hour/weekday filters).")
        else:
            sorted_counts = np.sort(counts.values)[::-1]
            idx = min(topk - 1, len(sorted_counts) - 1)
            suggested = int(max(1, sorted_counts[idx]))
            st.info(
                f"No OD edges for current filters. Try **Min rides per edge = {suggested}** "
                f"(there are {total_pairs:,} unique OD pairs; the {topk}-th heaviest has {suggested} rides)."
            )
        st.stop()

    # Keep a bounded square for readability/perf
    MAX_SIDE = 35
    value_col = "rides"
    by_o = edges.groupby("start_station_name")[value_col].sum().nlargest(MAX_SIDE).index
    by_d = edges.groupby("end_station_name")[value_col].sum().nlargest(MAX_SIDE).index
    mat_edges = edges[
        edges["start_station_name"].isin(by_o) & edges["end_station_name"].isin(by_d)
    ]

    mat = _matrix_from_edges(mat_edges, member_split=member_split)

    if mat.empty:
        st.info("No matrix after limiting size; increase Top-k or relax filters.")
    else:
        z = np.log10(mat.values + 1) if log_matrix else mat.values
        figm = px.imshow(
            z, aspect="auto", origin="lower", labels=dict(color=("log10(Rides+1)" if log_matrix else "Rides"))
        )
        figm.update_xaxes(
            tickmode="array", tickvals=list(range(len(mat.columns))), ticktext=[str(x)[:24] for x in mat.columns]
        )
        figm.update_yaxes(
            tickmode="array", tickvals=list(range(len(mat.index))), ticktext=[str(x)[:24] for x in mat.index]
        )
        figm.update_layout(height=640, title=f"Origin × Destination (top {len(mat)}×{len(mat.columns)})")
        st.plotly_chart(figm, use_container_width=True)

        # Export (matrix-friendly edges)
        export_cols = ["start_station_name", "end_station_name", "rides"]
        if member_split and "member_type_display" in mat_edges.columns:
            export_cols.insert(2, "member_type_display")

        csv_bytes = mat_edges[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download OD edges (CSV)", csv_bytes, "od_edges_matrix_view.csv", "text/csv")

elif page == "Station Popularity":
    st.header("🚉 Most popular start stations")

    if "start_station_name" not in df_f.columns:
        st.warning("`start_station_name` not found in sample.")
    else:
        # ── Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            topN = st.slider("Top N stations", 5, 100, 20, 5)
        with c2:
            metric = st.selectbox("Metric", ["Rides", "Share %"], help="Share is within current filters.")
        with c3:
            group_by = st.selectbox("Group by", ["Overall", "By Month", "By Hour"])

        c4, c5 = st.columns(2)
        with c4:
            stack_by_member = st.checkbox("Stack by Member Type", value=("member_type_display" in df_f.columns))
        with c5:
            wx_split = st.selectbox(
                "Weather split",
                ["None", "Wet vs Dry", "Temp bands (Cold/Mild/Hot)"],
                index=0 if "wet_day" not in df_f.columns and "avg_temp_c" not in df_f.columns else 1
            )

        st.markdown("---")

        # ── Prep base aggregations
        base = df_f.copy()
        base["station"] = base["start_station_name"].astype(str)

        # Weather split columns (optional)
        wx_col = None
        if wx_split == "Wet vs Dry" and "wet_day" in base.columns:
            wx_col = "wet_day_label"
            base[wx_col] = base["wet_day"].map({0: "Dry", 1: "Wet"})
        elif wx_split.startswith("Temp") and "avg_temp_c" in base.columns:
            wx_col = "temp_band"
            base[wx_col] = pd.cut(
                base["avg_temp_c"],
                bins=[-100, 5, 20, 200],
                labels=["Cold <5°C", "Mild 5–20°C", "Hot >20°C"],
                include_lowest=True
            )

        # Member split (optional)
        mcol = "member_type_display" if (stack_by_member and "member_type_display" in base.columns) else None

        # Top stations by total rides in current filter
        leaderboard = (base.groupby("station").size().rename("rides").sort_values(ascending=False).head(topN).reset_index())
        keep = set(leaderboard["station"])
        small = base[base["station"].isin(keep)].copy()

        # Share calculation helper
        def _maybe_to_share(df_grp, val_col="rides", by_cols=None):
            if metric == "Share %":
                if by_cols is None or not by_cols:
                    total = df_grp[val_col].sum()
                    df_grp[val_col] = np.where(total > 0, df_grp[val_col] / total * 100.0, 0.0)
                else:
                    # normalize within each group key (exclude station from the denominator)
                    denom = df_grp.groupby(by_cols)[val_col].transform(lambda s: s.sum() if s.sum() > 0 else np.nan)
                    df_grp[val_col] = (df_grp[val_col] / denom * 100.0).fillna(0.0)
            return df_grp

        # ───────────────────────── Overall (Bar) ─────────────────────────
        if group_by == "Overall":
            by = ["station"]
            if wx_col: by.append(wx_col)
            if mcol:   by.append(mcol)
            g = (small.groupby(by).size().rename("value").reset_index())

            # Share %
            share_keys = []
            if wx_col: share_keys.append(wx_col)
            if mcol:   share_keys.append(mcol)
            g = _maybe_to_share(g, val_col="value", by_cols=share_keys)

            # Build label and chart
            xlab = "Share (%)" if metric == "Share %" else "Rides (count)"
            color = mcol if mcol else wx_col
            fig = px.bar(
                g, x="station", y="value", color=color, barmode=("stack" if color else "relative"),
                labels={"station": "Station", "value": xlab, (color or ""): (MEMBER_LEGEND_TITLE if color == mcol else "Weather")},
                hover_data={ "station": True, "value": ":,.2f" if metric=="Share %" else ":," }
            )
            fig.update_layout(
                height=620,
                title=f"Top {len(keep)} start stations — {xlab}",
                xaxis_title="Station", yaxis_title=xlab,
                margin=dict(l=20, r=20, t=60, b=100),
                legend_title_text=(MEMBER_LEGEND_TITLE if color == mcol else ("Weather" if color else ""))
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=10), categoryorder="array", categoryarray=leaderboard["station"].tolist())
            st.plotly_chart(fig, use_container_width=True)

        # ───────────────────────── By Month ─────────────────────────
        elif group_by == "By Month":
            if "month" not in small.columns:
                st.info("Month column not available. Ensure `started_at` parsed in load_data.")
            else:
                by = ["month", "station"]
                if wx_col: by.append(wx_col)
                if mcol:   by.append(mcol)
                g = (small.groupby(by).size().rename("value").reset_index())

                # Share within each month (and weather/member group, if chosen)
                share_keys = ["month"]
                if wx_col: share_keys.append(wx_col)
                if mcol:   share_keys.append(mcol)
                g = _maybe_to_share(g, val_col="value", by_cols=share_keys)

                # If too many stations for a line chart, fallback to heatmap
                if len(keep) <= 10 and not wx_col and not mcol:
                    fig = px.line(
                        g, x="month", y="value", color="station", markers=True,
                        labels={"value": "Share (%)" if metric=="Share %" else "Rides", "month":"Month", "station":"Station"}
                    )
                    fig.update_layout(height=560, title="Monthly trend for top stations")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    mat = (g.pivot_table(index="station", columns="month", values="value", aggfunc="sum")
                             .loc[leaderboard["station"]]  # keep top order
                             .fillna(0))
                    fig = px.imshow(
                        mat, aspect="auto", origin="lower",
                        labels=dict(color=("Share (%)" if metric=="Share %" else "Rides"))
                    )
                    fig.update_layout(height=600, title="Monthly pattern — station × month")
                    st.plotly_chart(fig, use_container_width=True)

        # ───────────────────────── By Hour ─────────────────────────
        elif group_by == "By Hour":
            if "hour" not in small.columns:
                st.info("`hour` not available. Ensure `started_at` parsed in load_data.")
            else:
                by = ["hour", "station"]
                if wx_col: by.append(wx_col)
                if mcol:   by.append(mcol)
                g = (small.groupby(by).size().rename("value").reset_index())

                # Share within each hour (and weather/member group, if chosen)
                share_keys = ["hour"]
                if wx_col: share_keys.append(wx_col)
                if mcol:   share_keys.append(mcol)
                g = _maybe_to_share(g, val_col="value", by_cols=share_keys)

                mat = (g.pivot_table(index="station", columns="hour", values="value", aggfunc="sum")
                         .loc[leaderboard["station"]]
                         .reindex(columns=range(0,24))
                         .fillna(0))
                fig = px.imshow(
                    mat, aspect="auto", origin="lower",
                    labels=dict(color=("Share (%)" if metric=="Share %" else "Rides"))
                )
                fig.update_xaxes(title_text="Hour of day")
                fig.update_yaxes(title_text="Station")
                fig.update_layout(height=600, title="Hourly pattern — station × hour")
                st.plotly_chart(fig, use_container_width=True)

        # ───────────────────────── Map of Top Stations ─────────────────────────
        # Use start_lat/lng medians; scale radius by rides
        if {"start_lat","start_lng"}.issubset(df_f.columns):
            st.subheader("🗺️ Map — top stations sized by volume")
            import pydeck as pdk

            coords = (df_f.groupby("start_station_name")[["start_lat","start_lng"]]
                          .median().rename(columns={"start_lat":"lat","start_lng":"lon"}))
            geo = leaderboard.join(coords, on="station", how="left").dropna(subset=["lat","lon"]).copy()

            if len(geo):
                scale = st.slider("Bubble scale", 8, 40, 16)
                geo["radius"] = (60 + scale * np.sqrt(geo["rides"].clip(lower=1))).astype(float)
                geo["color"]  = [ [37,99,235,200] ] * len(geo)  # blue-ish

                view_state = pdk.ViewState(latitude=float(geo["lat"].median()),
                                           longitude=float(geo["lon"].median()),
                                           zoom=11, pitch=0)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=geo,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=True
                )
                deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                                map_style="mapbox://styles/mapbox/dark-v11",
                                tooltip={"text":"{station}\nRides: {rides}"})
                st.pydeck_chart(deck)
            else:
                st.info("No coordinates available for these stations.")

        # ───────────────────────── Station deep-dive ─────────────────────────
        st.subheader("🔎 Station deep-dive")
        picked = st.selectbox("Pick a station", leaderboard["station"].tolist())
        focus = df_f[df_f["start_station_name"].astype(str) == picked]
        cA, cB, cC = st.columns(3)

        # Hour profile
        with cA:
            if "hour" in focus.columns:
                gh = focus.groupby("hour").size().rename("rides").reset_index()
                figH = px.line(gh, x="hour", y="rides", markers=True,
                               labels={"hour":"Hour of day","rides":"Rides"})
                figH.update_layout(height=320, title="Hourly profile")
                st.plotly_chart(figH, use_container_width=True)

        # Weekday profile
        with cB:
            if "weekday" in focus.columns:
                gw = focus.groupby("weekday").size().rename("rides").reset_index()
                gw["weekday_name"] = gw["weekday"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
                figW = px.bar(gw, x="weekday_name", y="rides",
                              labels={"weekday_name":"Weekday","rides":"Rides"})
                figW.update_layout(height=320, title="Weekday profile")
                st.plotly_chart(figW, use_container_width=True)

        # Wet vs Dry impact (if available)
        with cC:
            if "wet_day" in focus.columns and focus["wet_day"].notna().any():
                gd = (focus.assign(day_type=lambda x: x["wet_day"].map({0:"Dry",1:"Wet"}))
                             .groupby("day_type").size().rename("rides").reset_index())
                figD = px.bar(gd, x="day_type", y="rides",
                              labels={"day_type":"Day type","rides":"Rides"})
                figD.update_layout(height=320, title="Wet vs Dry impact")
                st.plotly_chart(figD, use_container_width=True)

        # Download current leaderboard
        st.download_button(
            "Download leaderboard (CSV)",
            leaderboard.rename(columns={"rides":"rides_total"}).to_csv(index=False).encode("utf-8"),
            "top_stations_leaderboard.csv", "text/csv"
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

    if not {"started_at","hour","weekday"}.issubset(df_f.columns):
        st.info("Need `started_at` parsed into `hour` and `weekday` (done in load_data).")
    else:
        # ---- Controls ----
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            scale = st.selectbox("Scale", ["Absolute", "Row %", "Column %", "Z-score"], index=0,
                                 help="Row % shows distribution within a weekday; Column % shows distribution within an hour.")
        with c2:
            hour_bin = st.slider("Hour bin size", 1, 3, 1, help="Group hours into 1/2/3-hour buckets.")
        with c3:
            smooth = st.checkbox("Smooth across hours", value=False, help="Moving average per weekday (visual only).")
        with c4:
            wk_preset = st.selectbox("Preset", ["All days", "Weekdays only", "Weekend only"], index=0)
        with c5:
            member_mode = st.selectbox("Member view", ["All", "Member only", "Casual only", "Facet by Member Type"], index=0)

        # Apply day-of-week preset quickly (this stacks on your sidebar filter)
        subset = df_f.copy()
        if wk_preset == "Weekdays only":
            subset = subset[subset["weekday"].isin([0,1,2,3,4])]
        elif wk_preset == "Weekend only":
            subset = subset[subset["weekday"].isin([5,6])]

        # Member filter / facet
        facet = False
        if member_mode == "Member only" and "member_type" in subset.columns:
            subset = subset[subset["member_type"].astype(str) == "member"]
        elif member_mode == "Casual only" and "member_type" in subset.columns:
            subset = subset[subset["member_type"].astype(str) == "casual"]
        elif member_mode == "Facet by Member Type" and "member_type_display" in subset.columns:
            facet = True

        # Guard
        if subset.empty:
            st.info("No rows for current filters.")
            st.stop()

        # ---- Build main heatmap(s) ----
        def _render_heat(mat: pd.DataFrame, title: str):
            if mat.empty:
                st.info("Not enough data to render heatmap.")
                return
            # Smoothing (optional)
            if smooth:
                mat = _smooth_by_hour(mat, k=3)

            # Relabel Y-axis to weekday names
            mat_display = mat.copy()
            mat_display.index = _weekday_name(mat_display.index)

            # Imshow
            fig = px.imshow(
                mat_display,
                aspect="auto", origin="lower",
                labels=dict(x="Hour of day", y="Day of week", color=("Value" if scale=="Absolute" else scale)),
                text_auto=False, color_continuous_scale="Turbo" if scale=="Z-score" else "Viridis"
            )

            # Ticks: show real hour values (bin centers)
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(mat.columns))),
                ticktext=[f"{h:02d}:00" for h in mat.columns]
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(mat.index))),
                ticktext=mat_display.index.tolist()
            )
            fig.update_layout(height=600, title=title, margin=dict(l=20, r=20, t=50, b=50))

            # Hover with raw value and (if % scale) show formatted %
            hover = "<b>%{y}</b> @ <b>%{x}</b><br>Value: %{z}"
            if scale in ("Row %","Column %"):
                hover = "<b>%{y}</b> @ <b>%{x}</b><br>Share: %{z:.1f}%"
            fig.update_traces(hovertemplate=hover)

            # Peak annotation (only for Absolute / Row % to avoid odd Z-peak)
            if scale in ("Absolute","Row %"):
                fig = _add_peak_annotation(fig, mat)

            st.plotly_chart(fig, use_container_width=True)

        if facet:
            # Two panels: Member vs Casual
            cL, cR = st.columns(2)
            for label, col in [("Member 🧑‍💼", cL), ("Casual 🚲", cR)]:
                _sub = subset[subset["member_type"].astype(str).eq("member" if "Member" in label else "casual")]
                mat = _make_heat_grid(_sub, hour_bin=hour_bin, scale=scale)
                with col:
                    _render_heat(mat, f"Weekday × Hour — {label}")
        else:
            mat = _make_heat_grid(subset, hour_bin=hour_bin, scale=scale)
            _render_heat(mat, "Weekday × Hour — All riders")

        # ---- Marginal profiles (always useful) ----
        st.subheader("Marginal profiles")
        grid_all = _make_heat_grid(subset, hour_bin=hour_bin, scale="Absolute")
        if not grid_all.empty:
            # Hourly profile
            hourly = grid_all.sum(axis=0).rename("rides").reset_index().rename(columns={"index":"hour"})
            hourly["hour"] = hourly["hour"].astype(int)
            f1 = px.line(hourly, x="hour", y="rides", markers=True, labels={"hour":"Hour of day","rides":"Rides"})
            f1.update_layout(height=300, title="Hourly total rides")
            st.plotly_chart(f1, use_container_width=True)

            # Weekday profile
            weekday = grid_all.sum(axis=1).rename("rides").reset_index().rename(columns={0:"weekday"})
            weekday["weekday_name"] = _weekday_name(weekday["weekday"])
            f2 = px.bar(weekday, x="weekday_name", y="rides", labels={"weekday_name":"Weekday","rides":"Rides"})
            f2.update_layout(height=300, title="Weekday total rides")
            st.plotly_chart(f2, use_container_width=True)

        # ---- Wet vs Dry comparison (if available) ----
        if "wet_day" in subset.columns and subset["wet_day"].notna().any():
            st.subheader("Wet vs Dry comparison")
            cA, cB = st.columns(2)
            dry = subset[subset["wet_day"] == 0]
            wet = subset[subset["wet_day"] == 1]
            mat_dry = _make_heat_grid(dry, hour_bin=hour_bin, scale=scale)
            mat_wet = _make_heat_grid(wet, hour_bin=hour_bin, scale=scale)
            with cA: _render_heat(mat_dry, "Dry days")
            with cB: _render_heat(mat_wet, "Wet days")

        st.caption("Tips: try Row % to see *within-day* timing; Column % to see which days dominate each hour; Z-score to highlight anomalies.")

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
