# app/st_dashboard_Part_2.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from urllib.parse import quote, unquote
try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
except Exception:
    linkage = leaves_list = None
# Optional ML: linear simulator
try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None
    
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

def _one_hot(s, prefix):
    s = s.astype("int64")
    d = pd.get_dummies(s, prefix=prefix, drop_first=True)
    if d.shape[1] == 0:
        # Ensure at least one column for stability
        d[f"{prefix}_0"] = 0
    return d

def deweather_fit_predict(df_in: pd.DataFrame):
    """
    Fit a simple 'expected rides' model on rows where y is known, then predict for all rows.
    Returns: (yhat_all: pd.Series, resid_pct: pd.Series, coefs: pd.Series) or None if not enough data.
    """
    need = {"bike_rides_daily", "avg_temp_c"}
    if df_in is None or df_in.empty or not need.issubset(df_in.columns):
        return None

    df = df_in.copy()

    # ---------- y (train only on rows that have y & temp) ----------
    y = pd.to_numeric(df["bike_rides_daily"], errors="coerce")
    t = pd.to_numeric(df["avg_temp_c"], errors="coerce")
    train_mask = y.notna() & t.notna()
    if train_mask.sum() < 10:
        return None

    # ---------- feature builder (keeps columns consistent across train/predict) ----------
    def _build_X(frame: pd.DataFrame, wd_cols: list[str] | None = None):
        n = len(frame)
        parts = []
        names = []

        def add_col(arr, name):
            arr = pd.to_numeric(arr, errors="coerce").astype(float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(arr.reshape(-1, 1))
            names.append(name)

        # Intercept
        add_col(np.ones(n), "intercept")

        # Temp + quad
        tt = pd.to_numeric(frame["avg_temp_c"], errors="coerce")
        add_col(tt, "temp_c")
        add_col(tt**2, "temp_c_sq")

        # Optional weather covariates
        if "precip_mm" in frame.columns:
            add_col(frame["precip_mm"], "precip_mm")
        if "wind_kph" in frame.columns:
            add_col(frame["wind_kph"], "wind_kph")
        if "wet_day" in frame.columns:
            # 0/1 as float
            add_col(frame["wet_day"], "wet_day")

        # Weekday dummies (no drop-first; least squares handles redundancy)
        if "date" in frame.columns and pd.api.types.is_datetime64_any_dtype(frame["date"]):
            wd = frame["date"].dt.weekday.astype("Int64").fillna(0).astype(int)
            W = pd.get_dummies(wd, prefix="wd").astype(float)
            if wd_cols is not None:
                # ensure same dummy columns at predict time
                W = W.reindex(columns=wd_cols, fill_value=0.0)
            wd_cols_out = list(W.columns)
            if len(wd_cols_out):
                parts.append(W.to_numpy(dtype=float))
                names.extend(wd_cols_out)
        else:
            wd_cols_out = []

        X = np.hstack(parts).astype(float)
        # clean any remaining bad values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, names, wd_cols_out

    # --- Build train design matrix (captures weekday cols) ---
    X_train, names, wd_cols = _build_X(df.loc[train_mask], wd_cols=None)
    y_train = y.loc[train_mask].to_numpy(dtype=float)
    good = np.isfinite(y_train).flatten() & np.isfinite(X_train).all(axis=1)
    if good.sum() < 10:
        return None
    X_train = X_train[good]
    y_train = y_train[good]

    # --- Fit (robust to singularity) ---
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    coefs = pd.Series(beta, index=names)

    # --- Predict for ALL rows with the same columns ---
    X_all, _, _ = _build_X(df, wd_cols=wd_cols)
    yhat_all = pd.Series(X_all @ beta, index=df.index)

    # --- Residuals (% vs expected) only where y is known ---
    resid = pd.Series(np.nan, index=df.index, dtype=float)
    resid.loc[train_mask] = y.loc[train_mask] - yhat_all.loc[train_mask]
    denom = yhat_all.copy()
    denom.replace(0, np.nan, inplace=True)
    resid_pct = 100.0 * (resid / denom)

    return yhat_all, resid_pct, coefs

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
            border-radius: 24px;
            padding: 22px 24px;
            box-shadow: 0 8px 18px rgba(0,0,0,0.28);
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
            border-radius: 24px;
            padding: 16px 18px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.28);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            min-height: 160px;
            display: flex; flex-direction: column; justify-content: space-between;
        }
        .kpi-title {
            font-size: .95rem;
            color: #cbd5e1;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            margin-bottom: 6px;
            letter-spacing: .2px;
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

# --- quick presets row
col_p1, col_p2 = st.sidebar.columns([1,1])
with col_p1:
    if st.button("✨ Commuter preset"):
        # Weekdays 6–10 & 16–20, mild temps, members
        st.session_state["page_select"] = "Weekday × Hour Heatmap"
        if "weekday" in df.columns:
            weekdays = ["Mon","Tue","Wed","Thu","Fri"]; st.query_params.update(weekday=",".join(weekdays))
        if "hour" in df.columns: st.query_params.update(hour0="6", hour1="20")
        if "avg_temp_c" in df.columns:
            tmin, tmax = float(np.nanmin(df["avg_temp_c"])), float(np.nanmax(df["avg_temp_c"]))
            st.query_params.update(temp=f"{max(tmin,5)}:{min(tmax,25)}")
        if "member_type" in df.columns: st.query_params.update(usertype="member")

with col_p2:
    if st.button("🌧️ Rainy-day preset"):
        st.session_state["page_select"] = "Weather vs Bike Usage"
        if "wet_day" in df.columns: st.query_params.update(wet="1")

# --- reset / share
r1, r2 = st.sidebar.columns([1,1])
with r1:
    if st.button("♻️ Reset all"):
        st.cache_data.clear()
        if hasattr(st, "query_params"): st.query_params.clear()
        st.rerun()
with r2:
    if st.button("🔗 Copy current link"):
        st.sidebar.code(st.experimental_get_query_params() if not hasattr(st,"query_params") else dict(st.query_params))
        st.caption("The current state is in the URL — copy from your browser bar.")

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

# --- Time filters (kept visible) ---
hour_range = None
if "hour" in df.columns:
    hour_range = st.sidebar.slider("Hour of day", 0, 23, (6, 22), key="hour_slider")

# --- Collapsed: less-used filters ---
temp_range, weekdays = None, None
with st.sidebar.expander("More filters", expanded=False):
    # Temperature
    if "avg_temp_c" in df.columns:
        tmin = float(np.nanmin(df["avg_temp_c"]))
        tmax = float(np.nanmax(df["avg_temp_c"]))
        temp_range = st.slider("Temperature (°C)", tmin, tmax, (tmin, tmax), key="temp_slider")

    # Weekdays
    if "weekday" in df.columns:
        weekday_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekdays = st.multiselect("Weekday(s)", weekday_names, default=weekday_names, key="weekday_multi")

st.sidebar.markdown("---")

# ── URL state: read (on load) and write (after filters) ──
def _qp_get():
    if hasattr(st, "query_params"):  # Streamlit ≥1.31
        return dict(st.query_params)
    return st.experimental_get_query_params()

def _qp_set(**kv):
    try:
        if hasattr(st, "query_params"):
            st.query_params.update({k: str(v) for k, v in kv.items() if v is not None})
        else:
            st.experimental_set_query_params(**{k: str(v) for k, v in kv.items() if v is not None})
    except Exception:
        pass

PAGES = [
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
]

# Seed default page from URL (if present), otherwise first page
_qp = _qp_get()
_qp_page = None
if "page" in _qp:
    _qp_page = _qp["page"][0] if isinstance(_qp["page"], list) else _qp["page"]
if _qp_page not in PAGES:
    _qp_page = PAGES[0]

# The widget value drives the app; we do NOT override it afterwards
page = st.sidebar.selectbox(
    "📑 Analysis page",
    PAGES,
    index=PAGES.index(_qp_page),
    key="page_select",
)

# After filters chosen → write them to URL (safe)
try:
    _qp_set(
        page=page,
        date0=str(date_range[0]) if date_range else None,
        date1=str(date_range[1]) if date_range else None,
        usertype=usertype or "All",
        hour0=hour_range[0] if hour_range else None,
        hour1=hour_range[1] if hour_range else None
    )
except Exception:
    pass

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

st.sidebar.success(f"✅ {len(df_f):,} trips match")

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

    # Selection summary (everything below reflects this slice)
    st.caption(
        f"**Selection:** {date_range[0]} → {date_range[1]} · "
        f"{'All seasons' if seasons is None or set(seasons)==set(['Winter','Spring','Summer','Autumn']) else ', '.join(seasons)} · "
        f"{'All users' if (usertype in (None,'All')) else usertype.title()} · "
        f"{'All day' if hour_range is None else f'{hour_range[0]:02d}:00–{hour_range[1]:02d}:00'}"
    )

    show_cover(cover_path)
    st.caption("⚙️ Powered by NYC Citi Bike data • 365 days • Interactive visuals")

    # --- Core KPIs (robust/defensible) ---
    KPIs = compute_core_kpis(df_f, daily_f)  # total_rides, avg_day, corr_tr

    # Weather uplift: comfy (15–25°C) vs extreme (<5 or >30°C)
    weather_uplift_pct = None
    if daily_f is not None and not daily_f.empty and "avg_temp_c" in daily_f.columns:
        d_nonnull = daily_f.dropna(subset=["avg_temp_c", "bike_rides_daily"]).copy()
        if not d_nonnull.empty:
            comfy   = d_nonnull.loc[d_nonnull["avg_temp_c"].between(15, 25, inclusive="both"), "bike_rides_daily"].mean()
            extreme = d_nonnull.loc[~d_nonnull["avg_temp_c"].between(5, 30, inclusive="both"), "bike_rides_daily"].mean()
            if pd.notnull(comfy) and pd.notnull(extreme) and extreme not in (0, np.nan):
                weather_uplift_pct = (comfy - extreme) / extreme * 100.0
    weather_str = f"{weather_uplift_pct:+.0f}%" if weather_uplift_pct is not None else "—"

    # Coverage: % of selected days with usable weather (defensibility for weather KPIs)
    coverage = "—"
    if daily_f is not None and not daily_f.empty:
        if "avg_temp_c" in daily_f.columns:
            cov = 100.0 * daily_f["avg_temp_c"].notna().mean()
            coverage = f"{cov:.0f}%"
        else:
            coverage = "0%"

    # Peak Season (optional text if you want to keep it handy)
    peak_value, peak_sub = "—", ""
    if "season" in df_f.columns and daily_f is not None and not daily_f.empty:
        tmp = daily_f.copy()
        if "season" not in tmp.columns:
            s_map = (
                df_f.groupby("date")["season"]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
                    .reset_index()
            )
            tmp = tmp.merge(s_map, on="date", how="left")
        if "season" in tmp.columns and "bike_rides_daily" in tmp.columns:
            m = tmp.groupby("season")["bike_rides_daily"].mean().sort_values(ascending=False)
            if len(m):
                peak_value = f"{m.index[0]}"
                peak_sub   = f"{kfmt(m.iloc[0])} avg trips"

    # --- KPI cards ---
    cA, cB, cC, cD, cE = st.columns(5)
    with cA:
        kpi_card("Total Trips", kfmt(KPIs.get("total_rides", 0)), "Across all stations", "🧮")
    with cB:
        kpi_card("Daily Average", kfmt(KPIs["avg_day"]) if KPIs.get("avg_day") is not None else "—",
                 "Trips per day (selection)", "📅")
    with cC:
        kpi_card("Temp ↔ Rides (r)",
                 f"{KPIs['corr_tr']:+.3f}" if KPIs.get("corr_tr") is not None else "—",
                 "Pearson on daily agg", "🌡️")
    with cD:
        kpi_card("Weather Uplift", weather_str, "15–25°C vs extreme", "🌦️")
    with cE:
        # Prefer Coverage for credibility; swap to Peak Season if you want
        kpi_card("Coverage", coverage, "Weather data availability", "🧩")
        # kpi_card("Peak Season", peak_value, peak_sub, "🏆")  # ← alternative

    # --- Mini trend strip (adds motion right on the Intro) ---
    if daily_f is not None and not daily_f.empty and "avg_temp_c" in daily_f.columns:
        d = daily_f.sort_values("date").copy()
        n = 14  # gentle smoothing for both lines
        for col in ["bike_rides_daily", "avg_temp_c"]:
            d[f"{col}_roll"] = d[col].rolling(n, min_periods=max(2, n // 2), center=True).mean()

        fig_intro = make_subplots(specs=[[{"secondary_y": True}]])
        fig_intro.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["bike_rides_daily_roll"].fillna(d["bike_rides_daily"]),
                name="Daily rides",
                mode="lines",
                line=dict(color=RIDES_COLOR, width=2),
            ),
            secondary_y=False,
        )
        fig_intro.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["avg_temp_c_roll"].fillna(d["avg_temp_c"]),
                name="Avg temp (°C)",
                mode="lines",
                line=dict(color=TEMP_COLOR, width=2, dash="dot"),
                opacity=0.9,
            ),
            secondary_y=True,
        )
        fig_intro.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=0),
            hovermode="x unified",
            showlegend=True,
            title="Trend overview (14-day smoother)"
        )
        fig_intro.update_yaxes(title_text="Rides", secondary_y=False)
        fig_intro.update_yaxes(title_text="Temp (°C)", secondary_y=True)
        st.plotly_chart(fig_intro, use_container_width=True)

    # --- Sharpened “What you’ll find here” copy ---
    st.markdown("### What you’ll find here")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Decision-ready KPIs**\n\nTotals, avg/day, and a defensible temp↔rides correlation.")
    c2.info("**Weather impact**\n\nTrend, scatter with fit, and comfort bands for clear takeaways.")
    c3.info("**Station intelligence**\n\nTop stations, OD flows (Sankey/Matrix), and Pareto focus.")
    c4.info("**Time patterns**\n\nWeekday×Hour heatmap + marginal profiles for staffing windows.")
    st.caption("Use the sidebar filters; every view updates live.")

elif page == "Weather vs Bike Usage":
    st.header("🌤️ Daily bike rides vs weather")

    if daily_f is None or daily_f.empty:
        st.warning("Daily metrics aren’t available. Provide trip rows with `date` to aggregate.")
    else:
        # ---------- Prep once ----------
        d = daily_f.sort_values("date").copy()

        # Coverage ribbon (trust signal)
        cov = d["avg_temp_c"].notna().mean()*100 if "avg_temp_c" in d.columns else 0
        st.caption(f"Weather coverage in selection: **{cov:.0f}%** — metrics account for missing values.")

        # ---------- Tabs ----------
        tab_trend, tab_scatter, tab_dist, tab_lab, tab_resid = st.tabs(
            ["📈 Trend", "🔬 Scatter", "📦 Distributions", "🧪 Lab", "📉 De-weathered Index"]
        )

        # ======== Trend tab ========
        with tab_trend:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                roll_win = st.selectbox("Rolling window", ["Off","7d","14d","30d"], index=1)
            with c2:
                show_precip = st.checkbox("Show precipitation bars (mm)", value=("precip_mm" in d.columns))
            with c3:
                show_wind = st.checkbox("Show wind line (kph)", value=("wind_kph" in d.columns))
            with c4:
                st.caption("Use other tabs for residuals & elasticity")

            # Rolling smoother
            if roll_win != "Off":
                n = int(roll_win.replace("d", ""))
                for col in ["bike_rides_daily", "avg_temp_c", "wind_kph"]:
                    if col in d.columns:
                        d[f"{col}_roll"] = d[col].rolling(n, min_periods=max(2, n // 2), center=True).mean()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            # rides
            y_rides = d.get("bike_rides_daily_roll", d["bike_rides_daily"]) if roll_win != "Off" else d["bike_rides_daily"]
            fig.add_trace(go.Scatter(x=d["date"], y=y_rides, mode="lines", name="Daily bike rides",
                                     line=dict(color=RIDES_COLOR, width=2)), secondary_y=False)

            # temp
            if "avg_temp_c" in d.columns and d["avg_temp_c"].notna().any():
                y_temp = d.get("avg_temp_c_roll", d["avg_temp_c"]) if roll_win != "Off" else d["avg_temp_c"]
                fig.add_trace(go.Scatter(x=d["date"], y=y_temp, mode="lines",
                                         name="Average temperature (°C)",
                                         line=dict(color=TEMP_COLOR, width=2, dash="dot")), secondary_y=True)

            # wind
            if show_wind and "wind_kph" in d.columns and d["wind_kph"].notna().any():
                y_wind = d.get("wind_kph_roll", d["wind_kph"]) if roll_win != "Off" else d["wind_kph"]
                fig.add_trace(go.Scatter(x=d["date"], y=y_wind, mode="lines", name="Avg wind (kph)",
                                         line=dict(width=1), opacity=0.5), secondary_y=True)

            # precip
            if show_precip and "precip_mm" in d.columns and d["precip_mm"].notna().any():
                fig.add_trace(go.Bar(x=d["date"], y=d["precip_mm"], name="Precipitation (mm)",
                                     marker_color="rgba(100,100,120,0.35)", opacity=0.4), secondary_y=False)

            # Gentle comfort band cue behind rides (visual only)
            if len(d):
                fig.add_hrect(y0=float(d["bike_rides_daily"].min()), y1=float(d["bike_rides_daily"].max()),
                              line_width=0, fillcolor="rgba(34,197,94,0.05)", layer="below")

            fig.update_layout(hovermode="x unified", barmode="overlay", height=560)
            fig.update_yaxes(title_text="Bike rides (count)", secondary_y=False)
            y2_title = "Temperature (°C)" + (" + Wind (kph)" if show_wind and "wind_kph" in d.columns else "")
            fig.update_yaxes(title_text=y2_title, secondary_y=True)
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(title="Daily rides vs temperature, precipitation, and wind — NYC (2022)")
            st.plotly_chart(fig, use_container_width=True)

        # ======== Scatter tab ========
        with tab_scatter:
            c1, c2 = st.columns(2)
            with c1:
                color_scatter_by = st.selectbox("Color points by", ["None","wet_day","precip_bin","wind_bin"], index=1)
            with c2:
                split_wknd = st.checkbox("Show weekday vs weekend fits", value=True, help="Highlights commute vs leisure sensitivity")

            # choose temp column
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
                    chosen = None if color_scatter_by == "None" else color_scatter_by
                    color_arg = chosen if (chosen in scatter_df.columns) else None
                    labels = {
                        temp_col: temp_col.replace("_", " ").title().replace("C", "(°C)"),
                        "bike_rides_daily": "Bike rides (count)",
                        "wet_day": "Wet day",
                        "precip_bin": "Precipitation",
                        "wind_bin": "Wind",
                    }
                    fig2 = px.scatter(scatter_df, x=temp_col, y="bike_rides_daily", color=color_arg,
                                      labels=labels, opacity=0.85, trendline="ols")
                    fig2.update_layout(height=520, title="Rides vs Temperature")
                    st.plotly_chart(fig2, use_container_width=True)

                # Elasticity & rain penalty KPIs (quadratic fit)
                df_fit = d.dropna(subset=["avg_temp_c","bike_rides_daily"]) if "avg_temp_c" in d.columns else pd.DataFrame()
                if len(df_fit) >= 20:
                    Xq = np.c_[np.ones(len(df_fit)), df_fit["avg_temp_c"], df_fit["avg_temp_c"]**2]
                    a,b,c = np.linalg.lstsq(Xq, df_fit["bike_rides_daily"], rcond=None)[0]
                    t0 = 20.0
                    rides_t0 = a + b*t0 + c*t0*t0
                    slope_t0 = b + 2*c*t0
                    elasticity_pct = (slope_t0 / rides_t0) * 100 if rides_t0>0 else np.nan
                    rain_pen = None
                    if "wet_day" in d.columns and d["wet_day"].notna().any():
                        dry = d.loc[d["wet_day"]==0, "bike_rides_daily"].mean()
                        wet = d.loc[d["wet_day"]==1, "bike_rides_daily"].mean()
                        if pd.notnull(dry) and dry>0: rain_pen = (wet-dry)/dry*100
                    k1, k2 = st.columns(2)
                    with k1: st.metric("Temp elasticity @20°C", f"{elasticity_pct:+.1f}% / °C")
                    with k2: st.metric("Rain penalty (wet vs dry)", f"{rain_pen:+.0f}%" if rain_pen is not None else "—")

                # Weekday vs Weekend interaction fits
                if split_wknd and {"date","bike_rides_daily",temp_col}.issubset(d.columns):
                    dd = d.dropna(subset=[temp_col,"bike_rides_daily"]).copy()
                    dd["is_weekend"] = dd["date"].dt.weekday.isin([5,6]).map({True:"Weekend", False:"Weekday"})
                    figsw = px.scatter(dd, x=temp_col, y="bike_rides_daily", color="is_weekend", opacity=0.85, trendline="ols",
                                       labels={temp_col: "Avg temp (°C)", "bike_rides_daily": "Bike rides", "is_weekend":"Day type"})
                    figsw.update_layout(height=480, title="Rides vs Temp — Weekday vs Weekend")
                    st.plotly_chart(figsw, use_container_width=True)

        # ======== Distributions tab ========
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

        # ======== Lab tab (simulator + comfort point + backtest + forecast upload) ========
        with tab_lab:
            st.subheader("🔮 Quick ride simulator & comfort point")

            need_cols = {"bike_rides_daily","avg_temp_c"}
            if need_cols.issubset(d.columns) and len(d.dropna(subset=list(need_cols))) >= 10:
                d_fit = d.dropna(subset=["avg_temp_c","bike_rides_daily"]).copy()

                # Linear simulator (fast, intuitive)
                if LinearRegression is not None and len(d_fit) >= 3:
                    model = LinearRegression().fit(
                        d_fit[["avg_temp_c"]].to_numpy(), d_fit["bike_rides_daily"].to_numpy()
                    )
                    t_sel = st.slider(
                        "Forecast avg temp (°C)",
                        float(d_fit["avg_temp_c"].min()), float(d_fit["avg_temp_c"].max()),
                        20.0, 0.5
                    )
                    pred = int(model.predict([[t_sel]])[0])
                    st.metric("Expected rides (linear)", f"{pred:,}")
                else:
                    st.info("Install scikit-learn to enable the linear simulator.")

                # Quadratic “comfort” curve + optimum
                X = np.c_[np.ones(len(d_fit)), d_fit["avg_temp_c"], d_fit["avg_temp_c"]**2]
                try:
                    beta = np.linalg.lstsq(X, d_fit["bike_rides_daily"], rcond=None)[0]
                    a, b, c = map(float, beta)   # rides ≈ a + b*T + c*T^2
                    t_opt = (-b / (2*c)) if (c not in (0, np.nan) and np.isfinite(c)) else np.nan

                    if np.isfinite(t_opt):
                        st.success(f"Estimated **optimal temperature** for demand: **{t_opt:.1f}°C** (quadratic fit)")

                    # Mini visualization
                    t_grid = np.linspace(d_fit["avg_temp_c"].min(), d_fit["avg_temp_c"].max(), 100)
                    y_hat  = a + b*t_grid + c*(t_grid**2)
                    figq = go.Figure()
                    figq.add_trace(go.Scatter(x=d_fit["avg_temp_c"], y=d_fit["bike_rides_daily"],
                                              mode="markers", name="Observed", opacity=0.5))
                    figq.add_trace(go.Scatter(x=t_grid, y=y_hat, mode="lines", name="Quadratic fit"))
                    if np.isfinite(t_opt):
                        figq.add_vline(x=t_opt, line_dash="dot")
                    figq.update_layout(height=380, title="Comfort curve (rides vs temperature)")
                    figq.update_xaxes(title="Avg temp (°C)")
                    figq.update_yaxes(title="Bike rides per day")
                    st.plotly_chart(figq, use_container_width=True)
                except Exception:
                    st.info("Quadratic fit not available for this selection.")
            else:
                st.caption("Need daily rides + avg_temp_c for the simulator.")

            # Backtest (credibility)
            with st.expander("📏 Backtest (train/test)"):
                if len(d) >= 90 and "avg_temp_c" in d.columns and d["avg_temp_c"].notna().sum() >= 60:
                    d2 = d.dropna(subset=["avg_temp_c","bike_rides_daily"]).sort_values("date").copy()
                    cut = int(len(d2)*0.75)
                    tr, te = d2.iloc[:cut], d2.iloc[cut:]

                    # Baseline: rides ~ temp (linear)
                    Xtr = np.c_[np.ones(len(tr)), tr["avg_temp_c"]]
                    Xte = np.c_[np.ones(len(te)), te["avg_temp_c"]]
                    b_lin, *_ = np.linalg.lstsq(Xtr, tr["bike_rides_daily"], rcond=None)
                    pred_lin = Xte @ b_lin

                    # Enriched: quadratic + rain + wind + weekday/month dummies (via helper)
                    yhat_tr = None
                    yhat_te = None
                    out_tr = deweather_fit_predict(tr)
                    out_te = deweather_fit_predict(te)
                    if out_tr is not None: yhat_tr = out_tr[0]
                    if out_te is not None: yhat_te = out_te[0]

                    def MAE(y, yhat): 
                        y = np.asarray(y)
                        yhat = np.asarray(yhat)
                        m = np.isfinite(y) & np.isfinite(yhat)
                        return float(np.mean(np.abs(y[m] - yhat[m]))) if m.any() else np.nan

                    mae_lin = MAE(te["bike_rides_daily"].values, pred_lin)
                    mae_enr = MAE(te["bike_rides_daily"].values, yhat_te.values if yhat_te is not None else np.full(len(te), np.nan))

                    cbt1, cbt2 = st.columns(2)
                    with cbt1: st.metric("MAE — Linear(temp)", f"{mae_lin:,.0f}" if np.isfinite(mae_lin) else "—")
                    with cbt2: st.metric("MAE — Enriched", f"{mae_enr:,.0f}" if np.isfinite(mae_enr) else "—")
                    if np.isfinite(mae_lin) and np.isfinite(mae_enr):
                        st.caption(f"ΔMAE: {(mae_lin - mae_enr):,.0f} better with enriched model.")
                else:
                    st.info("Need ≥90 days with temperature to backtest.")

            # Bring-your-own forecast (product hook)
            st.markdown("#### 🔮 Bring your own forecast (CSV)")
            up = st.file_uploader("Upload 7–14 day forecast CSV with columns: date, avg_temp_c, precip_mm, wind_kph", type=["csv"])
            if up is not None:
                try:
                    df_fc = pd.read_csv(up, parse_dates=["date"])
                    df_fc = df_fc.reindex(columns=["date","avg_temp_c","precip_mm","wind_kph"])
                    # Concatenate to share design columns with helper
                    d3 = d.dropna(subset=["avg_temp_c","bike_rides_daily"]).copy()
                    tmp = pd.concat(
                        [d3[["date","avg_temp_c","precip_mm","wind_kph","bike_rides_daily"]],
                         df_fc.assign(bike_rides_daily=np.nan)],
                        ignore_index=True
                    )
                    yhat_tmp, _, _ = deweather_fit_predict(tmp)
                    mask_fc = tmp["bike_rides_daily"].isna()
                    fc = tmp.loc[mask_fc, ["date"]].copy()
                    fc["pred_rides"] = yhat_tmp.loc[mask_fc].values if yhat_tmp is not None else np.nan
                    figf = px.bar(fc, x="date", y="pred_rides", labels={"pred_rides":"Predicted rides","date":"Date"})
                    figf.update_layout(height=360, title="Forecast rides (uploaded weather)")
                    st.plotly_chart(figf, use_container_width=True)
                    st.dataframe(fc.assign(pred_rides=fc["pred_rides"].round(0).astype("Int64")))
                except Exception as e:
                    st.info(f"Could not parse forecast: {e}")

        # ======== De-weathered Index tab ========
        with tab_resid:
            out = deweather_fit_predict(d)
            if out is None:
                st.info("Need daily rides + avg_temp_c for de-weathering.")
            else:
                yhat, resid_pct, coefs = out
                dd = pd.DataFrame({"date": d["date"], "resid_pct": resid_pct, "expected": yhat}).dropna()
                figr = px.line(dd, x="date", y="resid_pct",
                               labels={"resid_pct":"Residual vs expected (%)", "date": "Date"})
                figr.add_hline(y=0, line_dash="dot")
                figr.update_layout(height=420, title="De-weathered demand index (over/under performance)")
                st.plotly_chart(figr, use_container_width=True)
                st.metric("Avg residual (last 30 days)", f"{dd.tail(30)['resid_pct'].mean():+.1f}%")
                with st.expander("Model drivers (top |β|)"):
                    st.write(coefs.sort_values(key=np.abs, ascending=False).head(10).round(3))

elif page == "Trip Metrics (Duration • Distance • Speed)":
    st.header("🚴 Trip metrics")

    need = {"duration_min","distance_km","speed_kmh"}
    if not need.issubset(df_f.columns):
        st.info("Need duration, distance, and speed (engineered in load_data).")
    else:
        # ── Controls
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            robust = st.checkbox("Robust clipping (99.5%)", value=True, help="Hide extreme outliers that crush the axes.")
        with c2:
            log_duration = st.checkbox("Log X: Duration", value=False)
        with c3:
            log_distance = st.checkbox("Log X: Distance", value=False)
        with c4:
            log_speed = st.checkbox("Log X: Speed", value=False)

        # ── Inlier masks + physical bounds
        m_dur = (inlier_mask(df_f, "duration_min", hi=0.995) if robust else pd.Series(True, index=df_f.index)) & \
                df_f["duration_min"].between(0.5, 240, inclusive="both")
        m_dst = (inlier_mask(df_f, "distance_km", hi=0.995) if robust else pd.Series(True, index=df_f.index)) & \
                df_f["distance_km"].between(0.01, 30, inclusive="both")
        m_spd = (inlier_mask(df_f, "speed_kmh", hi=0.995) if robust else pd.Series(True, index=df_f.index)) & \
                df_f["speed_kmh"].between(0.5, 60, inclusive="both")

        clipped_dur = int((~m_dur).sum()); clipped_dst = int((~m_dst).sum()); clipped_spd = int((~m_spd).sum())

        # ===== Histograms (robust) =====
        cA, cB, cC = st.columns(3)
        with cA:
            d = df_f.loc[m_dur, "duration_min"]
            ql, qh = d.quantile([0.01, 0.995]).tolist()
            fig = px.histogram(
                d, x="duration_min", nbins=60, labels={"duration_min":"Duration (min)"},
                log_x=log_duration, range_x=[ql, qh] if robust and not log_duration else None
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Clipped rows (duration): {clipped_dur:,}")

        with cB:
            d = df_f.loc[m_dst, "distance_km"]
            ql, qh = d.quantile([0.01, 0.995]).tolist()
            fig = px.histogram(
                d, x="distance_km", nbins=60, labels={"distance_km":"Distance (km)"},
                log_x=log_distance, range_x=[ql, qh] if robust and not log_distance else None
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Clipped rows (distance): {clipped_dst:,}")

        with cC:
            d = df_f.loc[m_spd, "speed_kmh"]
            ql, qh = d.quantile([0.01, 0.995]).tolist()
            fig = px.histogram(
                d, x="speed_kmh", nbins=60, labels={"speed_kmh":"Speed (km/h)"},
                log_x=log_speed, range_x=[ql, qh] if robust and not log_speed else None
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Clipped rows (speed): {clipped_spd:,}")

        # ===== Distance vs duration — operating envelope (JSON-safe Scattergl) =====
        st.subheader("Distance vs duration — feasibility & operating envelope")

        inliers_mask_all = m_dst & m_dur & m_spd
        cols_needed = ["distance_km", "duration_min", "speed_kmh"]
        inliers = df_f.loc[inliers_mask_all, cols_needed].copy()

        # Sanitize numerics
        for cnum in cols_needed:
            inliers[cnum] = pd.to_numeric(inliers[cnum], errors="coerce")
        inliers.replace([np.inf, -np.inf], np.nan, inplace=True)
        inliers.dropna(subset=cols_needed, inplace=True)

        nmax = 35000
        if len(inliers) > nmax:
            inliers = inliers.sample(n=nmax, random_state=13)

        # Plain Python lists → avoids JSON dtype traps
        x_vals = inliers["distance_km"].astype(float).tolist()
        y_vals = inliers["duration_min"].astype(float).tolist()
        c_vals = inliers["speed_kmh"].astype(float).tolist()

        fig2 = go.Figure()

        # Main cloud (WebGL)
        fig2.add_trace(
            go.Scattergl(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name="Trips",
                marker=dict(
                    size=6,
                    opacity=0.85,
                    color=c_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Speed (km/h)")
                ),
                hovertemplate=(
                    "Distance: %{x:.2f} km<br>"
                    "Duration: %{y:.1f} min<br>"
                    "Speed: %{marker.color:.1f} km/h"
                    "<extra></extra>"
                ),
            )
        )

        # Faint outliers (fully sanitized too)
        outliers = df_f.loc[~inliers_mask_all, ["distance_km", "duration_min"]].copy()
        for cnum in ["distance_km", "duration_min"]:
            outliers[cnum] = pd.to_numeric(outliers[cnum], errors="coerce")
        outliers.replace([np.inf, -np.inf], np.nan, inplace=True)
        outliers.dropna(subset=["distance_km", "duration_min"], inplace=True)
        if len(outliers):
            fig2.add_trace(
                go.Scattergl(
                    x=outliers["distance_km"].astype(float).tolist(),
                    y=outliers["duration_min"].astype(float).tolist(),
                    mode="markers",
                    name="Outliers",
                    marker=dict(size=5),
                    opacity=0.12,
                    hoverinfo="skip",
                )
            )

        # Feasibility guides: constant-speed lines (10/20/30 km/h)
        if len(inliers):
            x_min = max(0.01, float(np.nanmin(inliers["distance_km"])))
            x_max = max(x_min + 0.5, float(np.nanmax(inliers["distance_km"])))
            xs = np.linspace(x_min, x_max, 200).astype(float).tolist()
            for v in [10.0, 20.0, 30.0]:
                ys = [(x / v) * 60.0 for x in xs]
                fig2.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines", name=f"{int(v)} km/h guide",
                        line=dict(dash="dot", width=1), hoverinfo="skip",
                    )
                )
            # Tight axes (robust quantiles)
            xql, xqh = inliers["distance_km"].quantile([0.01, 0.995]).tolist()
            yql, yqh = inliers["duration_min"].quantile([0.01, 0.995]).tolist()
            if np.isfinite(xql) and np.isfinite(xqh) and xql < xqh:
                fig2.update_xaxes(range=[float(xql), float(xqh)])
            if np.isfinite(yql) and np.isfinite(yqh) and yql < yqh:
                fig2.update_yaxes(range=[float(yql), float(yqh)])

        fig2.update_layout(
            height=560,
            title="Trip operating envelope",
            xaxis_title="Distance (km)",
            yaxis_title="Duration (min)",
            margin=dict(l=20, r=20, t=60, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ===== Weather relationships =====
        st.subheader("Weather relationships")
        c1, c2 = st.columns(2)
        temp_ok = "avg_temp_c" in df_f.columns and df_f["avg_temp_c"].notna().any()
        wind_ok = "wind_kph" in df_f.columns and df_f["wind_kph"].notna().any()

        # Speed vs temperature
        with c1:
            if temp_ok:
                dat = df_f[m_spd & df_f["avg_temp_c"].notna()]
                nmax = 30000
                if len(dat) > nmax:
                    dat = dat.sample(n=nmax, random_state=4)
                figt = px.scatter(
                    dat, x="avg_temp_c", y="speed_kmh", opacity=0.7,
                    labels={"avg_temp_c":"Avg temperature (°C)", "speed_kmh":"Speed (km/h)"}
                )
                # Add fit
                x_ = pd.to_numeric(dat["avg_temp_c"], errors="coerce")
                y_ = pd.to_numeric(dat["speed_kmh"], errors="coerce")
                ok = x_.notna() & y_.notna()
                if ok.sum() >= 3 and x_[ok].nunique() >= 2:
                    a, b = np.polyfit(x_[ok], y_[ok], 1)
                    xs = np.linspace(x_[ok].min(), x_[ok].max(), 100)
                    ys = a * xs + b
                    figt.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Linear fit", line=dict(dash="dash")))
                figt.update_layout(height=480, title="Speed vs Temperature")
                st.plotly_chart(figt, use_container_width=True)
            else:
                st.info("No temperature column available for this view.")

        # Speed vs wind
        with c2:
            if wind_ok:
                dat = df_f[m_spd & df_f["wind_kph"].notna()]
                nmax = 30000
                if len(dat) > nmax:
                    dat = dat.sample(n=nmax, random_state=5)
                figw = px.scatter(
                    dat, x="wind_kph", y="speed_kmh", opacity=0.7,
                    labels={"wind_kph":"Wind (kph)", "speed_kmh":"Speed (km/h)"}
                )
                x_ = pd.to_numeric(dat["wind_kph"], errors="coerce")
                y_ = pd.to_numeric(dat["speed_kmh"], errors="coerce")
                ok = x_.notna() & y_.notna()
                if ok.sum() >= 3 and x_[ok].nunique() >= 2:
                    a, b = np.polyfit(x_[ok], y_[ok], 1)
                    xs = np.linspace(x_[ok].min(), x_[ok].max(), 100)
                    ys = a * xs + b
                    figw.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Linear fit", line=dict(dash="dash")))
                figw.update_layout(height=480, title="Speed vs Wind")
                st.plotly_chart(figw, use_container_width=True)
            else:
                st.info("No wind column available for this view.")

        # Distance/Duration vs Temperature (comfort story)
        c3, c4 = st.columns(2)
        with c3:
            if temp_ok:
                dat = df_f[m_dur & df_f["avg_temp_c"].notna()]
                nmax = 30000
                if len(dat) > nmax:
                    dat = dat.sample(n=nmax, random_state=6)
                figdt = px.scatter(
                    dat, x="avg_temp_c", y="duration_min", opacity=0.6,
                    labels={"avg_temp_c":"Avg temperature (°C)", "duration_min":"Duration (min)"}
                )
                x_ = pd.to_numeric(dat["avg_temp_c"], errors="coerce")
                y_ = pd.to_numeric(dat["duration_min"], errors="coerce")
                ok = x_.notna() & y_.notna()
                if ok.sum() >= 3 and x_[ok].nunique() >= 2:
                    a, b = np.polyfit(x_[ok], y_[ok], 1)
                    xs = np.linspace(x_[ok].min(), x_[ok].max(), 100)
                    ys = a * xs + b
                    figdt.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Linear fit", line=dict(dash="dash")))
                figdt.update_layout(height=420, title="Duration vs Temperature")
                st.plotly_chart(figdt, use_container_width=True)

        with c4:
            if temp_ok:
                dat = df_f[m_dst & df_f["avg_temp_c"].notna()]
                nmax = 30000
                if len(dat) > nmax:
                    dat = dat.sample(n=nmax, random_state=7)
                figDxT = px.scatter(
                    dat, x="avg_temp_c", y="distance_km", opacity=0.6,
                    labels={"avg_temp_c":"Avg temperature (°C)", "distance_km":"Distance (km)"}
                )
                x_ = pd.to_numeric(dat["avg_temp_c"], errors="coerce")
                y_ = pd.to_numeric(dat["distance_km"], errors="coerce")
                ok = x_.notna() & y_.notna()
                if ok.sum() >= 3 and x_[ok].nunique() >= 2:
                    a, b = np.polyfit(x_[ok], y_[ok], 1)
                    xs = np.linspace(x_[ok].min(), x_[ok].max(), 100)
                    ys = a * xs + b
                    figDxT.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Linear fit", line=dict(dash="dash")))
                figDxT.update_layout(height=420, title="Distance vs Temperature")
                st.plotly_chart(figDxT, use_container_width=True)

        # ===== 2D density: distance vs duration =====
        st.markdown("### 🔳 2D density: distance vs duration")
        try:
            inliers_all = df_f[m_dst & m_dur].copy()
            inliers_sample = inliers_all.sample(n=min(len(inliers_all), 60000), random_state=11) if len(inliers_all) > 60000 else inliers_all
            fig_hex = px.density_heatmap(
                inliers_sample, x="distance_km", y="duration_min",
                nbinsx=60, nbinsy=60, histfunc="count",
                labels={"distance_km":"Distance (km)", "duration_min":"Duration (min)"},
                color_continuous_scale="Viridis"
            )
            fig_hex.update_layout(height=520)
            st.plotly_chart(fig_hex, use_container_width=True)
        except Exception as e:
            st.caption(f"Density heatmap unavailable: {e}")

        # ===== Correlations (quick view) =====
        st.markdown("### 🔗 Correlations (quick view)")
        corr_cols = [c for c in ["duration_min","distance_km","speed_kmh","avg_temp_c","wind_kph"] if c in df_f.columns]
        if len(corr_cols) >= 2:
            corr_mat = df_f[corr_cols].corr(numeric_only=True)
            fig_corr = px.imshow(corr_mat, text_auto=True, aspect="auto", labels=dict(color="r"))
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.caption("Not enough numeric columns to compute a correlation matrix.")

        # ===== Rain/Wet impact on duration & speed =====
        st.subheader("Rain impact on trip characteristics")
        has_precip_bin = ("precip_bin" in df_f.columns) and df_f["precip_bin"].notna().any()
        has_wet_flag = ("wet_day" in df_f.columns)

        cc1, cc2 = st.columns(2)
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

        # ===== Quick weather deltas (KPIs) =====
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            if has_wet_flag and df_f["wet_day"].notna().any():
                dry_spd = df_f.loc[m_spd & (df_f["wet_day"]==0), "speed_kmh"].mean()
                wet_spd = df_f.loc[m_spd & (df_f["wet_day"]==1), "speed_kmh"].mean()
                if pd.notnull(dry_spd) and pd.notnull(wet_spd) and dry_spd>0:
                    st.metric("Speed: Wet vs Dry", f"{(wet_spd-dry_spd)/dry_spd*100:+.1f}%")
        with k2:
            if wind_ok:
                calm_spd = df_f.loc[m_spd & (df_f["wind_kph"]<10), "speed_kmh"].mean()
                windy_spd = df_f.loc[m_spd & (df_f["wind_kph"]>=20), "speed_kmh"].mean()
                if pd.notnull(calm_spd) and pd.notnull(windy_spd) and calm_spd>0:
                    st.metric("Speed: Windy (≥20) vs Calm (<10)", f"{(windy_spd-calm_spd)/calm_spd*100:+.1f}%")
        with k3:
            if temp_ok:
                comfy = df_f.loc[m_spd & df_f["avg_temp_c"].between(15,25), "speed_kmh"].mean()
                extreme = df_f.loc[m_spd & (~df_f["avg_temp_c"].between(5,30)), "speed_kmh"].mean()
                if pd.notnull(comfy) and pd.notnull(extreme) and comfy != 0:
                    st.metric("Speed: Comfy (15–25°C) vs Extreme", f"{(comfy-extreme)/comfy*100:+.1f}%")
        with k4:
            if has_precip_bin:
                low_dur = df_f.loc[m_dur & (df_f["precip_bin"]=="Low"), "duration_min"].mean()
                high_dur = df_f.loc[m_dur & (df_f["precip_bin"]=="High"), "duration_min"].mean()
                if pd.notnull(low_dur) and pd.notnull(high_dur) and low_dur>0:
                    st.metric("Duration: High rain vs Low", f"{(high_dur-low_dur)/low_dur*100:+.1f}%")

elif page == "Member vs Casual Profiles":
    st.header("👥 Member vs Casual riding patterns")

    # Guard
    if "member_type" not in df_f.columns or "hour" not in df_f.columns:
        st.info("Need member_type and started_at (engineered hour).")
    else:
        # ------------------------------------------------------------------
        # 1) Clean, ASCII-only member labels to avoid JSON encoding issues
        # ------------------------------------------------------------------
        SAFE_MEMBER_LABELS = {"member": "Member", "casual": "Casual"}
        def _asciify_member(s):
            s = str(s) if s is not None else ""
            return SAFE_MEMBER_LABELS.get(s.lower(), s.title())

        base = df_f.copy()
        base["member_type_clean"] = base["member_type"].astype(str).map(_asciify_member).astype("category")

        # ------------------------------------------------------------------
        # 2) Hourly and weekday profiles
        # ------------------------------------------------------------------
        st.subheader("Behavioral profiles")

        # Hourly
        g_hour = (
            base.groupby(["member_type_clean", "hour"])
            .size().rename("rides").reset_index()
        )
        fig_h = px.line(
            g_hour, x="hour", y="rides", color="member_type_clean",
            labels={"hour": "Hour of day", "rides": "Rides", "member_type_clean": "Member type"},
            markers=True
        )
        fig_h.update_layout(height=380, title="Hourly profile by rider type")
        st.plotly_chart(fig_h, use_container_width=True)

        # Weekday
        if "weekday" in base.columns:
            g_wk = (
                base.groupby(["member_type_clean", "weekday"])
                .size().rename("rides").reset_index()
            )
            wk_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            g_wk["weekday_name"] = g_wk["weekday"].map(wk_map)
            fig_w = px.line(
                g_wk, x="weekday_name", y="rides", color="member_type_clean",
                labels={"weekday_name": "Weekday", "rides": "Rides", "member_type_clean": "Member type"},
                markers=True
            )
            fig_w.update_layout(height=380, title="Weekday profile by rider type")
            st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("---")

        # ------------------------------------------------------------------
        # 3) Weather mix and temperate preference
        # ------------------------------------------------------------------
        st.subheader("Weather mix and temperature preference")

        cwx1, cwx2 = st.columns(2)

        # Rain / wet mix
        with cwx1:
            if "precip_bin" in base.columns and base["precip_bin"].notna().any():
                # keep a clear, stable order if present
                order_precip = ["Low", "Medium", "High"]
                g_rain = (
                    base.dropna(subset=["precip_bin"])
                    .assign(precip_bin=lambda x: pd.Categorical(x["precip_bin"], order_precip, ordered=True))
                    .groupby(["member_type_clean", "precip_bin"])
                    .size().rename("rides").reset_index()
                )
                fig_r = px.bar(
                    g_rain, x="precip_bin", y="rides", color="member_type_clean", barmode="group",
                    labels={"precip_bin": "Precipitation", "rides": "Rides", "member_type_clean": "Member type"}
                )
                fig_r.update_layout(height=380, title="Ride volume by precipitation bin")
                st.plotly_chart(fig_r, use_container_width=True)
            elif "wet_day" in base.columns and base["wet_day"].notna().any():
                g_wet = (
                    base.assign(day_type=lambda x: x["wet_day"].map({0: "Dry", 1: "Wet"}))
                    .groupby(["member_type_clean", "day_type"])
                    .size().rename("rides").reset_index()
                )
                fig_wd = px.bar(
                    g_wet, x="day_type", y="rides", color="member_type_clean", barmode="group",
                    labels={"day_type": "Day type", "rides": "Rides", "member_type_clean": "Member type"}
                )
                fig_wd.update_layout(height=380, title="Ride volume: Wet vs Dry")
                st.plotly_chart(fig_wd, use_container_width=True)
            else:
                st.info("No rain columns (precip_bin or wet_day) available.")

        # Temperature distribution by rider type
        with cwx2:
            if "avg_temp_c" in base.columns and base["avg_temp_c"].notna().any():
                vdat = base.dropna(subset=["avg_temp_c"])
                fig_v = px.violin(
                    vdat, x="member_type_clean", y="avg_temp_c", box=True, points=False,
                    labels={"member_type_clean": "Member type", "avg_temp_c": "Avg temperature during rides (C)"}
                )
                fig_v.update_layout(height=380, title="Where each group rides by temperature")
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.info("Temperature not available to plot distributions.")

        st.markdown("---")

        # ------------------------------------------------------------------
        # 4) Wind effect
        # ------------------------------------------------------------------
        st.subheader("Wind effect by rider type")
        if "wind_bin" in base.columns and base["wind_bin"].notna().any():
            order_wind = ["Calm", "Breeze", "Windy", "Very Windy"]
            present = [x for x in order_wind if x in base["wind_bin"].astype(str).unique().tolist()]
            g_wind = (
                base.dropna(subset=["wind_bin"])
                .assign(wind_bin=lambda x: pd.Categorical(x["wind_bin"], present if present else None, ordered=True))
                .groupby(["member_type_clean", "wind_bin"])
                .size().rename("rides").reset_index()
            )
            fig_wind = px.bar(
                g_wind, x="wind_bin", y="rides", color="member_type_clean", barmode="group",
                labels={"wind_bin": "Wind", "rides": "Rides", "member_type_clean": "Member type"}
            )
            fig_wind.update_layout(height=380, title="Ride volume by wind condition")
            st.plotly_chart(fig_wind, use_container_width=True)
        elif "wind_kph" in base.columns and base["wind_kph"].notna().any():
            bins = [-1, 10, 20, 30, 200]
            labels_w = ["<10", "10-20", "20-30", "30+"]
            g_wind2 = (
                base.assign(wind_bin=lambda x: pd.cut(x["wind_kph"], bins, labels=labels_w, include_lowest=True))
                .groupby(["member_type_clean", "wind_bin"])
                .size().rename("rides").reset_index()
            )
            fig_wind2 = px.bar(
                g_wind2, x="wind_bin", y="rides", color="member_type_clean", barmode="group",
                labels={"wind_bin": "Wind (kph)", "rides": "Rides", "member_type_clean": "Member type"}
            )
            fig_wind2.update_layout(height=380, title="Ride volume by wind (kph)")
            st.plotly_chart(fig_wind2, use_container_width=True)
        else:
            st.info("No wind columns available.")

        st.markdown("---")

        # ------------------------------------------------------------------
        # 5) Performance vs temperature (median speed and duration by band)
        # ------------------------------------------------------------------
        st.subheader("Performance vs temperature")
        if {"avg_temp_c", "speed_kmh", "duration_min"}.issubset(base.columns) and base["avg_temp_c"].notna().any():
            tbins = [-20, -5, 0, 5, 10, 15, 20, 25, 30, 35, 50]
            tlabs = ["<-5", "-5-0", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", ">35"]
            dat = base.dropna(subset=["avg_temp_c"]).copy()
            dat["temp_band"] = pd.cut(dat["avg_temp_c"], tbins, labels=tlabs, include_lowest=True)

            # Median speed
            gs = (
                dat.groupby(["member_type_clean", "temp_band"])["speed_kmh"]
                .median().reset_index().dropna(subset=["temp_band"])
            )
            figS = px.line(
                gs, x="temp_band", y="speed_kmh", color="member_type_clean", markers=True,
                labels={"temp_band": "Temperature band (C)", "speed_kmh": "Median speed (km/h)", "member_type_clean": "Member type"}
            )
            figS.update_layout(height=360, title="Median speed by temperature band")
            st.plotly_chart(figS, use_container_width=True)

            # Median duration
            gd = (
                dat.groupby(["member_type_clean", "temp_band"])["duration_min"]
                .median().reset_index().dropna(subset=["temp_band"])
            )
            figD = px.line(
                gd, x="temp_band", y="duration_min", color="member_type_clean", markers=True,
                labels={"temp_band": "Temperature band (C)", "duration_min": "Median duration (min)", "member_type_clean": "Member type"}
            )
            figD.update_layout(height=360, title="Median duration by temperature band")
            st.plotly_chart(figD, use_container_width=True)
        else:
            st.info("Need avg_temp_c, duration_min, and speed_kmh to chart performance vs temperature.")

        st.markdown("---")

        # ------------------------------------------------------------------
        # 6) Station preferences and lift (share vs overall) - ASCII safe
        # ------------------------------------------------------------------
        st.subheader("Station preferences (share vs overall)")
        if "start_station_name" in base.columns:
            topN = st.slider("Top N stations", 5, 40, 15, 5)
            # Overall shares
            overall = (
                base.groupby("start_station_name").size()
                .rename("rides").reset_index()
            )
            total_overall = float(overall["rides"].sum())
            overall["p_overall"] = overall["rides"] / total_overall if total_overall > 0 else 0.0

            # By rider type
            g = (
                base.groupby(["member_type_clean", "start_station_name"]).size()
                .rename("rides").reset_index()
            )
            g["p_type"] = (
                g.groupby("member_type_clean")["rides"].transform(lambda s: s / float(s.sum()) if float(s.sum()) > 0 else 0.0)
            )

            # Merge and compute lift; IMPORTANT: do NOT blanket fillna over categoricals
            mrg = g.merge(overall[["start_station_name", "p_overall"]], on="start_station_name", how="left")
            if "p_overall" in mrg.columns:
                mrg["p_overall"] = mrg["p_overall"].astype(float).fillna(0.0)

            mrg["lift"] = mrg["p_type"] - mrg["p_overall"]

            # Top positive lifts per rider type
            out_rows = []
            for typ, sub in mrg.groupby("member_type_clean"):
                top = sub.sort_values("lift", ascending=False).head(topN).copy()
                top["member_type_clean"] = typ
                out_rows.append(top[["member_type_clean", "start_station_name", "lift", "p_type", "p_overall"]])
            if out_rows:
                top_df = pd.concat(out_rows, ignore_index=True)
                fig_l = px.bar(
                    top_df, x="start_station_name", y="lift", color="member_type_clean", barmode="group",
                    labels={
                        "start_station_name": "Station",
                        "lift": "Share lift (type share - overall share)",
                        "member_type_clean": "Member type",
                    }
                )
                fig_l.update_layout(height=420, title="Top station share lifts by rider type")
                fig_l.update_xaxes(tickangle=40, tickfont=dict(size=10))
                st.plotly_chart(fig_l, use_container_width=True)
                st.caption("Lift = within-type share minus overall share. Positive means over-indexing for that rider type.")
        else:
            st.info("start_station_name not available.")

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
