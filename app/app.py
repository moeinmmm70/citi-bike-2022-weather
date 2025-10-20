# app/st_dashboard_Part_2.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Optional ML
try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None

# Optional clustering
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
except Exception:
    linkage = leaves_list = None

# ────────────────────────────── Page/Theming ──────────────────────────────
st.set_page_config(page_title="NYC Citi Bike — Strategy Dashboard", page_icon="🚲", layout="wide")
pio.templates.default = "plotly_white"

# ────────────────────────────── Paths/Constants ───────────────────────────
DATA_PATH = Path("data/processed/reduced_citibike_2022.csv")   # ≤25MB sample

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
    """Human friendly formatting: 1.2K, 3.4M, etc."""
    try:
        x = float(x)
    except Exception:
        return "—"
    units = ["", "K", "M", "B", "T"]
    for u in units:
        if abs(x) < 1000 or u == units[-1]:
            if u == "":
                return f"{x:,.0f}"
            return f"{x:.1f}{u}"
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
    """NaN-safe Pearson correlation on overlapping index; returns None if <3 points."""
    a, b = a.dropna(), b.dropna()
    j = a.index.intersection(b.index)
    if len(j) < 3:
        return None
    c = np.corrcoef(a.loc[j], b.loc[j])[0,1]
    return float(c)

def linear_fit(x: pd.Series, y: pd.Series):
    """Return slope/intercept/R2 if sklearn is available and n>=3; else None tuple."""
    if LinearRegression is None:
        return None, None, None
    valid = (~x.isna()) & (~y.isna())
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return None, None, None
    X = np.array(x).reshape(-1, 1)
    yv = np.array(y)
    try:
        model = LinearRegression().fit(X, yv)
        r2 = model.score(X, yv)
        return float(model.coef_[0]), float(model.intercept_), float(r2)
    except Exception:
        return None, None, None

@st.cache_data(show_spinner=False)
def load_data(path: Path, _mtime: float = None) -> pd.DataFrame:
    """Load CSV with basic parsing; _mtime only to make cache sensitive to file updates."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Light parsing
    if "started_at" in df.columns:
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        df["hour"] = df["started_at"].dt.hour
        df["date"] = df["started_at"].dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # Normalize typical columns if present
    if "member_type" in df.columns:
        df["member_type"] = df["member_type"].astype(str).str.lower()
    if "season" in df.columns:
        df["season"] = df["season"].astype(str).str.title()
    if "weekday" in df.columns:
        # ensure ints 0-6 if text exists
        df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").astype("Int64")
    return df

def ensure_daily(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty or "date" not in df.columns:
        return None
    # Aggregate to daily
    daily = df.groupby("date", as_index=False).size().rename(columns={"size": "bike_rides_daily"})
    # attach average temp if available
    for col in ["avg_temp_c", "precip_mm", "wind_kph", "wet_day"]:
        if col in df.columns:
            tmp = df.groupby("date", as_index=False)[col].mean(numeric_only=True)
            daily = daily.merge(tmp, on="date", how="left")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    return daily.sort_values("date")

def apply_filters(
    df: pd.DataFrame,
    date_range: tuple | None,
    seasons: list | None,
    usertype: str | None,
    temp_range: tuple | None,
    hour_range: tuple | None,
    weekdays: list | None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    if date_range and "date" in out.columns:
        d0 = pd.to_datetime(date_range[0], errors="coerce").date()
        d1 = pd.to_datetime(date_range[1], errors="coerce").date()
        out = out[(out["date"] >= d0) & (out["date"] <= d1)]

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
        # Expect weekdays as names e.g. ["Mon",...], convert if necessary
        name_to_idx = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        if isinstance(weekdays[0], str):
            idxs = [name_to_idx.get(w) for w in weekdays if w in name_to_idx]
        else:
            idxs = weekdays
        out = out[out["weekday"].isin(idxs)]

    return out

def compute_core_kpis(df_f: pd.DataFrame, daily_f: pd.DataFrame | None):
    total_rides = len(df_f) if df_f is not None else 0
    avg_day = float(daily_f["bike_rides_daily"].mean()) if daily_f is not None and not daily_f.empty else None
    corr_tr = safe_corr(daily_f.set_index("date")["bike_rides_daily"], daily_f.set_index("date")["avg_temp_c"]) \
              if daily_f is not None and "avg_temp_c" in (daily_f.columns if daily_f is not None else []) else None
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

# ────────────────────────────── URL helpers ────────────────────────────────
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

# ────────────────────────────── Page List (rational flow) ──────────────────
PAGES = [
    "Intro",
    "Weather vs Bike Usage",
    "Weekday × Hour Heatmap",
    "Trip Metrics (Duration • Distance • Speed)",
    "Station Popularity",
    "Station Imbalance (In vs Out)",
    "OD Matrix — Top Origins × Dest",
    "OD Flows — Sankey + Map",
    "Pareto: Share of Rides",
    "Member vs Casual Profiles",
    "Recommendations",
]

# ────────────────────────────── Sidebar / Data ─────────────────────────────
st.sidebar.header("⚙️ Controls")

# Status + load
if not DATA_PATH.exists():
    st.sidebar.error(f"Missing data: {DATA_PATH}")
    st.error("Data file not found. Create the ≤25MB sample CSV at data/processed/reduced_citibike_2022.csv.")
    st.stop()

df = load_data(DATA_PATH, DATA_PATH.stat().st_mtime)
st.sidebar.success(f"Data loaded: {len(df):,} rows")

# Core filters (ordered)
date_min = pd.to_datetime(df["date"]).min() if "date" in df.columns else None
date_max = pd.to_datetime(df["date"]).max() if "date" in df.columns else None
date_range = st.sidebar.date_input("Date range", value=(date_min, date_max)) if date_min is not None else None

hour_range = st.sidebar.slider("Hour of day", 0, 23, (6, 22), key="hour_slider") if "hour" in df.columns else None

usertype = None
if "member_type" in df.columns:
    raw_opts = ["All"] + sorted(df["member_type"].astype(str).unique().tolist())
    usertype = st.sidebar.selectbox(
        "User type",
        raw_opts,
        format_func=lambda v: "All" if v == "All" else MEMBER_LABELS.get(v, str(v).title())
    )

seasons_all = ["Winter","Spring","Summer","Autumn"]
seasons = st.sidebar.multiselect("Season(s)", seasons_all, default=seasons_all) if "season" in df.columns else None

# Advanced (collapsed)
temp_range, weekdays = None, None
with st.sidebar.expander("More filters", expanded=False):
    if "avg_temp_c" in df.columns:
        tmin = float(np.nanmin(df["avg_temp_c"]))
        tmax = float(np.nanmax(df["avg_temp_c"]))
        temp_range = st.slider("Temperature (°C)", tmin, tmax, (tmin, tmax), key="temp_slider")
    if "weekday" in df.columns:
        weekday_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekdays = st.multiselect("Weekday(s)", weekday_names, default=weekday_names, key="weekday_multi")

# Page selector (URL-aware seed)
_qp = _qp_get()
_qp_page = None
if "page" in _qp:
    _qp_page = _qp["page"][0] if isinstance(_qp["page"], list) else _qp["page"]
if _qp_page not in PAGES:
    _qp_page = PAGES[0]

page = st.sidebar.selectbox("📑 Analysis page", PAGES, index=PAGES.index(_qp_page), key="page_select")

# --- Quick presets (safe; after widgets exist) ---
with st.sidebar.expander("Presets", expanded=False):
    c1, c2 = st.columns(2)

    def _set_if_exists(key, value):
        if key in st.session_state:
            st.session_state[key] = value

    with c1:
        if st.button("✨ Commuter (weekday peaks)"):
            _set_if_exists("page_select", "Weekday × Hour Heatmap")
            _set_if_exists("hour_slider", (6, 20))
            if "weekday_multi" in st.session_state:
                st.session_state["weekday_multi"] = ["Mon","Tue","Wed","Thu","Fri"]
            if "member_type" in df.columns and "User type" in st.session_state:
                st.session_state["User type"] = "member"

    with c2:
        if st.button("🌧️ Rainy-day focus"):
            _set_if_exists("page_select", "Weather vs Bike Usage")

# Utilities
ut1, ut2 = st.sidebar.columns(2)
with ut1:
    if st.button("🔄 Reload data"):
        st.cache_data.clear()
        st.rerun()
with ut2:
    if st.button("♻️ Reset all"):
        st.cache_data.clear()
        if hasattr(st, "query_params"):
            st.query_params.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# Write chosen state to URL (safe)
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

# Build filtered data
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

# ────────────────────────────── Page Renderers ─────────────────────────────
def page_intro():
    st.title("NYC Citi Bike — Strategy Dashboard")
    st.markdown("""
    **What you'll find here:**  
    - The weather & time patterns that drive usage  
    - When & where rides concentrate (stations, OD)  
    - Member vs casual behaviors  
    - Practical recommendations for ops & growth
    """)
    k = compute_core_kpis(df_f, daily_f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rides (filtered)", kfmt(k["total_rides"]))
    c2.metric("Avg rides / day", "—" if k["avg_day"] is None else f"{k['avg_day']:,.0f}")
    c3.metric("Corr(temp, rides) daily", "—" if k["corr_tr"] is None else f"{k['corr_tr']:.2f}")

def page_weather_vs_usage():
    st.header("🌦️ Weather vs Bike Usage")
    if daily_f is None or daily_f.empty or "avg_temp_c" not in daily_f.columns:
        st.info("Need daily table with `avg_temp_c`. Add temperature to your CSV or backfill.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=daily_f["date"], y=daily_f["bike_rides_daily"], name="Daily rides"), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_f["date"], y=daily_f["avg_temp_c"], mode="lines", name="Avg Temp (°C)"), secondary_y=True)
    fig.update_yaxes(title_text="Rides/day", secondary_y=False)
    fig.update_yaxes(title_text="Temp (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

def page_weekday_hour_heatmap():
    st.header("🕒 Weekday × Hour Heatmap")
    need = {"weekday","hour"}
    if not need.issubset(df_f.columns):
        st.info("Need `weekday` (0=Mon..6=Sun) and `hour` columns.")
        return
    tbl = df_f.groupby(["weekday","hour"]).size().rename("rides").reset_index()
    # map weekday to name
    idx_to_name = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    tbl["weekday_name"] = tbl["weekday"].map(idx_to_name)
    heat = tbl.pivot(index="weekday_name", columns="hour", values="rides").fillna(0)
    fig = px.imshow(heat, aspect="auto", labels=dict(x="Hour", y="Weekday", color="Rides"))
    st.plotly_chart(fig, use_container_width=True)

def page_trip_metrics():
    st.header("🚴 Trip Metrics")
    cols = [c for c in ["ride_length_min","distance_km","speed_kmh"] if c in df_f.columns]
    if not cols:
        st.info("Add any of columns: `ride_length_min`, `distance_km`, `speed_kmh`.")
        return
    pick = st.selectbox("Metric", cols)
    msk = inlier_mask(df_f, pick, 0.01, 0.995)
    fig = px.histogram(df_f[msk], x=pick, nbins=40)
    friendly_axis(fig, x=pick.replace("_"," ").title(), y="Count")
    st.plotly_chart(fig, use_container_width=True)

def page_station_popularity():
    st.header("📍 Station Popularity")
    if "start_station_name" not in df_f.columns and "end_station_name" not in df_f.columns:
        st.info("Need `start_station_name` or `end_station_name`.")
        return
    mode = st.radio("Count by", ["Starts", "Ends"], horizontal=True)
    col = "start_station_name" if mode == "Starts" else "end_station_name"
    topn = st.slider("Top N", 5, 50, 20)
    agg = df_f.groupby(col).size().rename("rides").sort_values(ascending=False).head(topn).reset_index()
    agg[col] = agg[col].astype(str).map(lambda s: shorten_name(s, 28))
    fig = px.bar(agg, x="rides", y=col, orientation="h", height=600)
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    friendly_axis(fig, x="Rides", y="Station")
    st.plotly_chart(fig, use_container_width=True)

def page_station_imbalance():
    st.header("⚖️ Station Imbalance (In vs Out)")
    need = {"start_station_name","end_station_name"}
    if not need.issubset(df_f.columns):
        st.info("Need start/end station names.")
        return
    starts = df_f.groupby("start_station_name").size().rename("out")
    ends   = df_f.groupby("end_station_name").size().rename("in")
    bal = pd.concat([starts, ends], axis=1).fillna(0.0)
    bal["net_in"] = bal["in"] - bal["out"]
    bal = bal.sort_values("net_in", ascending=False).reset_index().rename(columns={"index":"station"})
    top = st.slider("Show top ±N by net inflow/outflow", 5, 50, 20)
    show = pd.concat([bal.head(top), bal.tail(top)])
    fig = px.bar(show, x="net_in", y="station", orientation="h", color=(show["net_in"]>0).map({True:"Net Inflow", False:"Net Outflow"}))
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    friendly_axis(fig, x="Net inflow (in - out)", y="Station")
    st.plotly_chart(fig, use_container_width=True)

def page_od_matrix():
    st.header("🔢 OD Matrix — Top Origins × Dest")
    need = {"start_station_name","end_station_name"}
    if not need.issubset(df_f.columns):
        st.info("Need start/end station names.")
        return
    # Top-k per side
    k = st.slider("Top-k stations", 5, 40, 15, 1)
    starts = df_f["start_station_name"].value_counts().head(k).index
    ends   = df_f["end_station_name"].value_counts().head(k).index
    sub = df_f[df_f["start_station_name"].isin(starts) & df_f["end_station_name"].isin(ends)]
    mat = sub.groupby(["start_station_name","end_station_name"]).size().rename("rides").reset_index()
    piv = mat.pivot(index="start_station_name", columns="end_station_name", values="rides").fillna(0)
    fig = px.imshow(piv, aspect="auto", labels=dict(x="Destination", y="Origin", color="Rides"))
    st.plotly_chart(fig, use_container_width=True)

def page_sankey():
    st.header("🔀 OD Flows — Sankey + Map")
    need = {"start_station_name","end_station_name"}
    if not need.issubset(df_f.columns):
        st.info("Need start/end station names.")
        return
    topk = st.slider("Top-k flows", 10, 200, 40, 10)
    flows = df_f.groupby(["start_station_name","end_station_name"]).size().rename("rides").reset_index()
    flows = flows.sort_values("rides", ascending=False).head(topk)
    # Build sankey
    nodes = pd.Index(sorted(set(flows["start_station_name"]) | set(flows["end_station_name"]))).tolist()
    node_index = {n:i for i,n in enumerate(nodes)}
    source = flows["start_station_name"].map(node_index).tolist()
    target = flows["end_station_name"].map(node_index).tolist()
    value  = flows["rides"].astype(float).tolist()
    sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=12, thickness=14, label=[shorten_name(n, 20) for n in nodes]),
        link=dict(source=source, target=target, value=value)
    )])
    sankey.update_layout(height=620)
    st.plotly_chart(sankey, use_container_width=True)
    st.caption("Map view not included in this fresh upload (no preset maps).")

def page_pareto():
    st.header("🧮 Pareto — Share of Rides")
    if "start_station_name" not in df_f.columns and "end_station_name" not in df_f.columns:
        st.info("Need station names.")
        return
    station_col = st.radio("Station role", ["Start", "End"], horizontal=True)
    station_col = "start_station_name" if station_col == "Start" else "end_station_name"
    normalize = st.selectbox("Normalization", ["Total rides (raw)", "Per day (avg/station)"])
    min_rides = st.slider("Min rides to include", 0, 500, 0, 10)
    target = st.slider("Target cumulative share (%)", 50, 95, 80, 5)
    show_lorenz = st.checkbox("Show Lorenz curve", value=False)

    subset = df_f.copy()
    subset[station_col] = subset[station_col].astype(str)

    if normalize == "Per day (avg/station)" and "date" in subset.columns:
        per_day = (
            subset.groupby([station_col, "date"])
            .size()
            .rename("rides_day")
            .reset_index()
        )
        totals = per_day.groupby(station_col)["rides_day"].mean().rename("rides")
    else:
        totals = subset.groupby(station_col).size().rename("rides")

    if min_rides > 0:
        totals = totals[totals >= float(min_rides)]

    if totals.empty:
        st.info("No stations left after filtering. Lower **Min rides**.")
        return

    totals = totals.sort_values(ascending=False)
    counts = totals.to_numpy(dtype=float)
    n = len(counts)
    cum_share = np.cumsum(counts) / counts.sum()

    target_frac = target / 100.0
    idx_target = int(np.searchsorted(cum_share, target_frac, side="left"))
    rank_needed = min(max(idx_target + 1, 1), n)

    # Gini & HHI
    x = np.sort(counts)
    cum_x = np.cumsum(x)
    gini = 1 - (2 / (n - 1)) * (n - (cum_x.sum() / (cum_x[-1] if cum_x[-1] else 1)))
    shares = counts / counts.sum()
    hhi = float(np.sum(shares ** 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, n + 1),
        y=cum_share,
        mode="lines",
        name="Cumulative share",
        hovertemplate="Rank: %{x}<br>Cumulative share: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=target_frac, line_dash="dot")
    fig.add_vline(x=rank_needed, line_dash="dot")
    fig.add_annotation(
        x=rank_needed, y=min(target_frac + 0.025, 0.98),
        showarrow=False,
        text=f"Top ~{rank_needed:,} / {n:,} stations ≈ {target}%",
        bgcolor="rgba(0,0,0,0.05)"
    )

    if show_lorenz:
        x_lor = np.linspace(0, 1, n, endpoint=True)
        y_lor = np.cumsum(np.sort(shares))
        fig.add_trace(go.Scatter(x=x_lor * n, y=y_lor, mode="lines", name="Lorenz (asc by size)"))
        fig.add_trace(go.Scatter(x=x_lor * n, y=x_lor, mode="lines", name="Equality", line=dict(dash="dash")))

    friendly_axis(fig, x="Station rank", y="Cumulative share")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Gini (0 even → 1 concentrated)", f"{gini:.2f}")
    c2.metric("HHI (0—1; higher=more concentrated)", f"{hhi:.3f}")

def page_member_vs_casual():
    st.header("👥 Member vs Casual Profiles")
    if "member_type" not in df_f.columns:
        st.info("Need `member_type`.")
        return
    col = st.selectbox("Compare by", [c for c in ["ride_length_min","distance_km","speed_kmh"] if c in df_f.columns] or ["—"])
    if col == "—":
        st.info("Add any ride metric column to compare.")
        return
    gr = df_f.groupby("member_type")[col].median().reset_index()
    gr["member_label"] = gr["member_type"].map(lambda v: MEMBER_LABELS.get(v, str(v).title()))
    fig = px.bar(gr, x="member_label", y=col)
    friendly_axis(fig, x=MEMBER_LEGEND_TITLE, y=f"Median {col.replace('_',' ').title()}")
    st.plotly_chart(fig, use_container_width=True)

def page_recommendations():
    st.header("🧭 Recommendations")
    st.markdown("""
    - **Staffing & rebalancing:** Focus on peak weekday **AM/PM** windows and stations with **chronic net inflow/outflow**.
    - **Weather playbooks:** On **rainy/cold days**, scale back staffing & bikes; on **mild/warm** days, add pop-up capacity.
    - **Product:** Tailor promotions to **casuals** on weekends around popular leisure corridors; retain **members** with commute-friendly perks.
    """)

# ────────────────────────────── Router ─────────────────────────────────────
ROUTES = {
    "Intro": page_intro,
    "Weather vs Bike Usage": page_weather_vs_usage,
    "Weekday × Hour Heatmap": page_weekday_hour_heatmap,
    "Trip Metrics (Duration • Distance • Speed)": page_trip_metrics,
    "Station Popularity": page_station_popularity,
    "Station Imbalance (In vs Out)": page_station_imbalance,
    "OD Matrix — Top Origins × Dest": page_od_matrix,
    "OD Flows — Sankey + Map": page_sankey,
    "Pareto: Share of Rides": page_pareto,
    "Member vs Casual Profiles": page_member_vs_casual,
    "Recommendations": page_recommendations,
}

ROUTES.get(page, page_intro)()
