# app/st_dashboard_Part_2.py
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static

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

    # Derive precip bins if precip exists
    for pcol in ["precip_mm", "precipitation_mm", "precip_mm_day", "precipitation"]:
        if pcol in df.columns:
            q = df[pcol].quantile([0.33, 0.66]).values if df[pcol].notna().sum() > 10 else [0.2, 2.0]
            labels = ["Low", "Medium", "High"]
            df["precip_bin"] = pd.cut(df[pcol], bins=[-1e9, q[0], q[1], 1e9], labels=labels, include_lowest=True)
            df["precip_bin"] = df["precip_bin"].astype("category")
            break

    # Very light Comfort Index (higher = nicer), if feasible
    temp_candidates = ["avg_temp_c", "avgTemp", "avg_temp", "temperature_c"]
    wind_candidates = ["wind_kph", "wind_speed_kph", "wind_mps"]
    hum_candidates  = ["humidity", "rel_humidity"]

    tcol = next((c for c in temp_candidates if c in df.columns), None)
    wcol = next((c for c in wind_candidates if c in df.columns), None)
    hcol = next((c for c in hum_candidates  if c in df.columns), None)

    if tcol is not None and wcol is not None:
        t = df[tcol].astype(float)
        w = df[wcol].astype(float)
        base = 1.0 - (np.abs(t - 22.0) / 30.0) - (w / (w.replace(0, np.nan).quantile(0.95) or 1.0)) * 0.3
        base = base.clip(-1.0, 1.0)
        if hcol is not None:
            h = (df[hcol].astype(float) / 100.0).clip(0, 1)
            base = base + 0.1 * (1 - np.abs(h - 0.5) * 2)
        df["comfort_index"] = base.astype(float)

    return df

def ensure_daily(df: pd.DataFrame) -> pd.DataFrame | None:
    """Guarantee a daily table with columns: date, bike_rides_daily, optional avg_temp_c/precip/comfort."""
    if df is None or df.empty:
        return None
    if {"date", "bike_rides_daily"}.issubset(df.columns):
        daily = df[["date", "bike_rides_daily"]].dropna().drop_duplicates()
    elif "date" in df.columns:
        daily = df.groupby("date", as_index=False).agg(bike_rides_daily=("date", "size"))
    else:
        return None

    # Attach daily means for available weather columns
    attach_cols = []
    for cand in ["avg_temp_c", "avgTemp", "avg_temp", "temperature_c", "precip_bin",
                 "precip_mm", "precipitation_mm", "wind_kph", "wind_speed_kph", "comfort_index"]:
        if cand in df.columns:
            attach_cols.append(cand)
    if attach_cols:
        agg = {}
        for c in attach_cols:
            if c == "precip_bin":
                agg[c] = lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
            else:
                agg[c] = "mean"
        extra = df.groupby("date", as_index=False).agg(agg)
        for talt in ["avgTemp", "avg_temp", "temperature_c"]:
            if talt in extra.columns and "avg_temp_c" not in extra.columns:
                extra = extra.rename(columns={talt: "avg_temp_c"})
        daily = daily.merge(extra, on="date", how="left")

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
    if usertype and usertype != "All":
        if "member_casual" in out.columns:
            out = out[out["member_casual"].astype(str).str.lower() == usertype.lower()]
        elif "usertype" in out.columns:
            out = out[out["usertype"].astype(str).str.lower() == usertype.lower()]
    if temp_range and any(col in out.columns for col in ["avg_temp_c", "avgTemp", "avg_temp", "temperature_c"]):
        tempcol = "avg_temp_c" if "avg_temp_c" in out.columns else \
                  "avgTemp" if "avgTemp" in out.columns else \
                  "avg_temp" if "avg_temp" in out.columns else "temperature_c"
        out = out[(out[tempcol] >= temp_range[0]) & (out[tempcol] <= temp_range[1])]
    return out

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

# ── UI helpers (Intro hero + KPI cards)
def kpi_card(title: str, value: str, sub: str = "", icon: str = "📊"):
    """Bigger card height, slightly smaller fonts so the text fits."""
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
    """Centered hero title panel with smaller text (original title)."""
    st.markdown(
        """
        <style>
        /* Hero Panel */
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
            font-size: clamp(1.4rem, 1.2rem + 1.6vw, 2.3rem); /* smaller than before */
            font-weight: 800;
            letter-spacing: .2px;
            margin: 2px 0 6px 0;
        }
        .hero-sub {
            color: #cbd5e1;
            font-size: clamp(.85rem, .8rem + .3vw, 1.0rem);  /* smaller subtitle */
            margin: 0;
        }

        /* KPI cards — physically bigger but fonts restrained so content fits */
        .kpi-card {
            background: linear-gradient(180deg, rgba(25,31,40,0.80) 0%, rgba(16,21,29,0.86) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 16px 18px;
            box-shadow: 0 10px 26px rgba(0,0,0,0.36);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            min-height: 190px;                /* taller cards */
            display: flex; flex-direction: column; justify-content: space-between;
        }
        .kpi-title {
            font-size: .95rem;                 /* slightly smaller */
            color: #cbd5e1;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: clamp(1.25rem, 1.0rem + 1.2vw, 2.0rem);  /* smaller to avoid overflow */
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.08;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .kpi-sub {
            font-size: .90rem;                 /* slightly smaller */
            color: #94a3b8;
            margin-top: 6px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }

        /* Rounded cover image below hero */
        .element-container img { border-radius: 16px; }
        </style>
        <div class="hero-panel">
            <h1 class="hero-title">NYC Citi Bike — Strategy Dashboard</h1>
            <p class="hero-sub">Seasonality • Weather–demand correlation • Station intelligence • Time patterns</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def compute_core_kpis(df_f: pd.DataFrame, daily_f: pd.DataFrame):
    total_rides = len(df_f) if "bike_rides_daily" not in df_f.columns else int(df_f["bike_rides_daily"].sum())
    avg_day = float(daily_f["bike_rides_daily"].mean()) if daily_f is not None and not daily_f.empty else None
    corr_tr = safe_corr(daily_f.set_index("date")["bike_rides_daily"], daily_f.set_index("date")["avg_temp_c"]) \
              if daily_f is not None and "avg_temp_c" in daily_f.columns else None
    return dict(total_rides=total_rides, avg_day=avg_day, corr_tr=corr_tr)

# ────────────────────────────── Sidebar / Data ─────────────────────────────
st.sidebar.header("⚙️ Controls")

if not DATA_PATH.exists():
    st.sidebar.error(f"Missing data: {DATA_PATH}")
    st.error("Data file not found. Create the ≤25MB sample CSV at data/processed/reduced_citibike_2022.csv.")
    st.stop()

df = load_data(DATA_PATH, DATA_PATH.stat().st_mtime)

# Filters
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
        "Correlation & Distributions",
        "Seasonal Patterns",
        "Station Popularity",
        "Pareto: Share of Rides",
        "Trip Flows Map",
        "Weekday × Hour Heatmap",
        "What-if: Temp → Rides",
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
)

daily_all = ensure_daily(df)
daily_f   = ensure_daily(df_f)

# ────────────────────────────── Pages ──────────────────────────────────────
if page == "Intro":
    # Centered hero panel (original title), smaller text
    render_hero_panel()

    # Cover image under hero
    show_cover(cover_path)
    st.caption("⚙️ Powered by NYC Citi Bike data • 365 days • Interactive visuals")

    # Compute KPIs for cards
    KPIs = compute_core_kpis(df_f, daily_f)

    # Weather Impact (% uplift good vs bad)
    weather_uplift_pct = None
    if daily_f is not None and not daily_f.empty:
        if "precip_bin" in daily_f.columns:
            good = daily_f.loc[daily_f["precip_bin"] == "Low", "bike_rides_daily"].mean()
            bad  = daily_f.loc[daily_f["precip_bin"] == "High", "bike_rides_daily"].mean()
            if pd.notnull(good) and pd.notnull(bad) and bad not in (0, np.nan):
                weather_uplift_pct = (good - bad) / bad * 100.0
        elif "avg_temp_c" in daily_f.columns:
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

    # KPI cards — bigger container, smaller fonts = clean fit
    cA, cB, cC, cD, cE = st.columns(5)
    with cA: kpi_card("Total Trips", kfmt(KPIs["total_rides"]), "Across all stations", "🧮")
    with cB: kpi_card("Daily Average", kfmt(KPIs["avg_day"]) if KPIs["avg_day"] is not None else "—", "Trips per day", "📅")
    with cC: kpi_card("Temp Impact", f"{KPIs['corr_tr']:+.3f}" if KPIs["corr_tr"] is not None else "—", "Correlation coeff.", "🌡️")
    with cD: kpi_card("Weather Impact", weather_str, "Good vs bad weather", "🌦️")
    with cE: kpi_card("Peak Season", peak_value, peak_sub, "🏆")

    st.markdown("### What you’ll find here")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Advanced KPIs**\n\nTotals, Avg/day, Temp↔Rides correlation.")
    c2.info("**Weather Deep-Dive**\n\nScatter + fit, comfort index, precipitation bins.")
    c3.info("**Station Intelligence**\n\nTop stations, Pareto, OD flows (Sankey/Kepler).")
    c4.info("**Time Patterns**\n\nWeekday×Hour heatmap, seasonal/monthly swings.")
    st.caption("Use the sidebar filters; every view updates live.")

elif page == "Weather vs Bike Usage":
    st.header("🌤️ Daily bike rides vs temperature")
    if daily_f is None or daily_f.empty:
        st.warning("Daily metrics aren’t available. Provide `bike_rides_daily` or raw trips with `date`.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=daily_f["date"], y=daily_f["bike_rides_daily"],
                mode="lines", name="Daily bike rides", line=dict(color=RIDES_COLOR, width=2)
            ),
            secondary_y=False
        )
        if "avg_temp_c" in daily_f.columns and daily_f["avg_temp_c"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=daily_f["date"], y=daily_f["avg_temp_c"],
                    mode="lines", name="Average temperature (°C)", line=dict(color=TEMP_COLOR, width=2, dash="dot")
                ),
                secondary_y=True
            )
            fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

        if "comfort_index" in daily_f.columns and daily_f["comfort_index"].notna().any():
            ci = (daily_f["comfort_index"] - daily_f["comfort_index"].min()) / (daily_f["comfort_index"].max() - daily_f["comfort_index"].min() + 1e-9)
            fig.add_trace(go.Scatter(x=daily_f["date"], y=ci, mode="lines", name="Comfort index (0–1)",
                                     line=dict(width=1, dash="dash"), opacity=0.5),
                          secondary_y=True)

        fig.update_layout(hovermode="x unified", height=520)
        friendly_axis(fig, x="Date", y="Bike rides (count)", title="Daily bike rides vs temperature — NYC (2022)")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        if "avg_temp_c" in daily_f.columns and daily_f["avg_temp_c"].notna().any():
            st.subheader("Temperature sensitivity (scatter)")
            x = daily_f["avg_temp_c"]; y = daily_f["bike_rides_daily"]
            a, b, yhat = linear_fit(x, y)
            fig2 = px.scatter(
                daily_f, x="avg_temp_c", y="bike_rides_daily",
                color=("precip_bin" if "precip_bin" in daily_f.columns else None),
                labels={"avg_temp_c": "Average temperature (°C)", "bike_rides_daily": "Bike rides (count)", "precip_bin":"Precipitation"},
            )
            if a is not None:
                xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                fig2.add_trace(go.Scatter(x=xs, y=yhat(xs), mode="lines", name=f"Fit: rides ≈ {a:.1f}×temp + {b:.0f}"))
            fig2.update_layout(height=480)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**So what?** Warm days lift demand; shoulder seasons show the steepest slope. Staff rebalancing accordingly.")

elif page == "Correlation & Distributions":
    st.header("📊 Correlation matrix & distributions")
    if daily_f is None or daily_f.empty:
        st.info("Need daily aggregates.")
    else:
        cand_cols = []
        for c in ["bike_rides_daily", "avg_temp_c", "precip_mm", "precipitation_mm",
                  "wind_kph", "wind_speed_kph", "comfort_index"]:
            if c in daily_f.columns:
                cand_cols.append(c)
        if len(cand_cols) >= 2:
            corr_df = daily_f[cand_cols].astype(float)
            corr = corr_df.corr()
            fig = px.imshow(
                corr, text_auto=True, aspect="auto", origin="lower",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                labels=dict(color="Correlation")
            )
            friendly_axis(fig, title="Correlation matrix (daily)", colorbar="Correlation")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric weather columns to compute a correlation matrix.")

        if "season" in daily_f.columns:
            st.subheader("Distribution of rides by season")
            figb = px.violin(
                daily_f, x="season", y="bike_rides_daily", box=True, points=False,
                labels={"season":"Season", "bike_rides_daily":"Bike rides per day"}
            )
            figb.update_layout(height=520)
            st.plotly_chart(figb, use_container_width=True)

        if "precip_bin" in daily_f.columns:
            st.subheader("Effect of precipitation")
            figp = px.box(
                daily_f, x="precip_bin", y="bike_rides_daily",
                labels={"precip_bin":"Precipitation", "bike_rides_daily":"Bike rides per day"}
            )
            figp.update_layout(height=420)
            st.plotly_chart(figp, use_container_width=True)

elif page == "Seasonal Patterns":
    st.header("🗓️ Monthly & seasonal patterns")
    if "date" not in df_f.columns:
        st.info("Need dates to compute monthly patterns.")
    else:
        d = df_f.copy()
        d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
        if "bike_rides_daily" in d.columns:
            monthly = d.groupby("month", as_index=False)["bike_rides_daily"].sum()
        else:
            monthly = d.groupby("month", as_index=False).size().rename(columns={"size": "bike_rides_daily"})
        monthly["ma3"] = monthly["bike_rides_daily"].rolling(3, center=True).mean()

        figm = go.Figure()
        figm.add_trace(go.Bar(x=monthly["month"], y=monthly["bike_rides_daily"], name="Monthly rides"))
        figm.add_trace(go.Scatter(x=monthly["month"], y=monthly["ma3"], mode="lines+markers", name="3-month trend"))
        figm.update_layout(height=520)
        friendly_axis(figm, x="", y="Rides per month", title="Monthly rides with 3-month trend")
        st.plotly_chart(figm, use_container_width=True)

        if {"start_station_name", "season"}.issubset(df_f.columns):
            st.subheader("Top stations by season (rank flips highlight)")
            g = (df_f.assign(n=1)
                    .groupby(["season","start_station_name"])["n"].sum()
                    .reset_index())
            topN = st.slider("Top N per season", 5, 20, 10)
            g = g.sort_values(["season","n"], ascending=[True,False]).groupby("season").head(topN)

            g["start_station_short"] = g["start_station_name"].astype(str).map(lambda s: shorten_name(s, 22))
            figfac = px.bar(
                g, x="start_station_short", y="n", facet_col="season", facet_col_wrap=2,
                height=700, color="season",
                labels={"start_station_short": "Station", "n": "Rides (count)", "season": "Season"},
                hover_data={"start_station_short": False, "season": True, "n": ":,", "start_station_name": True}
            )
            figfac.update_xaxes(matches=None, showticklabels=True, tickangle=45, tickfont=dict(size=10))
            figfac.for_each_yaxis(lambda a: a.update(title_text="Rides (count)"))
            figfac.update_layout(margin=dict(l=20, r=20, t=60, b=60), legend_title_text="")
            st.plotly_chart(figfac, use_container_width=True)

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

elif page == "Trip Flows Map":
    st.header("🗺️ Trip flows (Kepler.gl)")
    st.caption(
        "Now using live Kepler rendering. Choose a **Map preset** (JSON) to switch layers/filters/camera. "
        "Your **Analysis preset** below only affects the widgets/tables."
    )

    # ---------- Where to look for presets ----------
    ROOT = Path(repo_root) if "repo_root" in globals() else Path.cwd()
    preset_dirs = [
        ROOT / "map",                              
        ROOT / "data" / "reports" / "map",        
        ROOT / "data" / "reports" / "map" / "presets",
    ]
    preset_files = []
    for d in preset_dirs:
        if d.exists():
            preset_files.extend(sorted(d.glob("*.json")))
    presets = {p.stem: p for p in preset_files}

    if not presets:
        st.error("No Kepler presets found. Put JSON files in `map/` or `data/reports/map(/presets)/`.")
        st.stop()

    # ---------- Kepler map preset selector ----------
    colA, colB = st.columns([1.2, 2.3])
    with colA:
        map_preset_name = st.selectbox("Map preset (Kepler JSON)", list(presets.keys()), index=0)
    with colB:
        st.caption("Switching this updates layers, filters, and camera instantly.")

    # ---------- Load selected config and normalize dataset ids ----------
    def load_config(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def normalize_config_dataset_ids(cfg: dict, target_key: str = "trips") -> dict:
        """Force all layers/filters to reference the in-app dataset key."""
        cfg = copy.deepcopy(cfg)
        vis = cfg.get("config", {}).get("visState", {})
        for lyr in vis.get("layers", []):
            if isinstance(lyr, dict) and "config" in lyr and "dataId" in lyr:
                lyr["config"]["dataId"] = target_key
        for flt in vis.get("filters", []):
            if isinstance(flt, dict) and "dataId" in flt:
                # dataId may be a list in newer Kepler configs
                if isinstance(flt["dataId"], list):
                    flt["dataId"] = [target_key for _ in flt["dataId"]]
                else:
                    flt["dataId"] = target_key
        return cfg

    raw_cfg = load_config(presets[map_preset_name])
    cfg = normalize_config_dataset_ids(raw_cfg, target_key="trips")

    # ---------- Your existing scenario (analysis) preset ----------
    preset = st.selectbox(
        "Analysis preset (applies to the widgets below)",
        [
            "AM commute (Weekdays 07–10) • fair weather",
            "PM commute (Weekdays 16–19) • fair weather",
            "Weekend leisure (Sat–Sun 10–18) • warm & dry",
            "Rainy & cold days (all hours)",
            "No extra preset (use sidebar filters only)",
        ],
        index=0,
    )
    bullets = {
        "AM commute (Weekdays 07–10) • fair weather":
            "- **Inbound flows** to CBD/transit hubs\n- **Source→Sink pairs** for AM rebalancing\n- Stage trucks on corridors",
        "PM commute (Weekdays 16–19) • fair weather":
            "- **Outbound flows** to residential clusters\n- Evening sinks that fill up\n- Compare vs AM for asymmetry",
        "Weekend leisure (Sat–Sun 10–18) • warm & dry":
            "- **Scenic corridors** (waterfront/parks)\n- Longer casual trips\n- Event adjacency effects",
        "Rainy & cold days (all hours)":
            "- **Demand drop** vs resilient OD pairs\n- Scale staffing down; watch hotspots",
        "No extra preset (use sidebar filters only)":
            "- Use sidebar filters to define a scenario; widgets update live",
    }

    # ---------- Column availability ----------
    has_started_at = "started_at" in df_f.columns
    has_date = "date" in df_f.columns
    tempcol = next((c for c in ["avg_temp_c","avgTemp","avg_temp","temperature_c"] if c in df_f.columns), None)
    precip_bin_col = "precip_bin" if "precip_bin" in df_f.columns else None
    precip_raw = next((c for c in ["precip_mm","precipitation_mm","precip_mm_day","precipitation"] if c in df_f.columns), None)

    def fair_weather_mask(df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        if tempcol is not None:
            m &= df[tempcol].between(12, 26)
        if precip_bin_col is not None:
            m &= df[precip_bin_col].astype(str) != "High"
        elif precip_raw is not None and df[precip_raw].notna().sum() > 10:
            m &= (df[precip_raw] <= float(df[precip_raw].quantile(0.66)))
        return m

    def apply_preset_local(dfin: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        info: list[str] = []
        if dfin.empty:
            return dfin, ["No data after global (sidebar) filters."]
        dfx = dfin.copy()

        if has_started_at:
            dt = pd.to_datetime(dfx["started_at"], errors="coerce")
            dfx["_weekday"] = dt.dt.weekday
            dfx["_hour"] = dt.dt.hour
        elif has_date:
            dt = pd.to_datetime(dfx["date"], errors="coerce")
            dfx["_weekday"] = dt.dt.weekday
            dfx["_hour"] = np.nan
            info.append("No hour-level timestamps found — hour-based filters are skipped.")
        else:
            dfx["_weekday"] = np.nan
            dfx["_hour"] = np.nan
            info.append("No date/timestamp available — presets fall back to weather only.")

        m = pd.Series(True, index=dfx.index)

        if preset.startswith("AM commute"):
            if dfx["_weekday"].notna().any(): m &= dfx["_weekday"] <= 4
            else: info.append("Weekday filter skipped (no weekday).")
            if has_started_at: m &= dfx["_hour"].between(7, 10, inclusive="left")
            else: info.append("Hour window (07–10) skipped.")
            fm = fair_weather_mask(dfx); m &= fm if fm.any() else m

        elif preset.startswith("PM commute"):
            if dfx["_weekday"].notna().any(): m &= dfx["_weekday"] <= 4
            else: info.append("Weekday filter skipped (no weekday).")
            if has_started_at: m &= dfx["_hour"].between(16, 19, inclusive="left")
            else: info.append("Hour window (16–19) skipped.")
            fm = fair_weather_mask(dfx); m &= fm if fm.any() else m

        elif preset.startswith("Weekend leisure"):
            if dfx["_weekday"].notna().any(): m &= dfx["_weekday"] >= 5
            else: info.append("Weekend filter skipped (no weekday).")
            if has_started_at: m &= dfx["_hour"].between(10, 18, inclusive="both")
            else: info.append("Hour window (10–18) skipped.")
            if tempcol is not None: m &= dfx[tempcol].between(18, 28)
            if precip_bin_col is not None: m &= dfx[precip_bin_col].astype(str) == "Low"

        elif preset.startswith("Rainy & cold"):
            if tempcol is not None: m &= dfx[tempcol] < 8
            if precip_bin_col is not None: m &= dfx[precip_bin_col].astype(str) == "High"
            elif precip_raw is not None and dfx[precip_raw].notna().sum() > 10:
                m &= dfx[precip_raw] >= float(dfx[precip_raw].quantile(0.66))

        out = dfx[m] if m.any() else dfx.iloc[0:0].copy()
        if out.empty:
            info.append("Preset filters produced 0 rows. Try another preset or relax global filters.")
        return out, info

    dflow, preset_notes = apply_preset_local(df_f)
    if preset_notes:
        st.info("Preset notes:\n- " + "\n- ".join(preset_notes))

    # ---------- Render Kepler with the selected map preset ----------
if _KEPLER_OK:
    # Live interactive Kepler rendering
    data_dict = {"trips": df_f}  # must match the dataId normalized in the config
    m = KeplerGl(height=900, data=data_dict, config=cfg)
    keplergl_static(m)

else:
    # Fallback: static HTML export (so app doesn't crash on missing package)
    st.warning(
        "⚠️ Kepler live rendering is unavailable because the required packages aren't installed.\n\n"
        f"Import error: `{_KEPLER_ERR}`\n\n"
        "To enable presets, add these to your `app/requirements.txt`:\n\n"
        "```\n"
        "keplergl==0.3.2\n"
        "streamlit-keplergl==0.3.1\n"
        "```\n\n"
        "For now, showing the static exported map below."
    )

    path_to_html = None
    for p in MAP_HTMLS:
        if p.exists():
            path_to_html = p
            break

    if path_to_html:
        try:
            html_data = Path(path_to_html).read_text(encoding="utf-8")
            st.components.v1.html(html_data, height=900, scrolling=True)
        except Exception as e:
            st.error(f"Failed to load map HTML: {e}")
    else:
        st.info(
            "No Kepler.gl HTML found.\n\nExpected at one of:\n"
            "- `reports/map/citibike_trip_flows_2022.html`\n"
            "- `reports/map/NYC_Bike_Trips_Aggregated.html`"
        )

    # ---------- Context ----------
    st.markdown("### 🎯 What to look for")
    st.markdown(bullets.get(preset, ""))
    st.markdown("---")

    # ---------- Branch: do we have end_station_name? ----------
    have_od = {"start_station_name", "end_station_name"}.issubset(dflow.columns)
    if not have_od:
        st.warning(
            "This sample file lacks `end_station_name`. OD Sankey and station imbalance need both "
            "`start_station_name` **and** `end_station_name`.\n\n"
            "👉 **Workaround shown below:** Origin Hotspots & AM/PM origin load. "
            "To unlock full OD analysis, regenerate the reduced CSV keeping at least:\n"
            "`ride_id, started_at, start_station_name, end_station_name, date, avg_temp_c, member_type, season`"
        )

        if ("started_at" in dflow.columns) and not dflow.empty:
            st.subheader("⏰ AM vs PM origin load (weekdays)")
            dtx = pd.to_datetime(dflow["started_at"], errors="coerce")
            weekdays = dtx.dt.weekday <= 4
            hours = dtx.dt.hour
            bands = pd.cut(
                hours,
                bins=[0,7,10,16,19,24],
                labels=["pre-AM","AM(7-10)","midday","PM(16-19)","late"],
                right=False
            )
            tbl = dflow.loc[weekdays].assign(band=bands).groupby("band").size().reindex(
                ["pre-AM","AM(7-10)","midday","PM(16-19)","late"]
            ).fillna(0).rename("rides").reset_index()

            fig_b = go.Figure(go.Bar(x=tbl["band"], y=tbl["rides"], text=tbl["rides"], textposition="outside"))
            fig_b.update_layout(height=420, title="Weekday origin load by time band")
            fig_b.update_yaxes(title="Rides (count)")
            st.plotly_chart(fig_b, use_container_width=True)

        st.caption("Once you include `end_station_name`, this page will show OD Sankey and Source/Sink imbalance automatically.")
        st.stop()

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

elif page == "What-if: Temp → Rides":
    st.header("🧪 What-if: temperature impact on rides")
    if daily_all is None or daily_all.empty or "avg_temp_c" not in daily_all.columns:
        st.info("Requires a daily table with `avg_temp_c`.")
    else:
        x = daily_all["avg_temp_c"]; y = daily_all["bike_rides_daily"]
        a, b, yhat = linear_fit(x, y)
        tmin, tmax = float(np.nanmin(x)), float(np.nanmax(x))
        t = st.slider("Forecast average temperature (°C)", tmin, tmax, float(np.nanmedian(x)), 0.5)
        pred = yhat([t])[0] if a is not None else np.nan

        col1, col2, col3 = st.columns(3)
        col1.metric("Model", f"rides ≈ {a:.1f}×temp + {b:.0f}" if a is not None else "—")
        col2.metric("Temp (°C)", f"{t:.1f}")
        col3.metric("Predicted rides", kfmt(pred))

        if daily_f is not None and not daily_f.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_f["avg_temp_c"], y=daily_f["bike_rides_daily"],
                                     mode="markers", name="Filtered days"))
            xs = np.linspace(tmin, tmax, 100)
            fig.add_trace(go.Scatter(x=xs, y=yhat(xs), mode="lines", name="Fitted (all data)"))
            fig.update_layout(height=520)
            friendly_axis(fig, x="Average temperature (°C)", y="Bike rides (count)",
                          title="Observed (filtered) vs fitted line (all days)")
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Lightweight linear fit for planning; enrich later with rain, wind, weekday effects.")

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
