# Streamlit App: NYC Citi Bike Dashboard
# Author: Moein Mellat, 2025-10-21
# Purpose: Visualize and analyze NYC Citi Bike 2022 data with interactive controls.

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
import unicodedata
import hashlib
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
# Compact number formatter
def kfmt(x):
    """Format numbers like 1200 → 1.2K, 5_000_000 → 5.0M."""
    try:
        x = float(x)
    except Exception:
        return "—"
    units = ["", "K", "M", "B", "T"]
    for u in units:
        if abs(x) < 1000 or u == units[-1]:
            return f"{x:,.0f}{u}" if u == "" else f"{x:.1f}{u}"
        x /= 1000.0

# Shorten long strings with ellipsis
def shorten_name(s: str, max_len: int = 22) -> str:
    """Trim strings longer than max_len, e.g. 'Central Park West...'."""
    if not isinstance(s, str):
        return str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

# Apply readable axis titles and layout to Plotly figures
def friendly_axis(fig, x=None, y=None, title=None, colorbar=None):
    """Add friendly axis and colorbar titles to Plotly figures."""
    if x:
        fig.update_xaxes(title_text=x)
    if y:
        fig.update_yaxes(title_text=y)
    if title:
        fig.update_layout(title=title)
    if colorbar and hasattr(fig, "data"):
        for tr in fig.data:
            if hasattr(tr, "colorbar") and tr.colorbar:
                tr.colorbar.title = colorbar

# Compute correlation safely (handles NaNs, small samples)
def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    """Return Pearson correlation or None if insufficient overlap."""
    a, b = a.dropna(), b.dropna()
    j = a.index.intersection(b.index)
    if len(j) < 3:
        return None
    c = np.corrcoef(a.loc[j], b.loc[j])[0, 1]
    return float(c)

# Simple linear regression fit (returns slope, intercept, and predictor)
def linear_fit(x: pd.Series, y: pd.Series):
    """Return slope, intercept, and a prediction function y = a*x + b."""
    valid = (~x.isna()) & (~y.isna())
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None, None, lambda z: np.nan
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b), lambda z: a * np.asarray(z) + b

# Display project cover image with fallback handling
def show_cover(cover_path: Path):
    """Show the dashboard cover image with caption; warn if missing."""
    if not cover_path.exists():
        st.warning("⚠️ Cover image not found at reports/cover_bike.webp")
        return
    caption = (
        "🚲 Exploring one year of bike sharing in New York City. "
        "Photo © citibikenyc.com"
    )
    try:
        st.image(str(cover_path), use_container_width=True, caption=caption)
    except TypeError:  # for older Streamlit versions
        st.image(str(cover_path), use_column_width=True, caption=caption)

# Bin hour values (0–23) into uniform time buckets
def _bin_hour(h: pd.Series, bin_size: int) -> pd.Series:
    """Group hours into bins of given size (e.g., 2h → 0,2,4,...,22)."""
    b = (h // bin_size) * bin_size
    return b.clip(0, 23)

# Map weekday index (0–6) to short names
def _weekday_name(idx: pd.Series) -> pd.Series:
    """Convert weekday numbers to labels: 0→Mon, 6→Sun."""
    return idx.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})

# Build a 7 × (24/hour_bin) heat grid with optional scaling
def _make_heat_grid(
    df: pd.DataFrame,
    hour_col: str = "hour",
    weekday_col: str = "weekday",
    hour_bin: int = 1,
    scale: str = "Absolute",          # {"Absolute","Row %","Column %","Z-score"}
    value_col: str | None = None,     # if None → count rows; else aggregate this column
    agg: str = "sum",                 # used when value_col is not None
    label_weekdays: bool = False      # map 0..6 → Mon..Sun at the end
) -> pd.DataFrame:
    """
    Return a heat grid of shape (7, 24/hour_bin) from df using hour & weekday columns.

    - Accepts numeric weekday (0=Mon..6=Sun) OR datetime-like in `weekday_col`.
    - If `value_col` is None → counts trips; else aggregates `value_col` with `agg`.
    - `scale`: "Absolute" (int), "Row %", "Column %", "Z-score" (rowwise).
    """
    # Basic checks
    if hour_bin < 1 or 24 % hour_bin != 0:
        # bad bin size → nothing to do
        return pd.DataFrame()

    if hour_col not in df.columns or weekday_col not in df.columns:
        return pd.DataFrame()

    d = df[[hour_col, weekday_col] + ([value_col] if value_col else [])].dropna().copy()

    # Extract weekday if datetime-like provided
    if pd.api.types.is_datetime64_any_dtype(d[weekday_col]) or hasattr(d[weekday_col], "dt"):
        d[weekday_col] = d[weekday_col].dt.weekday

    # Enforce integer hour within 0..23 and bin
    h = pd.to_numeric(d[hour_col], errors="coerce").fillna(-1).astype(int)
    d[hour_col] = _bin_hour(h.clip(0, 23), hour_bin)

    # Enforce weekday 0..6 and categorical ordering
    wd = pd.to_numeric(d[weekday_col], errors="coerce").astype("Int64")
    d[weekday_col] = wd.clip(lower=0, upper=6).astype("Int64")
    d = d.dropna(subset=[weekday_col])  # in case of all-NaN after coercion

    # Group → either count or aggregate a value column
    if value_col is None:
        g = d.groupby([weekday_col, hour_col], observed=True).size().rename("val").reset_index()
    else:
        g = (
            d.groupby([weekday_col, hour_col], observed=True)[value_col]
              .agg(agg)
              .rename("val")
              .reset_index()
        )

    # Full grid with all weekdays/hours present
    hours = list(range(0, 24, hour_bin))
    weekdays = list(range(0, 7))
    mat = (
        g.pivot(index=weekday_col, columns=hour_col, values="val")
         .reindex(index=weekdays, columns=hours, fill_value=0)
    )

    if scale == "Absolute":
        out = mat.astype(int)
    elif scale == "Row %":
        denom = mat.sum(axis=1).replace(0, np.nan)
        out = (mat.div(denom, axis=0) * 100).fillna(0)
    elif scale == "Column %":
        denom = mat.sum(axis=0).replace(0, np.nan)
        out = (mat.div(denom, axis=1) * 100).fillna(0)
    elif scale == "Z-score":
        m = mat.mean(axis=1)
        s = mat.std(axis=1).replace(0, np.nan)
        out = ((mat.sub(m, axis=0)).div(s, axis=0)).fillna(0)
    else:
        # Unknown scale → return absolute
        out = mat.astype(int)

    if label_weekdays:
        out.index = _weekday_name(pd.Series(out.index)).values

    return out

# ── Shared helper: robust quadratic optimum (used by Weather vs Usage page)
if "_optimal_temp_quadratic" not in globals():
    def _optimal_temp_quadratic(
        daily: pd.DataFrame | None,
        tcol: str = "avg_temp_c",
        ycol: str = "bike_rides_daily",
        tmin: float = -5.0,
        tmax: float = 35.0,
    ) -> float | None:
        """
        Fit Y ~ a2*(T - Tm)^2 + a1*(T - Tm) + a0 (centered for numeric stability).
        Return vertex temp in °C if concave and inside [tmin, tmax]; else None.
        """
        if daily is None or daily.empty or not {tcol, ycol}.issubset(daily.columns):
            return None

        d = daily[[tcol, ycol]].dropna().copy()
        d = d[(d[tcol] >= tmin) & (d[tcol] <= tmax)]
        if len(d) < 20:
            return None

        T = d[tcol].to_numpy(dtype=float)
        Y = d[ycol].to_numpy(dtype=float)

        Tm = T.mean()
        Tc = T - Tm

        a2, a1, a0 = np.polyfit(Tc, Y, 2)  # concave if a2 < 0
        if a2 >= 0:
            return None

        Tc_opt = -a1 / (2 * a2)
        T_opt = float(Tc_opt + Tm)

        if T_opt < tmin or T_opt > tmax:
            return None
        return T_opt

# Apply moving-average smoothing across hourly values per weekday
def _smooth_by_hour(mat: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Smooth each weekday's hourly series with a centered moving average (window=k)."""
    if mat.empty or k <= 1:
        return mat

    # Ensure odd integer window
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1

    out = mat.copy()
    half = max(1, k // 2)

    for idx, row in out.iterrows():
        smoothed = row.rolling(window=k, center=True, min_periods=half).mean()
        out.loc[idx] = smoothed.values

    return out
    
# Add annotation for peak cell in heatmap (max value)
def _add_peak_annotation(fig, mat: pd.DataFrame, title_suffix: str = ""):
    """Annotate the heatmap with the weekday–hour peak cell and value."""
    if mat.empty:
        return fig

    # Identify position of maximum value
    try:
        r, c = np.unravel_index(np.nanargmax(mat.values), mat.shape)
    except ValueError:
        return fig  # handle all-NaN case gracefully

    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wk = weekdays[r] if r < len(weekdays) else str(r)
    hr = mat.columns[c]
    val = mat.iloc[r, c]

    label = (
        f"Peak: {wk} {hr:02d}:00<br>{val:,.0f}"
        if np.isfinite(val)
        else "Peak"
    )

    fig.add_annotation(
        x=c,
        y=r,
        text=label,
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        bgcolor="rgba(0,0,0,0.6)",
        font=dict(color="white", size=11)
    )

    # Safely update the title if it exists
    if title_suffix:
        current_title = fig.layout.title.text or ""
        fig.update_layout(title=f"{current_title}{title_suffix}")

    return fig

# Filter DataFrame by time-of-day or weekday mode
def _time_slice(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return subset of data by selected mode: AM, PM, Weekday, or Weekend."""
    if "hour" not in df.columns or "weekday" not in df.columns:
        return df

    match mode:
        case "AM (06–11)":
            return df[(df["hour"] >= 6) & (df["hour"] <= 11)]
        case "PM (16–20)":
            return df[(df["hour"] >= 16) & (df["hour"] <= 20)]
        case "Weekend":
            return df[df["weekday"].isin([5, 6])]
        case "Weekday":
            return df[df["weekday"].isin([0, 1, 2, 3, 4])]
        case _:
            return df

# Build OD edge list with optional per-origin top-k, thresholds, and member split
def _build_od_edges(
    df: pd.DataFrame,
    per_origin: bool,
    topk: int,
    min_rides: int,
    drop_self_loops: bool,
    member_split: bool
) -> pd.DataFrame:
    """
    Return OD edges with ride counts (and optional median distance/duration).
    - per_origin: top-k per start station (and member type if split)
    - topk: number of edges to keep (global or per-origin)
    - min_rides: drop edges below this count
    - drop_self_loops: remove start==end
    - member_split: split groups by `member_type_display` if present
    """
    need = {"start_station_name", "end_station_name"}
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame(columns=["start_station_name", "end_station_name", "rides"])

    # Clean/normalize names; blank → NaN
    d = df.copy()
    for col in ["start_station_name", "end_station_name"]:
        d[col] = (
            d[col]
            .astype("string", copy=False)
            .str.strip()
            .replace({"": pd.NA})
        )

    gb_cols = ["start_station_name", "end_station_name"]
    if member_split and "member_type_display" in d.columns:
        gb_cols.append("member_type_display")

    # Drop rows with missing endpoints
    d = d.dropna(subset=["start_station_name", "end_station_name"])
    if d.empty:
        return pd.DataFrame(columns=gb_cols + ["rides"])

    # Count rides
    g = (
        d.groupby(gb_cols, observed=True)
         .size()
         .rename("rides")
         .reset_index()
    )

    # Optional medians (only if numeric)
    if "distance_km" in d.columns and pd.api.types.is_numeric_dtype(d["distance_km"]):
        med_dist = (
            d.groupby(gb_cols, observed=True)["distance_km"]
             .median()
             .reset_index(name="med_distance_km")
        )
        g = g.merge(med_dist, on=gb_cols, how="left")

    if "duration_min" in d.columns and pd.api.types.is_numeric_dtype(d["duration_min"]):
        med_dur = (
            d.groupby(gb_cols, observed=True)["duration_min"]
             .median()
             .reset_index(name="med_duration_min")
        )
        g = g.merge(med_dur, on=gb_cols, how="left")

    # Drop self loops
    if drop_self_loops and not g.empty:
        g = g[g["start_station_name"] != g["end_station_name"]]

    # Threshold
    g = g[g["rides"] >= int(min_rides)]
    if g.empty:
        return g

    # Deterministic order (ties broken by names)
    sort_keys = ["rides"] + gb_cols
    g = g.sort_values(sort_keys, ascending=[False] + [True] * len(gb_cols), kind="mergesort")

    # Top-k selection
    topk = int(topk)
    if topk > 0:
        if per_origin:
            by = ["start_station_name"]
            if member_split and "member_type_display" in g.columns:
                by.append("member_type_display")
            # Use nlargest per group (fast + stable with mergesort pre-sort)
            g = (
                g.groupby(by, group_keys=False, observed=True)
                 .apply(lambda x: x.nlargest(topk, columns="rides"))
                 .reset_index(drop=True)
            )
        else:
            g = g.head(topk)

    # Keep integer dtype for rides
    g["rides"] = g["rides"].astype("int64", copy=False)
    return g.reset_index(drop=True)

# Cache OD edges; cache key depends only on the columns that matter
@st.cache_data(
    show_spinner=False,
    ttl=3600,  # 1h
    hash_funcs={
        pd.DataFrame: lambda d: hashlib.sha1(
            pd.util.hash_pandas_object(
                d.reindex(columns=[
                    "start_station_name", "end_station_name",
                    "member_type_display", "distance_km", "duration_min"
                ], fill_value=None),
                index=True
            ).values.tobytes()
        ).hexdigest()
    },
)
def _cached_edges(
    df: pd.DataFrame,
    per_origin: bool,
    topk: int,
    min_rides: int,
    drop_self_loops: bool,
    member_split: bool
) -> pd.DataFrame:
    """Cached wrapper around _build_od_edges with safe fallbacks."""
    try:
        if df is None or df.empty:
            return pd.DataFrame(columns=["start_station_name","end_station_name","rides"])
        # normalize numerics to avoid cache key fragmentation
        topk = int(topk)
        min_rides = int(min_rides)
        return _build_od_edges(df, per_origin, topk, min_rides, drop_self_loops, member_split)
    except Exception as e:
        st.warning(f"Edge build failed: {e}")
        return pd.DataFrame(columns=["start_station_name","end_station_name","rides"])

# Build OD matrix (rows=start, cols=end) from edge list
def _matrix_from_edges(
    edges: pd.DataFrame,
    member_split: bool,
    topn_rows: int | None = None,
    topn_cols: int | None = None
) -> pd.DataFrame:
    """
    Return an OD matrix of rides. If member_split is True and the column exists,
    edges are summed across member types before pivoting.

    Options:
      - topn_rows / topn_cols: keep only the top-N rows/cols by total (for readability).
    """
    if edges is None or edges.empty:
        return pd.DataFrame()

    need = {"start_station_name", "end_station_name", "rides"}
    if not need.issubset(edges.columns):
        return pd.DataFrame()

    base = edges.copy()

    # Clean station names; blank → NaN
    for col in ["start_station_name", "end_station_name"]:
        base[col] = (
            base[col].astype("string", copy=False)
                     .str.strip()
                     .replace({"": pd.NA})
        )
    base = base.dropna(subset=["start_station_name", "end_station_name"])

    # Collapse member types if requested
    gb_cols = ["start_station_name", "end_station_name"]
    if member_split and "member_type_display" in base.columns:
        base = (
            base.groupby(gb_cols, observed=True)["rides"]
                .sum()
                .reset_index()
        )

    # Pivot (robust to accidental duplicates)
    mat = (
        base.pivot_table(
            index="start_station_name",
            columns="end_station_name",
            values="rides",
            aggfunc="sum",
            fill_value=0,
            observed=True
        )
    )

    if mat.empty:
        return mat

    # Deterministic ordering: by totals desc, then alphabetical
    row_tot = mat.sum(axis=1)
    col_tot = mat.sum(axis=0)

    mat = (
        mat.loc[
            row_tot.sort_values(ascending=False)
                   .index.sort_values(kind="mergesort"),  # stable secondary
            col_tot.sort_values(ascending=False)
                   .index.sort_values(kind="mergesort")
        ]
    )

    # Optional top-N trimming for readability
    if isinstance(topn_rows, int) and topn_rows > 0:
        mat = mat.iloc[:topn_rows, :]
    if isinstance(topn_cols, int) and topn_cols > 0:
        mat = mat.iloc[:, :topn_cols]

    return mat

# One-hot encode a categorical/boolean Series with guaranteed output
def _one_hot(s: pd.Series, prefix: str) -> pd.DataFrame:
    """Return one-hot encoded DataFrame (drop_first=True) with a stable fallback."""
    if s is None or s.empty:
        return pd.DataFrame({f"{prefix}_0": []})

    # Coerce to int where possible; fallback to string to handle unexpected values
    try:
        s = s.astype("int64")
    except (ValueError, TypeError):
        s = s.astype("string")

    d = pd.get_dummies(s, prefix=prefix, drop_first=True)

    # Ensure stability: always at least one column
    if d.shape[1] == 0:
        d[f"{prefix}_0"] = 0

    return d

def deweather_fit_predict(
    df_in: pd.DataFrame,
    ridge_alpha: float = 0.0,     # 0 = OLS; >0 = Ridge
    clip_nonneg: bool = True      # clamp yhat to >= 0
):
    """
    Fit an 'expected rides' model where bike_rides_daily is known; predict for all rows.

    Returns:
        (yhat_all: pd.Series, resid_pct: pd.Series, coefs: pd.Series, r2: float | None)
        or None if not enough data.
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

    # ---------- feature builder (consistent columns across train/predict) ----------
    def _build_X(frame: pd.DataFrame):
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

        # Temp + quadratic
        tt = pd.to_numeric(frame["avg_temp_c"], errors="coerce")
        add_col(tt, "temp_c")
        add_col(tt**2, "temp_c_sq")

        # Optional weather covariates
        if "precip_mm" in frame.columns:
            add_col(frame["precip_mm"], "precip_mm")
        if "wind_kph" in frame.columns:
            add_col(frame["wind_kph"], "wind_kph")
        if "wet_day" in frame.columns:
            add_col(frame["wet_day"], "wet_day")  # expect 0/1

        # Weekday dummies (fixed columns wd_0..wd_6)
        wd = None
        if "date" in frame.columns:
            if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
                # try to parse if not datetime
                frame = frame.copy()
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            if pd.api.types.is_datetime64_any_dtype(frame["date"]):
                wd = frame["date"].dt.weekday

        if wd is not None:
            wd = wd.fillna(0).astype(int).clip(0, 6)
            W = pd.get_dummies(wd, prefix="wd").astype(float)
            # ensure fixed set of weekday columns
            fixed_cols = [f"wd_{i}" for i in range(7)]
            W = W.reindex(columns=fixed_cols, fill_value=0.0)
            if len(W.columns):
                parts.append(W.to_numpy(dtype=float))
                names.extend(list(W.columns))

        X = np.hstack(parts).astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, names

    # --- Build train design matrix ---
    X_train, names = _build_X(df.loc[train_mask])
    y_train = y.loc[train_mask].to_numpy(dtype=float)

    good = np.isfinite(y_train).flatten() & np.isfinite(X_train).all(axis=1)
    if good.sum() < 10:
        return None
    X_train = X_train[good]
    y_train = y_train[good]

    # --- Fit (OLS or Ridge) ---
    if ridge_alpha and ridge_alpha > 0:
        # Ridge: beta = (X'X + αI)^(-1) X'y (don't penalize intercept)
        XtX = X_train.T @ X_train
        I = np.eye(XtX.shape[0])
        I[0, 0] = 0.0  # don't penalize intercept
        beta = np.linalg.pinv(XtX + ridge_alpha * I) @ (X_train.T @ y_train)
    else:
        beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    coefs = pd.Series(beta, index=names)

    # --- Predict for ALL rows with the same columns ---
    X_all, _ = _build_X(df)
    yhat_all = pd.Series(X_all @ beta, index=df.index)
    if clip_nonneg:
        yhat_all = yhat_all.clip(lower=0)

    # --- Residuals (% vs expected) only where y is known ---
    resid = pd.Series(np.nan, index=df.index, dtype=float)
    resid.loc[train_mask] = y.loc[train_mask] - yhat_all.loc[train_mask]
    denom = yhat_all.replace(0, np.nan)
    resid_pct = 100.0 * (resid / denom)

    # --- Train R² for quick diagnostics ---
    try:
        y_hat_tr = (X_train @ beta)
        ss_res = float(np.sum((y_train - y_hat_tr) ** 2))
        ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    except Exception:
        r2 = None

    return yhat_all, resid_pct, coefs, r2

# Normalize text: remove accents/emojis, collapse spaces, lowercase
def _slug(s: str) -> str:
    """Return a clean lowercase ASCII slug (no accents, extra spaces, or emojis)."""
    if s is None:
        return ""
    s = str(s)
    # Strip accents and emojis
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii", "ignore")
    # Collapse multiple spaces and lowercase
    return " ".join(s.split()).lower()

# ---------- URL helpers (unified & future-proof) ----------
def _qp_get() -> dict:
    """
    Return current query params as a simple dict of scalars (not lists).
    Example: {"page": "Intro", "usertype": "All"}
    """
    try:
        qp = dict(st.query_params)  # Streamlit ≥1.31
    except Exception:
        qp = st.experimental_get_query_params()
    # unwrap one-element lists to scalars
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in qp.items()}

def _qp_set(qp: dict | None = None, **kv) -> None:
    """
    Accepts either a dict (_qp_set({"page":"Intro"})) or kwargs (_qp_set(page="Intro")).
    Merges both; kwargs win on key conflicts. Writes scalars to URL.
    """
    payload = {}
    if qp:
        # flatten one-element lists to scalars
        payload.update({k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in qp.items()})
    if kv:
        payload.update(kv)

    try:
        st.query_params.clear()
        st.query_params.update({k: "" if v is None else str(v) for k, v in payload.items()})
    except Exception:
        # Older Streamlit fallback
        st.experimental_set_query_params(**{k: "" if v is None else str(v) for k, v in payload.items()})

def _qp_clear() -> None:
    """Remove all query params."""
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()

# ──────────────── UI Helpers (Hero Panel + KPI Cards) ────────────────

def kpi_card(title: str, value: str, sub: str = "", icon: str = "📊"):
    """Render a stylized KPI card with title, main value, and optional subtitle."""
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


def render_hero_panel(
    title: str = "NYC Citi Bike — Strategy Dashboard",
    subtitle: str = "Seasonality • Weather–demand correlation • Station intelligence • Time patterns"
):
    """Render the top hero panel with title and subtitle."""
    st.markdown(
        f"""
        <style>
        .hero-panel {{
            background: linear-gradient(180deg, rgba(18,22,28,0.95) 0%, rgba(18,22,28,0.86) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 22px 24px;
            box-shadow: 0 8px 18px rgba(0,0,0,0.28);
            text-align: center;
        }}
        .hero-title {{
            color: #f8fafc;
            font-size: clamp(1.4rem, 1.2rem + 1.6vw, 2.3rem);
            font-weight: 800;
            letter-spacing: .2px;
            margin: 2px 0 6px 0;
        }}
        .hero-sub {{
            color: #cbd5e1;
            font-size: clamp(.85rem, .8rem + .3vw, 1.0rem);
            margin: 0;
        }}
        .kpi-card {{
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
        }}
        .kpi-title {{
            font-size: .95rem;
            color: #cbd5e1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 6px;
            letter-spacing: .2px;
        }}
        .kpi-value {{
            font-size: clamp(1.25rem, 1.0rem + 1.2vw, 2.0rem);
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.08;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .kpi-sub {{
            font-size: .90rem;
            color: #94a3b8;
            margin-top: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .element-container img {{
            border-radius: 16px;
        }}
        </style>
        <div class="hero-panel">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────── Data loading & features ────────────────────

# Small, readable constants
WEEKDAY_MAP = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
SEASON_MAP  = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
               6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}

def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance (km). Inputs can be scalars/arrays/Series."""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = (lat2 - lat1), (lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def load_data(path: Path, weather_path: Path | None = Path("data/processed/nyc_weather_2022_daily_full.csv")) -> pd.DataFrame:
    """Load trips CSV, parse timestamps, enrich weather, and derive core features."""
    df = pd.read_csv(path, low_memory=False)

    # --- Timestamps
    for col in ("date", "started_at", "ended_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "date" not in df.columns and "started_at" in df.columns:
        df["date"] = pd.to_datetime(df["started_at"], errors="coerce").dt.floor("D")

    # --- Weather (per-day) join on 'date'
    if weather_path and weather_path.exists() and "date" in df.columns:
        wx = pd.read_csv(weather_path, parse_dates=["date"])
        keep = [c for c in wx.columns if c in {
            "date","avg_temp_c","tmin_c","tmax_c","precip_mm","snow_mm","snow_depth_mm",
            "wind_mps","wind_kph","gust_mps","gust_kph","wet_day","precip_bin","wind_bin"
        } or c.startswith("wt")]
        df = df.merge(wx[keep].copy(), on="date", how="left")

    # --- Season (if missing)
    if "season" not in df.columns and "date" in df.columns:
        df["season"] = df["date"].dt.month.map(SEASON_MAP).astype("category")

    # --- Trip metrics
    if {"started_at","ended_at"}.issubset(df.columns):
        dur_min = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
        df["duration_min"] = pd.to_numeric(dur_min, errors="coerce").clip(lower=0)

    if {"start_lat","start_lng","end_lat","end_lng"}.issubset(df.columns):
        df["distance_km"] = _haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]).astype(float)

    if "duration_min" in df.columns and "distance_km" in df.columns:
        # Guard against divide-by-zero and hard cap to plausible city speeds
        spd = df["distance_km"] / (df["duration_min"] / 60.0)
        df["speed_kmh"] = pd.to_numeric(spd, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(upper=60)

    # --- Temporal fields (fast accessors)
    if "started_at" in df.columns:
        ts = df["started_at"]
        df["hour"]    = ts.dt.hour
        df["weekday"] = ts.dt.weekday
        df["month"]   = ts.dt.to_period("M").dt.to_timestamp()

    # --- Categories (compact memory + faster groupby)
    for c in ("start_station_name","end_station_name","member_type","rideable_type","season"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    # --- Pretty member labels
    if "member_type" in df.columns:
        # MEMBER_LABELS is expected to be defined elsewhere; fall back to title-case
        df["member_type_display"] = (
            df["member_type"].astype(str).map(MEMBER_LABELS).fillna(df["member_type"].astype(str).str.title())
        ).astype("category")

    return df


def ensure_daily(df: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate trip rows to daily table with weather attached."""
    if df is None or df.empty or "date" not in df.columns:
        return None

    out = df.copy()

    # Ensure avg_temp_c exists if min/max present
    if "avg_temp_c" not in out.columns and {"tmin_c","tmax_c"}.issubset(out.columns):
        out["avg_temp_c"] = (pd.to_numeric(out["tmin_c"], errors="coerce") + pd.to_numeric(out["tmax_c"], errors="coerce")) / 2.0

    # Base: rides/day
    daily = out.groupby("date", as_index=False).size().rename(columns={"size": "bike_rides_daily"})

    # Attach daily weather (means, max for booleans, mode for bins)
    agg: dict[str, str | callable] = {}
    for c in ("avg_temp_c","tmin_c","tmax_c","precip_mm","snow_mm","wind_kph","gust_kph"):
        if c in out.columns: agg[c] = "mean"
    if "wet_day" in out.columns:    agg["wet_day"]  = "max"
    if "precip_bin" in out.columns: agg["precip_bin"] = lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
    if "wind_bin" in out.columns:   agg["wind_bin"]   = lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan

    if agg:
        w = out.groupby("date", as_index=False).agg(agg)
        daily = daily.merge(w, on="date", how="left")

    # Season (mode per day)
    if "season" in out.columns:
        s = out.groupby("date")["season"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan).reset_index()
        daily = daily.merge(s, on="date", how="left")

    return daily.sort_values("date")


def apply_filters(
    df: pd.DataFrame,
    daterange: tuple[pd.Timestamp, pd.Timestamp] | None,
    seasons: list[str] | None,
    usertype: str | None,
    temp_range: tuple[float, float] | None,
    hour_range: tuple[int, int] | None = None,
    weekdays: list[str] | None = None,
) -> pd.DataFrame:
    """Apply common dashboard filters; returns a filtered copy."""
    out = df.copy()

    if daterange and "date" in out.columns:
        d0, d1 = map(pd.to_datetime, daterange)
        out = out[(out["date"] >= d0) & (out["date"] <= d1)]

    if seasons and "season" in out.columns:
        out = out[out["season"].isin(seasons)]

    if usertype and usertype != "All" and "member_type" in out.columns:
        out = out[out["member_type"].astype(str) == usertype]

    if temp_range and "avg_temp_c" in out.columns:
        lo, hi = temp_range
        out = out[(out["avg_temp_c"] >= lo) & (out["avg_temp_c"] <= hi)]

    if hour_range and "hour" in out.columns:
        lo, hi = hour_range
        out = out[(out["hour"] >= lo) & (out["hour"] <= hi)]

    if weekdays and "weekday" in out.columns:
        idxs = [WEEKDAY_MAP[w] for w in weekdays if w in WEEKDAY_MAP]
        out = out[out["weekday"].isin(idxs)]

    return out


def compute_core_kpis(df_f: pd.DataFrame, daily_f: pd.DataFrame | None) -> dict:
    """Return small set of KPIs used on the hero cards."""
    total_rides = int(len(df_f))
    avg_day = float(daily_f["bike_rides_daily"].mean()) if daily_f is not None and not daily_f.empty else None
    corr_tr = None
    if daily_f is not None and {"date","bike_rides_daily","avg_temp_c"}.issubset(daily_f.columns):
        s1 = daily_f.set_index("date")["bike_rides_daily"]
        s2 = daily_f.set_index("date")["avg_temp_c"]
        corr_tr = safe_corr(s1, s2)  # uses your earlier helper
    return dict(total_rides=total_rides, avg_day=avg_day, corr_tr=corr_tr)


# ───────── Robust plotting helpers ─────────

def quantile_bounds(s: pd.Series, lo: float = 0.01, hi: float = 0.995) -> tuple[float, float]:
    """Return (lo, hi) empirical quantiles; safe on empty or non-numeric Series."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return (float("nan"), float("nan"))
    ql, qh = s.quantile(lo), s.quantile(hi)
    return float(ql), float(qh)

def inlier_mask(df: pd.DataFrame, col: str, lo: float = 0.01, hi: float = 0.995) -> pd.Series:
    """Boolean mask for rows within [lo, hi] quantiles of column `col`."""
    if col not in df.columns or df.empty:
        return pd.Series([True] * len(df), index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    ql, qh = quantile_bounds(s, lo, hi)
    if not np.isfinite(ql) or not np.isfinite(qh):
        return pd.Series([True] * len(df), index=df.index)
    return (s >= ql) & (s <= qh)

# ────────────────────────────── Sidebar / Data ─────────────────────────────

st.sidebar.header("⚙️ Controls")

# ---------- Data presence ----------
if not DATA_PATH.exists():
    st.sidebar.error(f"Missing data: {DATA_PATH}")
    st.error("Data file not found. Create the ≤25MB sample CSV at data/processed/reduced_citibike_2022.csv.")
    st.stop()

# Load
df = load_data(DATA_PATH, weather_path=Path("data/processed/nyc_weather_2022_daily_full.csv"))

# ---------- Presets (top, to encourage use) ----------
with st.sidebar.expander("⚡ Quick presets", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✨ Commuter preset", key="btn_preset_commuter", help="Weekdays 06–10 & 16–20, mild temps, members"):
            _qp_set(page="Weekday × Hour Heatmap")
            if "weekday" in df.columns:
                _qp_set(weekday="Mon,Tue,Wed,Thu,Fri")
            if "hour" in df.columns:
                _qp_set(hour0=6, hour1=20)
            if "avg_temp_c" in df.columns:
                tmin, tmax = float(np.nanmin(df["avg_temp_c"])), float(np.nanmax(df["avg_temp_c"]))
                _qp_set(temp=f"{max(tmin,5)}:{min(tmax,25)}")
            if "member_type" in df.columns:
                _qp_set(usertype="member")
            st.rerun()
    with c2:
        if st.button("🌧️ Rainy-day preset", key="btn_preset_rain", help="Focus only on wet days (uses wet_day flag)"):
            _qp_set(page="Weather vs Bike Usage", wet=1)
            st.rerun()

# ---------- Actions ----------
with st.sidebar.expander("🛠 Actions", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("♻️ Reset filters", key="btn_reset", help="Clear URL state and sidebar selections"):
            st.cache_data.clear()
            _qp_clear()
            st.rerun()
    with c2:
        if st.button("🔗 Copy current link", key="btn_copy", help="Your current filters are already reflected in the URL"):
            st.sidebar.success("URL updated — copy it from your browser bar to share this exact view.")

# ---------- Read URL page seed ----------
PAGES = [
    "Intro",                                   # context, overview, scope
    "Weather vs Bike Usage",                   # external influence
    "Trip Metrics (Duration • Distance • Speed)", # behavioral metrics
    "Member vs Casual Profiles",               # segmentation
    "Pareto: Share of Rides",                  # demand concentration insight
    "Station Popularity",                      # top docks (where)
    "OD Flows — Sankey + Map",                 # trip movement overview
    "OD Matrix — Top Origins × Dest",          # detailed flow breakdown
    "Station Imbalance (In vs Out)",           # operational challenge
    "Weekday × Hour Heatmap",                  # temporal pattern synthesis
    "Time Series — Forecast & Decomposition",
    "Recommendations"                          # actions and strategy
]

# ---------- Maintain URL state ----------
_qp = _qp_get()
_qp_page = (
    _qp.get("page", [None])[0]
    if isinstance(_qp.get("page"), list)
    else _qp.get("page")
) or PAGES[0]

if _qp_page not in PAGES:
    _qp_page = PAGES[0]

page = st.sidebar.selectbox("📑 Analysis page", PAGES, index=PAGES.index(_qp_page), key="page_select")

# --- Sidebar paging controls (Prev / Next) ---
idx = PAGES.index(page)
col_prev, col_next = st.sidebar.columns(2)
prev_clicked = col_prev.button("◀ Prev", use_container_width=True)
next_clicked = col_next.button("Next ▶", use_container_width=True)

if prev_clicked or next_clicked:
    new_idx = (idx - 1) % len(PAGES) if prev_clicked else (idx + 1) % len(PAGES)
    new_page = PAGES[new_idx]

    # Preserve all existing params, just swap the page
    qp = _qp_get()
    qp["page"] = new_page
    _qp_set(qp)

    # Rerun to navigate
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

if _qp_page != page:
    qp = _qp_get()
    qp["page"] = page
    _qp_set(qp)

# ---------- Primary filters ----------
date_min = pd.to_datetime(df["date"].min()) if "date" in df.columns else None
date_max = pd.to_datetime(df["date"].max()) if "date" in df.columns else None
date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max) if (date_min is not None and date_max is not None) else None,
    help="Filter trips by trip date",
    key="date_range",
) if date_min is not None else None

seasons_all = ["Winter","Spring","Summer","Autumn"]
seasons = st.sidebar.multiselect(
    "Season(s)",
    seasons_all,
    default=seasons_all,
    help="Pick seasonal subsets",
    key="season_multi",
) if "season" in df.columns else None

usertype = None
if "member_type" in df.columns:
    raw_opts = ["All"] + sorted(df["member_type"].astype(str).unique().tolist())
    usertype = st.sidebar.selectbox(
        "User type",
        raw_opts,
        format_func=lambda v: "All" if v == "All" else MEMBER_LABELS.get(v, str(v).title()),
        help="Filter by rider category",
        key="usertype_select",
    )

hour_range = None
if "hour" in df.columns:
    hour_range = st.sidebar.slider("Hour of day", 0, 23, (6, 22), help="Trip start hour", key="hour_slider")

# ---------- Advanced filters ----------
temp_range, weekdays, rainy_only = None, None, False
with st.sidebar.expander("More filters", expanded=False):
    # Temperature
    if "avg_temp_c" in df.columns:
        tmin = float(np.nanmin(df["avg_temp_c"]))
        tmax = float(np.nanmax(df["avg_temp_c"]))
        temp_range = st.slider("Temperature (°C)", tmin, tmax, (tmin, tmax), key="temp_slider", help="Average trip-day temperature")

    # Weekdays
    if "weekday" in df.columns:
        weekday_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekdays = st.multiselect("Weekday(s)", weekday_names, default=weekday_names, key="weekday_multi")

    # Rainy day (wired to URL `wet=1`)
    if "wet_day" in df.columns:
        qp_wet = _qp.get("wet", ["0"])
        qp_wet = qp_wet[0] if isinstance(qp_wet, list) else qp_wet
        rainy_only = st.checkbox("Only rainy days", value=(str(qp_wet) == "1"), key="chk_rainy")

st.sidebar.markdown("---")

# ---------- Write URL state (after widgets resolve) ----------
try:
    _qp_set(
        page=page,
        date0=str(date_range[0]) if date_range else None,
        date1=str(date_range[1]) if date_range else None,
        usertype=usertype or "All",
        hour0=hour_range[0] if hour_range else None,
        hour1=hour_range[1] if hour_range else None,
        wet=1 if rainy_only else 0,
    )
except Exception:
    pass

# ---------- Filtered data ----------
df_f = apply_filters(
    df,
    (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])) if date_range else None,
    seasons,
    usertype,
    temp_range,
    hour_range=hour_range,
    weekdays=weekdays,
)

# Apply the rainy filter here (apply_filters doesn't handle it)
if rainy_only and "wet_day" in df_f.columns:
    df_f = df_f[df_f["wet_day"] == 1]

daily_all = ensure_daily(df)
daily_f   = ensure_daily(df_f)

st.sidebar.success(f"✅ {len(df_f):,} trips match")

# ---------- Backfill trip-level weather from daily (once) ----------
def _backfill_trip_weather(df_trips: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing trip-level weather columns from daily aggregates (by date)."""
    if daily_df is None or daily_df.empty or "date" not in df_trips.columns:
        return df_trips
    out = df_trips.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    lookups = {}
    for col in ["avg_temp_c","wind_kph","gust_kph","precip_mm","wet_day","precip_bin","wind_bin"]:
        if col in daily_df.columns:
            lookups[col] = daily_df.set_index("date")[col]

    for col, mapper in lookups.items():
        if col not in out.columns or out[col].notna().sum() == 0:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = out[col].where(out[col].notna(), out["date"].map(mapper))
    return out

df_f = _backfill_trip_weather(df_f, daily_all)

# ────────────────────────────── Sidebar forecasting controls ──────────────────────────────
st.sidebar.markdown("### ⏱ TS Controls")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 60, 21, 1)
show_last_n = st.sidebar.slider("Plot history window (days)", 60, 365, 180, 10)

model_name = st.sidebar.selectbox(
    "Model",
    [
        "Seasonal-Naive (t−7)",
        "Naive (t−1)",
        "7-day Moving Average",
        "SARIMAX (weekly)",
        "De-weathered + Seasonal-Naive",
    ],
    index=0
)

if model_name == "SARIMAX (weekly)" and not HAS_SARIMAX:
    st.sidebar.warning("`statsmodels` not available — SARIMAX disabled")

if model_name == "De-weathered + Seasonal-Naive" and t is None:
    st.sidebar.warning("No temperature column found — de-weathered option will fallback to Seasonal-Naive.")

# Future weather assumption for de-weathered model
if model_name == "De-weathered + Seasonal-Naive":
    fut_temp_assume = st.sidebar.selectbox(
        "Future temperature assumption",
        ["Repeat last 7 days", "Hold last day"],
        index=0
    )

# ────────────────────────────── Sidebar footer / credentials ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**👤 Moein Mellat, PhD**")
st.sidebar.markdown(
    """
    Environmental Engineer • Data Analyst  
    [🌐 GitHub](https://github.com/moeinmmm70)  
    [💼 LinkedIn](https://www.linkedin.com/in/moeinmellat/)  
    [📧 Email](mailto:moein.mellat@gmail.com)
    """
)
st.sidebar.caption("© 2025 Moein Mellat • Citi Bike NYC Weather Analytics")

# ────────────────────────────── Pages ──────────────────────────────────────
# ────────────────────────────── Page: Intro ──────────────────────────────

def _selection_summary(
    date_range: tuple | None,
    seasons: list[str] | None,
    usertype: str | None,
    hour_range: tuple[int, int] | None
) -> str:
    """Human-friendly summary of current filters."""
    # Dates
    if date_range and all(date_range):
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        date_str = f"{d0.date()} → {d1.date()}"
    else:
        date_str = "—"
    # Seasons
    all_seasons = {"Winter","Spring","Summer","Autumn"}
    if not seasons or set(seasons) == all_seasons:
        season_str = "All seasons"
    else:
        season_str = ", ".join(seasons)
    # Usertype
    user_str = "All users" if (usertype in (None, "All")) else str(usertype).title()
    # Hours
    if hour_range and len(hour_range) == 2:
        hr_str = f"{hour_range[0]:02d}:00–{hour_range[1]:02d}:00"
    else:
        hr_str = "All day"
    return f"**Selection:** {date_str} · {season_str} · {user_str} · {hr_str}"


def _weather_uplift(daily: pd.DataFrame | None) -> str:
    """
    +% uplift for 'comfy' (15–25°C) vs 'extreme' (<5 or >30°C) mean daily rides.
    Returns pretty string or '—'.
    """
    if daily is None or daily.empty or "avg_temp_c" not in daily.columns or "bike_rides_daily" not in daily.columns:
        return "—"
    d = daily.dropna(subset=["avg_temp_c", "bike_rides_daily"])
    if d.empty:
        return "—"
    comfy = d.loc[d["avg_temp_c"].between(15, 25, inclusive="both"), "bike_rides_daily"].mean()
    extreme = d.loc[~d["avg_temp_c"].between(5, 30, inclusive="both"), "bike_rides_daily"].mean()
    if pd.notnull(comfy) and pd.notnull(extreme) and extreme not in (0, np.nan):
        return f"{(comfy - extreme) / extreme * 100.0:+.0f}%"
    return "—"


def _weather_coverage(daily: pd.DataFrame | None) -> str:
    """% of days with usable temperature in the current selection."""
    if daily is None or daily.empty:
        return "—"
    if "avg_temp_c" in daily.columns:
        cov = 100.0 * daily["avg_temp_c"].notna().mean()
        return f"{cov:.0f}%"
    return "0%"


def page_intro(
    df_filtered: pd.DataFrame,
    daily_filtered: pd.DataFrame | None,
    *,
    date_range: tuple | None = None,
    seasons: list[str] | None = None,
    usertype: str | None = None,
    hour_range: tuple[int, int] | None = None,
    cover_path: Path | None = None
) -> None:
    """Intro page: hero, selection summary, KPIs, mini-trend, and overview copy."""
    render_hero_panel()

    # Selection summary
    st.caption(_selection_summary(date_range, seasons, usertype, hour_range))

    # Cover image + attribution
    if cover_path is not None:
        show_cover(cover_path)
    st.caption("⚙️ Powered by NYC Citi Bike data • 365 days • Interactive visuals")

    # KPIs
    kpis = compute_core_kpis(df_filtered, daily_filtered)  # total_rides, avg_day, corr_tr
    weather_str = _weather_uplift(daily_filtered)
    coverage_str = _weather_coverage(daily_filtered)

    # Optional: Peak season text (kept lightweight)
    peak_value, peak_sub = "—", ""
    if "season" in df_filtered.columns and daily_filtered is not None and not daily_filtered.empty:
        tmp = daily_filtered.copy()
        if "season" not in tmp.columns and "date" in df_filtered.columns:
            s_map = (
                df_filtered.groupby("date")["season"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
                .reset_index()
            )
            tmp = tmp.merge(s_map, on="date", how="left")
        if "season" in tmp.columns and "bike_rides_daily" in tmp.columns:
            m = tmp.groupby("season")["bike_rides_daily"].mean().sort_values(ascending=False)
            if len(m):
                peak_value = f"{m.index[0]}"
                peak_sub   = f"{kfmt(m.iloc[0])} avg trips"

    # KPI cards row
    cA, cB, cC, cD, cE = st.columns(5)
    with cA:
        kpi_card("Total Trips", kfmt(kpis.get("total_rides", 0)), "Across all stations", "🧮")
    with cB:
        kpi_card("Daily Average", kfmt(kpis["avg_day"]) if kpis.get("avg_day") is not None else "—",
                 "Trips per day (selection)", "📅")
    with cC:
        kpi_card("Temp ↔ Rides (r)",
                 f"{kpis['corr_tr']:+.3f}" if kpis.get("corr_tr") is not None else "—",
                 "Pearson on daily agg", "🌡️")
    with cD:
        kpi_card("Weather Uplift", weather_str, "15–25°C vs extreme", "🌦️")
    with cE:
        # Swap to Peak Season if you prefer that KPI:
        # kpi_card("Peak Season", peak_value, peak_sub, "🏆")
        kpi_card("Coverage", coverage_str, "Weather data availability", "🧩")

    # Mini trend strip (14-day smoother)
    if daily_filtered is not None and not daily_filtered.empty and "avg_temp_c" in daily_filtered.columns:
        d = daily_filtered.sort_values("date").copy()
        n = 14
        for col in ["bike_rides_daily", "avg_temp_c"]:
            d[f"{col}_roll"] = d[col].rolling(n, min_periods=max(2, n // 2), center=True).mean()

        # Fallback colors if constants not defined
        rides_color = globals().get("RIDES_COLOR", "#1f77b4")
        temp_color  = globals().get("TEMP_COLOR",  "#d62728")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["bike_rides_daily_roll"].fillna(d["bike_rides_daily"]),
                name="Daily rides",
                mode="lines",
                line=dict(color=rides_color, width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["avg_temp_c_roll"].fillna(d["avg_temp_c"]),
                name="Avg temp (°C)",
                mode="lines",
                line=dict(color=temp_color, width=2, dash="dot"),
                opacity=0.9,
            ),
            secondary_y=True,
        )
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=0),
            hovermode="x unified",
            showlegend=True,
            title="Trend overview (14-day smoother)",
        )
        fig.update_yaxes(title_text="Rides", secondary_y=False)
        fig.update_yaxes(title_text="Temp (°C)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Overview copy
    st.markdown("### What you’ll find here")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Decision-ready KPIs**\n\nTotals, avg/day, and a defensible temp↔rides correlation.")
    c2.info("**Weather impact**\n\nTrend, scatter with fit, and comfort bands for clear takeaways.")
    c3.info("**Station intelligence**\n\nTop stations, OD flows (Sankey/Matrix), and Pareto focus.")
    c4.info("**Time patterns**\n\nWeekday×Hour heatmap + marginal profiles for staffing windows.")
    st.caption("Use the sidebar filters; every view updates live.")

# ────────────────────────────── Page: Weather vs Bike Usage ──────────────────────────────

def _rolling_cols(d: pd.DataFrame, cols: list[str], win: int) -> pd.DataFrame:
    """Add centered rolling means for the given columns using window=win."""
    if win <= 1:
        return d
    d = d.copy()
    for c in cols:
        if c in d.columns:
            d[f"{c}_roll"] = d[c].rolling(win, min_periods=max(2, win // 2), center=True).mean()
    return d

def _first_existing(d: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from candidates or None."""
    for c in candidates:
        if c in d.columns:
            return c
    return None

def page_weather_vs_usage(daily_filtered: pd.DataFrame) -> None:
    """Daily rides vs weather: trend, scatter, distributions, lab, residual index."""
    st.header("🌤️ Daily bike rides vs weather")

    if daily_filtered is None or daily_filtered.empty:
        st.warning("Daily metrics aren’t available. Provide trip rows with `date` to aggregate.")
        return

    d = daily_filtered.sort_values("date").copy()

    # Coverage ribbon (trust signal)
    cov = d["avg_temp_c"].notna().mean() * 100 if "avg_temp_c" in d.columns else 0.0
    st.caption(f"Weather coverage in selection: **{cov:.0f}%** — metrics account for missing values.")

    # Tabs
    tab_trend, tab_scatter, tab_dist, tab_lab, tab_resid = st.tabs(
        ["📈 Trend", "🔬 Scatter", "📦 Distributions", "🧪 Lab", "📉 De-weathered Index"]
    )

    # ======== Trend tab ========
    with tab_trend:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            roll_win_label = st.selectbox("Rolling window", ["Off", "7d", "14d", "30d"], index=1)
        with c2:
            show_precip = st.checkbox("Show precipitation bars (mm)", value=("precip_mm" in d.columns))
        with c3:
            show_wind = st.checkbox("Show wind line (kph)", value=("wind_kph" in d.columns))
        with c4:
            st.caption("Use other tabs for residuals & elasticity")

        # Rolling smoother
        if roll_win_label != "Off":
            roll_n = int(roll_win_label.replace("d", ""))
            d_roll = _rolling_cols(d, ["bike_rides_daily", "avg_temp_c", "wind_kph"], roll_n)
        else:
            roll_n = 1
            d_roll = d

        # Fallback colors if constants not defined
        rides_color = globals().get("RIDES_COLOR", "#1f77b4")
        temp_color  = globals().get("TEMP_COLOR",  "#d62728")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # rides
        y_rides = d_roll.get("bike_rides_daily_roll", d["bike_rides_daily"]) if roll_n > 1 else d["bike_rides_daily"]
        fig.add_trace(
            go.Scatter(x=d["date"], y=y_rides, mode="lines", name="Daily bike rides",
                       line=dict(color=rides_color, width=2)),
            secondary_y=False,
        )

        # temp
        if "avg_temp_c" in d.columns and d["avg_temp_c"].notna().any():
            y_temp = d_roll.get("avg_temp_c_roll", d["avg_temp_c"]) if roll_n > 1 else d["avg_temp_c"]
            fig.add_trace(
                go.Scatter(x=d["date"], y=y_temp, mode="lines", name="Average temperature (°C)",
                           line=dict(color=temp_color, width=2, dash="dot"), opacity=0.9),
                secondary_y=True,
            )

        # wind
        if show_wind and "wind_kph" in d.columns and d["wind_kph"].notna().any():
            y_wind = d_roll.get("wind_kph_roll", d["wind_kph"]) if roll_n > 1 else d["wind_kph"]
            fig.add_trace(
                go.Scatter(x=d["date"], y=y_wind, mode="lines", name="Avg wind (kph)",
                           line=dict(width=1), opacity=0.5),
                secondary_y=True,
            )

        # precip
        if show_precip and "precip_mm" in d.columns and d["precip_mm"].notna().any():
            fig.add_trace(
                go.Bar(x=d["date"], y=d["precip_mm"], name="Precipitation (mm)",
                       marker_color="rgba(100,100,120,0.35)", opacity=0.4),
                secondary_y=False,
            )

        # Comfort band cue behind rides (visual only)
        if len(d):
            fig.add_hrect(
                y0=float(d["bike_rides_daily"].min()),
                y1=float(d["bike_rides_daily"].max()),
                line_width=0, fillcolor="rgba(34,197,94,0.05)", layer="below"
            )

        fig.update_layout(
            hovermode="x unified", barmode="overlay", height=560,
            title="Daily rides vs temperature, precipitation, and wind — NYC (2022)"
        )
        fig.update_yaxes(title_text="Bike rides (count)", secondary_y=False)
        y2_title = "Temperature (°C)" + (" + Wind (kph)" if show_wind and "wind_kph" in d.columns else "")
        fig.update_yaxes(title_text=y2_title, secondary_y=True)
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    # ======== Scatter tab ========
    with tab_scatter:
        c1, c2 = st.columns(2)
        with c1:
            color_scatter_by = st.selectbox("Color points by", ["None", "wet_day", "precip_bin", "wind_bin"], index=1)
        with c2:
            split_wknd = st.checkbox("Show weekday vs weekend fits", value=True,
                                     help="Highlights commute vs leisure sensitivity")

        temp_col = _first_existing(d, ["avg_temp_c", "tavg_c", "tmean_c", "tmin_c", "tmax_c"])

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
                fig2 = px.scatter(
                    scatter_df, x=temp_col, y="bike_rides_daily", color=color_arg,
                    labels=labels, opacity=0.85, trendline="ols"
                )
                fig2.update_layout(height=520, title="Rides vs Temperature")
                st.plotly_chart(fig2, use_container_width=True)

            # Elasticity & rain penalty (quadratic fit)
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
                with k1:
                    st.metric("Temp elasticity @20°C", f"{elasticity_pct:+.1f}% / °C")
                with k2:
                    st.metric("Rain penalty (wet vs dry)", f"{rain_pen:+.0f}%" if rain_pen is not None else "—")

            # Weekday vs Weekend interaction fits
            if split_wknd and {"date", "bike_rides_daily", temp_col}.issubset(d.columns):
                dd = d.dropna(subset=[temp_col, "bike_rides_daily"]).copy()
                dd["is_weekend"] = dd["date"].dt.weekday.isin([5, 6]).map({True: "Weekend", False: "Weekday"})
                figsw = px.scatter(
                    dd, x=temp_col, y="bike_rides_daily", color="is_weekend",
                    opacity=0.85, trendline="ols",
                    labels={temp_col: "Avg temp (°C)", "bike_rides_daily": "Bike rides", "is_weekend": "Day type"},
                )
                figsw.update_layout(height=480, title="Rides vs Temp — Weekday vs Weekend")
                st.plotly_chart(figsw, use_container_width=True)

    # ======== Distributions tab ========
    with tab_dist:
        st.subheader("Distribution by rainfall")
        if "precip_bin" in d.columns and d["precip_bin"].notna().any():
            fig3 = px.box(
                d, x="precip_bin", y="bike_rides_daily",
                labels={"precip_bin": "Precipitation", "bike_rides_daily": "Bike rides per day"},
                category_orders={"precip_bin": ["Low", "Medium", "High"]},
            )
            fig3.update_layout(height=420)
            st.plotly_chart(fig3, use_container_width=True)
        elif "wet_day" in d.columns:
            fig3 = px.box(
                d, x=d["wet_day"].map({0: "Dry", 1: "Wet"}), y="bike_rides_daily",
                labels={"x": "Day type", "bike_rides_daily": "Bike rides per day"},
            )
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

        need_cols = {"bike_rides_daily", "avg_temp_c"}
        if need_cols.issubset(d.columns) and len(d.dropna(subset=list(need_cols))) >= 10:
            d_fit = d.dropna(subset=["avg_temp_c", "bike_rides_daily"]).copy()

            # Linear simulator (fast, intuitive)
            try:
                from sklearn.linear_model import LinearRegression  # scoped import; optional dep
                if len(d_fit) >= 3:
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
            except Exception:
                st.info("Install scikit-learn to enable the linear simulator.")

            # ── Quadratic “comfort” curve + robust optimum
            # Plausible clamp window (intersect with observed range to avoid extrapolation)
            tmin = max(-5.0, float(d_fit["avg_temp_c"].min()))
            tmax = min(35.0, float(d_fit["avg_temp_c"].max()))

            # 1) Use robust helper to compute constrained vertex
            T_opt = _optimal_temp_quadratic(
                d_fit, tcol="avg_temp_c", ycol="bike_rides_daily", tmin=tmin, tmax=tmax
            )

            # 2) Build a numerically stable quadratic fit (centered) for plotting
            T = d_fit["avg_temp_c"].to_numpy().astype(float)
            Y = d_fit["bike_rides_daily"].to_numpy().astype(float)
            Tm = T.mean()
            Tc = T - Tm
            try:
                a2, a1, a0 = np.polyfit(Tc, Y, 2)  # Y ≈ a2*Tc^2 + a1*Tc + a0
                # Only show "optimum" if concave
                if a2 < 0 and T_opt is not None:
                    st.success(f"Estimated **optimal temperature** for demand: **{T_opt:.1f} °C** (quadratic fit)")
                else:
                    st.info("Optimal temperature not reliable for this selection.")

                # Mini visualization
                t_grid = np.linspace(tmin, tmax, 120)
                y_hat = a2*(t_grid - Tm)**2 + a1*(t_grid - Tm) + a0

                figq = go.Figure()
                figq.add_trace(go.Scatter(
                    x=T, y=Y, mode="markers", name="Observed", opacity=0.5
                ))
                figq.add_trace(go.Scatter(
                    x=t_grid, y=y_hat, mode="lines", name="Quadratic fit"
                ))
                if a2 < 0 and T_opt is not None:
                    figq.add_vline(x=T_opt, line_dash="dot")

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
                d2 = d.dropna(subset=["avg_temp_c", "bike_rides_daily"]).sort_values("date").copy()
                cut = int(len(d2) * 0.75)
                tr, te = d2.iloc[:cut], d2.iloc[cut:]

                # Baseline: rides ~ temp (linear)
                Xtr = np.c_[np.ones(len(tr)), tr["avg_temp_c"]]
                Xte = np.c_[np.ones(len(te)), te["avg_temp_c"]]
                b_lin, *_ = np.linalg.lstsq(Xtr, tr["bike_rides_daily"], rcond=None)
                pred_lin = Xte @ b_lin

                # Enriched model via helper
                yhat_te = None
                out_te = deweather_fit_predict(te)
                if out_te is not None:
                    yhat_te = out_te[0]

                def _mae(y, yhat):
                    y = np.asarray(y); yhat = np.asarray(yhat)
                    m = np.isfinite(y) & np.isfinite(yhat)
                    return float(np.mean(np.abs(y[m] - yhat[m]))) if m.any() else np.nan

                mae_lin = _mae(te["bike_rides_daily"].values, pred_lin)
                mae_enr = _mae(te["bike_rides_daily"].values, yhat_te.values if yhat_te is not None else np.full(len(te), np.nan))

                cbt1, cbt2 = st.columns(2)
                with cbt1:
                    st.metric("MAE — Linear(temp)", f"{mae_lin:,.0f}" if np.isfinite(mae_lin) else "—")
                with cbt2:
                    st.metric("MAE — Enriched", f"{mae_enr:,.0f}" if np.isfinite(mae_enr) else "—")
                if np.isfinite(mae_lin) and np.isfinite(mae_enr):
                    st.caption(f"ΔMAE: {(mae_lin - mae_enr):,.0f} better with enriched model.")
            else:
                st.info("Need ≥90 days with temperature to backtest.")

        # Bring-your-own forecast (product hook)
        st.markdown("#### 🔮 Bring your own forecast (CSV)")
        up = st.file_uploader(
            "Upload 7–14 day forecast CSV with columns: date, avg_temp_c, precip_mm, wind_kph",
            type=["csv"]
        )
        if up is not None:
            try:
                df_fc = pd.read_csv(up, parse_dates=["date"])
                df_fc = df_fc.reindex(columns=["date", "avg_temp_c", "precip_mm", "wind_kph"])

                # Share design with helper by concatenation
                d3 = d.dropna(subset=["avg_temp_c", "bike_rides_daily"]).copy()
                tmp = pd.concat(
                    [d3[["date", "avg_temp_c", "precip_mm", "wind_kph", "bike_rides_daily"]],
                     df_fc.assign(bike_rides_daily=np.nan)],
                    ignore_index=True
                )
                out_tmp = deweather_fit_predict(tmp)
                yhat_tmp = out_tmp[0] if out_tmp is not None else None

                mask_fc = tmp["bike_rides_daily"].isna()
                fc = tmp.loc[mask_fc, ["date"]].copy()
                fc["pred_rides"] = yhat_tmp.loc[mask_fc].values if yhat_tmp is not None else np.nan

                figf = px.bar(fc, x="date", y="pred_rides",
                              labels={"pred_rides": "Predicted rides", "date": "Date"})
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
            # handle 3- or 4-tuple returns
            yhat, resid_pct, coefs = out[:3]

            dd = pd.DataFrame({"date": d["date"], "resid_pct": resid_pct, "expected": yhat}).dropna()
            figr = px.line(dd, x="date", y="resid_pct",
                           labels={"resid_pct": "Residual vs expected (%)", "date": "Date"})
            figr.add_hline(y=0, line_dash="dot")
            figr.update_layout(height=420, title="De-weathered demand index (over/under performance)")
            st.plotly_chart(figr, use_container_width=True)
            
            st.metric("Avg residual (last 30 days)", f"{dd.tail(30)['resid_pct'].mean():+.1f}%")
            with st.expander("Model drivers (top |β|)"):
                st.write(coefs.sort_values(key=np.abs, ascending=False).head(10).round(3))


# ────────────────────────────── Page: Trip Metrics ──────────────────────────────

def _hist(df: pd.DataFrame, col: str, log_x: bool, robust: bool, nbins: int = 60, title: str | None = None):
    """Render a robust histogram for a single numeric column."""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    ql, qh = (s.quantile(0.01), s.quantile(0.995)) if len(s) else (None, None)
    fig = px.histogram(
        s, x=col, nbins=nbins,
        labels={col: col.replace("_", " ").title()},
        log_x=log_x,
        range_x=[float(ql), float(qh)] if robust and not log_x and np.isfinite(ql) and np.isfinite(qh) else None,
    )
    if title:
        fig.update_layout(title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _add_linear_fit(fig, x_vals, y_vals, name: str = "Linear fit"):
    """Add a simple OLS line to an existing scatter."""
    x = pd.to_numeric(x_vals, errors="coerce")
    y = pd.to_numeric(y_vals, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() >= 3 and x[ok].nunique() >= 2:
        a, b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 100)
        ys = a * xs + b
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, line=dict(dash="dash")))


def page_trip_metrics(df_filtered: pd.DataFrame) -> None:
    """Distributions and relationships for duration, distance, speed."""
    st.header("🚴 Trip metrics")

    needed = {"duration_min", "distance_km", "speed_kmh"}
    if df_filtered is None or df_filtered.empty or not needed.issubset(df_filtered.columns):
        st.info("Need duration, distance, and speed (engineered in load_data).")
        return

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
    m_dur = (inlier_mask(df_filtered, "duration_min", hi=0.995) if robust else pd.Series(True, index=df_filtered.index)) & \
            df_filtered["duration_min"].between(0.5, 240, inclusive="both")
    m_dst = (inlier_mask(df_filtered, "distance_km", hi=0.995) if robust else pd.Series(True, index=df_filtered.index)) & \
            df_filtered["distance_km"].between(0.01, 30, inclusive="both")
    m_spd = (inlier_mask(df_filtered, "speed_kmh", hi=0.995) if robust else pd.Series(True, index=df_filtered.index)) & \
            df_filtered["speed_kmh"].between(0.5, 60, inclusive="both")

    clipped_dur = int((~m_dur).sum()); clipped_dst = int((~m_dst).sum()); clipped_spd = int((~m_spd).sum())

    # ===== Histograms =====
    cA, cB, cC = st.columns(3)
    with cA:
        _hist(df_filtered.loc[m_dur], "duration_min", log_duration, robust, title="Duration (min)")
        st.caption(f"Clipped rows (duration): {clipped_dur:,}")
    with cB:
        _hist(df_filtered.loc[m_dst], "distance_km", log_distance, robust, title="Distance (km)")
        st.caption(f"Clipped rows (distance): {clipped_dst:,}")
    with cC:
        _hist(df_filtered.loc[m_spd], "speed_kmh", log_speed, robust, title="Speed (km/h)")
        st.caption(f"Clipped rows (speed): {clipped_spd:,}")

    # ===== Distance vs duration — operating envelope =====
    st.subheader("Distance vs duration — feasibility & operating envelope")
    cols_needed = ["distance_km", "duration_min", "speed_kmh"]
    inliers_mask_all = m_dst & m_dur & m_spd
    inliers = df_filtered.loc[inliers_mask_all, cols_needed].copy()

    # Sanitize numerics
    for c in cols_needed:
        inliers[c] = pd.to_numeric(inliers[c], errors="coerce")
    inliers.replace([np.inf, -np.inf], np.nan, inplace=True)
    inliers.dropna(subset=cols_needed, inplace=True)

    nmax = 35_000
    if len(inliers) > nmax:
        inliers = inliers.sample(n=nmax, random_state=13)

    x_vals = inliers["distance_km"].astype(float).tolist()
    y_vals = inliers["duration_min"].astype(float).tolist()
    c_vals = inliers["speed_kmh"].astype(float).tolist()

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scattergl(
            x=x_vals, y=y_vals, mode="markers", name="Trips",
            marker=dict(size=6, opacity=0.85, color=c_vals, colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Speed (km/h)")),
            hovertemplate="Distance: %{x:.2f} km<br>Duration: %{y:.1f} min<br>Speed: %{marker.color:.1f} km/h<extra></extra>",
        )
    )

    # Faint outliers
    outliers = df_filtered.loc[~inliers_mask_all, ["distance_km", "duration_min"]].copy()
    for c in ["distance_km", "duration_min"]:
        outliers[c] = pd.to_numeric(outliers[c], errors="coerce")
    outliers.replace([np.inf, -np.inf], np.nan, inplace=True)
    outliers.dropna(inplace=True)
    if len(outliers):
        fig2.add_trace(
            go.Scattergl(
                x=outliers["distance_km"].astype(float).tolist(),
                y=outliers["duration_min"].astype(float).tolist(),
                mode="markers", name="Outliers", marker=dict(size=5), opacity=0.12, hoverinfo="skip",
            )
        )

    # Constant-speed guides
    if len(inliers):
        x_min = max(0.01, float(np.nanmin(inliers["distance_km"])))
        x_max = max(x_min + 0.5, float(np.nanmax(inliers["distance_km"])))
        xs = np.linspace(x_min, x_max, 200).astype(float).tolist()
        for v in [10.0, 20.0, 30.0]:
            ys = [(x / v) * 60.0 for x in xs]
            fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{int(v)} km/h guide",
                                      line=dict(dash="dot", width=1), hoverinfo="skip"))
        # Tight axes (robust quantiles)
        xql, xqh = inliers["distance_km"].quantile([0.01, 0.995]).tolist()
        yql, yqh = inliers["duration_min"].quantile([0.01, 0.995]).tolist()
        if np.isfinite(xql) and np.isfinite(xqh) and xql < xqh:
            fig2.update_xaxes(range=[float(xql), float(xqh)])
        if np.isfinite(yql) and np.isfinite(yqh) and yql < yqh:
            fig2.update_yaxes(range=[float(yql), float(yqh)])

    fig2.update_layout(
        height=560, title="Trip operating envelope",
        xaxis_title="Distance (km)", yaxis_title="Duration (min)",
        margin=dict(l=20, r=20, t=60, b=30),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ===== Weather relationships =====
    st.subheader("Weather relationships")
    c1, c2 = st.columns(2)
    temp_ok = ("avg_temp_c" in df_filtered.columns) and df_filtered["avg_temp_c"].notna().any()
    wind_ok = ("wind_kph" in df_filtered.columns) and df_filtered["wind_kph"].notna().any()

    # Speed vs temperature
    with c1:
        if temp_ok:
            dat = df_filtered[m_spd & df_filtered["avg_temp_c"].notna()]
            if len(dat) > 30_000:
                dat = dat.sample(n=30_000, random_state=4)
            figt = px.scatter(
                dat, x="avg_temp_c", y="speed_kmh", opacity=0.7,
                labels={"avg_temp_c": "Avg temperature (°C)", "speed_kmh": "Speed (km/h)"},
            )
            _add_linear_fit(figt, dat["avg_temp_c"], dat["speed_kmh"])
            figt.update_layout(height=480, title="Speed vs Temperature")
            st.plotly_chart(figt, use_container_width=True)
        else:
            st.info("No temperature column available for this view.")

    # Speed vs wind
    with c2:
        if wind_ok:
            dat = df_filtered[m_spd & df_filtered["wind_kph"].notna()]
            if len(dat) > 30_000:
                dat = dat.sample(n=30_000, random_state=5)
            figw = px.scatter(
                dat, x="wind_kph", y="speed_kmh", opacity=0.7,
                labels={"wind_kph": "Wind (kph)", "speed_kmh": "Speed (km/h)"},
            )
            _add_linear_fit(figw, dat["wind_kph"], dat["speed_kmh"])
            figw.update_layout(height=480, title="Speed vs Wind")
            st.plotly_chart(figw, use_container_width=True)
        else:
            st.info("No wind column available for this view.")

    # Distance/Duration vs Temperature (comfort story)
    c3, c4 = st.columns(2)
    with c3:
        if temp_ok:
            dat = df_filtered[m_dur & df_filtered["avg_temp_c"].notna()]
            if len(dat) > 30_000:
                dat = dat.sample(n=30_000, random_state=6)
            figdt = px.scatter(
                dat, x="avg_temp_c", y="duration_min", opacity=0.6,
                labels={"avg_temp_c": "Avg temperature (°C)", "duration_min": "Duration (min)"},
            )
            _add_linear_fit(figdt, dat["avg_temp_c"], dat["duration_min"])
            figdt.update_layout(height=420, title="Duration vs Temperature")
            st.plotly_chart(figdt, use_container_width=True)
    with c4:
        if temp_ok:
            dat = df_filtered[m_dst & df_filtered["avg_temp_c"].notna()]
            if len(dat) > 30_000:
                dat = dat.sample(n=30_000, random_state=7)
            figDxT = px.scatter(
                dat, x="avg_temp_c", y="distance_km", opacity=0.6,
                labels={"avg_temp_c": "Avg temperature (°C)", "distance_km": "Distance (km)"},
            )
            _add_linear_fit(figDxT, dat["avg_temp_c"], dat["distance_km"])
            figDxT.update_layout(height=420, title="Distance vs Temperature")
            st.plotly_chart(figDxT, use_container_width=True)

    # ===== 2D density: distance vs duration =====
    st.markdown("### 🔳 2D density: distance vs duration")
    try:
        inliers_all = df_filtered[m_dst & m_dur].copy()
        inliers_sample = inliers_all.sample(n=min(len(inliers_all), 60_000), random_state=11) if len(inliers_all) > 60_000 else inliers_all
        fig_hex = px.density_heatmap(
            inliers_sample, x="distance_km", y="duration_min",
            nbinsx=60, nbinsy=60, histfunc="count",
            labels={"distance_km": "Distance (km)", "duration_min": "Duration (min)"},
            color_continuous_scale="Viridis",
        )
        fig_hex.update_layout(height=520)
        st.plotly_chart(fig_hex, use_container_width=True)
    except Exception as e:
        st.caption(f"Density heatmap unavailable: {e}")

    # ===== Correlations (quick view) =====
    st.markdown("### 🔗 Correlations (quick view)")
    corr_cols = [c for c in ["duration_min", "distance_km", "speed_kmh", "avg_temp_c", "wind_kph"] if c in df_filtered.columns]
    if len(corr_cols) >= 2:
        corr_mat = df_filtered[corr_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr_mat, text_auto=True, aspect="auto", labels=dict(color="r"))
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.caption("Not enough numeric columns to compute a correlation matrix.")

    # ===== Rain/Wet impact on duration & speed =====
    st.subheader("Rain impact on trip characteristics")
    has_precip_bin = ("precip_bin" in df_filtered.columns) and df_filtered["precip_bin"].notna().any()
    has_wet_flag = ("wet_day" in df_filtered.columns)

    cc1, cc2 = st.columns(2)
    with cc1:
        if has_precip_bin:
            figpb = px.box(
                df_filtered[m_dur], x="precip_bin", y="duration_min",
                category_orders={"precip_bin": ["Low", "Medium", "High"]},
                labels={"precip_bin": "Precipitation", "duration_min": "Duration (min)"},
            )
            figpb.update_layout(height=420, title="Duration by Precipitation")
            st.plotly_chart(figpb, use_container_width=True)
        elif has_wet_flag:
            figwd = px.box(
                df_filtered[m_dur].assign(day_type=lambda x: x["wet_day"].map({0: "Dry", 1: "Wet"})),
                x="day_type", y="duration_min",
                labels={"day_type": "Day type", "duration_min": "Duration (min)"},
            )
            figwd.update_layout(height=420, title="Duration: Wet vs Dry")
            st.plotly_chart(figwd, use_container_width=True)

    with cc2:
        if has_precip_bin:
            figpbs = px.box(
                df_filtered[m_spd], x="precip_bin", y="speed_kmh",
                category_orders={"precip_bin": ["Low", "Medium", "High"]},
                labels={"precip_bin": "Precipitation", "speed_kmh": "Speed (km/h)"},
            )
            figpbs.update_layout(height=420, title="Speed by Precipitation")
            st.plotly_chart(figpbs, use_container_width=True)
        elif has_wet_flag:
            figwds = px.box(
                df_filtered[m_spd].assign(day_type=lambda x: x["wet_day"].map({0: "Dry", 1: "Wet"})),
                x="day_type", y="speed_kmh",
                labels={"day_type": "Day type", "speed_kmh": "Speed (km/h)"},
            )
            figwds.update_layout(height=420, title="Speed: Wet vs Dry")
            st.plotly_chart(figwds, use_container_width=True)

    # ===== Quick weather deltas (KPIs) =====
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if has_wet_flag and df_filtered["wet_day"].notna().any():
            dry_spd = df_filtered.loc[m_spd & (df_filtered["wet_day"] == 0), "speed_kmh"].mean()
            wet_spd = df_filtered.loc[m_spd & (df_filtered["wet_day"] == 1), "speed_kmh"].mean()
            if pd.notnull(dry_spd) and pd.notnull(wet_spd) and dry_spd > 0:
                st.metric("Speed: Wet vs Dry", f"{(wet_spd - dry_spd) / dry_spd * 100:+.1f}%")
    with k2:
        if "wind_kph" in df_filtered.columns and df_filtered["wind_kph"].notna().any():
            calm_spd = df_filtered.loc[m_spd & (df_filtered["wind_kph"] < 10), "speed_kmh"].mean()
            windy_spd = df_filtered.loc[m_spd & (df_filtered["wind_kph"] >= 20), "speed_kmh"].mean()
            if pd.notnull(calm_spd) and pd.notnull(windy_spd) and calm_spd > 0:
                st.metric("Speed: Windy (≥20) vs Calm (<10)", f"{(windy_spd - calm_spd) / calm_spd * 100:+.1f}%")
    with k3:
        if temp_ok:
            comfy = df_filtered.loc[m_spd & df_filtered["avg_temp_c"].between(15, 25), "speed_kmh"].mean()
            extreme = df_filtered.loc[m_spd & (~df_filtered["avg_temp_c"].between(5, 30)), "speed_kmh"].mean()
            if pd.notnull(comfy) and pd.notnull(extreme) and comfy != 0:
                st.metric("Speed: Comfy (15–25°C) vs Extreme", f"{(comfy - extreme) / comfy * 100:+.1f}%")
    with k4:
        if has_precip_bin:
            low_dur = df_filtered.loc[m_dur & (df_filtered["precip_bin"] == "Low"), "duration_min"].mean()
            high_dur = df_filtered.loc[m_dur & (df_filtered["precip_bin"] == "High"), "duration_min"].mean()
            if pd.notnull(low_dur) and pd.notnull(high_dur) and low_dur > 0:
                st.metric("Duration: High rain vs Low", f"{(high_dur - low_dur) / low_dur * 100:+.1f}%")

# ───────────────────── Page: Member vs Casual Profiles ─────────────────────

def _clean_member_labels(df: pd.DataFrame, src_col: str = "member_type", out_col: str = "member_type_clean") -> pd.DataFrame:
    """Normalize member labels to ASCII-safe values used across charts."""
    out = df.copy()
    if src_col not in out.columns:
        out[out_col] = "Other"
        return out
    out[out_col] = (
        out[src_col].astype(str).str.strip().str.lower().map({"member": "Member", "casual": "Casual"})
    ).fillna("Other")
    return out

def _safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.median()) if s.notna().any() else float("nan")

def page_member_vs_casual(df_filtered: pd.DataFrame) -> None:
    """Compare Member vs Casual riding patterns: KPIs, rhythms, weather sensitivity, stations."""
    st.header("👥 Member vs Casual riding patterns")

    # Guards
    if df_filtered is None or df_filtered.empty:
        st.info("No trips in selection.")
        return
    need = {"member_type", "hour"}
    if not need.issubset(df_filtered.columns):
        st.info("Need `member_type` and `started_at` (engineered `hour`).")
        return

    legend_member_title = "Member Type"

    # Clean labels
    df_mc = _clean_member_labels(df_filtered)

    # ───────── KPIs ─────────
    grp = df_mc.groupby("member_type_clean", as_index=False).agg(
        rides=("member_type_clean", "size"),
        duration_med=("duration_min", _safe_median) if "duration_min" in df_mc.columns else ("member_type_clean", "size"),
        distance_med=("distance_km", _safe_median) if "distance_km" in df_mc.columns else ("member_type_clean", "size"),
        speed_med=("speed_kmh", _safe_median) if "speed_kmh" in df_mc.columns else ("member_type_clean", "size"),
    )

    # Rain penalty (wet vs dry) by group
    rain_penalty = None
    if "wet_day" in df_mc.columns:
        dry = df_mc.loc[df_mc["wet_day"] == 0].groupby("member_type_clean").size()
        wet = df_mc.loc[df_mc["wet_day"] == 1].groupby("member_type_clean").size()
        rain_penalty = (wet / dry - 1.0) * 100.0
        rain_penalty = rain_penalty.replace([np.inf, -np.inf], np.nan)

    # Temperature preference gap (Casual − Member)
    temp_gap = None
    if "avg_temp_c" in df_mc.columns:
        temp_meds = df_mc.groupby("member_type_clean")["avg_temp_c"].median()
        if {"Member", "Casual"}.issubset(set(temp_meds.index)):
            temp_gap = float(temp_meds["Casual"] - temp_meds["Member"])

    # KPI cards (uniform CSS)
    st.markdown("#### ✨ At-a-glance (selection)")
    st.markdown("""
    <style>
    .kpi-box{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:16px;
      padding:12px 14px;min-height:132px;display:flex;flex-direction:column;justify-content:space-between;
      box-shadow:0 2px 10px rgba(0,0,0,0.25);transition:transform .15s ease, box-shadow .15s ease;}
    .kpi-box:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(0,0,0,0.35);}
    .kpi-head{display:flex;align-items:center;gap:8px;color:#e5e7eb;font-weight:700;font-size:1.00rem;letter-spacing:.2px;opacity:.95;}
    .kpi-emoji{font-size:1.05rem;line-height:1;}
    .kpi-value{color:#f3f4f6;font-weight:800;font-size:1.06rem;margin:2px 0 0;line-height:1.1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .kpi-sub{color:#cbd5e1;font-size:.82rem;line-height:1.25;margin-top:4px;}
    </style>""", unsafe_allow_html=True)

    m = grp.set_index("member_type_clean") if not grp.empty else pd.DataFrame()
    total_member = int(m.loc["Member", "rides"]) if ("Member" in m.index) else 0
    total_casual = int(m.loc["Casual", "rides"]) if ("Casual" in m.index) else 0
    share_member = 100.0 * total_member / max(total_member + total_casual, 1)

    d_med_m = m.at["Member", "duration_med"] if ("Member" in m.index and "duration_med" in m.columns) else np.nan
    d_med_c = m.at["Casual", "duration_med"] if ("Casual" in m.index and "duration_med" in m.columns) else np.nan
    dur_txt = f"{d_med_m:.1f} vs {d_med_c:.1f} min" if np.isfinite(d_med_m) and np.isfinite(d_med_c) else "—"

    s_med_m = m.at["Member", "speed_med"] if ("Member" in m.index and "speed_med" in m.columns) else np.nan
    s_med_c = m.at["Casual", "speed_med"] if ("Casual" in m.index and "speed_med" in m.columns) else np.nan
    spd_txt = f"{s_med_m:.1f} vs {s_med_c:.1f} km/h" if np.isfinite(s_med_m) and np.isfinite(s_med_c) else "—"

    rain_txt = "—"
    if rain_penalty is not None and {"Member", "Casual"}.issubset(rain_penalty.index):
        rain_txt = f"M {rain_penalty['Member']:+.0f}% · C {rain_penalty['Casual']:+.0f}%"

    temp_txt = f"{temp_gap:+.1f}°C" if (temp_gap is not None and np.isfinite(temp_gap)) else "—"

    ca, cb, cc, cd, ce = st.columns(5)
    with ca:
        st.markdown(f"""
        <div class="kpi-box"><div class="kpi-head"><span class="kpi-emoji">🧑‍💼</span>Member share</div>
        <div class="kpi-value">{share_member:.1f}%</div><div class="kpi-sub">of total rides</div></div>""",
        unsafe_allow_html=True)
    with cb:
        st.markdown(f"""
        <div class="kpi-box"><div class="kpi-head"><span class="kpi-emoji">⏱️</span>Median duration</div>
        <div class="kpi-value">{dur_txt}</div><div class="kpi-sub">Member (M) vs Casual (C)</div></div>""",
        unsafe_allow_html=True)
    with cc:
        st.markdown(f"""
        <div class="kpi-box"><div class="kpi-head"><span class="kpi-emoji">🚴</span>Median speed</div>
        <div class="kpi-value">{spd_txt}</div><div class="kpi-sub">Member (M) vs Casual (C)</div></div>""",
        unsafe_allow_html=True)
    with cd:
        st.markdown(f"""
        <div class="kpi-box"><div class="kpi-head"><span class="kpi-emoji">🌧️</span>Rain penalty</div>
        <div class="kpi-value">{rain_txt}</div><div class="kpi-sub">Wet vs dry (group-wise)</div></div>""",
        unsafe_allow_html=True)
    with ce:
        st.markdown(f"""
        <div class="kpi-box"><div class="kpi-head"><span class="kpi-emoji">🌡️</span>Temp pref. gap</div>
        <div class="kpi-value">{temp_txt}</div><div class="kpi-sub">Casual − Member (median °C)</div></div>""",
        unsafe_allow_html=True)

    # ───────── Tabs ─────────
    tab_beh, tab_wx, tab_perf, tab_station = st.tabs(
        ["🕑 Behavior", "🌦️ Weather mix", "📈 Performance vs temp", "📍 Stations"]
    )

    # ======================= Behavior tab =======================
    with tab_beh:
        st.subheader("Daily rhythms")

        # Hourly profile
        g_hour = df_mc.groupby(["member_type_clean", "hour"]).size().rename("rides").reset_index()
        fig_h = px.line(
            g_hour, x="hour", y="rides", color="member_type_clean",
            labels={"hour": "Hour", "rides": "Rides", "member_type_clean": legend_member_title}
        )
        fig_h.update_traces(mode="lines+markers", hovertemplate="Hour %{x}:00<br>%{y:,} rides")
        fig_h.update_layout(height=380, title="Hourly profile — Member vs Casual")
        st.plotly_chart(fig_h, use_container_width=True)

        # Weekday profile
        if "weekday" in df_mc.columns:
            g_wd = df_mc.groupby(["member_type_clean", "weekday"]).size().rename("rides").reset_index()
            g_wd["weekday_name"] = g_wd["weekday"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
            fig_w = px.line(
                g_wd, x="weekday_name", y="rides", color="member_type_clean",
                labels={"weekday_name": "Weekday", "rides": "Rides", "member_type_clean": legend_member_title}
            )
            fig_w.update_traces(mode="lines+markers")
            fig_w.update_layout(height=380, title="Weekday profile — Member vs Casual")
            st.plotly_chart(fig_w, use_container_width=True)

    # ======================= Weather tab =======================
    with tab_wx:
        st.subheader("Weather sensitivity & mix")
        c1, c2 = st.columns(2)

        with c1:
            if "wet_day" in df_mc.columns:
                gg = (
                    df_mc.assign(day_type=lambda x: x["wet_day"].map({0: "Dry", 1: "Wet"}))
                    .groupby(["member_type_clean", "day_type"]).size().rename("rides").reset_index()
                )
                fig_r = px.bar(
                    gg, x="day_type", y="rides", color="member_type_clean", barmode="group",
                    labels={"day_type": "Day type", "rides": "Rides", "member_type_clean": legend_member_title}
                )
                fig_r.update_layout(height=360, title="Ride volume — Wet vs Dry")
                st.plotly_chart(fig_r, use_container_width=True)

        with c2:
            if "avg_temp_c" in df_mc.columns and df_mc["avg_temp_c"].notna().any():
                vdat = df_mc.dropna(subset=["avg_temp_c"])
                fig_v = px.violin(
                    vdat, x="member_type_clean", y="avg_temp_c", box=True, points=False,
                    labels={"member_type_clean": legend_member_title, "avg_temp_c": "Avg temp during rides (°C)"},
                )
                fig_v.update_layout(height=360, title="Temperature distribution by group")
                st.plotly_chart(fig_v, use_container_width=True)

    # ======================= Performance tab =======================
    with tab_perf:
        st.subheader("Speed & duration vs temperature")
        have_cols = {"avg_temp_c", "speed_kmh", "duration_min"}.issubset(df_mc.columns)
        if have_cols and df_mc["avg_temp_c"].notna().any():
            tbins = [-20, -5, 0, 5, 10, 15, 20, 25, 30, 35, 50]
            tlabs = ["<-5","-5–0","0–5","5–10","10–15","15–20","20–25","25–30","30–35",">35"]
            dat = df_mc.dropna(subset=["avg_temp_c"]).copy()
            dat["temp_band"] = pd.cut(dat["avg_temp_c"], tbins, labels=tlabs, include_lowest=True)

            gs = dat.groupby(["member_type_clean", "temp_band"])["speed_kmh"].median().reset_index()
            figS = px.line(
                gs, x="temp_band", y="speed_kmh", color="member_type_clean", markers=True,
                labels={"temp_band": "Temp band (°C)", "speed_kmh": "Median speed"},
            )
            figS.update_layout(height=360, title="Median speed by temperature band")
            st.plotly_chart(figS, use_container_width=True)

            gd = dat.groupby(["member_type_clean", "temp_band"])["duration_min"].median().reset_index()
            figD = px.line(
                gd, x="temp_band", y="duration_min", color="member_type_clean", markers=True,
                labels={"temp_band": "Temp band (°C)", "duration_min": "Median duration (min)"},
            )
            figD.update_layout(height=360, title="Median duration by temperature band")
            st.plotly_chart(figD, use_container_width=True)
        else:
            st.caption("Need avg_temp_c, speed_kmh, duration_min for this section.")

    # ======================= Stations tab =======================
    with tab_station:
        st.subheader("Top stations by user type")
        if "start_station_name" in df_mc.columns:
            topN = st.slider("Top N stations", 5, 40, 15, 5)
            by_group = df_mc.groupby(["member_type_clean", "start_station_name"]).size().rename("rides").reset_index()
            overall = df_mc.groupby("start_station_name").size().rename("rides_total").reset_index()
            overall["p_overall"] = overall["rides_total"] / float(overall["rides_total"].sum())

            g = by_group.sort_values("rides", ascending=False).groupby("member_type_clean").head(topN)
            mrg = g.merge(overall[["start_station_name", "p_overall"]], on="start_station_name", how="left")
            mrg["p_overall"] = mrg["p_overall"].fillna(0.0)

            cL, cR = st.columns(2)
            for label, col in [("Member", cL), ("Casual", cR)]:
                topk = mrg[mrg["member_type_clean"] == label].copy()
                if topk.empty:
                    with col:
                        st.info(f"No {label} rides in this selection.")
                    continue
                tot = float(by_group.loc[by_group["member_type_clean"] == label, "rides"].sum())
                topk["p_group"] = topk["rides"] / max(tot, 1.0)
                topk["lift"] = topk["p_group"] / topk["p_overall"].replace({0.0: np.nan})
                topk["station_s"] = topk["start_station_name"].astype(str).str.slice(0, 28)
                fig_b = px.bar(
                    topk.sort_values("lift", ascending=False).head(topN),
                    x="station_s", y="lift",
                    hover_data={"start_station_name": True, "rides": ":,", "lift": ":.2f"},
                    labels={"station_s": "Station", "lift": "Lift vs overall"},
                )
                fig_b.update_layout(height=420, title=f"Top stations with lift — {label}")
                fig_b.update_xaxes(tickangle=45)
                with col:
                    st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.caption("No station column available in this selection.")

# ───────────────────── Page: OD Flows — Sankey + Map ─────────────────────

def page_od_flows(df_filtered: pd.DataFrame) -> None:
    """Top origin→destination flows as a Sankey + Arc map, with time-slice and member split options."""
    st.header("🔀 Origin → Destination — Sankey + Map")

    need = {"start_station_name", "end_station_name"}
    if df_filtered is None or df_filtered.empty or not need.issubset(df_filtered.columns):
        st.info("Need start and end station names.")
        return

    # ───────── Controls ─────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        per_origin = st.checkbox("Top-k per origin", value=True)
    with c3:
        topk = st.slider("Top-k edges", 10, 250, 60, 10)
    with c4:
        member_split = st.checkbox("Split by member type", value=("member_type_display" in df_filtered.columns))

    c5, c6, c7 = st.columns(3)
    with c5:
        min_rides = st.number_input("Min rides per edge", min_value=1, max_value=1000, value=3, step=1)
    with c6:
        drop_loops = st.checkbox("Exclude self-loops", value=True)
    with c7:
        render_now = st.checkbox("Render visuals", value=False, help="Tick to build Sankey + Map")

    # Apply time slice & build edges
    subset = _time_slice(df_filtered, mode)
    edges = _cached_edges(subset, per_origin, int(topk), int(min_rides), drop_loops, member_split)
    if edges is None or not isinstance(edges, pd.DataFrame):
        edges = pd.DataFrame(columns=["start_station_name", "end_station_name", "rides"])

    # Early diagnostics if empty
    if edges.empty:
        gb_cols = ["start_station_name", "end_station_name"]
        if member_split and "member_type_display" in subset.columns:
            gb_cols.append("member_type_display")
        g_all = subset.groupby(gb_cols).size().rename("rides").reset_index()

        if drop_loops and not g_all.empty:
            s = g_all["start_station_name"].astype(str)
            e = g_all["end_station_name"].astype(str)
            g_all = g_all[s != e]

        if g_all.empty:
            st.info("No OD pairs in the current slice. Try widening the date/hour filters.")
            return

        counts = np.sort(g_all["rides"].to_numpy())[::-1]
        kth = int(counts[min(max(int(topk) - 1, 0), len(counts) - 1)])
        st.info(
            f"No edges after filters. Suggestions:\n\n"
            f"- Lower **Min rides per edge** to ≤ **{kth}**\n"
            f"- Increase **Top-k edges**\n"
            f"- Turn off **Exclude self-loops**"
        )
        with st.expander("Preview (before min-rides cut)"):
            st.dataframe(g_all.sort_values("rides", ascending=False).head(25), use_container_width=True)
        return

    st.success(f"{len(edges):,} edges match current filters.")
    if not render_now:
        st.caption("Tick **Render visuals** to draw the Sankey and Map (faster app when unchecked).")
        with st.expander("Preview top flows"):
            st.dataframe(edges.sort_values("rides", ascending=False).head(20), use_container_width=True)
        return

    # ───────── Sankey (with caps & stable labels) ─────────
    st.subheader("Sankey — top flows")

    MAX_LINKS = 350
    MAX_NODES = 110

    edges_vis = edges.nlargest(MAX_LINKS, "rides").copy()

    def ascii_safe(s: pd.Series) -> pd.Series:
        # Strip accents/emoji so Plotly labels stay tidy in all environments
        return s.astype(str).str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")

    edges_vis["start_s"] = ascii_safe(edges_vis["start_station_name"])
    edges_vis["end_s"]   = ascii_safe(edges_vis["end_station_name"])

    node_labels = pd.Index(pd.unique(edges_vis[["start_s", "end_s"]].values.ravel()))
    if len(node_labels) > MAX_NODES:
        deg = pd.concat(
            [edges_vis.groupby("start_s")["rides"].sum(),
             edges_vis.groupby("end_s")["rides"].sum()],
            axis=1,
        ).fillna(0).sum(axis=1).sort_values(ascending=False)
        keep = set(deg.head(MAX_NODES).index)
        edges_vis = edges_vis[edges_vis["start_s"].isin(keep) & edges_vis["end_s"].isin(keep)]
        node_labels = pd.Index(sorted(keep))
        st.info(f"Limited to {len(node_labels)} nodes / {len(edges_vis)} links for performance.")

    if edges_vis.empty:
        st.info("Nothing to render after caps; relax filters or increase caps.")
    else:
        idx_map = pd.Series(range(len(node_labels)), index=node_labels)
        src = edges_vis["start_s"].map(idx_map)
        tgt = edges_vis["end_s"].map(idx_map)
        vals = edges_vis["rides"].astype(float)

        link_colors = None
        if member_split and "member_type_display" in edges_vis.columns:
            cmap = {"Member 🧑‍💼": "rgba(34,197,94,0.60)", "Casual 🚲": "rgba(37,99,235,0.60)"}
            mt_vals = edges_vis["member_type_display"].to_numpy()
            link_colors = [cmap.get(str(v), "rgba(180,180,180,0.45)") if not pd.isna(v) else "rgba(180,180,180,0.45)" for v in mt_vals]

        sankey = go.Sankey(
            node=dict(
                label=node_labels.astype(str).tolist(),
                pad=6, thickness=12,
                color="rgba(240,240,255,0.85)",
                line=dict(color="rgba(80,80,120,0.4)", width=0.5),
            ),
            link=dict(source=src, target=tgt, value=vals, color=link_colors),
            arrangement="snap",
        )
        fig = go.Figure(sankey)
        fig.update_layout(height=560, title="Top OD flows", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ───────── Map (pydeck ArcLayer) ─────────
    st.subheader("🗺️ Map — OD arcs (width ∝ volume)")

    if {"start_lat", "start_lng", "end_lat", "end_lng"}.issubset(df_filtered.columns):
        import pydeck as pdk

        map_edges = edges.nlargest(250, "rides").copy()
        starts = (
            df_filtered.groupby("start_station_name")[["start_lat", "start_lng"]]
            .median()
            .rename(columns={"start_lat": "s_lat", "start_lng": "s_lng"})
        )
        ends = (
            df_filtered.groupby("end_station_name")[["end_lat", "end_lng"]]
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
            scale = st.slider("Arc width scale", 1, 30, 10)
            geo["width"] = (scale * (np.sqrt(geo["rides"]) / np.sqrt(vmax if vmax > 0 else 1))).clip(0.5, 14)

            # Robust color mapping
            if member_split and "member_type_display" in geo.columns:
                _cmap = {"Member 🧑‍💼": [34, 197, 94, 200], "Casual 🚲": [37, 99, 235, 200]}
                def _mk_color(v):
                    if pd.isna(v): return [160, 160, 160, 200]
                    return _cmap.get(str(v), [160, 160, 160, 200])
                geo["color"] = [_mk_color(v) for v in geo["member_type_display"].to_numpy()]
            else:
                geo["color"] = [[37, 99, 235, 200]] * len(geo)

            geo["start_s"] = ascii_safe(geo["start_station_name"])
            geo["end_s"]   = ascii_safe(geo["end_station_name"])

            layer = pdk.Layer(
                "ArcLayer",
                data=geo,
                get_source_position="[s_lng, s_lat]",
                get_target_position="[e_lng, e_lat]",
                get_width="width",
                get_source_color="color",
                get_target_color="color",
                pickable=True, auto_highlight=True,
            )

            center_lat = float(pd.concat([geo["s_lat"], geo["e_lat"]]).median())
            center_lon = float(pd.concat([geo["s_lng"], geo["e_lng"]]).median())
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=30, bearing=0)

            tooltip = {"html": "<b>{start_s}</b> → <b>{end_s}</b><br/>Rides: {rides}",
                       "style": {"backgroundColor": "rgba(17,17,17,0.9)", "color": "white"}}

            deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                            map_style="mapbox://styles/mapbox/dark-v11", tooltip=tooltip)
            st.pydeck_chart(deck)
    else:
        st.info("Trip coordinates not available for map.")

    # Diagnostics + download
    with st.expander("Diagnostics — top flows table"):
        showN = st.slider("Show first N rows", 10, 200, 40, 10, key="od_diag_rows")
        st.dataframe(edges.sort_values("rides", ascending=False).head(showN), use_container_width=True)

    st.download_button(
        "Download OD edges (CSV)",
        edges.to_csv(index=False).encode("utf-8"),
        "od_edges_current_view.csv",
        "text/csv",
    )                    

# ───────────────────── Page: OD Matrix — Top origins × destinations ─────────────────────
def page_od_matrix(df_filtered: pd.DataFrame) -> None:
    """Top-O × Top-D OD matrix with optional normalization, ordering, and member split."""
    st.header("🧮 OD Matrix — Top origins × destinations")

    # --- Guards
    need = {"start_station_name", "end_station_name"}
    if df_filtered is None or df_filtered.empty or not need.issubset(df_filtered.columns):
        st.info("Need start and end station names.")
        return

    # --- Member display column (uniform)
    mt_col = None
    if "member_type_display" in df_filtered.columns:
        mt_col = "member_type_display"
    elif "member_type" in df_filtered.columns:
        _map = {"member": "Member 🧑‍💼", "casual": "Casual 🚲", "Member": "Member 🧑‍💼", "Casual": "Casual 🚲"}
        df_filtered = df_filtered.copy()
        df_filtered["member_type_display"] = (
            df_filtered["member_type"].astype(str).map(_map).fillna(df_filtered["member_type"].astype(str))
        )
        mt_col = "member_type_display"

    # --- Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        member_split = st.checkbox("Split by member type", value=(mt_col is not None))
    with c3:
        top_orig = st.slider("Top origins", 10, 150, 40, 5)
    with c4:
        top_dest = st.slider("Top destinations", 10, 150, 40, 5)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        min_rides = st.number_input("Min rides to include (pair)", min_value=1, max_value=1000, value=3, step=1)
    with c6:
        norm = st.selectbox("Normalize", ["None", "Row (per origin)", "Column (per destination)"], index=0)
    with c7:
        sort_mode = st.selectbox("Order", ["By totals", "Alphabetical", "Clustered (if available)"], index=0)
    with c8:
        log_scale = st.checkbox("Log color scale", value=False, help="√(counts + 1) for smoother contrast")

    # --- Subset (time slice) & dtype safety
    subset = _time_slice(df_filtered, mode).copy()
    if subset.empty:
        st.info("No data in this time slice.")
        return
    subset["start_station_name"] = subset["start_station_name"].astype(str)
    subset["end_station_name"]   = subset["end_station_name"].astype(str)

    # --- Optional clustering imports (soft dependency)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
        _clust_ok = True
    except Exception:
        linkage = leaves_list = None
        _clust_ok = False

    # --- Builder (adds density & coverage diagnostics; clearer names)
    def build_matrix(df_src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
        dbg: dict[str, int | float | tuple[int, int]] = {}

        # Totals per endpoint
        o_tot = df_src.groupby("start_station_name").size().sort_values(ascending=False)
        d_tot = df_src.groupby("end_station_name").size().sort_values(ascending=False)
        dbg["origins_total_unique"] = int(o_tot.shape[0])
        dbg["dests_total_unique"] = int(d_tot.shape[0])

        # Keep top-N endpoints
        o_keep = set(o_tot.head(int(top_orig)).index)
        d_keep = set(d_tot.head(int(top_dest)).index)
        dbg["origins_kept"] = len(o_keep)
        dbg["dests_kept"] = len(d_keep)

        # Filter to the kept endpoints
        df2 = df_src[df_src["start_station_name"].isin(o_keep) & df_src["end_station_name"].isin(d_keep)]
        dbg["raw_rows_after_topN_before_pivot"] = int(df2.shape[0])
        if df2.empty:
            dbg.update({
                "unique_pairs_after_groupby": 0,
                "pairs_after_minrides": 0,
                "matrix_shape": (0, 0),
                "matrix_nonzero_density_pct": 0.0,
                "coverage_rides_captured_pct": 0.0,
            })
            return pd.DataFrame(), pd.DataFrame(), o_tot, d_tot, dbg

        # Group to unique OD pairs
        pairs = (
            df2.groupby(["start_station_name", "end_station_name"])
               .size().rename("rides").reset_index()
        )
        dbg["unique_pairs_after_groupby"] = int(pairs.shape[0])

        # Apply min_rides threshold
        if min_rides > 1:
            pairs = pairs[pairs["rides"] >= int(min_rides)]
        dbg["pairs_after_minrides"] = int(pairs.shape[0])
        if pairs.empty:
            total_trips_in_slice = int(df_src.shape[0])
            dbg.update({
                "matrix_shape": (0, 0),
                "matrix_nonzero_density_pct": 0.0,
                "coverage_rides_captured_pct": 0.0 if total_trips_in_slice > 0 else 0.0,
            })
            return pd.DataFrame(), pd.DataFrame(), o_tot, d_tot, dbg

        # Pivot to matrix
        mat = pairs.pivot_table(
            index="start_station_name", columns="end_station_name",
            values="rides", aggfunc="sum", fill_value=0
        )

        # Normalize (optional)
        if norm == "Row (per origin)":
            denom = mat.sum(axis=1).replace(0, np.nan)
            mat = (mat.T / denom).T.fillna(0.0)
        elif norm == "Column (per destination)":
            denom = mat.sum(axis=0).replace(0, np.nan)
            mat = (mat / denom).fillna(0.0)

        # Sort
        if sort_mode == "Alphabetical":
            mat = mat.sort_index(axis=0).sort_index(axis=1)
        elif sort_mode == "By totals":
            if norm == "None":
                mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
                mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]
            else:
                _o = o_tot.reindex(mat.index).fillna(0).sort_values(ascending=False).index
                _d = d_tot.reindex(mat.columns).fillna(0).sort_values(ascending=False).index
                mat = mat.loc[_o, _d]
        elif sort_mode == "Clustered (if available)":
            try:
                if _clust_ok and mat.shape[0] > 2 and mat.shape[1] > 2:
                    rZ = linkage(mat.values, method="average", metric="euclidean")
                    cZ = linkage(mat.values.T, method="average", metric="euclidean")
                    mat = mat.loc[mat.index[leaves_list(rZ)], mat.columns[leaves_list(cZ)]]
                else:
                    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
                    mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]
            except Exception:
                mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
                mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]

        # Diagnostics
        dbg["matrix_shape"] = (int(mat.shape[0]), int(mat.shape[1]))
        matrix_cells = int(mat.shape[0] * mat.shape[1])
        nonzero_cells = int((mat.values > 0).sum()) if matrix_cells else 0
        dbg["matrix_nonzero_density_pct"] = round(100.0 * nonzero_cells / matrix_cells, 1) if matrix_cells else 0.0

        total_trips_in_slice = int(df_src.shape[0])  # trips that entered this OD page after endpoint filter
        rides_in_pairs = int(pairs["rides"].sum())
        dbg["coverage_rides_captured_pct"] = round(100.0 * rides_in_pairs / total_trips_in_slice, 1) if total_trips_in_slice else 0.0

        return mat, pairs, o_tot, d_tot, dbg

    # --- Heatmap renderer
    def render_heatmap(mat: pd.DataFrame, title: str):
        if mat.empty:
            st.info(
                "Nothing to show with current filters. Try lowering **Min rides**, "
                "increasing **Top origins/destinations**, or switching order to **Alphabetical**."
            )
            return

        z = mat.values.astype(float)
        if log_scale and norm == "None":
            z = np.sqrt(z + 1.0)

        if norm == "None":
            hovertemplate = "Origin: %{y}<br>Destination: %{x}<br>Rides: %{z}<extra></extra>"
            colorbar_title = "rides" if not log_scale else "√(rides+1)"
        elif norm.startswith("Row"):
            hovertemplate = "Origin: %{y}<br>Destination: %{x}<br>Share (row): %{z:.2%}<extra></extra>"
            colorbar_title = "row share"
        else:
            hovertemplate = "Origin: %{y}<br>Destination: %{x}<br>Share (col): %{z:.2%}<extra></extra>"
            colorbar_title = "col share"

        fig = go.Figure(go.Heatmap(
            z=z,
            x=mat.columns.astype(str).tolist(),
            y=mat.index.astype(str).tolist(),
            colorbar=dict(title=colorbar_title),
            hovertemplate=hovertemplate,
        ))
        fig.update_layout(
            title=title, xaxis_title="Destination", yaxis_title="Origin",
            height=720, margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Diagnostics renderer (nice KPI layout + downloads)
    def render_diag(dbg: dict, mat: pd.DataFrame, pairs: pd.DataFrame, label: str | None = None):
        with st.expander("🧮 Diagnostics & Download", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🧭 Origins kept", dbg.get("origins_kept", 0))
            c2.metric("🎯 Dests kept", dbg.get("dests_kept", 0))
            c3.metric("🔗 Unique OD pairs", dbg.get("unique_pairs_after_groupby", dbg.get("unique_pairs", 0)))
            c4.metric("📊 After min rides", dbg.get("pairs_after_minrides", 0))

            c5, c6, c7, c8 = st.columns(4)
            ms = dbg.get("matrix_shape", (0, 0))
            c5.metric("🧱 Matrix shape", f"{ms[0]} × {ms[1]}")
            c6.metric("🌐 Non-zero cells", f"{dbg.get('matrix_nonzero_density_pct', 0.0)} %")
            c7.metric("📈 Rides coverage", f"{dbg.get('coverage_rides_captured_pct', 0.0)} %")
            c8.metric("🧾 Rows after topN", dbg.get("raw_rows_after_topN_before_pivot", dbg.get("pairs_rows_after_topN", 0)))

            # Gentle guidance
            cov = float(dbg.get("coverage_rides_captured_pct", 0.0))
            den = float(dbg.get("matrix_nonzero_density_pct", 0.0))
            if cov < 60:
                st.warning("Coverage is low. Increase Top-N or reduce **Min rides** to capture more traffic.")
            elif den < 25:
                st.info("Matrix is quite sparse. Consider lowering Top-N or raising **Min rides** for stronger corridors.")

            # Peek at pairs + downloads
            if not pairs.empty:
                st.dataframe(pairs.sort_values("rides", ascending=False).head(40), use_container_width=True)
                base = (label or "all").replace(" ", "_")
                st.download_button(
                    f"⬇️ Download pairs ({label or 'All'}) CSV",
                    pairs.to_csv(index=False).encode("utf-8"),
                    f"od_pairs_{base}.csv", "text/csv", key=f"dl_pairs_{base}"
                )
                st.download_button(
                    f"⬇️ Download matrix ({label or 'All'}) CSV",
                    mat.reset_index().rename(columns={"start_station_name": "origin"}).to_csv(index=False).encode("utf-8"),
                    f"od_matrix_{base}.csv", "text/csv", key=f"dl_matrix_{base}"
                )

    # --- Render (split or combined)
    if member_split and (mt_col is not None) and (mt_col in subset.columns):
        tabs = st.tabs(["Member 🧑‍💼", "Casual 🚲", "All"])
        segments = [
            ("Member 🧑‍💼", subset[subset[mt_col].astype(str) == "Member 🧑‍💼"]),
            ("Casual 🚲",   subset[subset[mt_col].astype(str) == "Casual 🚲"]),
            ("All",         subset),
        ]
        for (label, seg_df), tab in zip(segments, tabs):
            with tab:
                mat, pairs, o_tot, d_tot, dbg = build_matrix(seg_df)
                render_heatmap(mat, f"Top {mat.shape[0]} origins × Top {mat.shape[1]} destinations — {label}")
                render_diag(dbg, mat, pairs, label)
    else:
        mat, pairs, o_tot, d_tot, dbg = build_matrix(subset)
        render_heatmap(mat, f"Top {mat.shape[0]} origins × Top {mat.shape[1]} destinations")
        render_diag(dbg, mat, pairs, None)

# ───────────────────── Page: Station Popularity (Top start stations) ─────────────────────

def page_station_popularity(df_filtered: pd.DataFrame) -> None:
    """Top start stations by rides or share, with optional weather/member splits and monthly/hourly breakdowns."""
    st.header("🚉 Most popular start stations")

    if df_filtered is None or df_filtered.empty or "start_station_name" not in df_filtered.columns:
        st.warning("`start_station_name` not found in selection.")
        return

    # Pretty member display if only raw member_type exists
    df = df_filtered.copy()
    if "member_type_display" not in df.columns and "member_type" in df.columns:
        _map = {"member": "Member 🧑‍💼", "casual": "Casual 🚲", "Member": "Member 🧑‍💼", "Casual": "Casual 🚲"}
        df["member_type_display"] = df["member_type"].astype(str).map(_map).fillna(df["member_type"].astype(str))

    MEMBER_LEGEND_TITLE = globals().get("MEMBER_LEGEND_TITLE", "Member type")

    # ── Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        topN = st.slider("Top N stations", 5, 150, 30, 5)
    with c2:
        metric = st.selectbox("Metric", ["Rides", "Share %"], help="Share is within current filters.")
    with c3:
        group_by = st.selectbox("Group by", ["Overall", "By Month", "By Hour"])

    c4, c5 = st.columns(2)
    with c4:
        stack_by_member = st.checkbox("Stack by Member Type", value=("member_type_display" in df.columns))
    with c5:
        wx_split = st.selectbox(
            "Weather split",
            ["None", "Wet vs Dry", "Temp bands (Cold/Mild/Hot)"],
            index=0 if ("wet_day" not in df.columns and "avg_temp_c" not in df.columns) else 1
        )

    st.markdown("---")

    # ── Prep base aggregations
    base = df.copy()
    base["station"] = base["start_station_name"].astype(str)

    # Weather split columns (optional)
    wx_col = None
    if wx_split == "Wet vs Dry" and "wet_day" in base.columns:
        wx_col = "wet_day_label"
        base[wx_col] = base["wet_day"].map({0: "Dry", 1: "Wet"})
    elif wx_split.startswith("Temp") and "avg_temp_c" in base.columns:
        wx_col = "temp_band"
        base[wx_col] = pd.cut(
            base["avg_temp_c"], bins=[-100, 5, 20, 200],
            labels=["Cold <5°C", "Mild 5–20°C", "Hot >20°C"], include_lowest=True
        )

    # Member split (optional)
    mcol = "member_type_display" if (stack_by_member and "member_type_display" in base.columns) else None

    # Top stations by total rides in current filter
    leaderboard = (
        base.groupby("station").size().rename("rides")
        .sort_values(ascending=False).head(int(topN)).reset_index()
    )
    keep = set(leaderboard["station"])
    small = base[base["station"].isin(keep)].copy()

    # Share calculation helper
    def _to_share(df_grp: pd.DataFrame, val_col="value", by_cols: list[str] | None = None) -> pd.DataFrame:
        if metric == "Share %":
            if not by_cols:
                total = df_grp[val_col].sum()
                df_grp[val_col] = np.where(total > 0, df_grp[val_col] / total * 100.0, 0.0)
            else:
                denom = df_grp.groupby(by_cols)[val_col].transform(lambda s: s.sum() if s.sum() > 0 else np.nan)
                df_grp[val_col] = (df_grp[val_col] / denom * 100.0).fillna(0.0)
        return df_grp

    # ───────────────────────── Overall (Bar) ─────────────────────────
    if group_by == "Overall":
        by = ["station"]
        if wx_col: by.append(wx_col)
        if mcol:   by.append(mcol)

        g = small.groupby(by).size().rename("value").reset_index()

        share_keys: list[str] = []
        if wx_col: share_keys.append(wx_col)
        if mcol:   share_keys.append(mcol)
        g = _to_share(g, val_col="value", by_cols=share_keys or None)

        xlab = "Share (%)" if metric == "Share %" else "Rides (count)"
        color = mcol if mcol else wx_col

        fig = px.bar(
            g, x="station", y="value", color=color, barmode=("stack" if color else "relative"),
            labels={"station": "Station", "value": xlab, (color or ""): (MEMBER_LEGEND_TITLE if color == mcol else "Weather")},
            hover_data={"station": True, "value": ":,.2f" if metric == "Share %" else ":,"}
        )
        fig.update_layout(
            height=620,
            title=f"Top {len(keep)} start stations — {xlab}",
            xaxis_title="Station", yaxis_title=xlab,
            margin=dict(l=20, r=20, t=60, b=100),
            legend_title_text=(MEMBER_LEGEND_TITLE if color == mcol else ("Weather" if color else "")),
        )
        fig.update_xaxes(
            tickangle=45, tickfont=dict(size=10),
            categoryorder="array", categoryarray=leaderboard["station"].tolist()
        )
        st.plotly_chart(fig, use_container_width=True)

    # ───────────────────────── By Month ─────────────────────────
    elif group_by == "By Month":
        if "month" not in small.columns:
            st.info("Month column not available. Ensure `started_at` parsed in load_data.")
        else:
            by = ["month", "station"]
            if wx_col: by.append(wx_col)
            if mcol:   by.append(mcol)

            g = small.groupby(by).size().rename("value").reset_index()

            share_keys = ["month"]
            if wx_col: share_keys.append(wx_col)
            if mcol:   share_keys.append(mcol)
            g = _to_share(g, val_col="value", by_cols=share_keys)

            # ≤10 stations and no splits → simple line; else heatmap
            if len(keep) <= 10 and not wx_col and not mcol:
                fig = px.line(
                    g, x="month", y="value", color="station", markers=True,
                    labels={"value": "Share (%)" if metric == "Share %" else "Rides", "month": "Month", "station": "Station"}
                )
                fig.update_layout(height=560, title="Monthly trend for top stations")
                st.plotly_chart(fig, use_container_width=True)
            else:
                mat = (
                    g.pivot_table(index="station", columns="month", values="value", aggfunc="sum")
                    .loc[leaderboard["station"]]
                    .fillna(0)
                )
                fig = px.imshow(mat, aspect="auto", origin="lower",
                                labels=dict(color=("Share (%)" if metric == "Share %" else "Rides")))
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
            g = small.groupby(by).size().rename("value").reset_index()

            share_keys = ["hour"]
            if wx_col: share_keys.append(wx_col)
            if mcol:   share_keys.append(mcol)
            g = _to_share(g, val_col="value", by_cols=share_keys)

            mat = (
                g.pivot_table(index="station", columns="hour", values="value", aggfunc="sum")
                .loc[leaderboard["station"]]
                .reindex(columns=range(0, 24))
                .fillna(0)
            )
            fig = px.imshow(mat, aspect="auto", origin="lower",
                            labels=dict(color=("Share (%)" if metric == "Share %" else "Rides")))
            fig.update_xaxes(title_text="Hour of day")
            fig.update_yaxes(title_text="Station")
            fig.update_layout(height=600, title="Hourly pattern — station × hour")
            st.plotly_chart(fig, use_container_width=True)

    # ───────────────────────── Map of Top Stations ─────────────────────────
    if {"start_lat", "start_lng"}.issubset(df.columns):
        st.subheader("🗺️ Map — top stations sized by volume")
        import pydeck as pdk

        coords = (
            df.groupby("start_station_name")[["start_lat", "start_lng"]]
              .median().rename(columns={"start_lat": "lat", "start_lng": "lon"})
        )
        geo = leaderboard.join(coords, on="station", how="left").dropna(subset=["lat", "lon"]).copy()

        if len(geo):
            scale = st.slider("Bubble scale", 1, 15, 5)
            vmax = float(geo["rides"].max())
            geo["radius"] = (20 + scale * (np.sqrt(geo["rides"]) / np.sqrt(vmax if vmax > 0 else 1)) * 100).astype(float)
            geo["color"] = [[37, 99, 235, 200]] * len(geo)

            view_state = pdk.ViewState(
                latitude=float(geo["lat"].median()),
                longitude=float(geo["lon"].median()),
                zoom=11, pitch=0
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=geo,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="color",
                pickable=True
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v11",
                tooltip={"text": "{station}\nRides: {rides}"}
            )
            st.pydeck_chart(deck)
        else:
            st.info("No coordinates available for these stations.")

    # ───────────────────────── Station deep-dive ─────────────────────────
    st.subheader("🔎 Station deep-dive")
    picked = st.selectbox("Pick a station", leaderboard["station"].tolist() if len(leaderboard) else [])
    if picked:
        focus = df[df["start_station_name"].astype(str) == picked]
        cA, cB, cC = st.columns(3)

        with cA:
            if "hour" in focus.columns and not focus.empty:
                gh = focus.groupby("hour").size().rename("rides").reset_index()
                figH = px.line(gh, x="hour", y="rides", markers=True,
                               labels={"hour": "Hour of day", "rides": "Rides"})
                figH.update_layout(height=320, title="Hourly profile")
                st.plotly_chart(figH, use_container_width=True)

        with cB:
            if "weekday" in focus.columns and not focus.empty:
                gw = focus.groupby("weekday").size().rename("rides").reset_index()
                gw["weekday_name"] = gw["weekday"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
                figW = px.bar(gw, x="weekday_name", y="rides",
                              labels={"weekday_name": "Weekday", "rides": "Rides"})
                figW.update_layout(height=320, title="Weekday profile")
                st.plotly_chart(figW, use_container_width=True)

        with cC:
            if "wet_day" in focus.columns and focus["wet_day"].notna().any():
                gd = (
                    focus.assign(day_type=lambda x: x["wet_day"].map({0: "Dry", 1: "Wet"}))
                         .groupby("day_type").size().rename("rides").reset_index()
                )
                figD = px.bar(gd, x="day_type", y="rides",
                              labels={"day_type": "Day type", "rides": "Rides"})
                figD.update_layout(height=320, title="Wet vs Dry impact")
                st.plotly_chart(figD, use_container_width=True)

    st.download_button(
        "Download leaderboard (CSV)",
        leaderboard.rename(columns={"rides": "rides_total"}).to_csv(index=False).encode("utf-8"),
        "top_stations_leaderboard.csv", "text/csv"
    )



# ───────────────────── Page: Station Imbalance (arrivals − departures) ─────────────────────

def page_station_imbalance(df_filtered: pd.DataFrame) -> None:
    """Show stations with largest net arrivals minus departures, optionally per-day normalized, split by member type, with map."""
    st.header("⚖️ Station imbalance (arrivals − departures)")

    need = {"start_station_name", "end_station_name"}
    if df_filtered is None or df_filtered.empty or not need.issubset(df_filtered.columns):
        st.info("Need start/end station names.")
        return

    # Pretty member labels if only raw exists
    df = df_filtered.copy()
    mt_col = None
    if "member_type_display" in df.columns:
        mt_col = "member_type_display"
    elif "member_type" in df.columns:
        _map = {"member": "Member 🧑‍💼", "casual": "Casual 🚲", "Member": "Member 🧑‍💼", "Casual": "Casual 🚲"}
        df["member_type_display"] = df["member_type"].astype(str).map(_map).fillna(df["member_type"].astype(str))
        mt_col = "member_type_display"

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        normalize = st.selectbox("Normalize", ["None", "Per day (avg in/out)"], index=0, help="Uses `date` if available.")
    with c3:
        topK = st.slider("Show top ±K stations", 5, 60, 15, 5)
    with c4:
        min_total = st.number_input("Min total traffic at station (in+out)", 0, 10000, 20, 5)

    c5, c6 = st.columns(2)
    with c5:
        member_split = st.checkbox("Split by member type", value=(mt_col is not None))
    with c6:
        show_map = st.checkbox(
            "Show map",
            value=({"start_lat", "start_lng"}.issubset(df.columns) or {"end_lat", "end_lng"}.issubset(df.columns))
        )

    # Subset by time slice
    subset = _time_slice(df, mode).copy()
    if subset.empty:
        st.info("No rides in this time slice.")
        return

    # Ensure strings
    subset["start_station_name"] = subset["start_station_name"].astype(str)
    subset["end_station_name"]   = subset["end_station_name"].astype(str)

    # ---- Core builder
    def build_imbalance(df_src: pd.DataFrame) -> pd.DataFrame:
        if normalize == "Per day (avg in/out)" and "date" in df_src.columns:
            dep = (
                df_src.groupby(["start_station_name", "date"]).size().rename("cnt").reset_index()
                      .groupby("start_station_name")["cnt"].mean().rename("out").reset_index()
            )
            arr = (
                df_src.groupby(["end_station_name", "date"]).size().rename("cnt").reset_index()
                      .groupby("end_station_name")["cnt"].mean().rename("in").reset_index()
            )
            to_float = True
        else:
            dep = df_src.groupby("start_station_name").size().rename("out").reset_index()
            arr = df_src.groupby("end_station_name").size().rename("in").reset_index()
            to_float = False

        s = dep.merge(arr, left_on="start_station_name", right_on="end_station_name", how="outer")
        s["station"] = s["start_station_name"].fillna(s["end_station_name"])
        s = s.drop(columns=["start_station_name", "end_station_name"])

        if to_float:
            s["in"]  = s["in"].fillna(0.0).astype(float)
            s["out"] = s["out"].fillna(0.0).astype(float)
        else:
            s["in"]  = s["in"].fillna(0).astype(int)
            s["out"] = s["out"].fillna(0).astype(int)

        s["total"] = s["in"] + s["out"]
        if min_total > 0:
            s = s[s["total"] >= (float(min_total) if to_float else int(min_total))]

        s["imbalance"] = s["in"] - s["out"]
        return s.sort_values("imbalance", ascending=False).reset_index(drop=True)

    # ---- Plot helper
    def render_bar(df_in: pd.DataFrame, suffix: str = "") -> pd.DataFrame | None:
        if df_in.empty:
            st.info("Nothing to show with current filters. Lower **Min total traffic** or change the time slice.")
            return None
        top_pos = df_in.nlargest(int(topK), "imbalance")
        top_neg = df_in.nsmallest(int(topK), "imbalance")
        biggest = pd.concat([top_pos, top_neg], ignore_index=True)

        biggest["label"] = biggest["station"].astype(str).str.slice(0, 28)
        colors = np.where(biggest["imbalance"] >= 0, "rgba(34,197,94,0.85)", "rgba(220,38,38,0.85)")

        fig = go.Figure(go.Bar(
            x=biggest["label"],
            y=biggest["imbalance"],
            marker=dict(color=colors),
            customdata=np.stack([biggest["in"], biggest["out"]], axis=1),
            hovertemplate="Station: %{x}<br>IN: %{customdata[0]}<br>OUT: %{customdata[1]}<br>Δ: %{y}<extra></extra>",
        ))
        y_title = "Avg Δ (in − out) / day" if normalize.startswith("Per day") else "Δ (in − out)"
        fig.update_layout(
            height=560,
            title=f"Stations with largest net IN (green) / OUT (red) {suffix}".strip(),
            xaxis_title="", yaxis_title=y_title, margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)
        return biggest

    # ---- Map helper
    def render_map(df_in: pd.DataFrame, suffix: str = "") -> None:
        if not show_map or df_in is None or df_in.empty:
            return

        # Build station coords (from starts and/or ends)
        coords = pd.DataFrame(index=pd.Index(df_in["station"].unique(), name="station"))
        if {"start_lat", "start_lng"}.issubset(df.columns):
            coords_s = df.groupby("start_station_name")[["start_lat", "start_lng"]].median().rename(
                columns={"start_lat": "lat", "start_lng": "lon"}
            )
            coords = coords.join(coords_s.rename_axis("station"), how="left")
        if {"end_lat", "end_lng"}.issubset(df.columns):
            coords_e = df.groupby("end_station_name")[["end_lat", "end_lng"]].median().rename(
                columns={"end_lat": "lat", "end_lng": "lon"}
            )
            coords = coords.combine_first(coords_e.rename_axis("station"))

        geo = df_in.set_index("station").join(coords, how="left").reset_index().dropna(subset=["lat", "lon"])
        if geo.empty:
            st.info("No coordinates for the selected stations.")
            return

        import pydeck as pdk

        vmax = float(np.abs(geo["imbalance"]).max())
        suffix_key = "".join(ch for ch in str(suffix) if ch.isalnum()).lower() or "all"
        scale = st.slider("Map bubble scale", 1, 12, 5, key=f"imb_map_scale_{suffix_key}")

        geo["radius"] = (10 + scale * (np.sqrt(np.abs(geo["imbalance"])) / np.sqrt(vmax if vmax > 0 else 1)) * 120).astype(float)
        geo["color"]  = [[34, 197, 94, 210] if v >= 0 else [220, 38, 38, 210] for v in geo["imbalance"].to_numpy()]

        def ascii_safe(s: pd.Series) -> pd.Series:
            return s.astype(str).str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
        geo["name_s"] = ascii_safe(geo["station"])

        tooltip = {"html": "<b>{name_s}</b><br>IN: {in}<br>OUT: {out}<br>&Delta;: {imbalance}",
                   "style": {"backgroundColor": "rgba(17,17,17,0.85)", "color": "white"}}

        view_state = pdk.ViewState(
            latitude=float(geo["lat"].median()), longitude=float(geo["lon"].median()),
            zoom=11, pitch=0, bearing=0
        )
        layer = pdk.Layer(
            "ScatterplotLayer", data=geo,
            get_position="[lon, lat]", get_radius="radius", get_fill_color="color",
            pickable=True, auto_highlight=True,
        )
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                        map_style="mapbox://styles/mapbox/dark-v11", tooltip=tooltip)
        st.subheader("🗺️ Map — stations sized by |Δ| and colored by sign" + (f" {suffix}" if suffix else ""))
        st.pydeck_chart(deck)

    # ---- Render (split or combined)
    if member_split and (mt_col is not None) and (mt_col in subset.columns):
        tabs = st.tabs(["Member 🧑‍💼", "Casual 🚲", "All"])
        segments = [
            ("Member 🧑‍💼", subset[subset[mt_col].astype(str) == "Member 🧑‍💼"]),
            ("Casual 🚲",   subset[subset[mt_col].astype(str) == "Casual 🚲"]),
            ("All",         subset),
        ]
        for (label, seg_df), tab in zip(segments, tabs):
            with tab:
                m = build_imbalance(seg_df)
                biggest = render_bar(m, f"— {label}")
                with st.expander("Preview & Download — " + label):
                    st.dataframe(m.sort_values("imbalance", ascending=False).head(120), use_container_width=True)
                    st.download_button(
                        f"Download imbalance ({label}) CSV",
                        m.to_csv(index=False).encode("utf-8"),
                        f"station_imbalance_{label.replace(' ', '_')}.csv",
                        "text/csv",
                        key=f"dl_imb_{label}",
                    )
                render_map(biggest, f"— {label}")
    else:
        m = build_imbalance(subset)
        biggest = render_bar(m)
        with st.expander("Preview & Download"):
            st.dataframe(m.sort_values("imbalance", ascending=False).head(120), use_container_width=True)
            st.download_button(
                "Download imbalance CSV", m.to_csv(index=False).encode("utf-8"),
                "station_imbalance.csv", "text/csv",
            )
        render_map(biggest)

    st.caption("Tip: Use **Time slice**, **Normalize → Per day**, and **Min total traffic** to isolate AM vs PM redistribution cleanly.")

# ───────────────────── Page: Pareto (Share of rides) ─────────────────────

def page_pareto(df_filtered: pd.DataFrame) -> None:
    """Pareto concentration of rides across start/end stations, with per-day normalization, member filter, and Lorenz option."""
    st.header("📈 Pareto curve — demand concentration")

    if df_filtered is None or df_filtered.empty:
        st.info("No data in current selection.")
        return

    if "start_station_name" not in df_filtered.columns and "end_station_name" not in df_filtered.columns:
        st.warning("Need `start_station_name` or `end_station_name`.")
        return

    # ---- Controls
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        basis = st.selectbox("Count rides by", ["Start stations", "End stations"], index=0)
    with c1:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
    with c2:
        normalize = st.selectbox(
            "Normalize counts", ["Total rides", "Per day (avg/station)"], index=0,
            help="Per-day uses the `date` column if present."
        )
    with c3:
        target = st.slider("Target cumulative share", 50, 95, 80, 1)

    c4, c5, c6 = st.columns(3)
    with c4:
        member_filter = st.selectbox("Member filter", ["All", "Member only", "Casual only"], index=0)
    with c5:
        min_rides = st.number_input("Min rides per station (pre-Pareto filter)", 0, 10000, 0, 10)
    with c6:
        show_lorenz = st.checkbox("Show Lorenz curve (cum. stations vs cum. rides)", value=False)

    # ---- Subset by time slice + member filter
    subset = _time_slice(df_filtered, mode).copy()
    if subset.empty:
        st.info("No rides for current filters.")
        return

    # Pretty member labels if you only have raw member_type (non-destructive)
    if "member_type_display" not in subset.columns and "member_type" in subset.columns:
        _map = {"member": "Member 🧑‍💼", "casual": "Casual 🚲", "Member": "Member 🧑‍💼", "Casual": "Casual 🚲"}
        subset["member_type_display"] = subset["member_type"].astype(str).map(_map).fillna(subset["member_type"].astype(str))

    if member_filter != "All" and "member_type" in subset.columns:
        if member_filter == "Member only":
            subset = subset[subset["member_type"].astype(str) == "member"]
        else:
            subset = subset[subset["member_type"].astype(str) == "casual"]

    if subset.empty:
        st.info("No rides after member filter.")
        return

    # ---- Pick station column and build totals
    station_col = "start_station_name" if basis.startswith("Start") else "end_station_name"
    if station_col not in subset.columns:
        st.warning(f"`{station_col}` not found.")
        return
    subset[station_col] = subset[station_col].astype(str)

    if normalize == "Per day (avg/station)" and "date" in subset.columns:
        per_day = (
            subset.groupby([station_col, "date"]).size().rename("rides_day").reset_index()
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

    # ---- Hit target share
    target_frac = target / 100.0
    idx_target = int(np.searchsorted(cum_share, target_frac, side="left"))
    rank_needed = min(max(idx_target + 1, 1), n)  # 1-based

    # ---- Diagnostics: Gini & HHI (robust)
    def _gini_from_counts(x: np.ndarray) -> float:
        """Gini (0=equal, 1=concentrated) for nonnegative counts."""
        x = np.asarray(x, dtype=float)
        x = x[x >= 0]
        if x.size == 0 or x.sum() == 0:
            return 0.0
        x_sorted = np.sort(x)
        cumx = np.cumsum(x_sorted)
        # G = 1 - 2 * (area under Lorenz); area via trapezoids on cum shares
        lor = cumx / cumx[-1]
        area = (lor[:-1] + lor[1:]).sum() / (2 * (x.size - 1))
        return float(1 - 2 * area)

    def _hhi_from_counts(x: np.ndarray) -> float:
        """Herfindahl–Hirschman Index (0–1), higher = more concentrated."""
        s = x / x.sum() if x.sum() > 0 else np.zeros_like(x, dtype=float)
        return float(np.sum(s ** 2))

    gini = _gini_from_counts(counts)
    hhi = _hhi_from_counts(counts)
    shares = counts / counts.sum()

    # ---- Plot
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
        # Lorenz: cumulative stations share on X (ascending size), cumulative rides on Y
        x_lor = np.linspace(0, 1, n, endpoint=True)
        y_lor = np.cumsum(np.sort(shares))
        fig.add_trace(go.Scatter(
            x=x_lor * n, y=y_lor, mode="lines", name="Lorenz (asc by size)",
            hovertemplate="Cum stations: %{x:.0f}<br>Cum rides: %{y:.1%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x_lor * n, y=x_lor, mode="lines", name="Equality", line=dict(dash="dash"), hoverinfo="skip",
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=40, b=40),
        title=f"Demand concentration (Pareto) — {basis.lower()}",
    )
    friendly_axis(fig, x="Stations (ranked by rides)", y="Cumulative share of rides", title=None)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Stats
    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.metric("Stations to reach target", f"{rank_needed:,} / {n:,}")
    with cB:
        st.metric("Top station share", f"{shares.max():.1%}")
    with cC:
        st.metric("Gini coefficient", f"{gini:.3f}")
    with cD:
        st.metric("HHI (0–1)", f"{hhi:.3f}")

    # ---- Download
    out = totals.reset_index().rename(columns={station_col: "station", "rides": "value"})
    out["rank"] = np.arange(1, len(out) + 1)
    out["cum_share"] = cum_share
    st.download_button(
        "Download Pareto table (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        f"pareto_{'start' if station_col=='start_station_name' else 'end'}_stations.csv",
        "text/csv",
    )

    st.caption("Tip: if the curve is very steep, a small set of hubs carries most demand—prioritize rebalancing, maintenance, and inventory there.")

# ───────────── Page: Weekday × Hour Heatmap ─────────────

def page_weekday_hour_heatmap(df_filtered: pd.DataFrame) -> None:
    """Temporal load heatmap (weekday × start hour) with scaling, binning, smoothing, presets, and member facets."""
    st.header("⏰ Temporal load — weekday × start hour")

    need = {"started_at", "hour", "weekday"}
    if df_filtered is None or df_filtered.empty or not need.issubset(df_filtered.columns):
        st.info("Need `started_at` parsed into `hour` and `weekday` (done in load_data).")
        return

    # ---- Controls ----
    c0, c1, c2, c3, c4, c5 = st.columns(6)
    with c0:
        mode = st.selectbox("Time slice", ["All", "Weekday", "Weekend", "AM (06–11)", "PM (16–20)"], index=0)
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

    # Base subset using global time-slice helper
    subset = _time_slice(df_filtered, mode).copy()

    # Apply day-of-week preset (stacks on sidebar filter + mode)
    if wk_preset == "Weekdays only":
        subset = subset[subset["weekday"].isin([0, 1, 2, 3, 4])]
    elif wk_preset == "Weekend only":
        subset = subset[subset["weekday"].isin([5, 6])]

    # Member filter / facet
    facet = False
    if member_mode == "Member only" and "member_type" in subset.columns:
        subset = subset[subset["member_type"].astype(str) == "member"]
    elif member_mode == "Casual only" and "member_type" in subset.columns:
        subset = subset[subset["member_type"].astype(str) == "casual"]
    elif member_mode == "Facet by Member Type" and "member_type_display" in subset.columns:
        facet = True

    if subset.empty:
        st.info("No rows for current filters.")
        return

    # ---- Heatmap renderer ----
    def _render_heat(mat: pd.DataFrame, title: str):
        if mat.empty:
            st.info("Not enough data to render heatmap.")
            return
        # Optional smoothing (per weekday across hours)
        if smooth:
            mat = _smooth_by_hour(mat, k=3)

        # Relabel Y-axis to weekday names
        mat_display = mat.copy()
        mat_display.index = _weekday_name(mat_display.index)

        # Use imshow for speed; annotate hover smartly
        fig = px.imshow(
            mat_display,
            aspect="auto", origin="lower",
            labels=dict(x="Hour of day", y="Day of week", color=("Value" if scale == "Absolute" else scale)),
            text_auto=False, color_continuous_scale="Turbo" if scale == "Z-score" else "Viridis"
        )
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

        # Hover text
        hover = "<b>%{y}</b> @ <b>%{x}</b><br>Value: %{z}"
        if scale in ("Row %", "Column %"):
            hover = "<b>%{y}</b> @ <b>%{x}</b><br>Share: %{z:.1f}%"
        fig.update_traces(hovertemplate=hover)

        # Peak annotation (only for Absolute / Row %)
        if scale in ("Absolute", "Row %"):
            _ = _add_peak_annotation(fig, mat)

        st.plotly_chart(fig, use_container_width=True)

    # ---- Build & render ----
    if facet:
        cL, cR = st.columns(2)
        for label, col in [("Member 🧑‍💼", cL), ("Casual 🚲", cR)]:
            _sub = subset[subset["member_type"].astype(str).eq("member" if "Member" in label else "casual")]
            mat = _make_heat_grid(_sub, hour_bin=hour_bin, scale=scale)
            with col:
                _render_heat(mat, f"Weekday × Hour — {label}")
    else:
        mat = _make_heat_grid(subset, hour_bin=hour_bin, scale=scale)
        _render_heat(mat, "Weekday × Hour — All riders")

    # ---- Marginal profiles ----
    st.subheader("Marginal profiles")
    grid_all = _make_heat_grid(subset, hour_bin=hour_bin, scale="Absolute")
    if not grid_all.empty:
        # Hourly totals (columns)
        hourly = grid_all.sum(axis=0).rename("rides").reset_index()
        hourly = hourly.rename(columns={hourly.columns[0]: "hour"})
        hourly["hour"] = pd.to_numeric(hourly["hour"], errors="coerce").fillna(0).astype(int)
        f1 = px.line(hourly, x="hour", y="rides", markers=True,
                     labels={"hour": "Hour of day", "rides": "Rides"})
        f1.update_layout(height=300, title="Hourly total rides")
        st.plotly_chart(f1, use_container_width=True)

        # Weekday totals (rows)
        weekday = grid_all.sum(axis=1).rename("rides").reset_index()
        weekday = weekday.rename(columns={weekday.columns[0]: "weekday"})
        weekday["weekday_name"] = _weekday_name(weekday["weekday"])
        f2 = px.bar(weekday, x="weekday_name", y="rides",
                    labels={"weekday_name": "Weekday", "rides": "Rides"})
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
        with cA:
            _render_heat(mat_dry, "Dry days")
        with cB:
            _render_heat(mat_wet, "Wet days")

    st.caption("Tips: try Row % to see within-day timing; Column % to see which days dominate each hour; Z-score to highlight anomalies.")

# ───────────────────── Page: Time Series — Forecast & Decomposition ─────────────────────
def page_time_series_forecast(daily_all: pd.DataFrame | None,
                              daily_filtered: pd.DataFrame | None) -> None:
    """
    Time series page: STL decomposition, baseline forecasts (Naive / Seasonal-Naive / 7MA),
    optional SARIMAX(weekly), and De-weathered + Seasonal-Naive.
    Includes rolling-origin backtest for any selected model.
    """
    # Optional libs
    try:
        from statsmodels.tsa.seasonal import STL  # type: ignore
        HAS_STL = True
    except Exception:
        HAS_STL = False

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        HAS_SARIMAX = True
    except Exception:
        HAS_SARIMAX = False

    st.header("📆 Time Series — Forecast & Decomposition")

    # ----- Guardrails & data prep -----
    if daily_all is None or daily_all.empty or "date" not in daily_all.columns:
        st.info("Need daily table with `date` and `bike_rides_daily`.")
        return

    d = (daily_filtered if (daily_filtered is not None and not daily_filtered.empty) else daily_all).copy()
    if "bike_rides_daily" not in d.columns:
        st.info("`bike_rides_daily` is missing. Build daily aggregation first.")
        return

    # Build continuous daily index; safe imputation
    s = (d[["date", "bike_rides_daily"]]
          .dropna()
          .sort_values("date")
          .set_index("date")["bike_rides_daily"]
          .asfreq("D"))
    s = s.interpolate(limit_direction="both")

    # Optional temperature for de-weather
    temp_col = None
    for c in ["avg_temp_c", "avgTemp", "t_mean_c", "temp_c"]:
        if c in d.columns:
            temp_col = c
            break
    t = None
    if temp_col is not None:
        t = (d[["date", temp_col]].dropna().drop_duplicates()
               .sort_values("date").set_index("date")[temp_col].asfreq("D"))
        t = t.interpolate(limit_direction="both")
        # align to s
        t = t.reindex(s.index).interpolate(limit_direction="both")

    st.caption(f"Series coverage: **{len(s):,} days** — {int(np.isfinite(s).sum())} usable after interpolation")
    
    # ----- Helpers -----
    def _pi_from_resid(fc: np.ndarray, resid: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        resid = resid[np.isfinite(resid)]
        if resid.size < 10:
            return fc, fc
        q = np.quantile(np.abs(resid), 1 - alpha/2.0)
        lo = fc - q
        hi = fc + q
        return lo, hi

    def forecast_naive(series: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yhat = np.full(h, float(series.iloc[-1]))
        resid = series.diff().dropna().to_numpy(dtype=float)
        lo, hi = _pi_from_resid(yhat, resid)
        return yhat, lo, hi

    def forecast_seasonal_naive(series: pd.Series, h: int, season: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(series) < season:
            return forecast_naive(series, h)
        last_season = series.iloc[-season:].to_numpy(dtype=float)
        reps = int(np.ceil(h / season))
        yhat = np.tile(last_season, reps)[:h]
        resid = (series.iloc[-season*10:-season] - series.shift(season).iloc[-season*10:-season]).dropna().to_numpy(dtype=float)
        lo, hi = _pi_from_resid(yhat, resid)
        return yhat, lo, hi

    def forecast_ma(series: pd.Series, h: int, k: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = series.rolling(k, min_periods=max(2, k//2)).mean().iloc[-1]
        if not np.isfinite(base):
            base = float(series.iloc[-k:].mean())
        yhat = np.full(h, float(base))
        resid = (series - series.rolling(k, min_periods=max(2, k//2)).mean()).dropna().to_numpy(dtype=float)
        lo, hi = _pi_from_resid(yhat, resid)
        return yhat, lo, hi

    def _sarimax_best(series: pd.Series, h: int, s_period: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Tiny grid search over (p,d,q)x(P,D,Q, s=7). Fast enough for Streamlit Cloud.
        """
        if not HAS_SARIMAX or len(series) < 30:
            yhat, lo, hi = forecast_seasonal_naive(series, h, season=s_period)
            return yhat, lo, hi, {"note": "SARIMAX unavailable or series too short; used Seasonal-Naive."}

        # Define a small grid
        pdq = [(p, d, q) for p in (0, 1) for d in (0, 1) for q in (0, 1)]
        PDQ = [(P, D, Q, s_period) for P in (0, 1) for D in (0, 1) for Q in (0, 1)]

        best = {"aic": np.inf, "order": None, "sorder": None, "result": None}
        y = series.astype(float)

        for (p, d, q) in pdq:
            for (P, D, Q, S) in PDQ:
                try:
                    mod = SARIMAX(
                        y,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, S),
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                        freq="D",
                    )
                    res = mod.fit(disp=False, maxiter=200)
                    if np.isfinite(res.aic) and res.aic < best["aic"]:
                        best.update({"aic": float(res.aic), "order": (p, d, q), "sorder": (P, D, Q, S), "result": res})
                except Exception:
                    continue

        if best["result"] is None:
            yhat, lo, hi = forecast_seasonal_naive(series, h, season=s_period)
            return yhat, lo, hi, {"note": "All SARIMAX candidates failed; used Seasonal-Naive."}

        res = best["result"]
        fc = res.get_forecast(steps=h)
        yhat = fc.predicted_mean.to_numpy(dtype=float)
        conf = fc.conf_int(alpha=0.10)  # 90% PI
        lo = conf.iloc[:, 0].to_numpy(dtype=float)
        hi = conf.iloc[:, 1].to_numpy(dtype=float)
        meta = {"aic": best["aic"], "order": best["order"], "seasonal_order": best["sorder"]}
        return yhat, lo, hi, meta

    def _deweather_residual(series: pd.Series, temp: pd.Series | None, h: int, season: int = 7,
                            fut_temp_mode: str = "Repeat last 7 days") -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Simple de-weather model:
        1) Fit y = b0 + b1*temp (OLS)
        2) Residuals r = y - (b0 + b1*temp)
        3) Forecast r via Seasonal-Naive
        4) Forecast temp via assumption, add back: yhat = b0 + b1*temp_future + rhat
        """
        if temp is None or not np.all(series.index == temp.index):
            # Fallback to seasonal-naive if temp missing or misaligned
            yhat, lo, hi = forecast_seasonal_naive(series, h, season=season)
            return yhat, lo, hi, {"note": "No temperature available — used Seasonal-Naive."}

        y = series.astype(float).to_numpy()
        x = temp.astype(float).to_numpy()
        # Add intercept
        X = np.column_stack([np.ones_like(x), x])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            yhat, lo, hi = forecast_seasonal_naive(series, h, season=season)
            return yhat, lo, hi, {"note": "OLS failed — used Seasonal-Naive."}

        fitted = X @ beta
        resid = y - fitted
        # Residual forecast
        rhat, rlo, rhi = forecast_seasonal_naive(pd.Series(resid, index=series.index), h, season=season)

        # Future temp assumption
        if fut_temp_mode.startswith("Repeat"):
            if len(x) >= season:
                last = x[-season:]
                reps = int(np.ceil(h / season))
                tfut = np.tile(last, reps)[:h]
            else:
                tfut = np.full(h, x[-1])
        else:  # Hold last day
            tfut = np.full(h, x[-1])

        base = beta[0] + beta[1] * tfut
        yhat = base + rhat
        # PI from residuals only
        lo, hi = _pi_from_resid(yhat, resid)
        meta = {"beta0": float(beta[0]), "beta1_temp": float(beta[1]), "fut_temp_mode": fut_temp_mode}
        return yhat, lo, hi, meta

    # ----- Decomposition tab -----
    tab_dec, tab_fc, tab_bt = st.tabs(["🔍 Decompose", "📈 Forecast", "🧪 Backtest"])

    with tab_dec:
        st.subheader("STL decomposition (weekly seasonality)")
        if HAS_STL and len(s) >= 28:
            try:
                res = STL(s, period=7, robust=True).fit()
                comp = pd.DataFrame({
                    "date": s.index,
                    "observed": s.values,
                    "trend": res.trend,
                    "seasonal": res.seasonal,
                    "resid": res.resid
                })
                # Observed vs Trend
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=comp["date"], y=comp["observed"], name="Observed", mode="lines"))
                fig.add_trace(go.Scatter(x=comp["date"], y=comp["trend"], name="Trend", mode="lines"))
                fig.update_layout(height=380, title="Observed vs Trend")
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    fig_s = px.line(comp, x="date", y="seasonal", title="Seasonal (period=7)")
                    fig_s.update_layout(height=300)
                    st.plotly_chart(fig_s, use_container_width=True)
                with c2:
                    fig_r = px.line(comp, x="date", y="resid", title="Residuals")
                    fig_r.update_layout(height=300)
                    st.plotly_chart(fig_r, use_container_width=True)

                peak_dow = (res.seasonal
                            .groupby(pd.Series(s.index.dayofweek, index=s.index))
                            .mean()
                            .rename(index={0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
                            .sort_values(ascending=False))
                st.caption(f"Biggest positive weekly uplift: **{peak_dow.index[0]}**.")
            except Exception as e:
                st.warning(f"STL failed ({e}). Showing observed only.")
                st.line_chart(s)
        else:
            st.info("`statsmodels` not available or series too short — showing observed only.")
            st.line_chart(s)

    # ----- Forecast tab -----
    with tab_fc:
        st.subheader("Short-term forecast (choose model)")
        # Select model and forecast
        if model_name == "Seasonal-Naive (t−7)":
            yhat, lo, hi = forecast_seasonal_naive(s, horizon, season=7)
            meta_text = "Seasonal-Naive (weekly)"
        elif model_name == "Naive (t−1)":
            yhat, lo, hi = forecast_naive(s, horizon)
            meta_text = "Naive (last value)"
        elif model_name == "7-day Moving Average":
            yhat, lo, hi = forecast_ma(s, horizon, k=7)
            meta_text = "7-day moving average level"
        elif model_name == "SARIMAX (weekly)" and HAS_SARIMAX:
            yhat, lo, hi, meta = _sarimax_best(s, horizon, s_period=7)
            meta_text = f"SARIMAX best by AIC: order={meta.get('order')} seasonal={meta.get('seasonal_order')} AIC={meta.get('aic', 'n/a'):.0f}" if 'aic' in meta else meta.get('note', '')
        elif model_name == "De-weathered + Seasonal-Naive":
            yhat, lo, hi, meta = _deweather_residual(s, t, horizon, season=7, fut_temp_mode=fut_temp_assume)
            meta_text = f"y = b0 + b1*temp; beta1={meta.get('beta1_temp','n/a'):.3f} ({meta.get('fut_temp_mode','')})" if 'beta1_temp' in meta else meta.get('note','')
        else:
            yhat, lo, hi = forecast_seasonal_naive(s, horizon, season=7)
            meta_text = "Fallback to Seasonal-Naive"

        idx_fc = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        df_fc = pd.DataFrame({"date": idx_fc, "yhat": yhat, "lo": lo, "hi": hi})

        # Plot last N days + forecast
        hist = s.iloc[-show_last_n:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History", mode="lines"))
        fig.add_trace(go.Scatter(x=df_fc["date"], y=df_fc["yhat"], name=f"Forecast — {model_name}", mode="lines"))
        fig.add_trace(go.Scatter(x=pd.concat([df_fc["date"], df_fc["date"][::-1]]),
                                 y=pd.concat([df_fc["hi"], df_fc["lo"][::-1]]),
                                 fill="toself", name="Interval", opacity=0.2, line=dict(width=0)))
        fig.update_layout(height=460, title=f"{model_name} — next {horizon} days • {meta_text}", hovermode="x unified")
        fig.update_yaxes(title_text="Bike rides (daily)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_fc.head(10), use_container_width=True)

    # ----- Backtest tab -----
    with tab_bt:
        st.subheader("Rolling-origin backtest")
        c1, c2, c3 = st.columns(3)
        with c1:
            initial = st.number_input("Initial train window (days)", 90, max(365, len(s)-horizon-7), 180, step=7)
        with c2:
            step = st.number_input("Step size (days)", 1, 30, 7, step=1)
        with c3:
            bt_h = st.number_input("Horizon (days)", 7, 30, min(horizon, 14), step=1)

        def _roll_forecast(series: pd.Series, start: int, h: int, model: str) -> np.ndarray:
            train = series.iloc[:start]
            if model == "Seasonal-Naive (t−7)":
                yhat, _, _ = forecast_seasonal_naive(train, h, season=7)
            elif model == "Naive (t−1)":
                yhat, _, _ = forecast_naive(train, h)
            elif model == "7-day Moving Average":
                yhat, _, _ = forecast_ma(train, h, k=7)
            elif model == "SARIMAX (weekly)" and HAS_SARIMAX:
                yhat, _, _, _ = _sarimax_best(train, h, s_period=7)
            elif model == "De-weathered + Seasonal-Naive":
                # For backtest, use historical temp aligned to series if available
                tt = None
                if t is not None:
                    tt = t.iloc[:start]
                yhat, _, _, _ = _deweather_residual(train, tt, h, season=7, fut_temp_mode="Repeat last 7 days")
            else:
                yhat, _, _ = forecast_seasonal_naive(train, h, season=7)
            return yhat

        actuals, preds = [], []
        starts = range(int(initial), len(s) - int(bt_h) + 1, int(step))
        for st_ix in starts:
            fc = _roll_forecast(s, st_ix, int(bt_h), model_name)
            preds.append(fc)
            actuals.append(s.iloc[st_ix: st_ix + int(bt_h)].to_numpy(dtype=float))

        if len(preds) == 0:
            st.info("Backtest window too short for selected settings.")
            return

        y_true = np.concatenate(actuals)
        y_pred = np.concatenate(preds)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1.0, y_true))) * 100.0)

        k1, k2 = st.columns(2)
        with k1:
            st.metric("RMSE", f"{rmse:,.0f}")
        with k2:
            st.metric("MAPE", f"{mape:.1f}%")

        err = y_true - y_pred
        fige = px.histogram(x=err, nbins=30)
        fige.update_layout(
            height=320,
            title=None,                 # remove the printed title
            xaxis_title="Error (y_true − y_pred)",
            yaxis_title="Count",
            hovermode="x"
        )
        # Optional polish: show zero reference
        fige.update_xaxes(zeroline=True, zerolinewidth=1)
        st.plotly_chart(fige, use_container_width=True)

# ───────────────────── Page: Recommendations ─────────────────────
def page_recommendations(df_filtered: pd.DataFrame, daily_filtered: pd.DataFrame | None) -> None:
    """Evidence-driven recommendations + KPIs, auto-derived from the current selection (polished UI)."""

    # ── Guards
    if df_filtered is None or df_filtered.empty:
        st.header("🚀 Recommendations — evidence from this selection")
        st.info("No data in the current selection. Adjust filters on the left.")
        return

    # ── Local helpers (kept inside to avoid name clashes)
    def _rain_penalty(daily: pd.DataFrame | None) -> float | None:
        if daily is None or daily.empty or "wet_day" not in daily.columns or "bike_rides_daily" not in daily.columns:
            return None
        d = daily.dropna(subset=["wet_day", "bike_rides_daily"])
        if d.empty or (d["wet_day"] == 0).sum() == 0:
            return None
        dry = d.loc[d["wet_day"] == 0, "bike_rides_daily"].mean()
        wet = d.loc[d["wet_day"] == 1, "bike_rides_daily"].mean()
        return float((wet - dry) / dry * 100.0) if pd.notnull(dry) and dry > 0 else None

    def _optimal_temp_quadratic(
        daily: pd.DataFrame | None,
        tcol: str = "avg_temp_c",
        ycol: str = "bike_rides_daily",
        tmin: float = -5.0,
        tmax: float = 35.0,
    ) -> float | None:
        """
        Fit Y ≈ a2*(T - Tm)^2 + a1*(T - Tm) + a0 (centered for stability).
        Return vertex temp if concave (a2<0) and inside [tmin, tmax]; else None.
        """
        if daily is None or daily.empty or not {tcol, ycol}.issubset(daily.columns):
            return None
        d = daily[[tcol, ycol]].dropna().copy()
        d = d[(d[tcol] >= tmin) & (d[tcol] <= tmax)]
        if len(d) < 20:
            return None
        T = d[tcol].to_numpy(dtype=float)
        Y = d[ycol].to_numpy(dtype=float)
        Tm = T.mean()
        Tc = T - Tm
        a2, a1, a0 = np.polyfit(Tc, Y, 2)
        if a2 >= 0:
            return None
        Tc_opt = -a1 / (2 * a2)
        T_opt = float(Tc_opt + Tm)
        if T_opt < tmin or T_opt > tmax:
            return None
        return T_opt

    def _temp_elasticity_at_20c(daily: pd.DataFrame | None) -> float | None:
        """Elasticity (%/°C) at 20 °C from quadratic y = a + bT + cT^2."""
        if daily is None or daily.empty or not {"avg_temp_c", "bike_rides_daily"}.issubset(daily.columns):
            return None
        d = daily.dropna(subset=["avg_temp_c", "bike_rides_daily"]).copy()
        if len(d) < 20:
            return None
        T = d["avg_temp_c"].to_numpy(dtype=float)
        Y = d["bike_rides_daily"].to_numpy(dtype=float)
        X = np.c_[np.ones(len(T)), T, T**2]
        try:
            a, b, c = np.linalg.lstsq(X, Y, rcond=None)[0]
            t = 20.0
            y_hat = a + b*t + c*t*t
            dy_dt = b + 2*c*t
            if y_hat <= 0:
                return None
            return float(100.0 * dy_dt / y_hat)
        except Exception:
            return None

    def _top_share(df: pd.DataFrame, col: str, top_k: int = 20) -> float | None:
        if col not in df.columns or df.empty:
            return None
        vc = df[col].astype(str).value_counts()
        tot = int(vc.sum())
        if tot == 0:
            return None
        return float(100.0 * vc.head(top_k).sum() / tot)

    def _peak_cell(df: pd.DataFrame) -> tuple[str, str] | None:
        """Return ('Weekday','Hour') for absolute peak in weekday×hour grid."""
        mat = _make_heat_grid(df, hour_col="hour", weekday_col="weekday", hour_bin=1, scale="Absolute")
        if mat.empty or not np.isfinite(mat.to_numpy()).any():
            return None
        r, c = np.unravel_index(np.nanargmax(mat.values), mat.shape)
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][r]
        hour = mat.columns[c]
        return weekday, f"{int(hour):02d}:00"

    def _imbalance_table(df: pd.DataFrame, topk: int = 8) -> pd.DataFrame:
        need = {"start_station_name", "end_station_name"}
        if not need.issubset(df.columns):
            return pd.DataFrame()
        g_start = df.groupby("start_station_name").size().rename("out")
        g_end = df.groupby("end_station_name").size().rename("in")
        m = pd.concat([g_start, g_end], axis=1).fillna(0).astype(int)
        m["Δ (in−out)"] = (m["in"] - m["out"]).astype(int)
        if m.empty:
            return m
        hi = m.sort_values("Δ (in−out)", ascending=False).head(topk)
        lo = m.sort_values("Δ (in−out)", ascending=True).head(topk)
        return pd.concat([hi, lo])

    # ── Core evidence
    kpis = compute_core_kpis(df_filtered, daily_filtered)  # expects {total_rides, avg_day, corr_tr}
    rain_pen = _rain_penalty(daily_filtered)
    temp_el  = _temp_elasticity_at_20c(daily_filtered)
    share_top20_start = _top_share(df_filtered, "start_station_name", 20)
    share_top20_end   = _top_share(df_filtered, "end_station_name", 20)
    peak = _peak_cell(df_filtered)
    imb  = _imbalance_table(df_filtered, topk=10)  # 10 for pilot plan

    # safe formatting
    p_start = 0.0 if share_top20_start is None or (isinstance(share_top20_start, float) and np.isnan(share_top20_start)) else float(share_top20_start)
    p_end   = 0.0 if share_top20_end   is None or (isinstance(share_top20_end,   float) and np.isnan(share_top20_end))   else float(share_top20_end)
    rain_txt = f"{rain_pen:+.0f}%" if rain_pen is not None else "lower"

    # ── Optional: compute optimal temperature here (kept lean & safe)
    T_opt = None
    if daily_filtered is not None and not daily_filtered.empty and "avg_temp_c" in daily_filtered.columns:
        try:
            tmin = max(-5.0, float(daily_filtered["avg_temp_c"].min()))
            tmax = min(35.0, float(daily_filtered["avg_temp_c"].max()))
            T_opt = _optimal_temp_quadratic(daily_filtered, "avg_temp_c", "bike_rides_daily", tmin, tmax)
        except Exception:
            T_opt = None

    # ── Styles
    st.markdown(
        """
        <style>
        .hero{
          background: radial-gradient(1000px 300px at 10% -20%, rgba(56, 189, 248,.18), transparent),
                      radial-gradient(1000px 300px at 110% -40%, rgba(99,102,241,.14), transparent);
          border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:18px 18px; margin-bottom:12px;
        }
        .badge{display:inline-block; padding:4px 8px; border-radius:999px; font-size:.78rem;
               border:1px solid rgba(255,255,255,.16); color:#e5e7eb; background:rgba(255,255,255,.05); margin-right:6px;}
        .rec-card {background:linear-gradient(180deg, rgba(25,31,40,.82), rgba(16,21,29,.90));
                   border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:14px 16px;}
        .rec-title{color:#cbd5e1; font-size:.90rem;}
        .rec-val  {color:#f8fafc; font-weight:800; font-size:1.18rem; margin-top:2px;}
        .rec-sub  {color:#94a3b8; font-size:.80rem; margin-top:2px;}
        .callout{border-left:4px solid #60a5fa; padding:10px 12px; background:rgba(59,130,246,.08);
                 border-radius:8px; color:#dbeafe;}
        .risk{border-left:4px solid #f59e0b; padding:10px 12px; background:rgba(245,158,11,.08);
                 border-radius:8px; color:#fde68a;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.subheader("🚀 Recommendations — evidence from this selection")
    badges = []
    if peak:
        badges.append(f'<span class="badge">Peak: {peak[0]} {peak[1]}</span>')
    if kpis.get("corr_tr") is not None:
        badges.append(f'<span class="badge">Temp↔Rides r: {kpis["corr_tr"]:+.2f}</span>')
    if rain_pen is not None:
        badges.append(f'<span class="badge">Rain impact: {rain_txt}</span>')
    if p_start or p_end:
        badges.append(f'<span class="badge">Top-20 coverage: {p_start:.0f}% / {p_end:.0f}%</span>')
    if T_opt is not None:
        badges.append(f'<span class="badge">Comfort: {T_opt:.1f} °C</span>')
    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Executive summary cards
    st.markdown("#### Executive summary (current selection)")
    cols = st.columns(5)

    def _metric(col, title, val, sub=""):
        with col:
            st.markdown(
                f"""<div class="rec-card">
                    <div class="rec-title">{title}</div>
                    <div class="rec-val">{val}</div>
                    <div class="rec-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    _metric(cols[0], "Total trips", f"{kfmt(kpis.get('total_rides', 0))}", "Scope of evidence")
    _metric(cols[1], "Avg/day", f"{kfmt(kpis.get('avg_day')) if kpis.get('avg_day') else '—'}", "Daily volume")
    _metric(cols[2], "Temp ↔ rides (r)", f"{kpis['corr_tr']:+.3f}" if kpis.get("corr_tr") is not None else "—", "Daily correlation")
    _metric(cols[3], "Rain penalty", rain_txt if rain_pen is not None else "—", "Wet vs dry days")
    _metric(cols[4], "Top-20 share", f"{p_start:.0f}% / {p_end:.0f}%", "Start / End station coverage")

    # ── Insights that adapt to data
    bullets = []
    if kpis.get("corr_tr") is not None and abs(kpis["corr_tr"]) >= 0.30:
        bullets.append("• Temperature is a **material driver** of demand in this selection.")
    if rain_pen is not None and rain_pen <= -5:
        bullets.append("• **Wet days depress rides** — plan pre-positioning and recovery sweeps.")
    if p_start >= 60 or p_end >= 60:
        bullets.append("• Demand is **concentrated**: Hot-20 covers most rides — optimize here first.")
    if peak:
        bullets.append(f"• Peak window: **{peak[0]} {peak[1]}**. Time the last replenishment **before** this window.")
    if T_opt is not None:
        bullets.append(f"• Comfort temperature ≈ **{T_opt:.1f} °C**; up-weight AM/PM fills on mild days.")
    if bullets:
        st.markdown('<div class="callout">', unsafe_allow_html=True)
        st.markdown("**Insights at a glance**  \n" + "\n".join(bullets))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Strategic recommendations (4–8 weeks)
    st.markdown("### 📋 What to do next (4–8 weeks)")
    st.markdown(
        f"""
1) **Morning readiness at hotspot stations**  
   - Target **≥85% fill before AM** at top origins; **≥70% before PM** at top destinations.  
   - Use the Weekday×Hour peak to time the **last pre-peak sweep**.

2) **Weather-aware stocking**  
   - On **mild days (15–25 °C)**, lift dock targets in **AM + PM** windows;  
     on **wet days**, pre-position trucks near **high-loss stations** and expect demand **{rain_txt} vs dry**.

3) **Corridor-based rebalancing**  
   - Stage trucks near **repeat high-flow endpoints** and run **loop routes** (origin→dest clusters).  
   - Prioritize stations with the **largest |Δ (in−out)|** below.

4) **Rider nudges**  
   - Offer **credits** for returns to **under-stocked docks** during commute windows.  
   - Show in-app banners only when **dock-out risk** exceeds threshold within **60–90 min**.

5) **Focus scope (Pareto)**  
   - Maintain a **Hot-20** list covering ~**{p_start:.0f}% of starts / {p_end:.0f}% of ends**;  
     raise service quality here **first**.
        """
    )

    # ── KPI tracker
    st.markdown("### 🎯 KPIs to track")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Dock-out rate @ peaks (Hot-20)", "< 5%")
    with k2:
        st.metric("Empty/Full complaints (MoM)", "−30%")
    with k3:
        st.metric("Truck km / rebalanced bike", "−15%")
    with k4:
        st.metric("On-time dock readiness", "≥ 90%")

    # ── Evidence snapshots
    tab1, tab2, tab3, tab4 = st.tabs(["🧭 Imbalance focus", "🚚 Hot-20 Pilot Plan", "📈 Trend (rides vs temp)", "📦 Download evidence"])

    # Imbalance focus
    with tab1:
        if not imb.empty:
            show = imb.reset_index().rename(columns={"index": "Station"}).copy()
            show["Station"] = show["Station"].astype(str).str.slice(0, 36)
            st.dataframe(show, use_container_width=True)
        else:
            st.info("Need start/end station names to compute imbalance.")

    # Pilot Plan
    with tab2:
        if not imb.empty:
            pilot = imb.reindex(imb["Δ (in−out)"].abs().sort_values(ascending=False).head(10).index).reset_index()
            pilot = pilot.rename(columns={"index": "Station"})
            pilot["AM target fill"] = 85
            pilot["PM target fill"] = 70
            pilot["Rationale"] = np.where(
                pilot["Δ (in−out)"] >= 0,
                "Net arrivals (needs PM relief / more docks free)",
                "Net departures (needs AM stock / more bikes ready)",
            )
            st.caption("Two-week pilot: last sweep **30–45 min before peak**; monitor KPIs daily.")
            st.dataframe(
                pilot[["Station", "in", "out", "Δ (in−out)", "AM target fill", "PM target fill", "Rationale"]],
                use_container_width=True,
            )
            st.download_button(
                "Download Hot-20 Pilot (CSV)",
                pilot.to_csv(index=False).encode("utf-8"),
                "hot20_pilot_plan.csv",
                "text/csv",
            )
        else:
            st.info("Pilot requires station in/out data (start & end station names).")

    # Trend tab
    with tab3:
        if daily_filtered is not None and not daily_filtered.empty and {"date", "bike_rides_daily"}.issubset(daily_filtered.columns):
            d = daily_filtered.sort_values("date").copy()
            n = 14
            for col in ["bike_rides_daily", "avg_temp_c"]:
                if col in d.columns:
                    d[f"{col}_roll"] = d[col].rolling(n, min_periods=max(2, n // 2), center=True).mean()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=d["date"], y=d.get("bike_rides_daily_roll", d["bike_rides_daily"]),
                           name="Daily rides", mode="lines", line=dict(width=2)),
                secondary_y=False,
            )
            if "avg_temp_c" in d.columns:
                fig.add_trace(
                    go.Scatter(x=d["date"], y=d.get("avg_temp_c_roll", d["avg_temp_c"]),
                               name="Avg temp (°C)", mode="lines", line=dict(width=2, dash="dot"), opacity=0.9),
                    secondary_y=True,
                )
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", title="Trend overview (14-day smoother)")
            fig.update_yaxes(title_text="Rides", secondary_y=False)
            fig.update_yaxes(title_text="Temp (°C)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Daily table not available for trend (need `date` → daily aggregation).")

    # Downloads
    with tab4:
        if not imb.empty:
            st.download_button(
                "Download imbalance stations (CSV)",
                imb.reset_index().to_csv(index=False).encode("utf-8"),
                "imbalance_focus.csv",
                "text/csv",
            )
        summary = {
            "total_trips": [kpis.get("total_rides", 0)],
            "avg_per_day": [kpis.get("avg_day") if kpis.get("avg_day") is not None else np.nan],
            "corr_temp_rides": [kpis.get("corr_tr") if kpis.get("corr_tr") is not None else np.nan],
            "rain_penalty_pct": [rain_pen if rain_pen is not None else np.nan],
            "temp_elasticity_20c_pct_per_c": [temp_el if temp_el is not None else np.nan],
            "top20_share_start_pct": [p_start],
            "top20_share_end_pct": [p_end],
            "peak_weekday": [peak[0] if peak else ""],
            "peak_hour": [peak[1] if peak else ""],
            "comfort_temp_c": [T_opt if T_opt is not None else np.nan],
        }
        csv = pd.DataFrame(summary).to_csv(index=False).encode("utf-8")
        st.download_button("Download summary (CSV)", csv, "recommendations_summary.csv", "text/csv")

    # ── Risk flags & assumptions
    risk_msgs = []
    if daily_filtered is None or daily_filtered.empty or "avg_temp_c" not in (daily_filtered.columns if daily_filtered is not None else []):
        risk_msgs.append("No daily temperature series in this selection (trend elasticity unavailable).")
    if kpis.get("total_rides", 0) < 5000:
        risk_msgs.append("Thin sample size; station metrics may be noisy.")
    if risk_msgs:
        st.markdown('<div class="risk">', unsafe_allow_html=True)
        st.markdown("**Risk flags & assumptions**  \n" + "\n".join([f"- {m}" for m in risk_msgs]))
        st.markdown("</div>", unsafe_allow_html=True)

    # Supporting media (optional)
    with st.expander("CitiBike experience"):
        st.video("https://www.youtube.com/watch?v=vm37IuX7UPQ")

    st.caption("Built from the current selection — share this view via the URL to reproduce the plan.")

# ────────────────────────────── Page routing ─────────────────────────────────────

if page == "Intro":
    page_intro(
        df_filtered=df_f,
        daily_filtered=daily_f,
        date_range=date_range,
        seasons=seasons,
        usertype=usertype,
        hour_range=hour_range,
        cover_path=cover_path,
    )

elif page == "Weather vs Bike Usage":
    page_weather_vs_usage(daily_f)

elif page == "Trip Metrics (Duration • Distance • Speed)":
    page_trip_metrics(df_f)

elif page == "Member vs Casual Profiles":
    page_member_vs_casual(df_f)

elif page == "OD Flows — Sankey + Map":
    page_od_flows(df_f)

elif page == "OD Matrix — Top Origins × Dest":
    page_od_matrix(df_f)

elif page == "Station Popularity":
    page_station_popularity(df_f)

elif page == "Station Imbalance (In vs Out)":
    page_station_imbalance(df_f)

elif page == "Pareto: Share of Rides":
    page_pareto(df_f)

elif page == "Weekday × Hour Heatmap":
    page_weekday_hour_heatmap(df_f)

elif page == "Time Series — Forecast & Decomposition":
    page_time_series_forecast(daily_all, daily_f)

elif page == "Recommendations":
    page_recommendations(df_f, daily_f)

# ────────────────────────────── Footer ─────────────────────────────────────
st.markdown("---")
st.caption("Built for stakeholder decisions. Data: Citi Bike (2022) + reduced daily weather sample.")
