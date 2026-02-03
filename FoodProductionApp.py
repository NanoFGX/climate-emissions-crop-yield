import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from io import BytesIO

# ============================================================
# OPTIONAL: click events (sidebar selection is source of truth)
# ============================================================
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

# NEW: training utilities
from sklearn.ensemble import RandomForestRegressor

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Climate–Emissions–Agriculture Decision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Paths
# =========================
BASE = Path(__file__).parent

DATA = BASE / "data"
MODELS = BASE / "models"

# IMPORTANT: some of your screenshots show a folder named "assests" by mistake.
# We support BOTH "assets" and "assests" so images always load.
ASSETS = BASE / "assets"
ASSETS_ALT = BASE / "assests"

# Support both CSV and XLSX (your earlier screenshots show Excel files)
MASTER_PATH_XLSX = DATA / "Master_Fused_Dataset_2000_2013.xlsx"
MASTER_PATH_CSV = DATA / "Master_Fused_Dataset_2000_2013.csv"

MODEL_PATH = MODELS / "best_food_model.pkl"
FEATS_PATH = MODELS / "model_features.json"

# =========================
# Styling (glass / neon-ish)
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 800px at 30% 10%, rgba(40,120,255,0.25), rgba(0,0,0,0) 60%),
                    radial-gradient(1200px 800px at 80% 20%, rgba(0,255,180,0.18), rgba(0,0,0,0) 55%),
                    linear-gradient(180deg, #070A12 0%, #05060A 100%);
        color: #EAF2FF;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

    h1, h2, h3 {letter-spacing: 0.2px;}
    h1 {font-weight: 800;}
    h2 {font-weight: 750;}
    h3 {font-weight: 700;}

    .glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    [data-testid="stMetricLabel"] {opacity: 0.9;}
    [data-testid="stMetricValue"] {font-weight: 900;}

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
@st.cache_data
def load_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def get_master_path() -> Path:
    if MASTER_PATH_XLSX.exists():
        return MASTER_PATH_XLSX
    if MASTER_PATH_CSV.exists():
        return MASTER_PATH_CSV
    return MASTER_PATH_CSV

def safe_load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(
            f"❌ Could not load model: {path.name}\n\n"
            f"Error: {e}\n\n"
            "Likely NumPy/sklearn mismatch. Fix by pinning versions in requirements.txt "
            "to match Colab, or re-save the model after upgrading libraries."
        )
        return None

def plotly_dark(fig):
    fig.update_layout(template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color="#EAF2FF"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1
        )
    )
    return fig

def ensure_temp_category(df: pd.DataFrame) -> pd.DataFrame:
    # If Temp_Category missing, derive from Avg_Temp bins:
    # <10 Cold, 10-20 Moderate, >20 Hot
    if "Temp_Category" not in df.columns and "Avg_Temp" in df.columns:
        def cat(t):
            if pd.isna(t):
                return np.nan
            if t < 10:
                return "Cold"
            if t <= 20:
                return "Moderate"
            return "Hot"
        out = df.copy()
        out["Temp_Category"] = out["Avg_Temp"].apply(cat)
        return out
    return df

def try_import_pycountry():
    try:
        import pycountry  # type: ignore
        return pycountry
    except Exception:
        return None

def to_iso3_series(country_series: pd.Series) -> pd.Series:
    """
    Robust Country -> ISO3 mapping.
    Returns NaN for unknowns.
    """
    pycountry = try_import_pycountry()
    if pycountry is None:
        return pd.Series([np.nan] * len(country_series), index=country_series.index)

    manual = {
        "United States": "USA",
        "USA": "USA",
        "United States of America": "USA",
        "UK": "GBR",
        "United Kingdom": "GBR",
        "Russia": "RUS",
        "Russian Federation": "RUS",

        "Iran": "IRN",
        "Syria": "SYR",
        "Venezuela": "VEN",
        "Bolivia": "BOL",
        "Tanzania": "TZA",

        "Viet Nam": "VNM",
        "Vietnam": "VNM",
        "Lao PDR": "LAO",
        "Laos": "LAO",

        "Moldova": "MDA",
        "Czechia": "CZE",
        "Czech Republic": "CZE",
        "Myanmar": "MMR",
        "Brunei": "BRN",

        "South Korea": "KOR",
        "Korea, Rep.": "KOR",
        "North Korea": "PRK",
        "Korea, Dem. Rep.": "PRK",

        # Congo variants
        "Congo": "COG",
        "Republic of the Congo": "COG",
        "Congo, Rep.": "COG",
        "Democratic Republic of the Congo": "COD",
        "Congo, Dem. Rep.": "COD",
        "DR Congo": "COD",
        "D.R. Congo": "COD",

        # Ivory Coast variants
        "Ivory Coast": "CIV",
        "Côte d’Ivoire": "CIV",
        "Côte d'Ivoire": "CIV",
        "Cote d'Ivoire": "CIV",

        # Palestine
        "Palestine": "PSE",
        "State of Palestine": "PSE",

        # Common country display tweaks
        "Cabo Verde": "CPV",
        "Cape Verde": "CPV",
        "Swaziland": "SWZ",
        "Eswatini": "SWZ",
        "Macedonia": "MKD",
        "North Macedonia": "MKD",
    }

    def lookup(name: str):
        if pd.isna(name):
            return np.nan
        n = str(name).strip()
        if n in manual:
            return manual[n]
        try:
            c = pycountry.countries.lookup(n)
            return c.alpha_3
        except Exception:
            return np.nan

    return country_series.apply(lookup)

def compute_forecast_features_by_trend(cdf: pd.DataFrame, year_col: str, feature_cols: list, future_years: list) -> pd.DataFrame:
    """
    Forecast each feature using a simple linear trend vs year (per country).
    This is only to extend beyond dataset end year when future feature data doesn't exist.
    """
    out_rows = []
    cdf = cdf.dropna(subset=[year_col]).copy()
    cdf[year_col] = pd.to_numeric(cdf[year_col], errors="coerce")

    for y in future_years:
        row = {year_col: int(y)}
        for f in feature_cols:
            if f not in cdf.columns:
                row[f] = np.nan
                continue

            series = cdf[[year_col, f]].dropna()
            if len(series) < 3:
                last_val = cdf[f].dropna().iloc[-1] if cdf[f].dropna().shape[0] else np.nan
                row[f] = float(last_val) if pd.notna(last_val) else np.nan
                continue

            x = series[year_col].values.astype(float)
            v = series[f].values.astype(float)

            try:
                m, b = np.polyfit(x, v, 1)
                row[f] = float(m * float(y) + b)
            except Exception:
                row[f] = float(series[f].iloc[-1])
        out_rows.append(row)

    return pd.DataFrame(out_rows)

def metrics_regression(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "R2": np.nan}

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}

def resolve_asset_path(filename: str) -> Path:
    """
    Fixes the 'XAI images not showing' problem by:
    - supporting both assets/ and assests/
    - trying exact filename
    - trying common alternative names if needed
    """
    candidates = [
        ASSETS / filename,
        ASSETS_ALT / filename,
        # Sometimes people accidentally prefix with "./assets/"
        BASE / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def safe_pct_change(series: pd.Series) -> pd.Series:
    """
    Year-over-year % change: (x_t - x_{t-1}) / x_{t-1} * 100
    """
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(1)
    denom = prev.replace(0, np.nan)
    return ((s - prev) / denom) * 100

# =========================
# Load data
# =========================
MASTER_PATH = get_master_path()
if not MASTER_PATH.exists():
    st.error(
        "❌ Missing dataset file.\n\n"
        f"Put it inside /data as one of:\n- {MASTER_PATH_XLSX.name}\n- {MASTER_PATH_CSV.name}"
    )
    st.stop()

df = load_file(MASTER_PATH)
df = ensure_temp_category(df)

country_col = "Country" if "Country" in df.columns else None
year_col = "Year" if "Year" in df.columns else None
if country_col is None or year_col is None:
    st.error("Dataset must contain 'Country' and 'Year' columns.")
    st.stop()

df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

# =========================
# Session state
# =========================
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None
if "selected_year" not in st.session_state:
    st.session_state.selected_year = int(df[year_col].dropna().min())

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "trained_feats" not in st.session_state:
    st.session_state.trained_feats = None

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="glass">
      <h1>🌍 Climate–Emissions–Agriculture Decision Dashboard</h1>
      <div style="opacity:0.9; font-size: 0.98rem;">
        Global overview → Country drilldown → Prediction (tonnes) + growth → XAI → Clustering → Retraining.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# =========================
# Sidebar
# =========================
st.sidebar.header("Controls")

miny = int(df[year_col].dropna().min())
maxy = int(df[year_col].dropna().max())

# Keep selected_year valid
if st.session_state.selected_year < miny:
    st.session_state.selected_year = miny
if st.session_state.selected_year > maxy:
    st.session_state.selected_year = maxy

yr = st.sidebar.slider("Year (global snapshot)", miny, maxy, int(st.session_state.selected_year))
st.session_state.selected_year = int(yr)

st.sidebar.markdown("---")

# Country selection (sidebar = source of truth)
all_countries = sorted(df[country_col].dropna().unique().tolist())
if not all_countries:
    st.error("No countries found in dataset.")
    st.stop()

if st.session_state.selected_country not in all_countries:
    st.session_state.selected_country = all_countries[0]

picked = st.sidebar.selectbox(
    "Country (accurate selection)",
    all_countries,
    index=all_countries.index(st.session_state.selected_country)
)
st.session_state.selected_country = picked

st.sidebar.markdown("---")

# Dataset explanation (stakeholder-friendly)
with st.sidebar.expander("📚 Datasets Used (What & Why)", expanded=False):
    st.markdown(
        """
**1) Food Production / Crop Output (Target Variable)**  
- **What:** `Food_Production_Tonnes` (the value we predict).  
- **Why:** Used for **food security planning** (stock buffer, import/export decisions, subsidy targeting).

**2) Emissions (Main Predictors)**  
- **What:** `co2`, `methane`, `nitrous_oxide`, `total_ghg`, plus engineered signals like `Carbon_Intensity_Index`.  
- **Why:** Emissions relate to industrialization and environmental pressure that can affect production patterns.

**3) Climate (Context + Signal)**  
- **What:** `Avg_Temp`, and derived `Temp_Category` (Cold/Moderate/Hot).  
- **Why:** Provides climate context that supports interpretation and policy decisions.

**Fusion step:**  
All sources are aligned by **Country + Year**, producing a unified master dataset for modelling and dashboards.
        """
    )

st.sidebar.caption("Tip: If map hover is unreliable, use the global table + sidebar selection (always correct).")

# =========================
# Snapshot KPIs for selected year
# =========================
df_year = df[df[year_col] == st.session_state.selected_year].copy()

def safe_mean(col):
    return float(df_year[col].mean()) if col in df_year.columns else np.nan

def safe_sum(col):
    return float(df_year[col].sum()) if col in df_year.columns else np.nan

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Countries (year)", f"{df_year[country_col].nunique():,}")
with kpi_cols[1]:
    st.metric("Avg Temp (mean)", f"{safe_mean('Avg_Temp'):.2f} °C" if "Avg_Temp" in df_year.columns else "N/A")
with kpi_cols[2]:
    st.metric("Total GHG (sum)", f"{safe_sum('total_ghg'):,.2f}" if "total_ghg" in df_year.columns else "N/A")
with kpi_cols[3]:
    st.metric("Food Production (sum)", f"{safe_sum('Food_Production_Tonnes'):,.0f}" if "Food_Production_Tonnes" in df_year.columns else "N/A")

st.write("")

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "🌐 Global Overview (Stable Map + Fallback)",
    "📈 Country Detail",
    "🤖 Prediction (Actual vs Predicted + Growth)",
    "🧠 Explainable AI (XAI)",
    "🧩 Clustering Insight",
    "🛠️ Train / Retrain Model"
])

# ============================================================
# TAB 1: Global Overview (RE-DESIGNED MAP ARCHITECTURE)
# - Uses go.Choropleth with ISO3 + aligned arrays to stop hover mismatch
# - If mapping is weak, fallback to ranked charts + table
# ============================================================
with tabs[0]:
    st.markdown('<div class="glass"><h2>🌐 Global Overview</h2></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
This section is designed to be **reliable**:
- Preferred: a stable world map using **ISO3 codes** with **aligned hover data** (prevents Russia≠Bangladesh hover bugs).
- Fallback: if too many country names cannot be mapped to ISO3, we show **ranked charts + a sortable table**.
</div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    metric_choices = []
    if "Temp_Category" in df_year.columns:
        metric_choices.append("Temp_Category")
    for m in ["Avg_Temp", "total_ghg", "Food_Production_Tonnes", "Carbon_Intensity_Index"]:
        if m in df_year.columns:
            metric_choices.append(m)

    if not metric_choices:
        st.warning("No usable global metrics found for the selected year.")
        st.stop()

    chosen_metric = st.selectbox("Global view metric", metric_choices, index=0)

    # Prepare grouped frame: 1 row per country
    map_df = df_year.dropna(subset=[country_col]).copy()
    numeric_cols = map_df.select_dtypes(include="number").columns.tolist()
    agg_dict = {c: "mean" for c in numeric_cols if c != year_col}
    grouped = map_df.groupby(country_col, as_index=False).agg(agg_dict)
    grouped = ensure_temp_category(grouped)

    # ISO3 mapping
    grouped["ISO3"] = to_iso3_series(grouped[country_col])

    unmapped = grouped["ISO3"].isna().sum()
    total = len(grouped)
    unmapped_ratio = (unmapped / total) if total else 1.0

    pycountry_ok = try_import_pycountry() is not None
    map_ok = pycountry_ok and total > 0 and unmapped_ratio <= 0.20  # allow up to 20% unmapped

    left, right = st.columns([1.25, 0.75])

    with left:
        if map_ok:
            plot_df = grouped.dropna(subset=["ISO3"]).copy()

            # Sort by ISO3 so locations/customdata/text are aligned
            plot_df = plot_df.sort_values("ISO3").reset_index(drop=True)

            # Build aligned arrays
            locs = plot_df["ISO3"].astype(str).tolist()
            names = plot_df[country_col].astype(str).tolist()

            # common hover fields
            avg_temp = plot_df["Avg_Temp"].tolist() if "Avg_Temp" in plot_df.columns else [np.nan]*len(plot_df)
            total_ghg = plot_df["total_ghg"].tolist() if "total_ghg" in plot_df.columns else [np.nan]*len(plot_df)
            food_t = plot_df["Food_Production_Tonnes"].tolist() if "Food_Production_Tonnes" in plot_df.columns else [np.nan]*len(plot_df)

            customdata = np.array(list(zip(names, avg_temp, total_ghg, food_t)), dtype=object)

            if chosen_metric == "Temp_Category":
                # discrete coloring
                cat = plot_df["Temp_Category"].astype(str).fillna("Unknown")
                cat_to_num = {"Cold": 0, "Moderate": 1, "Hot": 2, "Unknown": -1}
                z = cat.map(cat_to_num).tolist()

                colorscale = [
                    [0.0, "#3B82F6"], [0.33, "#3B82F6"],   # Cold
                    [0.34, "#22C55E"], [0.66, "#22C55E"], # Moderate
                    [0.67, "#F97316"], [1.0, "#F97316"],  # Hot
                ]

                # NOTE: z is only for coloring; hover shows the real category
                # Put category as text
                text = cat.tolist()

                fig = go.Figure(data=go.Choropleth(
                    locations=locs,
                    locationmode="ISO-3",
                    z=z,
                    text=text,
                    customdata=customdata,
                    colorscale=colorscale,
                    showscale=False,
                    marker_line_color="rgba(255,255,255,0.18)",
                    marker_line_width=0.5,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "ISO3: %{location}<br>"
                        "Temp_Category: %{text}<br>"
                        "Avg_Temp: %{customdata[1]:.2f} °C<br>"
                        "total_ghg: %{customdata[2]:.2f}<br>"
                        "Food_Production_Tonnes: %{customdata[3]:,.0f}"
                        "<extra></extra>"
                    )
                ))
                fig.update_layout(title=f"Temperature Category — {st.session_state.selected_year}")

            else:
                # numeric metric coloring
                z = plot_df[chosen_metric].astype(float).tolist()

                fig = go.Figure(data=go.Choropleth(
                    locations=locs,
                    locationmode="ISO-3",
                    z=z,
                    text=names,
                    customdata=customdata,
                    colorscale="Turbo",
                    colorbar=dict(title=chosen_metric),
                    marker_line_color="rgba(255,255,255,0.18)",
                    marker_line_width=0.5,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "ISO3: %{location}<br>"
                        f"{chosen_metric}: %{{z:.3f}}<br>"
                        "Avg_Temp: %{customdata[1]:.2f} °C<br>"
                        "total_ghg: %{customdata[2]:.2f}<br>"
                        "Food_Production_Tonnes: %{customdata[3]:,.0f}"
                        "<extra></extra>"
                    )
                ))
                fig.update_layout(title=f"{chosen_metric} — {st.session_state.selected_year}")

            fig.update_layout(height=640, dragmode=False, uirevision="fixed_map")
            fig.update_geos(
                showcountries=True,
                countrycolor="rgba(255,255,255,0.25)",
                showcoastlines=False,
                showframe=False,
                bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(plotly_dark(fig), use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

        else:
            st.markdown('<div class="glass"><h3>Fallback View (No Map)</h3></div>', unsafe_allow_html=True)
            st.write("")
            why = []
            if not pycountry_ok:
                why.append("- `pycountry` not installed (needed for ISO3 mapping).")
            if total == 0:
                why.append("- No country rows found for the selected year.")
            if unmapped_ratio > 0.20:
                why.append(f"- Too many unmapped countries: {unmapped}/{total} (~{unmapped_ratio*100:.1f}%).")

            st.warning("Map disabled to avoid wrong countries / buggy rendering.\n\n" + "\n".join(why))

            if chosen_metric == "Temp_Category":
                counts = grouped["Temp_Category"].value_counts(dropna=False).reset_index()
                counts.columns = ["Temp_Category", "Count"]
                figb = px.bar(counts, x="Temp_Category", y="Count", title="Temp Category Distribution")
                st.plotly_chart(plotly_dark(figb), use_container_width=True)
            else:
                tmp = grouped[[country_col, chosen_metric]].dropna().sort_values(chosen_metric, ascending=False).head(25)
                figb = px.bar(tmp, x=chosen_metric, y=country_col, orientation="h", title=f"Top 25 Countries by {chosen_metric}")
                st.plotly_chart(plotly_dark(figb), use_container_width=True)

    with right:
        st.markdown('<div class="glass"><h3>Global Table (Always Correct)</h3></div>', unsafe_allow_html=True)
        st.write("")
        show_cols = [country_col]
        for c in ["Temp_Category", "Avg_Temp", "total_ghg", "Food_Production_Tonnes", "Carbon_Intensity_Index"]:
            if c in grouped.columns and c not in show_cols:
                show_cols.append(c)

        table_df = grouped[show_cols].copy()
        if chosen_metric in table_df.columns and chosen_metric != "Temp_Category":
            table_df = table_df.sort_values(chosen_metric, ascending=False)

        st.dataframe(table_df, use_container_width=True, height=620)

    st.write("")
    if not map_ok:
        with st.expander("Show unmapped countries list"):
            bad = grouped[grouped["ISO3"].isna()][country_col].astype(str).sort_values().unique().tolist()
            st.write(bad)

    st.info(f"Selected country (sidebar): **{st.session_state.selected_country}**")

# ============================================================
# TAB 2: Country Detail
# ============================================================
with tabs[1]:
    st.markdown('<div class="glass"><h2>📈 Country Detail</h2></div>', unsafe_allow_html=True)
    st.write("")

    ctry = st.session_state.selected_country
    cdf = df[df[country_col] == ctry].copy().sort_values(year_col)

    if cdf.empty:
        st.warning("No rows for selected country.")
        st.stop()

    latest = cdf.dropna(subset=[year_col]).sort_values(year_col).tail(1).iloc[0]
    c_kpis = st.columns(4)
    with c_kpis[0]:
        st.metric("Country", ctry)
    with c_kpis[1]:
        st.metric("Temp Category", str(latest.get("Temp_Category", "N/A")))
    with c_kpis[2]:
        st.metric("Avg Temp (latest)", f"{float(latest['Avg_Temp']):.2f} °C" if "Avg_Temp" in cdf.columns and pd.notna(latest.get("Avg_Temp")) else "N/A")
    with c_kpis[3]:
        st.metric("Food Production (latest)", f"{float(latest['Food_Production_Tonnes']):,.0f}" if "Food_Production_Tonnes" in cdf.columns and pd.notna(latest.get("Food_Production_Tonnes")) else "N/A")

    st.write("")
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="glass"><h3>Climate & Emissions Trend</h3></div>', unsafe_allow_html=True)
        st.write("")

        if "Avg_Temp" in cdf.columns:
            fig_temp = px.line(cdf, x=year_col, y="Avg_Temp", markers=True, title="Avg Temperature Over Time")
            st.plotly_chart(plotly_dark(fig_temp), use_container_width=True)

        em_cols = [c for c in ["co2", "methane", "nitrous_oxide", "total_ghg"] if c in cdf.columns]
        if em_cols:
            fig_em = px.line(cdf, x=year_col, y=em_cols, title="Emissions Over Time (multi-series)")
            st.plotly_chart(plotly_dark(fig_em), use_container_width=True)

    with right:
        st.markdown('<div class="glass"><h3>Production & Efficiency</h3></div>', unsafe_allow_html=True)
        st.write("")

        if "Food_Production_Tonnes" in cdf.columns:
            fig_prod = px.area(cdf, x=year_col, y="Food_Production_Tonnes", title="Food Production Over Time")
            st.plotly_chart(plotly_dark(fig_prod), use_container_width=True)

        if "Carbon_Intensity_Index" in cdf.columns:
            fig_cii = px.line(cdf, x=year_col, y="Carbon_Intensity_Index", markers=True, title="Carbon Intensity Index Over Time")
            st.plotly_chart(plotly_dark(fig_cii), use_container_width=True)

    st.write("")
    s1, s2 = st.columns([1.1, 0.9])

    with s1:
        st.markdown('<div class="glass"><h3>Scatter: GHG vs Production</h3></div>', unsafe_allow_html=True)
        st.write("")
        if "total_ghg" in cdf.columns and "Food_Production_Tonnes" in cdf.columns:
            hover_cols = [year_col]
            for cc in ["Avg_Temp", "co2", "methane", "nitrous_oxide", "Carbon_Intensity_Index"]:
                if cc in cdf.columns:
                    hover_cols.append(cc)

            fig_scatter = px.scatter(
                cdf,
                x="total_ghg",
                y="Food_Production_Tonnes",
                color="Temp_Category" if "Temp_Category" in cdf.columns else None,
                size="Avg_Temp" if "Avg_Temp" in cdf.columns else None,
                hover_data=hover_cols,
                title="GHG vs Food Production (per-year points)"
            )
            st.plotly_chart(plotly_dark(fig_scatter), use_container_width=True)
        else:
            st.info("Need 'total_ghg' and 'Food_Production_Tonnes' columns for scatter plot.")

    with s2:
        st.markdown('<div class="glass"><h3>Correlation Heatmap</h3></div>', unsafe_allow_html=True)
        st.write("")
        keep = [c for c in ["Avg_Temp", "co2", "methane", "nitrous_oxide", "total_ghg", "Carbon_Intensity_Index", "Food_Production_Tonnes"] if c in cdf.columns]
        if len(keep) >= 3:
            corr = cdf[keep].corr(numeric_only=True)
            fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, hoverongaps=False))
            fig_corr.update_layout(title="Feature Correlations (Selected Country)")
            st.plotly_chart(plotly_dark(fig_corr), use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation view.")

# ============================================================
# TAB 3: Prediction (Actual vs Predicted + Growth %)
# ============================================================
with tabs[2]:
    st.markdown('<div class="glass"><h2>🤖 Prediction (Actual vs Predicted + Growth)</h2></div>', unsafe_allow_html=True)
    st.write("")

    # Load features
    if not FEATS_PATH.exists():
        st.error("❌ model_features.json not found in models/")
        st.stop()
    feats = json.loads(FEATS_PATH.read_text())

    # Prefer trained model (if any), else load saved model
    model = st.session_state.trained_model
    if model is None:
        if not MODEL_PATH.exists():
            st.error("❌ best_food_model.pkl not found in models/")
            st.stop()
        model = safe_load_model(MODEL_PATH)
    if model is None:
        st.stop()

    target_col = "Food_Production_Tonnes"
    if target_col not in df.columns:
        st.error("❌ Dataset missing target column 'Food_Production_Tonnes'.")
        st.stop()

    missing_feats = [f for f in feats if f not in df.columns]
    if missing_feats:
        st.error(f"❌ Dataset missing model feature columns: {missing_feats}")
        st.stop()

    ctry = st.session_state.selected_country
    cdf = df[df[country_col] == ctry].copy().sort_values(year_col)
    cdf = cdf.dropna(subset=[year_col])
    if cdf.empty:
        st.warning("No rows for selected country.")
        st.stop()

    y_min = int(cdf[year_col].min())
    y_max = int(cdf[year_col].max())
    yr_range = st.slider(
        "Year range (historical comparison)",
        min_value=y_min,
        max_value=y_max,
        value=(y_min, y_max)
    )

    cdf_range = cdf[(cdf[year_col] >= yr_range[0]) & (cdf[year_col] <= yr_range[1])].copy()

    X_hist = cdf_range[feats].copy()
    y_true = cdf_range[target_col].copy()

    valid_mask = np.isfinite(X_hist.to_numpy()).all(axis=1) & np.isfinite(y_true.to_numpy())
    cdf_hist = cdf_range.loc[valid_mask].copy()

    if cdf_hist.empty:
        st.warning("No fully valid rows (features+target) for this year range.")
        st.stop()

    X_hist = cdf_hist[feats]
    y_true = cdf_hist[target_col]
    y_pred = model.predict(X_hist)

    m = metrics_regression(y_true, y_pred)
    met_cols = st.columns(4)
    met_cols[0].metric("MAE", f"{m['MAE']:,.2f}" if np.isfinite(m["MAE"]) else "N/A")
    met_cols[1].metric("RMSE", f"{m['RMSE']:,.2f}" if np.isfinite(m["RMSE"]) else "N/A")
    met_cols[2].metric("MAPE (%)", f"{m['MAPE_%']:.2f}%" if np.isfinite(m["MAPE_%"]) else "N/A")
    met_cols[3].metric("R²", f"{m['R2']:.3f}" if np.isfinite(m["R2"]) else "N/A")

    st.write("")

    plot_df = pd.DataFrame({
        "Year": cdf_hist[year_col].astype(int).values,
        "Actual": y_true.values.astype(float),
        "Predicted": np.array(y_pred, dtype=float),
    }).sort_values("Year").reset_index(drop=True)

    # NEW: Growth % (YoY)
    plot_df["Actual_Growth_%"] = safe_pct_change(plot_df["Actual"])
    plot_df["Predicted_Growth_%"] = safe_pct_change(plot_df["Predicted"])

    # Metrics for latest year in selected range
    latest_row = plot_df.tail(1).iloc[0]
    gcols = st.columns(2)
    gcols[0].metric(
        "Latest YoY Change (Actual)",
        f"{latest_row['Actual_Growth_%']:.2f}%" if np.isfinite(latest_row["Actual_Growth_%"]) else "N/A"
    )
    gcols[1].metric(
        "Latest YoY Change (Predicted)",
        f"{latest_row['Predicted_Growth_%']:.2f}%" if np.isfinite(latest_row["Predicted_Growth_%"]) else "N/A"
    )

    st.write("")

    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=plot_df["Year"],
        y=plot_df["Actual"],
        mode="lines+markers",
        name="Actual",
        customdata=np.array(list(zip(plot_df["Actual_Growth_%"])), dtype=object),
        hovertemplate="Year: %{x}<br>Actual: %{y:,.0f}<br>YoY: %{customdata[0]:.2f}%<extra></extra>"
    ))
    fig_ap.add_trace(go.Scatter(
        x=plot_df["Year"],
        y=plot_df["Predicted"],
        mode="lines+markers",
        name="Predicted",
        customdata=np.array(list(zip(plot_df["Predicted_Growth_%"])), dtype=object),
        hovertemplate="Year: %{x}<br>Predicted: %{y:,.0f}<br>YoY: %{customdata[0]:.2f}%<extra></extra>"
    ))
    fig_ap.update_layout(
        title=f"Actual vs Predicted Food Production — {ctry} ({yr_range[0]}–{yr_range[1]})",
        xaxis_title="Year",
        yaxis_title="Food_Production_Tonnes",
    )
    st.plotly_chart(plotly_dark(fig_ap), use_container_width=True)

    st.write("")

    # Small table showing growth clearly
    st.markdown('<div class="glass"><h3>Year-over-Year Growth Table</h3></div>', unsafe_allow_html=True)
    show_growth = plot_df[["Year", "Actual", "Actual_Growth_%", "Predicted", "Predicted_Growth_%"]].copy()
    st.dataframe(show_growth, use_container_width=True, height=260)

    st.write("")

    # Future Projection beyond dataset end year
    st.markdown('<div class="glass"><h3>Future Projection (Scenario Preview)</h3></div>', unsafe_allow_html=True)
    st.caption(
        "Future years estimate input features using a simple trend per feature per country, then apply the model. "
        "Use this as a scenario preview, not a guaranteed forecast."
    )

    future_end = st.slider("Project until year", min_value=int(y_max), max_value=2050, value=min(2030, 2050))
    if future_end > y_max:
        future_years = list(range(int(y_max) + 1, int(future_end) + 1))
        future_features = compute_forecast_features_by_trend(cdf, year_col, feats, future_years)

        if future_features.empty:
            st.info("Not enough data to build future projections.")
        else:
            future_pred = model.predict(future_features[feats])
            future_plot = pd.DataFrame({
                "Year": future_features[year_col].astype(int),
                "Projected": np.array(future_pred, dtype=float)
            }).sort_values("Year").reset_index(drop=True)

            # NEW: projected growth
            future_plot["Projected_Growth_%"] = safe_pct_change(future_plot["Projected"])

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=plot_df["Year"],
                y=plot_df["Actual"],
                mode="lines+markers",
                name="Actual (historical)"
            ))
            fig_f.add_trace(go.Scatter(
                x=plot_df["Year"],
                y=plot_df["Predicted"],
                mode="lines+markers",
                name="Predicted (historical)"
            ))
            fig_f.add_trace(go.Scatter(
                x=future_plot["Year"],
                y=future_plot["Projected"],
                mode="lines",
                name=f"Projected ({future_plot['Year'].min()}–{future_plot['Year'].max()})",
                line=dict(dash="dash"),
                customdata=np.array(list(zip(future_plot["Projected_Growth_%"])), dtype=object),
                hovertemplate="Year: %{x}<br>Projected: %{y:,.0f}<br>YoY: %{customdata[0]:.2f}%<extra></extra>"
            ))
            fig_f.update_layout(
                title=f"Historical vs Projected Food Production — {ctry}",
                xaxis_title="Year",
                yaxis_title="Food_Production_Tonnes",
            )
            st.plotly_chart(plotly_dark(fig_f), use_container_width=True)

    st.write("")
    st.markdown('<div class="glass"><h3>Residuals (Actual − Predicted)</h3></div>', unsafe_allow_html=True)
    res = plot_df.copy()
    res["Residual"] = res["Actual"] - res["Predicted"]
    fig_res = px.bar(res, x="Year", y="Residual", title="Residuals by Year (Positive = under-predicted)")
    st.plotly_chart(plotly_dark(fig_res), use_container_width=True)

# ============================================================
# TAB 4: Explainable AI (XAI)
# - FIX: asset path resolution supports assets/ and assests/
# - FIX: filenames exactly match what you showed
# ============================================================
with tabs[3]:
    st.markdown('<div class="glass"><h2>🧠 Explainable AI (XAI)</h2></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
<h3>Why XAI matters for GeoFoodSec</h3>
This dashboard supports **decision-making**, so stakeholders must understand **why** a prediction changes.
XAI helps validate that the model uses sensible signals (emissions, temperature, carbon intensity) and supports trust.
</div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    shap_file = "final_shap_beeswarm.png"
    cluster_file = "cluster_visualization_final.png"

    shap_path = resolve_asset_path(shap_file)
    cluster_path = resolve_asset_path(cluster_file)

    st.markdown('<div class="glass"><h3>SHAP Beeswarm</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
- Features at the top have the strongest overall effect across samples.<br>
- Points to the right increase predicted production; to the left decrease it.<br>
- Color typically indicates whether feature values are high/low.
</div>
        """,
        unsafe_allow_html=True
    )

    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True, caption=shap_file)
    else:
        st.warning(f"Missing asset: {shap_file}. Put it in assets/ (or assests/).")

    st.write("")
    st.markdown('<div class="glass"><h3>Clustering Visualization</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
This view groups countries with similar climate/emissions/production patterns.
It supports **segmentation** (e.g., identify country groups that behave similarly) and adds interpretability.
</div>
        """,
        unsafe_allow_html=True
    )

    if cluster_path.exists():
        st.image(str(cluster_path), use_container_width=True, caption=cluster_file)
    else:
        st.warning(f"Missing asset: {cluster_file}. Put it in assets/ (or assests/).")

# ============================================================
# TAB 5: Clustering Insight (image + summary file)
# ============================================================
with tabs[4]:
    st.markdown('<div class="glass"><h2>🧩 Clustering Insight</h2></div>', unsafe_allow_html=True)
    st.write("")

    cluster_file = "cluster_visualization_final.png"
    cluster_path = resolve_asset_path(cluster_file)

    if cluster_path.exists():
        st.image(str(cluster_path), use_container_width=True, caption=cluster_file)
    else:
        st.warning(f"Missing: {cluster_file} (place it in assets/ or assests/)")

    st.write("")
    st.markdown('<div class="glass"><h3>Cluster Summary</h3></div>', unsafe_allow_html=True)

    cluster_summary_xlsx = DATA / "clustering_results_summary.xlsx"
    cluster_summary_csv = DATA / "clustering_results_summary.csv"
    summary_path = cluster_summary_xlsx if cluster_summary_xlsx.exists() else (cluster_summary_csv if cluster_summary_csv.exists() else None)

    if summary_path is None:
        st.info("No clustering summary file found in /data (optional).")
    else:
        try:
            csum = load_file(summary_path)
            st.dataframe(csum, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load clustering summary: {e}")

# ============================================================
# TAB 6: Train / Retrain Model
# - FIX: stakeholder-friendly labels instead of n_estimators/max_depth/min_samples_leaf
# ============================================================
with tabs[5]:
    st.markdown('<div class="glass"><h2>🛠️ Train / Retrain Model</h2></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
<div class="glass">
<h3>What this section does</h3>
<p style="opacity:0.92;">
Retrain a <b>Random Forest</b> model using the fused dataset.
You control training/testing year ranges to respect time (avoid data leakage).
After training, the Prediction tab will automatically use the new model.
</p>
</div>
        """,
        unsafe_allow_html=True
    )

    if not FEATS_PATH.exists():
        st.error("❌ model_features.json not found in models/")
        st.stop()
    feats = json.loads(FEATS_PATH.read_text())

    target_col = "Food_Production_Tonnes"
    if target_col not in df.columns:
        st.error("❌ Dataset missing target column 'Food_Production_Tonnes'.")
        st.stop()

    missing_feats = [f for f in feats if f not in df.columns]
    if missing_feats:
        st.error(f"❌ Dataset missing model feature columns: {missing_feats}")
        st.stop()

    global_min_year = int(df[year_col].dropna().min())
    global_max_year = int(df[year_col].dropna().max())

    st.write("")
    st.markdown('<div class="glass"><h3>Year Split</h3></div>', unsafe_allow_html=True)

    # Safe defaults
    default_train_start = global_min_year
    default_train_end = min(global_max_year, global_min_year + 10)
    if default_train_start > default_train_end:
        default_train_end = global_min_year

    default_test_start = min(global_max_year, default_train_end + 1)
    default_test_end = global_max_year
    if default_test_start > default_test_end:
        default_test_start = global_max_year
        default_test_end = global_max_year

    train_range = st.slider(
        "Training years (learn patterns from)",
        min_value=global_min_year,
        max_value=global_max_year,
        value=(int(default_train_start), int(default_train_end))
    )
    test_range = st.slider(
        "Testing years (evaluate on)",
        min_value=global_min_year,
        max_value=global_max_year,
        value=(int(default_test_start), int(default_test_end))
    )

    st.write("")
    st.markdown('<div class="glass"><h3>Model Settings (Stakeholder-Friendly)</h3></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_estimators = st.number_input(
            "Number of Trees (more trees = more stable, slower)",
            min_value=50, max_value=2000, value=400, step=50
        )
    with c2:
        max_depth = st.number_input(
            "Tree Depth Limit (higher = more complex; 0 = unlimited)",
            min_value=0, max_value=100, value=0, step=1
        )
    with c3:
        min_samples_leaf = st.number_input(
            "Minimum Data per Leaf (higher = smoother, less overfitting)",
            min_value=1, max_value=50, value=1, step=1
        )

    train_df = df[(df[year_col] >= train_range[0]) & (df[year_col] <= train_range[1])].copy()
    test_df = df[(df[year_col] >= test_range[0]) & (df[year_col] <= test_range[1])].copy()

    needed_cols = feats + [target_col]
    train_df = train_df.dropna(subset=needed_cols)
    test_df = test_df.dropna(subset=needed_cols)

    st.write("")
    tcols = st.columns(3)
    tcols[0].metric("Train rows", f"{len(train_df):,}")
    tcols[1].metric("Test rows", f"{len(test_df):,}")
    tcols[2].metric("Features", f"{len(feats):,}")

    st.write("")
    train_btn = st.button("🚀 Train Model", use_container_width=True)

    if train_btn:
        if len(train_df) < 50 or len(test_df) < 10:
            st.error("Not enough rows to train/test reliably. Try widening your year ranges.")
        else:
            X_train = train_df[feats].to_numpy()
            y_train = train_df[target_col].to_numpy(dtype=float)

            X_test = test_df[feats].to_numpy()
            y_test = test_df[target_col].to_numpy(dtype=float)

            rf = RandomForestRegressor(
                n_estimators=int(n_estimators),
                max_depth=None if int(max_depth) == 0 else int(max_depth),
                min_samples_leaf=int(min_samples_leaf),
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)

            pred_test = rf.predict(X_test)
            m = metrics_regression(y_test, pred_test)

            st.session_state.trained_model = rf
            st.session_state.trained_feats = feats

            st.success("✅ Training complete. Prediction tab will now use this trained model.")

            mc = st.columns(4)
            mc[0].metric("MAE (test)", f"{m['MAE']:,.2f}" if np.isfinite(m["MAE"]) else "N/A")
            mc[1].metric("RMSE (test)", f"{m['RMSE']:,.2f}" if np.isfinite(m["RMSE"]) else "N/A")
            mc[2].metric("MAPE% (test)", f"{m['MAPE_%']:.2f}%" if np.isfinite(m["MAPE_%"]) else "N/A")
            mc[3].metric("R² (test)", f"{m['R2']:.3f}" if np.isfinite(m["R2"]) else "N/A")

            # Download trained model
            buf = BytesIO()
            joblib.dump(rf, buf)
            buf.seek(0)

            st.download_button(
                "⬇️ Download trained model (.pkl)",
                data=buf.getvalue(),
                file_name="trained_food_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )

    st.write("")
    st.markdown(
        """
<div class="glass">
<h3>Note</h3>
<p style="opacity:0.92;">
If your saved Colab model was trained with different library versions, you might see loading issues.
For best reproducibility, pin versions in requirements.txt and keep feature columns consistent.
</p>
</div>
        """,
        unsafe_allow_html=True
    )
