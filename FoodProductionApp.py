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
# IMPORTANT (requirements.txt)
# ------------------------------------------------------------
# streamlit-plotly-events is optional now (map click kept but
# sidebar selection is the source of truth).
#
# Add to requirements.txt:
#   streamlit==<your_version>
#   pandas
#   numpy
#   plotly
#   scikit-learn
#   joblib
#   pycountry
#   streamlit-plotly-events   (optional)
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

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"
ASSETS = BASE / "assets"

MASTER_PATH = DATA / "Master_Fused_Dataset_2000_2013.csv"
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
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

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
        df = df.copy()
        df["Temp_Category"] = df["Avg_Temp"].apply(cat)
    return df

def to_iso3_series(country_series: pd.Series) -> pd.Series:
    """
    Robust Country -> ISO3 mapping.
    We FORCE ISO3 on the map to prevent wrong hover/country mismatches.
    Any unmapped countries are dropped from the map_df (but still exist in dataset).
    """
    try:
        import pycountry
    except Exception:
        # If pycountry not installed, return NaNs so we can warn clearly
        return pd.Series([np.nan] * len(country_series), index=country_series.index)

    manual = {
        "United States": "USA",
        "USA": "USA",
        "United States of America": "USA",
        "UK": "GBR",
        "United Kingdom": "GBR",
        "Russia": "RUS",
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

        # Palestine variants
        "Palestine": "PSE",
        "State of Palestine": "PSE",
    }

    def norm(s: str) -> str:
        return str(s).strip()

    def lookup(name: str):
        if pd.isna(name):
            return np.nan
        n = norm(name)
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
    This is only to extend beyond 2013 when future feature data doesn't exist.
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

# =========================
# Load data
# =========================
if not MASTER_PATH.exists():
    st.error(f"❌ Missing dataset file: {MASTER_PATH}")
    st.stop()

df = load_csv(MASTER_PATH)
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

# NEW: training session state
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
        Global heatmap → choose country (sidebar) → drilldown analytics → actual vs predicted forecasting + XAI + retraining.
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

# NOTE: keep selected_year valid
if st.session_state.selected_year < miny:
    st.session_state.selected_year = miny
if st.session_state.selected_year > maxy:
    st.session_state.selected_year = maxy

yr = st.sidebar.slider("Year (map snapshot)", miny, maxy, int(st.session_state.selected_year))
st.session_state.selected_year = int(yr)

st.sidebar.markdown("---")

# NEW: accurate sidebar country selection
all_countries = sorted(df[country_col].dropna().unique().tolist())
if not all_countries:
    st.error("No countries found in dataset.")
    st.stop()

default_idx = 0
if st.session_state.selected_country in all_countries:
    default_idx = all_countries.index(st.session_state.selected_country)
else:
    default_idx = 0
    st.session_state.selected_country = all_countries[0]

picked = st.sidebar.selectbox("Country (accurate selection)", all_countries, index=default_idx)
st.session_state.selected_country = picked

st.sidebar.markdown("---")

# NEW: dataset explanation
with st.sidebar.expander("📚 Datasets Used (What & Why)", expanded=False):
    st.markdown(
        """
**1) Crop Yield / Food Production Dataset (Target)**  
- **Purpose:** Main output variable: **Food_Production_Tonnes** (what we predict).  
- **Why important:** Supports food security decisions like import/export planning, stock planning, and subsidy targeting.

**2) Emissions Dataset (Key Predictors)**  
- **Purpose:** GHG indicators (**CO₂, methane, nitrous oxide, total GHG**) + engineered features like **Carbon Intensity Index**.  
- **Why important:** Strong drivers for prediction; Random Forest captures non-linear relationships between emissions and production.

**3) Climate / Temperature Dataset (Context + Secondary Signal)**  
- **Purpose:** Adds **Avg_Temp** and derived **Temp_Category** (Cold/Moderate/Hot).  
- **Why important:** Useful for interpretation and context; less critical than emissions + yield for core regression accuracy in our results.

**Fusion (data mining step):**  
We align by **Country + Year** and join into one master dataset, ensuring temporal consistency for forecasting.
        """
    )

st.sidebar.caption("Map stays as heatmap. Country selection uses sidebar to avoid hover/click mismatches.")

# =========================
# Snapshot (chosen year)
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

# =========================
# Tabs
# =========================
tabs = st.tabs([
    "🗺️ Global Heatmap",
    "📈 Country Detail (Trends + Scatter + Correlation)",
    "🤖 Prediction (Actual vs Predicted + Year Range)",
    "🧠 Explainable AI (XAI)",
    "🛠️ Train / Retrain Model"
])

# ============================================================
# TAB 1: Global Heatmap (FORCE ISO3 -> fixes wrong hover names)
# ============================================================
with tabs[0]:
    st.markdown('<div class="glass"><h2>🗺️ Global Temperature Category Heatmap</h2></div>', unsafe_allow_html=True)
    st.write("")

    # Build map df: one row per country for selected year
    map_df = df_year.dropna(subset=[country_col]).copy()

    # Aggregate numeric columns by mean
    numeric_cols = map_df.select_dtypes(include="number").columns.tolist()
    agg_dict = {c: "mean" for c in numeric_cols if c != year_col}
    map_df = map_df.groupby(country_col, as_index=False).agg(agg_dict)

    # Ensure temperature category after aggregation
    map_df = ensure_temp_category(map_df)

    # FORCE ISO3 mapping to prevent Plotly guessing wrong country shapes
    map_df["ISO3"] = to_iso3_series(map_df[country_col])

    # Show unmapped countries (won't appear in map)
    bad = map_df["ISO3"].isna()
    if bad.any():
        with st.expander("⚠️ Unmapped countries (not shown on map)", expanded=False):
            st.write(sorted(map_df.loc[bad, country_col].astype(str).unique().tolist()))

    map_df = map_df.loc[~bad].copy()

    if map_df.empty:
        st.error("Map cannot render: no countries could be mapped to ISO3. Install pycountry or fix country names.")
        st.stop()

    # Color map fixed to Temp_Category
    color_map = {"Cold": "#3B82F6", "Moderate": "#22C55E", "Hot": "#F97316"}

    # Prepare customdata for hover
    custom_cols = [country_col, "Temp_Category", "Avg_Temp", "total_ghg", "Food_Production_Tonnes", "ISO3"]
    for c in custom_cols:
        if c not in map_df.columns:
            map_df[c] = np.nan
    map_df["customdata"] = list(map_df[custom_cols].itertuples(index=False, name=None))

    # Always plot using ISO3
    loc_col = "ISO3"
    loc_mode = "ISO-3"

    fig_map = px.choropleth(
        map_df,
        locations=loc_col,
        locationmode=loc_mode,
        color="Temp_Category",
        hover_name=country_col,
        title=f"Temperature Category — {st.session_state.selected_year}",
        color_discrete_map=color_map
    )

    hover_lines = []
    i = custom_cols.index(country_col); hover_lines.append(f"<b>%{{customdata[{i}]}}</b>")
    i = custom_cols.index("ISO3"); hover_lines.append(f"ISO3: %{{customdata[{i}]}}")
    i = custom_cols.index("Temp_Category"); hover_lines.append(f"Temp_Category: %{{customdata[{i}]}}")
    i = custom_cols.index("Avg_Temp"); hover_lines.append(f"Avg_Temp: %{{customdata[{i}]:.2f}} °C")
    i = custom_cols.index("total_ghg"); hover_lines.append(f"total_ghg: %{{customdata[{i}]:.2f}}")
    i = custom_cols.index("Food_Production_Tonnes"); hover_lines.append(f"Food_Production_Tonnes: %{{customdata[{i}]:,.0f}}")

    fig_map.update_traces(
        customdata=np.array(map_df["customdata"].tolist(), dtype=object),
        hovertemplate="<br>".join(hover_lines) + "<extra></extra>"
    )

    # Keep map stable (no drag/zoom)
    fig_map.update_layout(height=640, dragmode=False, uirevision="fixed_map")
    fig_map.update_geos(
        showcountries=True,
        countrycolor="rgba(255,255,255,0.25)",
        showcoastlines=False,
        showframe=False,
        bgcolor="rgba(0,0,0,0)"
    )
    fig_map = plotly_dark(fig_map)

    plotly_config = {"scrollZoom": False, "displayModeBar": False, "doubleClick": False, "staticPlot": False}

    # IMPORTANT FIX:
    # - Always render the figure.
    # - If plotly_events exists, use it to capture click events.
    if HAS_PLOTLY_EVENTS:
        selected_points = plotly_events(
            fig_map,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=640,
            override_width="100%",
            key="global_map_click",
        )

        if selected_points:
            ev = selected_points[0]
            clicked_country = None

            # location should be ISO3 now
            if "location" in ev:
                loc = ev["location"]
                match = map_df[map_df["ISO3"] == loc]
                if len(match):
                    clicked_country = match.iloc[0][country_col]

            if clicked_country is not None:
                st.session_state.selected_country = str(clicked_country)
                st.success(f"Selected country (map click): **{st.session_state.selected_country}**")
            else:
                st.warning("Clicked point couldn’t be mapped. Use the sidebar selector.")
    else:
        st.plotly_chart(fig_map, use_container_width=True, config=plotly_config)

    # Always show current selection
    st.info(f"Selected country (sidebar): **{st.session_state.selected_country}**")

# ============================================================
# TAB 2: Country Detail (charts not tables)
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
            fig_temp = px.line(
                cdf,
                x=year_col,
                y="Avg_Temp",
                markers=True,
                title="Avg Temperature Over Time"
            )
            st.plotly_chart(plotly_dark(fig_temp), use_container_width=True)

        em_cols = [c for c in ["co2", "methane", "nitrous_oxide", "total_ghg"] if c in cdf.columns]
        if em_cols:
            fig_em = px.line(
                cdf,
                x=year_col,
                y=em_cols,
                title="Emissions Over Time (multi-series)"
            )
            st.plotly_chart(plotly_dark(fig_em), use_container_width=True)

    with right:
        st.markdown('<div class="glass"><h3>Production & Efficiency</h3></div>', unsafe_allow_html=True)
        st.write("")

        if "Food_Production_Tonnes" in cdf.columns:
            fig_prod = px.area(
                cdf,
                x=year_col,
                y="Food_Production_Tonnes",
                title="Food Production Over Time"
            )
            st.plotly_chart(plotly_dark(fig_prod), use_container_width=True)

        if "Carbon_Intensity_Index" in cdf.columns:
            fig_cii = px.line(
                cdf,
                x=year_col,
                y="Carbon_Intensity_Index",
                markers=True,
                title="Carbon Intensity Index Over Time"
            )
            st.plotly_chart(plotly_dark(fig_cii), use_container_width=True)

    st.write("")

    s1, s2 = st.columns([1.1, 0.9])

    with s1:
        st.markdown('<div class="glass"><h3>Interactive Scatter: GHG vs Production</h3></div>', unsafe_allow_html=True)
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
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    hoverongaps=False
                )
            )
            fig_corr.update_layout(title="Feature Correlations (Selected Country)")
            st.plotly_chart(plotly_dark(fig_corr), use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation view.")

# ============================================================
# TAB 3: Prediction (Actual vs Predicted + YEAR RANGE)
# ============================================================
with tabs[2]:
    st.markdown('<div class="glass"><h2>🤖 Prediction (Actual vs Predicted + Year Range)</h2></div>', unsafe_allow_html=True)
    st.write("")

    # Load features
    if not FEATS_PATH.exists():
        st.error("❌ model_features.json not found in models/")
        st.stop()

    feats = json.loads(FEATS_PATH.read_text())

    # Prefer trained model (if any), else load saved model
    model = st.session_state.trained_model
    if model is None:
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

    # NEW: year range for historical comparison
    y_min = int(cdf[year_col].min())
    y_max = int(cdf[year_col].max())
    yr_range = st.slider("Year range (historical comparison)", min_value=y_min, max_value=y_max, value=(y_min, y_max))

    cdf_range = cdf[(cdf[year_col] >= yr_range[0]) & (cdf[year_col] <= yr_range[1])].copy()

    X_hist = cdf_range[feats].copy()
    y_true = cdf_range[target_col].copy()

    valid_mask = np.isfinite(X_hist.to_numpy()).all(axis=1) & np.isfinite(y_true.to_numpy())
    cdf_hist = cdf_range.loc[valid_mask].copy()

    if cdf_hist.empty:
        st.warning("No fully valid rows (features+target) for actual vs predicted chart in this year range.")
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
    }).sort_values("Year")

    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=plot_df["Year"],
        y=plot_df["Actual"],
        mode="lines+markers",
        name="Actual"
    ))
    fig_ap.add_trace(go.Scatter(
        x=plot_df["Year"],
        y=plot_df["Predicted"],
        mode="lines+markers",
        name="Predicted"
    ))

    fig_ap.update_layout(
        title=f"Actual vs Predicted Food Production — {ctry} ({yr_range[0]}–{yr_range[1]})",
        xaxis_title="Year",
        yaxis_title="Food_Production_Tonnes",
    )
    st.plotly_chart(plotly_dark(fig_ap), use_container_width=True)

    st.write("")

    # Future Projection beyond 2013
    st.markdown('<div class="glass"><h3>Future Projection (Beyond 2013)</h3></div>', unsafe_allow_html=True)
    st.caption(
        "You don’t have real climate/emissions inputs beyond 2013 in this dataset, "
        "so this section estimates future features using a simple trend per feature per country, "
        "then applies the regression model to get projected production. Use as scenario preview."
    )

    future_end = st.slider("Project until year", min_value=2013, max_value=2050, value=2030)
    if future_end > 2013:
        future_years = list(range(2014, int(future_end) + 1))
        future_features = compute_forecast_features_by_trend(cdf, year_col, feats, future_years)

        if future_features.empty:
            st.info("Not enough data to build future projections.")
        else:
            future_pred = model.predict(future_features[feats])
            future_plot = pd.DataFrame({
                "Year": future_features[year_col].astype(int),
                "Projected": np.array(future_pred, dtype=float)
            }).sort_values("Year")

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
                line=dict(dash="dash")
            ))
            fig_f.update_layout(
                title=f"Historical vs Projected Food Production — {ctry}",
                xaxis_title="Year",
                yaxis_title="Food_Production_Tonnes",
            )
            st.plotly_chart(plotly_dark(fig_f), use_container_width=True)

    st.write("")

    # Residuals
    st.markdown('<div class="glass"><h3>Residuals (Actual − Predicted)</h3></div>', unsafe_allow_html=True)
    res = plot_df.copy()
    res["Residual"] = res["Actual"] - res["Predicted"]
    fig_res = px.bar(res, x="Year", y="Residual", title="Residuals by Year (Positive = under-predicted)")
    st.plotly_chart(plotly_dark(fig_res), use_container_width=True)

# ============================================================
# TAB 4: Explainable AI (XAI) — uses exact assets you showed
# ============================================================
with tabs[3]:
    st.markdown('<div class="glass"><h2>🧠 Explainable AI (XAI)</h2></div>', unsafe_allow_html=True)
    st.write("")

    shap_path = ASSETS / "final_shap_beeswarm.png"
    cluster_path = ASSETS / "cluster_visualization_final.png"

    st.markdown(
        """
<div class="glass">
<h3>Why XAI matters for GEO FOOD SEC</h3>
<p style="opacity:0.92;">
Our system is a <b>decision-support</b> tool. So we must explain <b>why</b> the model predicts a certain production value.
Explainable AI helps policymakers trust the system and verify whether the model is using sensible drivers
(e.g., previous-year production, emissions, carbon intensity).
</p>

<h3>What decision-makers should still know</h3>
<ul style="opacity:0.92;">
  <li><b>Top drivers:</b> Lag feature (previous year production) is often a dominant predictor in time-dependent food data.</li>
  <li><b>Environmental cost:</b> Carbon Intensity and emissions can shift predictions in non-linear ways.</li>
  <li><b>Uncertainty:</b> Natural disasters, policy changes, wars, and sudden economic shocks are not fully captured here.</li>
</ul>
</div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown('<div class="glass"><h3>SHAP Beeswarm (Global Feature Impact)</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
This plot summarizes how each feature contributes to predictions across all countries/years.
Features at the top have the strongest overall effect.
Points to the right increase predicted production; points to the left decrease it.
Color shows whether the feature value is high or low.
</div>
        """,
        unsafe_allow_html=True
    )

    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True, caption="final_shap_beeswarm.png")
    else:
        st.warning("Missing asset: final_shap_beeswarm.png (put it in assets/)")

    st.write("")

    st.markdown('<div class="glass"><h3>Clustering Visualization (Country Group Profiles)</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
This visualization groups countries into clusters with similar emissions/climate/production profiles.
It is mainly used for segmentation and pattern discovery (secondary insight),
while the Random Forest regression model remains the main predictor for production in tonnes.
</div>
        """,
        unsafe_allow_html=True
    )

    if cluster_path.exists():
        st.image(str(cluster_path), use_container_width=True, caption="cluster_visualization_final.png")
    else:
        st.warning("Missing asset: cluster_visualization_final.png (put it in assets/)")

# ============================================================
# TAB 5: Train / Retrain model + year split controls
# ============================================================
with tabs[4]:
    st.markdown('<div class="glass"><h2>🛠️ Train / Retrain Model</h2></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
<div class="glass">
<h3>What this section does</h3>
<p style="opacity:0.92;">
This section lets you retrain a <b>Random Forest Regressor</b> directly from the fused dataset.
You can control the <b>training year range</b> and <b>testing year range</b> so evaluation respects time (no leakage).
When trained, the Prediction tab will automatically use the new model.
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

    # IMPORTANT FIX:
    # Slider defaults must be within [global_min_year, global_max_year] or Streamlit throws ValueError.
    default_train_start = max(global_min_year, 2000)
    default_train_end = min(global_max_year, 2012)
    if default_train_start > default_train_end:
        default_train_start = global_min_year
        default_train_end = min(global_max_year, global_min_year)

    default_test_start = max(global_min_year, 2013)
    default_test_end = min(global_max_year, 2013)
    if default_test_start > default_test_end:
        default_test_start = global_max_year
        default_test_end = global_max_year

    train_range = st.slider(
        "Train years",
        min_value=global_min_year,
        max_value=global_max_year,
        value=(int(default_train_start), int(default_train_end))
    )
    test_range = st.slider(
        "Test years",
        min_value=global_min_year,
        max_value=global_max_year,
        value=(int(default_test_start), int(default_test_end))
    )

    st.write("")
    st.markdown('<div class="glass"><h3>Random Forest Settings</h3></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_estimators = st.number_input("n_estimators", min_value=50, max_value=2000, value=400, step=50)
    with c2:
        max_depth = st.number_input("max_depth (0 = None)", min_value=0, max_value=100, value=0, step=1)
    with c3:
        min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=50, value=1, step=1)

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
