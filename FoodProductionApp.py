import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="GeoFoodSec — Climate–Emissions–Food Production Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Robust project root detection (FIXES missing models/images)
# ============================================================
def find_project_root() -> Path:
    """
    Streamlit Cloud / local runs sometimes change working dir.
    This finds the first parent folder that contains our expected folders.
    """
    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)
    for p in candidates:
        if (p / "data").exists() and (p / "models").exists():
            return p
        if (p / "data").exists() and (p / "models").exists() and (p / "assests").exists():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() and (cwd / "models").exists():
        return cwd
    return here.parent


BASE = find_project_root()
DATA = BASE / "data"
MODELS = BASE / "models"

# tolerate both assets/ and assests/
ASSETS = BASE / "assets"
if not ASSETS.exists():
    ASSETS = BASE / "assests"
if not ASSETS.exists():
    ASSETS = Path.cwd() / "assets"

MASTER_PATH_XLSX = DATA / "Master_Fused_Dataset_2000_2013.xlsx"
MASTER_PATH_CSV = DATA / "Master_Fused_Dataset_2000_2013.csv"

FOOD_MODEL_PATH = MODELS / "best_food_model.pkl"
FOOD_FEATS_PATH = MODELS / "model_features.json"

# ============================================================
# Styling
# ============================================================
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
    h1 {font-weight: 850;}
    h2 {font-weight: 800;}
    h3 {font-weight: 750;}
    .glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .chip {
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding:6px 10px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,0.16);
        background: rgba(255,255,255,0.06);
        margin-right:8px;
        font-size: 0.90rem;
        opacity: 0.95;
    }
    .dot {
        width:12px;height:12px;border-radius:50%;
        display:inline-block;
        border:1px solid rgba(255,255,255,0.25);
    }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    div[data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        font-size: 1.7rem !important;
        line-height: 1.15 !important;
    }
    thead tr th {
        background: rgba(255,255,255,0.05) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================
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
    return MASTER_PATH_XLSX


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


def ensure_temp_category(df_: pd.DataFrame) -> pd.DataFrame:
    if "Temp_Category" not in df_.columns and "Avg_Temp" in df_.columns:
        def cat(t):
            if pd.isna(t):
                return np.nan
            if t < 10:
                return "Cold"
            if t <= 20:
                return "Moderate"
            return "Hot"
        out = df_.copy()
        out["Temp_Category"] = out["Avg_Temp"].apply(cat)
        return out
    return df_


def try_import_pycountry():
    try:
        import pycountry  # type: ignore
        return pycountry
    except Exception:
        return None


def to_iso3_series(country_series: pd.Series) -> pd.Series:
    pycountry = try_import_pycountry()
    if pycountry is None:
        return pd.Series([np.nan] * len(country_series), index=country_series.index)

    manual = {
        "United States": "USA", "USA": "USA", "United States of America": "USA",
        "UK": "GBR", "United Kingdom": "GBR",
        "Russia": "RUS",
        "Iran": "IRN", "Syria": "SYR",
        "Venezuela": "VEN", "Bolivia": "BOL", "Tanzania": "TZA",
        "Viet Nam": "VNM", "Vietnam": "VNM",
        "Lao PDR": "LAO", "Laos": "LAO",
        "Moldova": "MDA", "Czechia": "CZE", "Czech Republic": "CZE",
        "Myanmar": "MMR", "Brunei": "BRN",
        "South Korea": "KOR", "Korea, Rep.": "KOR",
        "North Korea": "PRK", "Korea, Dem. Rep.": "PRK",
        "Congo": "COG", "Republic of the Congo": "COG", "Congo, Rep.": "COG",
        "Democratic Republic of the Congo": "COD", "Congo, Dem. Rep.": "COD", "DR Congo": "COD", "D.R. Congo": "COD",
        "Ivory Coast": "CIV", "Côte d’Ivoire": "CIV", "Côte d'Ivoire": "CIV", "Cote d'Ivoire": "CIV",
        "Palestine": "PSE", "State of Palestine": "PSE",
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


def compute_forecast_features_by_trend(cdf: pd.DataFrame, year_col: str, feature_cols: list, future_years: list) -> pd.DataFrame:
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


def year_over_year_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(1)
    pct = (s - prev) / prev.replace(0, np.nan) * 100
    return pct


def safe_load_model(path: Path, label: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(
            f"❌ Could not load **{label}** ({path.name}).\n\n"
            f"Error: {e}\n\n"
            "This usually happens when the model was saved under different NumPy / scikit-learn versions.\n"
            "Fix: pin versions in `requirements.txt` to match the training environment, or re-save the model after upgrading."
        )
        return None


def find_asset(filename: str) -> Path | None:
    candidates = [
        ASSETS / filename,
        BASE / filename,
        Path.cwd() / filename,
        Path(filename),
    ]
    for p in candidates:
        if p.exists():
            return p

    if ASSETS.exists() and ASSETS.is_dir():
        target = filename.lower()
        for p in ASSETS.iterdir():
            if p.is_file() and p.name.lower() == target:
                return p

    stem = Path(filename).stem.lower()
    if ASSETS.exists() and ASSETS.is_dir():
        for p in ASSETS.iterdir():
            if p.is_file() and p.stem.lower() == stem:
                return p

    return None


# ============================================================
# Load data
# ============================================================
MASTER_PATH = get_master_path()
if not MASTER_PATH.exists():
    st.error(
        "❌ Missing dataset file in /data.\n"
        f"Expected one of:\n- {MASTER_PATH_XLSX.name}\n- {MASTER_PATH_CSV.name}\n\n"
        f"Detected BASE folder: {BASE}\n"
        f"DATA folder exists: {DATA.exists()}\nMODELS folder exists: {MODELS.exists()}\nASSETS folder exists: {ASSETS.exists()}"
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

# ============================================================
# Session state
# ============================================================
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None
if "selected_year" not in st.session_state:
    st.session_state.selected_year = int(df[year_col].dropna().min())

if "trained_food_model" not in st.session_state:
    st.session_state.trained_food_model = None
if "trained_food_feats" not in st.session_state:
    st.session_state.trained_food_feats = None

# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="glass">
      <h1>🌍 GeoFoodSec — Decision Dashboard</h1>
      <div style="opacity:0.9; font-size: 0.98rem;">
        One view connecting <b>Climate → Emissions → Food Production</b>.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Controls")

miny = int(df[year_col].dropna().min())
maxy = int(df[year_col].dropna().max())

yr = st.sidebar.slider("Year (global snapshot)", miny, maxy, st.session_state.selected_year)
st.session_state.selected_year = int(yr)

st.sidebar.markdown("---")

all_countries = sorted(df[country_col].dropna().unique().tolist())
if not all_countries:
    st.error("No countries found in dataset.")
    st.stop()

if st.session_state.selected_country not in all_countries:
    st.session_state.selected_country = all_countries[0]

picked = st.sidebar.selectbox(
    "Country (for detail + forecasts)",
    all_countries,
    index=all_countries.index(st.session_state.selected_country),
)
st.session_state.selected_country = picked

st.sidebar.markdown("---")
with st.sidebar.expander("What this tool answers"):
    st.markdown(
        """
- **Food production forecasting:** estimated tonnes by country  
- **Drivers:** temperature + emissions + carbon intensity  
- **Insights:** clustering-based peer groups for comparison  
        """
    )

# ============================================================
# Snapshot KPIs
# ============================================================
df_year = df[df[year_col] == st.session_state.selected_year].copy()

def safe_mean(col):
    return float(df_year[col].mean()) if col in df_year.columns else np.nan

def safe_sum(col):
    return float(df_year[col].sum()) if col in df_year.columns else np.nan

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Countries in snapshot", f"{df_year[country_col].nunique():,}")
with kpi_cols[1]:
    st.metric("Average temperature", f"{safe_mean('Avg_Temp'):.2f} °C" if "Avg_Temp" in df_year.columns else "N/A")
with kpi_cols[2]:
    st.metric("Total greenhouse gas (sum)", f"{safe_sum('total_ghg'):,.2f}" if "total_ghg" in df_year.columns else "N/A")
with kpi_cols[3]:
    st.metric("Food production (sum)", f"{safe_sum('Food_Production_Tonnes'):,.0f}" if "Food_Production_Tonnes" in df_year.columns else "N/A")

st.write("")

# ============================================================
# Tabs (✅ Clustering Insight removed)
# ============================================================
tabs = st.tabs([
    "🌐 Global Overview",
    "📈 Country Insights",
    "📊 Forecast & Change",
    "🧠 Explainability (XAI)",
    "🛠️ Model Tuning",
])

# ============================================================
# TAB 1: Global Overview
# ============================================================
with tabs[0]:
    st.markdown('<div class="glass"><h2>🌐 Global Overview</h2></div>', unsafe_allow_html=True)
    st.write("")

    metric_choices = [m for m in ["Temp_Category", "Avg_Temp", "total_ghg", "Food_Production_Tonnes"] if m in df_year.columns]
    if not metric_choices:
        st.warning("No usable global metrics found for the selected year.")
        st.stop()

    chosen_metric = st.selectbox("What to display on the world map", metric_choices, index=0)

    if "Temp_Category" in metric_choices:
        st.markdown(
            """
            <div style="margin-top:6px; margin-bottom: 6px;">
              <span class="chip"><span class="dot" style="background:#3B82F6;"></span> Cold</span>
              <span class="chip"><span class="dot" style="background:#22C55E;"></span> Moderate</span>
              <span class="chip"><span class="dot" style="background:#F97316;"></span> Hot</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    map_df = df_year.dropna(subset=[country_col]).copy()
    numeric_cols = map_df.select_dtypes(include="number").columns.tolist()
    agg_dict = {c: "mean" for c in numeric_cols if c != year_col}
    grouped = map_df.groupby(country_col, as_index=False).agg(agg_dict)
    grouped = ensure_temp_category(grouped)
    grouped["ISO3"] = to_iso3_series(grouped[country_col])

    unmapped = grouped["ISO3"].isna().sum()
    total = len(grouped)
    unmapped_ratio = (unmapped / total) if total else 1.0

    pycountry_ok = try_import_pycountry() is not None
    map_ok = pycountry_ok and total > 0 and unmapped_ratio <= 0.20

    st.markdown('<div class="glass"><h3>World Map</h3></div>', unsafe_allow_html=True)
    st.write("")

    if map_ok:
        plot_df = grouped.dropna(subset=["ISO3"]).copy()

        if chosen_metric == "Temp_Category":
            color_map = {"Cold": "#3B82F6", "Moderate": "#22C55E", "Hot": "#F97316"}
            fig = px.choropleth(
                plot_df,
                locations="ISO3",
                locationmode="ISO-3",
                color="Temp_Category",
                hover_name=country_col,
                title=f"Temperature Category — {st.session_state.selected_year}",
                color_discrete_map=color_map,
            )
        else:
            fig = px.choropleth(
                plot_df,
                locations="ISO3",
                locationmode="ISO-3",
                color=chosen_metric,
                hover_name=country_col,
                title=f"{chosen_metric} — {st.session_state.selected_year}",
                color_continuous_scale="Turbo",
            )

        fig.update_layout(height=640, dragmode=False, uirevision="fixed_map")
        fig.update_geos(
            showcountries=True,
            countrycolor="rgba(255,255,255,0.25)",
            showcoastlines=False,
            showframe=False,
            bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(plotly_dark(fig), use_container_width=True, config={"displayModeBar": False})
    else:
        why = []
        if not pycountry_ok:
            why.append("• `pycountry` is not installed (needed for ISO3 mapping).")
        if total == 0:
            why.append("• No country rows found for the selected year.")
        if unmapped_ratio > 0.20:
            why.append(f"• Too many unmapped countries: {unmapped}/{total} (~{unmapped_ratio*100:.1f}%).")
        st.warning("Map is disabled to avoid incorrect hover labels.\n\n" + "\n".join(why))

    st.write("")
    st.markdown('<div class="glass"><h3>Country Summary Table</h3></div>', unsafe_allow_html=True)
    st.write("")

    show_cols = [country_col]
    for c in ["Temp_Category", "Avg_Temp", "total_ghg", "Food_Production_Tonnes", "Carbon_Intensity_Index"]:
        if c in grouped.columns and c not in show_cols:
            show_cols.append(c)

    table_df = grouped[show_cols].copy()
    if chosen_metric in table_df.columns and chosen_metric != "Temp_Category":
        table_df = table_df.sort_values(chosen_metric, ascending=False)

    def style_temp_category(val):
        if str(val) == "Cold":
            return "background-color: rgba(59,130,246,0.25);"
        if str(val) == "Moderate":
            return "background-color: rgba(34,197,94,0.20);"
        if str(val) == "Hot":
            return "background-color: rgba(249,115,22,0.20);"
        return ""

    styler = table_df.style
    if "Temp_Category" in table_df.columns:
        styler = styler.applymap(style_temp_category, subset=["Temp_Category"])

    st.dataframe(styler, use_container_width=True, height=520)

# ============================================================
# TAB 2: Country Insights
# ============================================================
with tabs[1]:
    st.markdown('<div class="glass"><h2>📈 Country Insights</h2></div>', unsafe_allow_html=True)
    st.write("")

    ctry = st.session_state.selected_country
    cdf = df[df[country_col] == ctry].copy().sort_values(year_col)
    if cdf.empty:
        st.warning("No rows for selected country.")
        st.stop()

    latest = cdf.dropna(subset=[year_col]).sort_values(year_col).tail(1).iloc[0]

    c_kpis = st.columns(4)
    c_kpis[0].metric("Selected country", ctry)
    c_kpis[1].metric("Latest temperature category", str(latest.get("Temp_Category", "N/A")))
    c_kpis[2].metric(
        "Latest average temperature",
        f"{float(latest['Avg_Temp']):.2f} °C" if "Avg_Temp" in cdf.columns and pd.notna(latest.get("Avg_Temp")) else "N/A"
    )
    c_kpis[3].metric(
        "Latest food production",
        f"{float(latest['Food_Production_Tonnes']):,.0f} tonnes" if "Food_Production_Tonnes" in df.columns and pd.notna(latest.get("Food_Production_Tonnes")) else "N/A"
    )

    st.write("")
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="glass"><h3>Climate & Emissions over time</h3></div>', unsafe_allow_html=True)
        st.write("")

        if "Avg_Temp" in cdf.columns:
            fig_temp = px.line(cdf, x=year_col, y="Avg_Temp", markers=True, title="Average Temperature")
            st.plotly_chart(plotly_dark(fig_temp), use_container_width=True)

        em_cols = [c for c in ["co2", "methane", "nitrous_oxide", "total_ghg"] if c in cdf.columns]
        if em_cols:
            fig_em = px.line(cdf, x=year_col, y=em_cols, title="Emissions (multiple gases)")
            st.plotly_chart(plotly_dark(fig_em), use_container_width=True)

    with right:
        st.markdown('<div class="glass"><h3>Production & efficiency</h3></div>', unsafe_allow_html=True)
        st.write("")

        if "Food_Production_Tonnes" in cdf.columns:
            fig_prod = px.area(cdf, x=year_col, y="Food_Production_Tonnes", title="Food Production")
            st.plotly_chart(plotly_dark(fig_prod), use_container_width=True)

        if "Carbon_Intensity_Index" in cdf.columns:
            fig_cii = px.line(cdf, x=year_col, y="Carbon_Intensity_Index", markers=True, title="Carbon Intensity Index")
            st.plotly_chart(plotly_dark(fig_cii), use_container_width=True)

    st.write("")
    s1, s2 = st.columns([1.1, 0.9])

    with s1:
        st.markdown('<div class="glass"><h3>Relationship: emissions vs production</h3></div>', unsafe_allow_html=True)
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
                title="GHG vs Food Production (each point is a year)",
            )
            st.plotly_chart(plotly_dark(fig_scatter), use_container_width=True)
        else:
            st.info("Need 'total_ghg' and 'Food_Production_Tonnes' columns for this chart.")

    with s2:
        st.markdown('<div class="glass"><h3>Correlation (quick check)</h3></div>', unsafe_allow_html=True)
        st.write("")
        keep = [c for c in ["Avg_Temp", "co2", "methane", "nitrous_oxide", "total_ghg", "Carbon_Intensity_Index", "Food_Production_Tonnes"] if c in cdf.columns]
        if len(keep) >= 3:
            corr = cdf[keep].corr(numeric_only=True)
            fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, hoverongaps=False))
            fig_corr.update_layout(title="Correlation heatmap (selected country)")
            st.plotly_chart(plotly_dark(fig_corr), use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation view.")

# ============================================================
# TAB 3: Forecast & Change
# ============================================================
with tabs[2]:
    st.markdown('<div class="glass"><h2>📊 Forecast & Change</h2></div>', unsafe_allow_html=True)
    st.write("")

    if not FOOD_FEATS_PATH.exists():
        st.error(f"❌ Missing {FOOD_FEATS_PATH} (model_features.json)")
        st.stop()
    food_feats = json.loads(FOOD_FEATS_PATH.read_text())

    food_model = st.session_state.trained_food_model
    if food_model is None:
        if not FOOD_MODEL_PATH.exists():
            st.error(f"❌ Missing {FOOD_MODEL_PATH} (best_food_model.pkl)")
            st.stop()
        food_model = safe_load_model(FOOD_MODEL_PATH, "Food production model")
    if food_model is None:
        st.stop()

    target_col = "Food_Production_Tonnes"
    if target_col not in df.columns:
        st.error("❌ Dataset missing target column 'Food_Production_Tonnes'.")
        st.stop()

    missing_feats = [f for f in food_feats if f not in df.columns]
    if missing_feats:
        st.error(f"❌ Dataset missing required feature columns: {missing_feats}")
        st.stop()

    ctry = st.session_state.selected_country
    cdf = df[df[country_col] == ctry].copy().sort_values(year_col).dropna(subset=[year_col])
    if cdf.empty:
        st.warning("No rows for selected country.")
        st.stop()

    y_min = int(cdf[year_col].min())
    y_max = int(cdf[year_col].max())

    yr_range = st.slider("Choose historical years for comparison", min_value=y_min, max_value=y_max, value=(y_min, y_max))
    cdf_range = cdf[(cdf[year_col] >= yr_range[0]) & (cdf[year_col] <= yr_range[1])].copy()

    X_hist = cdf_range[food_feats].copy()
    y_true = cdf_range[target_col].copy()
    valid_mask = np.isfinite(X_hist.to_numpy()).all(axis=1) & np.isfinite(y_true.to_numpy())
    cdf_hist = cdf_range.loc[valid_mask].copy()

    if cdf_hist.empty:
        st.warning("No fully valid rows (features + target) for the selected range.")
        st.stop()

    X_hist = cdf_hist[food_feats]
    y_true = cdf_hist[target_col]
    y_pred = food_model.predict(X_hist)

    m = metrics_regression(y_true, y_pred)
    met_cols = st.columns(4)
    met_cols[0].metric("Average error (MAE)", f"{m['MAE']:,.2f}" if np.isfinite(m["MAE"]) else "N/A")
    met_cols[1].metric("Typical error size (RMSE)", f"{m['RMSE']:,.2f}" if np.isfinite(m["RMSE"]) else "N/A")
    met_cols[2].metric("Percentage error (MAPE)", f"{m['MAPE_%']:.2f}%" if np.isfinite(m["MAPE_%"]) else "N/A")
    met_cols[3].metric("Fit score (R²)", f"{m['R2']:.3f}" if np.isfinite(m["R2"]) else "N/A")

    plot_df = pd.DataFrame({
        "Year": cdf_hist[year_col].astype(int).values,
        "Actual production (tonnes)": y_true.values.astype(float),
        "Model estimate (tonnes)": np.array(y_pred, dtype=float),
    }).sort_values("Year")

    plot_df["Actual year-over-year change (%)"] = year_over_year_pct(plot_df["Actual production (tonnes)"])
    plot_df["Estimated year-over-year change (%)"] = year_over_year_pct(plot_df["Model estimate (tonnes)"])

    latest_actual_yoy = plot_df["Actual year-over-year change (%)"].dropna().iloc[-1] if plot_df["Actual year-over-year change (%)"].dropna().shape[0] else np.nan
    latest_pred_yoy = plot_df["Estimated year-over-year change (%)"].dropna().iloc[-1] if plot_df["Estimated year-over-year change (%)"].dropna().shape[0] else np.nan

    yoy_cols = st.columns(2)
    yoy_cols[0].metric("Latest year-over-year change (Actual)", f"{latest_actual_yoy:.2f}%" if np.isfinite(latest_actual_yoy) else "N/A")
    yoy_cols[1].metric("Latest year-over-year change (Model estimate)", f"{latest_pred_yoy:.2f}%" if np.isfinite(latest_pred_yoy) else "N/A")

    st.write("")
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(x=plot_df["Year"], y=plot_df["Actual production (tonnes)"], mode="lines+markers", name="Actual"))
    fig_ap.add_trace(go.Scatter(x=plot_df["Year"], y=plot_df["Model estimate (tonnes)"], mode="lines+markers", name="Model estimate"))
    fig_ap.update_layout(
        title=f"Food Production — Actual vs Model Estimate ({ctry})",
        xaxis_title="Year",
        yaxis_title="Tonnes",
    )
    st.plotly_chart(plotly_dark(fig_ap), use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.markdown('<div class="glass"><h3>Projection (future estimate)</h3></div>', unsafe_allow_html=True)
    st.caption("For years beyond the dataset, the dashboard estimates missing inputs using a simple trend per feature, then runs the same model.")

    future_end = st.slider("Project until year", min_value=int(y_max), max_value=2050, value=min(2030, 2050))
    future_plot = None
    if future_end > y_max:
        future_years = list(range(int(y_max) + 1, int(future_end) + 1))
        future_features = compute_forecast_features_by_trend(cdf, year_col, food_feats, future_years)
        if not future_features.empty:
            future_pred = food_model.predict(future_features[food_feats])
            future_plot = pd.DataFrame({
                "Year": future_features[year_col].astype(int),
                "Projected production (tonnes)": np.array(future_pred, dtype=float),
            }).sort_values("Year")
            future_plot["Projected year-over-year change (%)"] = year_over_year_pct(future_plot["Projected production (tonnes)"])

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=plot_df["Year"], y=plot_df["Actual production (tonnes)"], mode="lines+markers", name="Actual (historical)"))
            fig_f.add_trace(go.Scatter(x=plot_df["Year"], y=plot_df["Model estimate (tonnes)"], mode="lines+markers", name="Model estimate (historical)"))
            fig_f.add_trace(go.Scatter(
                x=future_plot["Year"], y=future_plot["Projected production (tonnes)"],
                mode="lines", name=f"Projection ({future_plot['Year'].min()}–{future_plot['Year'].max()})",
                line=dict(dash="dash")
            ))
            fig_f.update_layout(
                title=f"Historical + Projection — Food Production ({ctry})",
                xaxis_title="Year",
                yaxis_title="Tonnes",
            )
            st.plotly_chart(plotly_dark(fig_f), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Not enough data to build projections for this country.")

    st.write("")
    st.markdown('<div class="glass"><h3>Model gap (Residuals)</h3></div>', unsafe_allow_html=True)
    st.caption("Residual = Actual − Model estimate. Residuals are only shown for years where actual data exists.")

    res = plot_df[["Year", "Actual production (tonnes)", "Model estimate (tonnes)"]].copy()
    res["Residual (tonnes)"] = res["Actual production (tonnes)"] - res["Model estimate (tonnes)"]

    fig_res = px.bar(res, x="Year", y="Residual (tonnes)", title="Residuals by Year (tonnes)")
    st.plotly_chart(plotly_dark(fig_res), use_container_width=True, config={"displayModeBar": False})

    with st.expander("Show change table (year-over-year %)"):
        show = plot_df[["Year", "Actual year-over-year change (%)", "Estimated year-over-year change (%)"]].copy()
        st.dataframe(show, use_container_width=True)

        if future_plot is not None:
            st.write("")
            st.markdown("**Projection change (year-over-year %)**")
            st.dataframe(future_plot[["Year", "Projected year-over-year change (%)"]], use_container_width=True)

# ============================================================
# TAB 4: Explainability (XAI) — ONLY ONE IMAGE
# ============================================================
with tabs[3]:
    st.markdown('<div class="glass"><h2>🧠 Explainability (XAI)</h2></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
<h3 style="margin-top:0;">What this XAI section explains</h3>
This dashboard uses a trained model to estimate <b>Food Production (tonnes)</b> from climate and emissions features.
Explainability helps answer:
<ul>
  <li><b>Which features matter most overall?</b></li>
  <li><b>When a prediction goes up/down, which feature pushes it?</b></li>
  <li><b>Why can two countries with similar emissions have different predicted production?</b> (Because other inputs differ.)</li>
</ul>

<b>Important:</b> Explainability shows <i>model behavior</i>, not direct causation.
It tells us what the model learned from dataset patterns.
</div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    st.markdown('<div class="glass"><h3>Food production: feature impact summary (SHAP)</h3></div>', unsafe_allow_html=True)
    st.write("")

    fp_shap = find_asset("final_shap_beeswarm.png")
    if fp_shap is not None:
        st.image(str(fp_shap), use_container_width=True, caption=f"Loaded from: {fp_shap}")

        st.markdown(
            """
<div class="glass" style="opacity:0.92;">
<h3 style="margin-top:0;">How to read the SHAP beeswarm</h3>

<b>1) Each dot = one country-year record.</b><br>
The plot summarizes how each input feature influenced the model prediction.

<br><br>
<b>2) Left vs right = decreases vs increases prediction.</b><br>
• Dots on the <b>right</b> push predicted food production <b>higher</b><br>
• Dots on the <b>left</b> push predicted food production <b>lower</b>

<br><br>
<b>3) Color shows whether the feature value is high or low.</b><br>
In most SHAP plots: <b>red = high</b>, <b>blue = low</b>.  
So if a feature has many <b>red dots on the right</b>, high values tend to increase predicted production.

<br><br>
<b>4) Features are sorted by importance.</b><br>
Top rows have the biggest average impact across the dataset.

<br><br>
<b>What you can write in your report (example interpretation):</b><br>
• The model is most sensitive to the top 3–5 features shown.<br>
• Production is influenced by a combination of climate conditions (e.g., temperature) and emissions-related indicators.<br>
• Some features may have mixed effects (dots on both sides), meaning the effect changes by country/year context.
</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(
            "Missing image: final_shap_beeswarm.png\n\n"
            f"Checked assets folder: {ASSETS}\n"
            f"Assets exists: {ASSETS.exists()}"
        )

# ============================================================
# TAB 5: Model Tuning
# ============================================================
with tabs[4]:
    st.markdown('<div class="glass"><h2>🛠️ Model Tuning</h2></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
<div class="glass" style="opacity:0.92;">
Use this section only if you want to test a different Random Forest setup.
You can always restore the <b>original baseline model</b> with one click.
</div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
    except Exception:
        st.warning("Training requires scikit-learn installed. You can still use the saved baseline model.")
        RandomForestRegressor = None  # type: ignore

    top_actions = st.columns([1, 1, 1])
    with top_actions[0]:
        if st.button("✅ Restore baseline model (recommended)"):
            st.session_state.trained_food_model = None
            st.session_state.trained_food_feats = None
            st.success("Baseline model restored. Forecasts will use models/best_food_model.pkl")

    st.write("")
    st.markdown('<div class="glass"><h3>Training controls (simple wording)</h3></div>', unsafe_allow_html=True)
    st.write("")

    if RandomForestRegressor is None:
        st.stop()

    trees = st.number_input("Model strength (number of decision trees)", min_value=50, max_value=1500, value=400, step=50)
    depth = st.number_input("Model detail level (0 means no limit)", min_value=0, max_value=60, value=0, step=1)
    smooth = st.number_input("Smoothing level (minimum samples per split endpoint)", min_value=1, max_value=50, value=12, step=1)

    model_target = "Food_Production_Tonnes"
    if model_target not in df.columns:
        st.error("Missing Food_Production_Tonnes in dataset.")
        st.stop()

    if not FOOD_FEATS_PATH.exists():
        st.error(f"Missing {FOOD_FEATS_PATH} in models/.")
        st.stop()

    train_feats = json.loads(FOOD_FEATS_PATH.read_text())
    missing = [c for c in train_feats + [model_target] if c not in df.columns]
    if missing:
        st.error(f"Dataset missing columns required for training: {missing}")
        st.stop()

    train_df = df.dropna(subset=train_feats + [model_target]).copy()
    if train_df.empty:
        st.error("No usable rows for training (too many missing values).")
        st.stop()

    X = train_df[train_feats].astype(float)
    y = train_df[model_target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

    train_stats = st.columns(3)
    train_stats[0].metric("Training rows", f"{len(X_train):,}")
    train_stats[1].metric("Testing rows", f"{len(X_test):,}")
    train_stats[2].metric("Inputs used", f"{len(train_feats)}")

    if st.button("🚀 Train tuned model"):
        max_depth = None if int(depth) == 0 else int(depth)
        rf = RandomForestRegressor(
            n_estimators=int(trees),
            max_depth=max_depth,
            min_samples_leaf=int(smooth),
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        mt = metrics_regression(y_test, preds)

        st.session_state.trained_food_model = rf
        st.session_state.trained_food_feats = train_feats

        st.success("Tuned model trained and activated for forecasting.")
        perf = st.columns(4)
        perf[0].metric("MAE", f"{mt['MAE']:,.2f}")
        perf[1].metric("RMSE", f"{mt['RMSE']:,.2f}")
        perf[2].metric("MAPE", f"{mt['MAPE_%']:.2f}%")
        perf[3].metric("R²", f"{mt['R2']:.3f}")
