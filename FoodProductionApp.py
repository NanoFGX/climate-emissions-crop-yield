import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(page_title="Crop Yield Dashboard", layout="wide")

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"
ASSETS = BASE / "assets"

st.title("🌍 Climate–Emissions–Agriculture Decision Dashboard")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Could not load model: {path.name}\n\nError: {e}\n\n"
                 "This is usually a NumPy / scikit-learn version mismatch.\n"
                 "Fix by pinning the exact versions used in Colab inside requirements.txt, "
                 "or re-saving the model after upgrading libraries in Colab.")
        return None

# -----------------------
# Sidebar: choose dataset
# -----------------------
st.sidebar.header("Dataset Viewer")

dataset_options = {
    "Master Fused Dataset (2000–2013)": "Master_Fused_Dataset_2000_2013.csv",
    "Setting 1 (ChiSquare)": "Dataset_Setting1_ChiSquare.csv",
    "Setting 2 (IQR)": "Dataset_Setting2_IQR.csv",
    "Setting 3 (OneHot)": "Dataset_Setting3_OneHot.csv",
    "Setting 4 (Discretized)": "Dataset_Setting4_Discretized.csv",
}

selected_name = st.sidebar.selectbox("Pick a dataset", list(dataset_options.keys()))
df = load_csv(DATA / dataset_options[selected_name])

# Detect basic columns
country_col = "Country" if "Country" in df.columns else None
year_col = "Year" if "Year" in df.columns else None

# Filters
filtered = df.copy()

if country_col:
    countries = sorted(filtered[country_col].dropna().unique())
    chosen_countries = st.sidebar.multiselect("Country", countries, default=countries[:10] if len(countries) > 10 else countries)
    if chosen_countries:
        filtered = filtered[filtered[country_col].isin(chosen_countries)]

if year_col:
    miny, maxy = int(filtered[year_col].min()), int(filtered[year_col].max())
    yr = st.sidebar.slider("Year Range", miny, maxy, (miny, maxy))
    filtered = filtered[(filtered[year_col] >= yr[0]) & (filtered[year_col] <= yr[1])]

st.sidebar.markdown("---")
rows_show = st.sidebar.slider("Rows to show", 10, 300, 50)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (filtered)", f"{len(filtered):,}")
k2.metric("Columns", f"{filtered.shape[1]:,}")
k3.metric("Missing Cells", f"{int(filtered.isna().sum().sum()):,}")
k4.metric("Countries", f"{filtered[country_col].nunique() if country_col else 'N/A'}")

tabs = st.tabs(["📄 Data", "🗺️ Global Map", "🤖 Predict (Regression)", "🧠 XAI (SHAP)", "🧩 Clustering", "✅ Classification Results"])

# -----------------------
# TAB 1: Data Table
# -----------------------
with tabs[0]:
    st.subheader(f"Dataset: {selected_name}")
    st.dataframe(filtered.head(rows_show), use_container_width=True)

# -----------------------
# TAB 2: Global Map
# -----------------------
with tabs[1]:
    st.subheader("🗺️ Global Choropleth Map")

    if not country_col:
        st.warning("No 'Country' column found in this dataset.")
    else:
        numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available to visualize.")
        else:
            value_col = st.selectbox("Value to map", numeric_cols)

            map_df = filtered.copy()
            if year_col:
                chosen_year = st.selectbox("Year for map", sorted(map_df[year_col].dropna().unique()))
                map_df = map_df[map_df[year_col] == chosen_year]

            # Aggregate by country (important for mapping)
            map_df = map_df.groupby(country_col, as_index=False)[value_col].mean()

            fig = px.choropleth(
                map_df,
                locations=country_col,
                locationmode="country names",
                color=value_col,
                hover_name=country_col,
                projection="natural earth"
            )
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# TAB 3: Regression Prediction (Food Production)
# -----------------------
with tabs[2]:
    st.subheader("🤖 Food Production Prediction (Regression)")

    model = safe_load_model(MODELS / "best_food_model.pkl")
    if model is not None:
        # Load expected features
        with open(MODELS / "model_features.json") as f:
            feats = json.load(f)

        st.write("Model expects features:", feats)

        # Choose a row from filtered dataset to predict
        st.caption("Pick a row (country-year) to run prediction using the model features.")
        idx = st.number_input("Row index (from filtered view)", min_value=0, max_value=max(0, len(filtered)-1), value=0)

        row = filtered.iloc[int(idx)]
        # Build input vector
        missing = [c for c in feats if c not in filtered.columns]
        if missing:
            st.error(f"Dataset missing required columns: {missing}")
        else:
            X = pd.DataFrame([row[feats].to_dict()])
            pred = model.predict(X)[0]
            st.success(f"Predicted Food Production (tonnes): **{pred:,.2f}**")
            st.dataframe(X, use_container_width=True)

# -----------------------
# TAB 4: XAI (SHAP)
# -----------------------
with tabs[3]:
    st.subheader("🧠 Explainable AI (SHAP Summary)")
    shap_img = ASSETS / "final_shap_beeswarm.png"
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
        st.info("This plot shows which features push predictions higher or lower, and how strongly.")
    else:
        st.warning("SHAP image not found in assets/")

# -----------------------
# TAB 5: Clustering
# -----------------------
with tabs[4]:
    st.subheader("🧩 Clustering Results")

    clust_img = ASSETS / "cluster_visualization_final.png"
    if clust_img.exists():
        st.image(str(clust_img), use_container_width=True)

    # Summary table
    summary_path = DATA / "clustering_results_summary.csv"
    if summary_path.exists():
        summary = load_csv(summary_path)
        st.dataframe(summary, use_container_width=True)
    else:
        st.warning("clustering_results_summary.csv not found.")

# -----------------------
# TAB 6: Classification Results
# -----------------------
with tabs[5]:
    st.subheader("✅ Classification Results (All Models / Settings)")
    cls_path = DATA / "classification_full_results.csv"
    if cls_path.exists():
        cls = load_csv(cls_path)
        st.dataframe(cls, use_container_width=True)
    else:
        st.warning("classification_full_results.csv not found.")
