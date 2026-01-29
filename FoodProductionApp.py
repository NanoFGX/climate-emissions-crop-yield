import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page + Theme
# -----------------------------
st.set_page_config(page_title="Water Quality Dashboard", page_icon="💧", layout="wide")

def set_app_style():
    st.markdown(
        """
        <style>
          :root{
            --bg: #0B0F17;
            --panel: rgba(255,255,255,0.04);
            --panel2: rgba(255,255,255,0.06);
            --text: rgba(255,255,255,0.92);
            --muted: rgba(255,255,255,0.65);
            --accent: #7DF9FF;
          }
          html, body, [class*="css"]  { color: var(--text); }
          .stApp { background: radial-gradient(900px 500px at 15% 10%, rgba(125,249,255,0.08), transparent 50%),
                           radial-gradient(900px 500px at 85% 0%, rgba(179,136,255,0.10), transparent 50%),
                           var(--bg); }
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
          h1,h2,h3 { letter-spacing: -0.02em; }
          .small-note { color: var(--muted); font-size: 0.9rem; }
          .chip { display:inline-block; padding:6px 10px; border-radius:999px; background: var(--panel); border: 1px solid rgba(255,255,255,0.08); margin-right:8px; color: var(--muted); }
          div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 14px;
          }
          section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            border-right: 1px solid rgba(255,255,255,0.06);
          }
          .stTabs [data-baseweb="tab"]{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 999px;
            padding: 10px 14px;
            margin-right: 8px;
          }
          .stTabs [aria-selected="true"]{
            border-color: rgba(125,249,255,0.35);
            box-shadow: 0 0 0 2px rgba(125,249,255,0.08) inset;
          }
          .panel {
            background: var(--panel);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 14px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_app_style()

# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_data(mat_path):
    if not os.path.exists(mat_path):
        st.error(f"Error: {mat_path} not found. Please upload it in the same folder.")
        st.stop()

    data = loadmat(mat_path)
    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    X_te, Y_te = data["X_te"], data["Y_te"]

    Xtr = np.stack([X_tr[0, t] for t in range(X_tr.shape[1])], axis=0)
    Xte = np.stack([X_te[0, t] for t in range(X_te.shape[1])], axis=0)
    Ytr = Y_tr.T
    Yte = Y_te.T

    # Standardization (z-score) using train stats
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr_s = (Xtr - mu) / std
    Xte_s = (Xte - mu) / std

    L, F = 37, 11
    feature_cols = [f"L{l+1}_F{f+1}" for l in range(L) for f in range(F)]
    target_cols = [f"L{l+1}_pH" for l in range(L)]

    train_df = pd.concat([
        pd.DataFrame(Xtr_s.reshape(Xtr_s.shape[0], -1), columns=feature_cols),
        pd.DataFrame(Ytr, columns=target_cols)
    ], axis=1)

    test_df = pd.concat([
        pd.DataFrame(Xte_s.reshape(Xte_s.shape[0], -1), columns=feature_cols),
        pd.DataFrame(Yte, columns=target_cols)
    ], axis=1)

    return train_df, test_df, feature_cols, target_cols

# -----------------------------
# Label function
# -----------------------------
def ph_to_class_ui(ph, th0, th1):
    if ph < th0:
        return 0
    elif ph <= th1:
        return 1
    else:
        return 2

# -----------------------------
# Models
# -----------------------------
@st.cache_resource
def train_classification_models(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    reports, predictions, trained = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reports[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        predictions[name] = y_pred
        trained[name] = model

    return reports, predictions, trained

@st.cache_resource
def load_ts_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Error: {model_path} not found. Please ensure the model exists.")
        st.stop()
    return joblib.load(model_path)

def build_performance_df(classification_reports):
    rows = []
    for model_name, report in classification_reports.items():
        for cls in ["0", "1", "2"]:
            if cls in report:
                rows.append({
                    "Model": model_name,
                    "Class": int(cls),
                    "Precision": report[cls]["precision"],
                    "Recall": report[cls]["recall"],
                    "F1": report[cls]["f1-score"],
                    "Support": report[cls]["support"],
                })
        rows.append({
            "Model": model_name,
            "Class": "Overall",
            "Precision": report["macro avg"]["precision"],
            "Recall": report["macro avg"]["recall"],
            "F1": report["macro avg"]["f1-score"],
            "Accuracy": report.get("accuracy", np.nan),
        })
    return pd.DataFrame(rows)

# -----------------------------
# Country Heatmap (fixed scale + highlight only)
# -----------------------------
def make_country_heatmap_fixed(selected_country: str):
    """
    Heatmap remains fixed (same colors/scale).
    Only adds a highlight marker for the selected country.
    """

    # Example fixed “risk index” per country (replace with your real mapping if you have it)
    # Keeping constant range and constant color scale ensures the colors never shift.
    countries = [
        "Malaysia","Singapore","Thailand","Indonesia","Philippines","Vietnam",
        "Japan","South Korea","China","India","Australia","United States","United Kingdom"
    ]
    # Fixed values (demo). Replace with your real country values if available.
    risk = np.array([0.62, 0.66, 0.61, 0.64, 0.63, 0.60, 0.68, 0.67, 0.65, 0.59, 0.62, 0.66, 0.64])

    df = pd.DataFrame({"country": countries, "risk": risk})

    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color="risk",
        color_continuous_scale="Turbo",  # futuristic vibe
        range_color=(0.55, 0.70),        # FIXED SCALE => colors stay as-is
    )
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showcoastlines=False,
        showland=True,
        landcolor="rgba(255,255,255,0.02)",
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title="Index", tickformat=".2f")
    )

    # Highlight selected country WITHOUT changing heatmap colors
    if selected_country:
        fig.add_trace(
            go.Scattergeo(
                locations=[selected_country],
                locationmode="country names",
                mode="markers",
                marker=dict(size=10, symbol="circle-open", line=dict(width=2)),
                name="Selected"
            )
        )

    return fig

# -----------------------------
# Year mapping helper
# -----------------------------
def index_to_year_series(n, start_year=2000, end_year=2013):
    """
    Map sample indices to years (evenly). Useful when dataset has no real timestamps.
    """
    years = np.linspace(start_year, end_year, n)
    return years.round().astype(int)

# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("💧 Water Quality Analysis Dashboard")
    st.markdown('<div class="small-note">Futuristic minimal UI • Interactive visuals • Training + XAI sections</div>', unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Controls")
        dataset_path = st.text_input("Dataset (.mat) path", "water_dataset.mat")
        model_path = st.text_input("Time-series model (.pkl) path", "gbr_model.pkl")

        st.divider()
        st.subheader("Geo View")
        selected_country = st.selectbox(
            "Select Country",
            ["Malaysia","Singapore","Thailand","Indonesia","Philippines","Vietnam","Japan","South Korea","China","India","Australia","United States","United Kingdom"],
            index=0
        )

        st.divider()
        st.subheader("Location + Thresholds")
        location = st.selectbox("Select pH Location", [f"L{i}_pH" for i in range(1, 38)], index=0)
        compare_location = st.selectbox("Compare with another location", [f"L{i}_pH" for i in range(1, 38)], index=9)

        th0 = st.slider("Class 0/1 threshold", 0.50, 0.90, 0.65, 0.01)
        th1 = st.slider("Class 1/2 threshold", 0.50, 0.90, 0.70, 0.01)
        show_conf = st.checkbox("Show confusion matrix", True)

        st.divider()
        st.subheader("Forecast Year Range")
        year_start = st.slider("Start year", 1995, 2025, 2000, 1)
        year_end = st.slider("End year", 1995, 2025, 2013, 1)

    # ---------------- Load Data ----------------
    if "train_df" not in st.session_state or st.session_state.get("dataset_path") != dataset_path:
        with st.spinner("Loading and preprocessing data..."):
            train_df, test_df, feature_cols, target_cols = load_data(dataset_path)
            st.session_state.train_df = train_df
            st.session_state.test_df = test_df
            st.session_state.feature_cols = feature_cols
            st.session_state.target_cols = target_cols
            st.session_state.dataset_path = dataset_path

    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    feature_cols = st.session_state.feature_cols

    # Labels based on selected location + thresholds
    label_col = f"{location}_label"
    train_df[label_col] = train_df[location].apply(lambda x: ph_to_class_ui(x, th0, th1))
    test_df[label_col] = test_df[location].apply(lambda x: ph_to_class_ui(x, th0, th1))

    X_train_cls = train_df[feature_cols]
    y_train_cls = train_df[label_col]
    X_test_cls = test_df[feature_cols]
    y_test_cls = test_df[label_col]

    # ---------------- Train baseline classification models ----------------
    cache_key = (location, th0, th1, dataset_path)
    if st.session_state.get("cls_cache_key") != cache_key:
        with st.spinner("Training classification models..."):
            reports, preds, trained = train_classification_models(X_train_cls, y_train_cls, X_test_cls, y_test_cls)
            st.session_state.classification_reports = reports
            st.session_state.classification_predictions = preds
            st.session_state.trained_classification_models = trained
            st.session_state.performance_df = build_performance_df(reports)
            st.session_state.cls_cache_key = cache_key

    reports = st.session_state.classification_reports
    preds = st.session_state.classification_predictions
    trained_models = st.session_state.trained_classification_models
    perf_df = st.session_state.performance_df

    # ---------------- Time-series forecast (loaded model) ----------------
    ts_key = (location, model_path, dataset_path)
    if st.session_state.get("ts_key") != ts_key:
        with st.spinner("Preparing time series + predicting..."):
            ph_train = train_df[location]
            ph_test = test_df[location]
            full = pd.concat([ph_train, ph_test], axis=0)
            lag1 = full.shift(1)

            lag1_test = lag1.iloc[ph_train.shape[0]:]
            y_test = ph_test.iloc[1:]
            X_test_lag = lag1_test.iloc[1:].values.reshape(-1, 1)

            gbr = load_ts_model(model_path)
            y_pred = gbr.predict(X_test_lag)

            st.session_state.y_test_ts = y_test
            st.session_state.y_pred_ts = y_pred
            st.session_state.ts_mse = float(mean_squared_error(y_test, y_pred))
            st.session_state.ts_r2 = float(r2_score(y_test, y_pred))
            st.session_state.ts_key = ts_key

    y_test_ts = st.session_state.y_test_ts
    y_pred_ts = st.session_state.y_pred_ts
    ts_mse = st.session_state.ts_mse
    ts_r2 = st.session_state.ts_r2

    # ---------------- KPIs ----------------
    overall = perf_df[perf_df["Class"] == "Overall"][["Model", "Accuracy"]].dropna()
    best_model_row = overall.sort_values("Accuracy", ascending=False).head(1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Country", selected_country)
    c2.metric("Best Classifier", best_model_row["Model"].iloc[0] if not best_model_row.empty else "—")
    c3.metric("TS MSE", f"{ts_mse:.4f}")
    c4.metric("TS R²", f"{ts_r2:.4f}")

    st.markdown(
        """
        <div style="margin-top:10px;">
          <span class="chip">Fixed map colors</span>
          <span class="chip">Country highlight only</span>
          <span class="chip">XAI images supported</span>
          <span class="chip">Train & save model</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ---------------- Tabs ----------------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Dataset", "Geo", "EDA", "Classification", "Forecasting", "Explainable AI + Training"]
    )

    # ============ Dataset Tab ============
    with tab0:
        st.subheader("Dataset Selected + Why")
        st.markdown(
            """
            <div class="panel">
            <b>Chosen dataset:</b> UCI Water Quality Prediction-1 (MATLAB .mat format)<br><br>
            <b>Why this dataset:</b>
            <ul>
              <li><b>Multi-location sensing:</b> 37 locations (L1–L37) emulate distributed IoT sampling points.</li>
              <li><b>Multi-sensor features:</b> 11 features per location capture environmental & sensor variation.</li>
              <li><b>Real AIoT pipeline fit:</b> preprocessing → classification (status) → forecasting (trend prediction).</li>
              <li><b>Decision support:</b> can drive alerts, compliance checks, and operational response planning.</li>
            </ul>
            <b>Targets:</b> pH values per location (Lx_pH).<br>
            <b>Inputs:</b> standardized features (Lx_F1…Lx_F11).
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        left, right = st.columns(2)
        with left:
            st.caption("Train sample preview")
            st.dataframe(train_df[[location, compare_location]].head(12), use_container_width=True)
        with right:
            st.caption("Test sample preview")
            st.dataframe(test_df[[location, compare_location]].head(12), use_container_width=True)

    # ============ Geo Tab ============
    with tab1:
        st.subheader("Country Heatmap (Fixed Scale) + Highlight")
        st.caption("The heatmap colors stay fixed. Changing country only updates the highlight.")
        fig_map = make_country_heatmap_fixed(selected_country)
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown(
            """
            <div class="panel">
            <b>Note:</b> If you have a real country-to-value mapping (e.g., average pH risk index per country),
            replace the demo values inside <code>make_country_heatmap_fixed()</code>. Keep <code>range_color</code>
            fixed to ensure the colors never shift.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============ EDA Tab ============
    with tab2:
        st.subheader("pH Distributions")
        dist_df = pd.DataFrame({
            "pH": pd.concat([train_df[location], train_df[compare_location]], axis=0),
            "Location": ([location] * len(train_df)) + ([compare_location] * len(train_df))
        })
        fig = px.histogram(dist_df, x="pH", color="Location", nbins=35, barmode="overlay", opacity=0.6)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("pH Trends")
        trend_df = pd.DataFrame({
            "t": np.arange(len(train_df)),
            location: train_df[location].values,
            compare_location: train_df[compare_location].values
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trend_df["t"], y=trend_df[location], name=location))
        fig2.add_trace(go.Scatter(x=trend_df["t"], y=trend_df[compare_location], name=compare_location))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="Time step", yaxis_title="pH")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Class Distribution")
        cd = train_df[label_col].value_counts().sort_index().reset_index()
        cd.columns = ["Class", "Count"]
        fig3 = px.bar(cd, x="Class", y="Count")
        fig3.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    # ============ Classification Tab ============
    with tab3:
        st.subheader("Model Performance")
        overall_acc = perf_df[perf_df["Class"] == "Overall"][["Model", "Accuracy"]].dropna()
        fig = px.bar(overall_acc, x="Model", y="Accuracy")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Class-wise Metrics")
        class_metrics = perf_df[perf_df["Class"] != "Overall"].copy()
        class_metrics["Class"] = class_metrics["Class"].astype(int)
        metric = st.selectbox("Metric", ["Precision", "Recall", "F1"], index=2, key="metric_pick")
        fig2 = px.bar(class_metrics, x="Model", y=metric, color="Class", barmode="group")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Feature Importance (Random Forest)")
        rf = trained_models.get("Random Forest")
        if rf and hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            topk = 12
            idx = np.argsort(importances)[-topk:][::-1]
            fi_df = pd.DataFrame({"Feature": np.array(feature_cols)[idx], "Importance": importances[idx]})
            fig3 = px.bar(fi_df[::-1], x="Importance", y="Feature", orientation="h")
            fig3.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Random Forest model not available for feature importance.")

        if show_conf and not overall_acc.empty:
            st.subheader("Confusion Matrix (Best Model)")
            best = overall_acc.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
            y_pred_best = preds[best]
            cm = confusion_matrix(y_test_cls, y_pred_best, labels=[0, 1, 2])
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1", "Actual 2"], columns=["Pred 0", "Pred 1", "Pred 2"])
            fig4 = px.imshow(cm_df, text_auto=True, aspect="auto")
            fig4.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

    # ============ Forecasting Tab ============
    with tab4:
        st.subheader("Time Series Forecast (Year Range Applied)")

        # Map the test index to years (synthetic timeline)
        years = index_to_year_series(len(y_test_ts), start_year=year_start, end_year=year_end)
        ts_df = pd.DataFrame({"Year": years, "Actual": y_test_ts.values, "Predicted": y_pred_ts})

        # Apply year filter (so user can “zoom” by year)
        min_y, max_y = min(year_start, year_end), max(year_start, year_end)
        ts_df_f = ts_df[(ts_df["Year"] >= min_y) & (ts_df["Year"] <= max_y)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df_f["Year"], y=ts_df_f["Actual"], name="Actual"))
        fig.add_trace(go.Scatter(x=ts_df_f["Year"], y=ts_df_f["Predicted"], name="Predicted", line=dict(dash="dash")))
        fig.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Year",
            yaxis_title=location
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Threshold Lines (Classification Logic)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=train_df[location].values, x=np.arange(len(train_df)), name="pH"))
        fig2.add_hline(y=th0, line_dash="dash", annotation_text="0/1 threshold")
        fig2.add_hline(y=th1, line_dash="dash", annotation_text="1/2 threshold")
        fig2.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time step",
            yaxis_title="pH"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ============ Explainable AI + Training Tab ============
    with tab5:
        colA, colB = st.columns([1.1, 0.9])

        with colA:
            st.subheader("Explainable AI (XAI)")
            st.caption("Place your XAI images inside the assets/ folder to auto-display here.")

            xai_images = [
                ("assets/shap_summary.png", "SHAP Summary Plot — shows which features push predictions up/down."),
                ("assets/feature_importance.png", "Global Feature Importance — overall most influential sensors."),
                ("assets/lime_example.png", "LIME Explanation — local explanation for one prediction."),
            ]

            shown_any = False
            for path, cap in xai_images:
                if os.path.exists(path):
                    st.image(path, use_container_width=True, caption=cap)
                    shown_any = True

            if not shown_any:
                st.info("No images found in /assets yet. Add your XAI plots there (png/jpg) to show them automatically.")

            st.markdown(
                """
                <div class="panel">
                <b>How XAI supports your project:</b>
                <ul>
                  <li><b>Trust:</b> shows which sensor features drive the water safety prediction.</li>
                  <li><b>Debug:</b> detect if model relies on noisy/irrelevant sensors.</li>
                  <li><b>Decision support:</b> helps explain alerts to stakeholders.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with colB:
            st.subheader("Train a Model (Interactive)")
            st.caption("Train and save a classifier inside the dashboard.")

            model_choice = st.selectbox("Choose model to train", ["Random Forest", "SVM (RBF)", "Logistic Regression", "KNN"])

            # Minimal hyperparameters
            if model_choice == "Random Forest":
                n_estimators = st.slider("n_estimators", 50, 600, 250, 25)
            elif model_choice == "KNN":
                k = st.slider("n_neighbors", 1, 25, 7, 1)
            elif model_choice == "SVM (RBF)":
                C = st.slider("C", 0.1, 50.0, 10.0, 0.1)
            else:
                max_iter = st.slider("max_iter", 200, 3000, 1200, 100)

            train_btn = st.button("🚀 Train Now", use_container_width=True)

            if train_btn:
                with st.spinner("Training..."):
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    elif model_choice == "KNN":
                        model = KNeighborsClassifier(n_neighbors=k)
                    elif model_choice == "SVM (RBF)":
                        model = SVC(kernel="rbf", C=C, gamma="scale", random_state=42)
                    else:
                        model = LogisticRegression(max_iter=max_iter, random_state=42)

                    model.fit(X_train_cls, y_train_cls)
                    y_pred = model.predict(X_test_cls)
                    acc = accuracy_score(y_test_cls, y_pred)

                st.success(f"Training complete ✅  Accuracy: {acc:.4f}")

                save_name = st.text_input("Save model as", value="trained_model.pkl")
                if st.button("💾 Save Model", use_container_width=True):
                    joblib.dump(model, save_name)
                    st.success(f"Saved to: {save_name}")

    st.divider()
    st.caption("Futuristic minimal UI • Fixed heatmap scale • XAI section • Training + Year range forecasting")

if __name__ == "__main__":
    main()
