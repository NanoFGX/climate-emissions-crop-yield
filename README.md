# 🌍 GEOFOODSEC  
### Global Food Security Intelligence under Climate Change

**GEOFOODSEC** is an interactive decision-support system that analyzes the interconnected relationship between **climate change**, **greenhouse gas emissions**, and **global food production**.  
Built using **multi-source data fusion** and **data mining techniques**, the project transforms fragmented environmental datasets into actionable insights for forecasting, risk analysis, and policy support.

The system is implemented as a **Streamlit dashboard**, combining visualization, predictive modeling, explainable AI (XAI), and clustering to support **proactive food security planning** rather than reactive crisis response.

---

## 🔗 Live Dashboard (Online Demo)

You can access the deployed Streamlit dashboard here:

👉 **https://climate-emissions-crop-yield-ideynb6gsjxyvmieizojus.streamlit.app/**

No installation is required to explore the dashboard online.

---

## 🎯 Project Motivation

Global food production is under increasing pressure due to:
- Rising greenhouse gas (GHG) emissions  
- Climate volatility and temperature anomalies  
- Inefficient and uneven agricultural productivity  

Despite abundant data, decision-makers often lack a **unified analytical tool** that links:

> **Human activity (Emissions) → Environmental change (Climate) → Food availability (Production)**

**GEOFOODSEC** addresses this gap by fusing climate, emissions, and agricultural datasets into a single analytical framework aligned with:
- **SDG 2 – Zero Hunger**
- **SDG 13 – Climate Action**

---

## 🧠 What GEOFOODSEC Does

### 1. Multi-Source Data Fusion  
Fuses global datasets into a **Country–Year panel (2000–2013)** using:
- Temporal harmonisation  
- Feature-level fusion  
- Index-based fusion (e.g. Carbon Intensity Index)

### 2. Data Mining & Modeling  
Systematic evaluation of **24 experimental configurations** across:
- Regression (food production prediction)
- Classification (food security risk)
- Clustering (environmental & production profiles)

### 3. Explainable AI (XAI)  
Uses **SHAP** and feature importance analysis to:
- Move from “black-box” to **interpretable models**
- Justify predictions for reporting and decision-making

### 4. Interactive Decision Dashboard  
All results are deployed into a **Streamlit dashboard** for:
- Global overview
- Country-level insights
- Forecasting & residual analysis
- Model tuning & restoration

---

## 🚀 Dashboard Features

### 🌐 Global Overview
- Interactive world choropleth map by year  
- Summary table by country  
- High-level KPIs (temperature, total GHG, food production)

### 📊 Country Insights
- Multi-year trends (climate, emissions, production)  
- GHG vs food production scatter analysis  
- Correlation heatmap for quick dependency checks  

### 🔮 Forecast & Change Analysis
- Model performance metrics (MAE, RMSE, MAPE, R²)  
- Actual vs predicted food production  
- Future projection using trend-based feature estimation  
- Residual analysis and year-over-year change tables  

### 🧩 Explainability (XAI)
- SHAP beeswarm visualization  
- Clear interpretation notes for academic and policy use  

### ⚙️ Model Tuning
- Retrain Random Forest models directly in the app  
- Restore the baseline (best) model at any time  

---

## 🧪 Experimental Framework

The project benchmarks **four preprocessing settings**:

| Setting | Technique |
|------|---------|
| Setting 1 | Chi-Square Feature Selection |
| Setting 2 | IQR Outlier Removal |
| Setting 3 | One-Hot Encoding |
| Setting 4 | Data Discretization (Binning) |

Models are evaluated using **temporal splits** to simulate real-world forecasting:
- Train ≤ 2011 → Test 2012–2013  
- Train ≤ 2012 → Test 2013  

---

## 🏆 Key Results

- **Best Regression Model:**  
  Random Forest Regressor (Setting 3 – One-Hot Encoding)  
  - R² ≈ **0.98**
  - Strong generalization on unseen future data

- **Best Classification Model:**  
  Decision Tree (Risk-focused, optimized for Recall)  
  - High detection of **High-Risk food security states**
  - AUC > **0.90**

- **Best Clustering Model:**  
  K-Means (Setting 2 – IQR)  
  - Identifies environmental & production “fingerprints”  
  - Separates industrialized emitters from vulnerable regions

---

## ▶️ How to Run the Dashboard

### Option 1: Run Online (Recommended)
1. Open the live app link:  
   👉 https://climate-emissions-crop-yield-ideynb6gsjxyvmieizojus.streamlit.app/
2. Use the sidebar to:
   - Select year and country
   - Switch between overview, insights, forecasting, and XAI
3. Interact with charts, tables, and model outputs in real time

---
