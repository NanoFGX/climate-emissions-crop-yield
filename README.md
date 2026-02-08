# GeoFoodSec — Climate–Emissions–Food Production Dashboard (Streamlit)

GeoFoodSec is a decision-support dashboard that connects **Climate → Emissions → Food Production** using a fused country-year dataset (2000–2013).  
It provides global visualization, country-level insights, forecasting, model explainability (XAI), model tuning, and clustering-based comparison.

---

## Features
**Global Overview**
- World map (choropleth) for the selected year
- Summary table by country
- Quick snapshot KPIs (temperature, total GHG, food production)

**Country Insights**
- Trends across years (temperature, emissions, production, carbon intensity)
- Scatter relationship (GHG vs production)
- Correlation heatmap (quick checks)

**Forecast & Change**
- Model evaluation metrics (MAE, RMSE, MAPE, R²)
- Actual vs predicted production
- Future projection using simple feature trend estimation
- Residual analysis + year-over-year change table

**Explainability (XAI)**
- SHAP beeswarm image to explain feature impact on predictions
- Clear interpretation notes for reporting

**Model Tuning**
- Train a new Random Forest configuration from inside the app
- Restore baseline model anytime
---

## Project Structure (Required)
Your repository should look like this:

