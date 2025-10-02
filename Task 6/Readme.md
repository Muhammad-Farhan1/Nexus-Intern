# Sales Forecasting with XGBoost + LightGBM (Blended Model)

This project predicts daily sales using **Machine Learning models** (XGBoost & LightGBM) blended together.  
It uses historical sales data, lag features, rolling averages, and date-based features for better forecasting.  
A **Streamlit interface** is also provided to visualize predictions vs actual values in a business-friendly dashboard.

---

## 🚀 Features
- Data preprocessing & feature engineering:
  - Lag features (previous days’ sales)
  - Rolling averages (7-day, 30-day trends)
  - Date-based features (month, weekday, quarter)
- Model training:
  - XGBoost and LightGBM regression
  - Blending of both models for improved accuracy
- Evaluation:
  - Metrics: MAE, RMSE, R², MAPE
- Streamlit dashboard:
  - Interactive UI
  - Actual vs Predicted Sales comparison (bar chart)
  - Key metrics displayed in KPI cards

---
# To Run This Project Use This Commands 
pip install -r requirements.txt

# To View Model Interface Use This Command
streamlit run app.py
