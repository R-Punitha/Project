%%writefile app.py

# ------------------- Imports -------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ------------------- Page Config -------------------
st.set_page_config(page_title="Comparative Analysis of Time series Algorithm",
                   layout="wide",
                   page_icon="🌍")

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    data = pd.read_csv("india_daily_temperature (3).csv")
    # Use lowercase 'date' since that's the actual column name
    data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y", errors="coerce")
    data.set_index('date', inplace=True)
    return data




data = load_data()

# ------------------- Train-Test Split -------------------
train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

# ------------------- Fit Models -------------------
arima_model = sm.tsa.ARIMA(train['india_daily_temperature (3)'], order=(1,1,1))
arima_results = arima_model.fit()

sarima_model = sm.tsa.statespace.SARIMAX(train['india_daily_temperature (3)'],
                                         order=(1,1,1),
                                         seasonal_order=(1,1,1,12))
sarima_results = sarima_model.fit()

# Forecasts
arima_forecast = arima_results.forecast(steps=len(test))
sarima_forecast = sarima_results.forecast(steps=len(test))

# ------------------- Evaluation Function -------------------
def evaluate_forecast(actual, forecast, model_name):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    r2 = r2_score(actual, forecast)
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R²": r2
    }

arima_metrics = evaluate_forecast(test['india_daily_temperature (3)'], arima_forecast, "ARIMA")
sarima_metrics = evaluate_forecast(test['india_daily_temperature (3)'], sarima_forecast, "SARIMA")
comparison = pd.DataFrame([arima_metrics, sarima_metrics])

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Data Exploration",
    "Model Training",
    "Model Comparison",
    "Future Forecast",
    "Conclusion"
])

# ------------------- Pages -------------------

# 1. Overview
if page == "Overview":
    st.title("Comparative Analysis of Time series Algorithm")
    st.markdown("""
    ### Problem Statement
    Accurate forecasting of daily temperature is essential for climate analysis and planning.
    
    ### Methodology Flow
    1. Data Collection  
    2. Data Cleaning  
    3. Data Pre-Processing  
    4. Exploratory Data Analysis (EDA)  
    5. Train–Test Split  
    6. Model Development  
    7. Model Tuning  
    8. Model Evaluation  
    """)

# 2. Data Exploration
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.subheader("Historical Temperature Trends")
    fig = px.line(data, x=data.index, y="temperature", title="Historical Temperature")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stationarity Check (ADF Test)")
    result = adfuller(data['temperature'])
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")

# 3. Model Training
elif page == "Model Training":
    st.title("⚙️ Model Training")
    st.subheader("ARIMA Model")
    st.write(f"Order: (1,1,1), AIC: {arima_results.aic:.2f}")
    st.subheader("SARIMA Model")
    st.write(f"Order: (1,1,1), Seasonal Order: (1,1,1,12), AIC: {sarima_results.aic:.2f}")

# 4. Model Comparison
elif page == "Model Comparison":
    st.title("📈 Model Comparison")
    st.subheader("Forecasts vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test['temperature'], name="Actual"))
    fig.add_trace(go.Scatter(x=test.index, y=arima_forecast, name="ARIMA Forecast"))
    fig.add_trace(go.Scatter(x=test.index, y=sarima_forecast, name="SARIMA Forecast"))
    fig.update_layout(title="ARIMA vs SARIMA Forecasts")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evaluation Metrics")
    st.dataframe(comparison.style.highlight_min(color="lightcoral").highlight_max(color="lightgreen"))

    st.subheader("Metrics Visualization")
    metrics_long = comparison.melt(id_vars="Model", 
                                   value_vars=["MAE", "RMSE", "MAPE (%)", "R²"],
                                   var_name="Metric", 
                                   value_name="Value")
    fig = px.bar(metrics_long, x="Metric", y="Value", color="Model", barmode="group",
                 title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

# 5. Future Forecast
elif page == "Future Forecast":
    st.title("🔮 Future Forecast (SARIMA)")
    future_forecast = sarima_results.get_forecast(steps=365)
    future_values = future_forecast.predicted_mean

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['india_daily_temperature (3)'], name="Historical"))
    fig.add_trace(go.Scatter(x=future_values.index, y=future_values, name="Future Forecast"))
    fig.update_layout(title="Next 365 Days Forecast")
    st.plotly_chart(fig, use_container_width=True)

# 6. Conclusion
elif page == "Conclusion":
    st.title("✅ Conclusion")
    st.markdown("""
    - SARIMA captures seasonality better than ARIMA.  
    - SARIMA produces lower AIC and more accurate forecasts.  
    - Recommended model for climate forecasting in this dataset.  
    """)

    # Download button
    st.download_button("📥 Download Forecast Results (CSV)",
                       data=future_values.to_csv(),
                       file_name="future_forecast.csv",
                       mime="text/csv")
