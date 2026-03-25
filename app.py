import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("india_daily_temperature.csv")
    # Parse date column correctly
    data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y", errors="coerce")
    data.set_index('date', inplace=True)
    return data

data = load_data()

st.title("🌡️ India Daily Temperature Dashboard")

# -------------------------------
# Data Overview
# -------------------------------
st.subheader("Dataset Preview")
st.write(data.head())

st.subheader("Summary Statistics")
st.write(data.describe())

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Temperature Trends")
fig = px.line(data, x=data.index, y="temperature_2m_max", title="Daily Max Temperature")
st.plotly_chart(fig)

fig2 = px.line(data, x=data.index, y="temperature_2m_min", title="Daily Min Temperature")
st.plotly_chart(fig2)

# -------------------------------
# Forecasting (ARIMA Example)
# -------------------------------
st.subheader("ARIMA Forecast (Max Temperature)")

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

# Fit ARIMA model
arima_model = sm.tsa.ARIMA(train["temperature_2m_max"], order=(1,1,1))
arima_result = arima_model.fit()

# Forecast
forecast = arima_result.forecast(steps=len(test))

# Metrics
mae = mean_absolute_error(test["temperature_2m_max"], forecast)
rmse = mean_squared_error(test["temperature_2m_max"], forecast, squared=False)
r2 = r2_score(test["temperature_2m_max"], forecast)

st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}, **R²:** {r2:.2f}")

# Plot forecast vs actual
fig3 = px.line(title="ARIMA Forecast vs Actual")
fig3.add_scatter(x=test.index, y=test["temperature_2m_max"], mode="lines", name="Actual")
fig3.add_scatter(x=test.index, y=forecast, mode="lines", name="Forecast")
st.plotly_chart(fig3)
