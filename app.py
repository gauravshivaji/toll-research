import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import ttest_ind
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap

st.set_page_config(layout="wide")
st.title("Toll Traffic Research Dashboard (Leakage Analysis Included)")

# -----------------------------
# Upload Dataset
# -----------------------------

uploaded_file = st.file_uploader("Upload Dataset")

if uploaded_file is None:
    st.stop()

if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# Feature Engineering
# -----------------------------

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear
df["Time_Index"] = np.arange(len(df))

for lag in [1,7,14,30]:
    df[f"Lag_{lag}"] = df["Total_Vehicles"].shift(lag)

for roll in [7,14,30]:
    df[f"Rolling_{roll}"] = df["Total_Vehicles"].rolling(roll).mean()

df = df.dropna().reset_index(drop=True)

# -----------------------------
# Leakage Sensitivity Section
# -----------------------------

st.header("Leakage Sensitivity Experiment")

# Scenario A: With Leakage
features_leaky = [col for col in df.columns 
                  if col not in ["Date","Total_Vehicles","Total_Revenue"]]

# Scenario B: Without Leakage
leakage_columns = [
    "Car_Count","Bus_Count","LCV_Count","MAV_Count",
    "Car_Revenue","Bus_Revenue","LCV_Revenue","MAV_Revenue"
]

features_clean = [col for col in features_leaky 
                  if col not in leakage_columns]

X_leaky = df[features_leaky]
X_clean = df[features_clean]

y = df["Total_Vehicles"]

split = int(len(df)*0.8)

def evaluate_model(X):
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = XGBRegressor(n_estimators=300, random_state=42)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    r2 = r2_score(y_test,preds)
    
    return mae, rmse, r2, model

mae_leaky, rmse_leaky, r2_leaky, model_leaky = evaluate_model(X_leaky)
mae_clean, rmse_clean, r2_clean, model_clean = evaluate_model(X_clean)

inflation = r2_leaky - r2_clean

results_df = pd.DataFrame({
    "Model Type":["With Leakage","Without Leakage"],
    "MAE":[mae_leaky,mae_clean],
    "RMSE":[rmse_leaky,rmse_clean],
    "R2":[r2_leaky,r2_clean]
})

st.subheader("Performance Comparison")
st.write(results_df)

st.subheader("Performance Inflation Due to Leakage")
st.write("R2 Inflation:", round(inflation,4))

# -----------------------------
# Hypothesis Testing
# -----------------------------

st.header("Weekend Effect Hypothesis Test")

weekend = df[df["Is_Weekend"]==1]["Total_Vehicles"]
weekday = df[df["Is_Weekend"]==0]["Total_Vehicles"]

t_stat, p_val = ttest_ind(weekend,weekday)

st.write("Mean Weekend:", weekend.mean())
st.write("Mean Weekday:", weekday.mean())
st.write("p-value:", p_val)

# -----------------------------
# ADF Test
# -----------------------------

st.header("ADF Stationarity Test")

adf = adfuller(df["Total_Vehicles"])
st.write("ADF Statistic:", adf[0])
st.write("p-value:", adf[1])

# -----------------------------
# Seasonal Decomposition
# -----------------------------

st.header("Seasonal Decomposition")

decomp = seasonal_decompose(df["Total_Vehicles"], period=7)
fig = decomp.plot()
st.pyplot(fig)

# -----------------------------
# SHAP Analysis (Clean Model Only)
# -----------------------------

st.header("SHAP Feature Importance (Clean Model)")

X_train_clean = X_clean.iloc[:split]
X_test_clean = X_clean.iloc[split:]

explainer = shap.TreeExplainer(model_clean)
shap_values = explainer.shap_values(X_test_clean)

plt.figure()
shap.summary_plot(shap_values, X_test_clean, show=False)
st.pyplot(plt.gcf())

st.success("Full Leakage Analysis Complete")
