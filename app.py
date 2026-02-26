import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from scipy.stats import ttest_ind
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide")
st.title("Toll Traffic Research Dashboard (Full Publication Version)")

# -------------------------
# Upload Dataset
# -------------------------

uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)")

if uploaded_file is None:
    st.stop()

if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# -------------------------
# Feature Engineering
# -------------------------

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear
df["Time_Index"] = np.arange(len(df))

for lag in [1,7,14,30]:
    df[f"Lag_{lag}"] = df["Total_Vehicles"].shift(lag)

for roll in [7,14,30]:
    df[f"Rolling_{roll}"] = df["Total_Vehicles"].rolling(roll).mean()

df = df.dropna().reset_index(drop=True)

st.subheader("Basic Statistics")
st.write(df[["Total_Vehicles","Total_Revenue"]].describe())

# -------------------------
# Hypothesis Testing
# -------------------------

st.header("Weekend Effect Hypothesis Test")

weekend = df[df["Is_Weekend"]==1]["Total_Vehicles"]
weekday = df[df["Is_Weekend"]==0]["Total_Vehicles"]

t_stat, p_val = ttest_ind(weekend,weekday)

st.write("Mean Weekend:", weekend.mean())
st.write("Mean Weekday:", weekday.mean())
st.write("p-value:", p_val)

# -------------------------
# ADF Test
# -------------------------

st.header("ADF Stationarity Test")

adf = adfuller(df["Total_Vehicles"])
st.write("ADF Statistic:", adf[0])
st.write("p-value:", adf[1])

# -------------------------
# Seasonal Decomposition
# -------------------------

st.header("Seasonal Decomposition")

decomp = seasonal_decompose(df["Total_Vehicles"], period=7)
fig = decomp.plot()
st.pyplot(fig)

# -------------------------
# Prepare Clean Features (No Leakage)
# -------------------------

leakage_cols = [
    "Car_Count","Bus_Count","LCV_Count","MAV_Count",
    "Car_Revenue","Bus_Revenue","LCV_Revenue","MAV_Revenue"
]

features = [col for col in df.columns 
            if col not in ["Date","Total_Vehicles","Total_Revenue"] + leakage_cols]

X = df[features]
y = df["Total_Vehicles"]

split = int(len(df)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# -------------------------
# Multi-Model Comparison
# -------------------------

st.header("Multi-Model Comparison")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(n_estimators=300, random_state=42),
    "LightGBM": LGBMRegressor(),
  
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    mape = np.mean(np.abs((y_test - preds)/y_test))*100
    r2 = r2_score(y_test,preds)

    results.append([name,mae,rmse,mape,r2])

# SARIMAX
sarimax = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,7))
sarimax_fit = sarimax.fit(disp=False)
sarimax_pred = sarimax_fit.forecast(len(y_test))

mae = mean_absolute_error(y_test,sarimax_pred)
rmse = np.sqrt(mean_squared_error(y_test,sarimax_pred))
mape = np.mean(np.abs((y_test - sarimax_pred)/y_test))*100
r2 = r2_score(y_test,sarimax_pred)

results.append(["SARIMAX",mae,rmse,mape,r2])

# -------------------------
# LSTM Model
# -------------------------

def create_lstm_dataset(series, window=14):
    X_lstm, y_lstm = [], []
    for i in range(len(series)-window):
        X_lstm.append(series[i:i+window])
        y_lstm.append(series[i+window])
    return np.array(X_lstm), np.array(y_lstm)

series = df["Total_Vehicles"].values
X_lstm, y_lstm = create_lstm_dataset(series)

split_lstm = int(len(X_lstm)*0.8)

X_train_lstm = X_lstm[:split_lstm]
X_test_lstm = X_lstm[split_lstm:]
y_train_lstm = y_lstm[:split_lstm]
y_test_lstm = y_lstm[split_lstm:]

X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0],X_train_lstm.shape[1],1))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0],X_test_lstm.shape[1],1))

model_lstm = Sequential()
model_lstm.add(LSTM(50,activation="relu",input_shape=(14,1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam",loss="mse")

model_lstm.fit(X_train_lstm,y_train_lstm,epochs=10,verbose=0)

lstm_pred = model_lstm.predict(X_test_lstm)

mae = mean_absolute_error(y_test_lstm,lstm_pred)
rmse = np.sqrt(mean_squared_error(y_test_lstm,lstm_pred))
mape = np.mean(np.abs((y_test_lstm - lstm_pred.flatten())/y_test_lstm))*100
r2 = r2_score(y_test_lstm,lstm_pred)

results.append(["LSTM",mae,rmse,mape,r2])

# -------------------------
# Show Results
# -------------------------

results_df = pd.DataFrame(results,
        columns=["Model","MAE","RMSE","MAPE","R2"])

st.write(results_df.sort_values("R2",ascending=False))

# -------------------------
# Walk Forward Validation
# -------------------------

st.header("Walk Forward Validation (XGBoost)")

tscv = TimeSeriesSplit(n_splits=5)
mae_scores = []

for train_idx,test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    model = XGBRegressor()
    model.fit(X_tr,y_tr)
    pred = model.predict(X_te)
    mae_scores.append(mean_absolute_error(y_te,pred))

st.write("Average Walk Forward MAE:",np.mean(mae_scores))
# ============================================================
# HYBRID LINEAR + XGBOOST RESIDUAL MODEL (Novel Contribution)
# ============================================================

st.header("Hybrid Linear-Boosted Forecasting Model")

# Step 1: Fit Ridge (Base Linear Model)
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

ridge_pred_train = ridge_model.predict(X_train)
ridge_pred_test = ridge_model.predict(X_test)

# Step 2: Compute Residuals
residual_train = y_train - ridge_pred_train

# Step 3: Fit XGBoost on Residuals
xgb_residual = XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
xgb_residual.fit(X_train, residual_train)

residual_pred_test = xgb_residual.predict(X_test)

# Step 4: Final Hybrid Prediction
hybrid_pred = ridge_pred_test + residual_pred_test

# Metrics
hybrid_mae = mean_absolute_error(y_test, hybrid_pred)
hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
hybrid_r2 = r2_score(y_test, hybrid_pred)

st.write("Hybrid MAE:", hybrid_mae)
st.write("Hybrid RMSE:", hybrid_rmse)
st.write("Hybrid R2:", hybrid_r2)
# ============================================================
# TRAFFIC SURGE INDEX (TSI)
# ============================================================

st.header("Traffic Surge Index (Holiday Impact)")

baseline = df[df["Is_Holiday"]==0]["Total_Vehicles"].mean()
holiday_mean = df[df["Is_Holiday"]==1]["Total_Vehicles"].mean()

TSI = (holiday_mean - baseline) / baseline

st.write("Baseline Traffic:", baseline)
st.write("Holiday Traffic:", holiday_mean)
st.write("Traffic Surge Index (TSI):", TSI)

# ============================================================
# REGIME DETECTION USING KMEANS
# ============================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.header("Regime Detection (Traffic States)")

regime_features = df[["Total_Vehicles","Lag_1","Lag_7","Is_Weekend","Is_Holiday"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(regime_features)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Regime"] = kmeans.fit_predict(scaled)

st.write(df["Regime"].value_counts())

# Visualize regimes
plt.figure(figsize=(12,4))
plt.scatter(df["Date"], df["Total_Vehicles"], c=df["Regime"])
plt.title("Traffic Regimes")
st.pyplot(plt.gcf())


# ============================================================
# QUANTILE REGRESSION (Prediction Intervals)
# ============================================================

st.header("Prediction Intervals (Quantile LightGBM)")

lower_model = LGBMRegressor(objective="quantile", alpha=0.1)
upper_model = LGBMRegressor(objective="quantile", alpha=0.9)

lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

lower_pred = lower_model.predict(X_test)
upper_pred = upper_model.predict(X_test)

plt.figure(figsize=(10,4))
plt.plot(y_test.values, label="Actual")
plt.plot(lower_pred, label="Lower 10%")
plt.plot(upper_pred, label="Upper 90%")
plt.legend()
st.pyplot(plt.gcf())


# Cohen's d
mean_diff = weekend.mean() - weekday.mean()
pooled_std = np.sqrt((weekend.std()**2 + weekday.std()**2)/2)
cohen_d = mean_diff / pooled_std

st.write("Cohen's d:", cohen_d)


# ============================================================
# ECONOMIC IMPACT ANALYSIS
# ============================================================

st.header("Economic Impact Estimation")

avg_toll = 110  # change if needed
daily_revenue_error = hybrid_mae * avg_toll

st.write("Average Daily Revenue Forecast Error (₹):", daily_revenue_error)
# -------------------------
# SHAP
# -------------------------

st.header("SHAP Feature Importance (XGBoost)")

xgb_model = XGBRegressor()
xgb_model.fit(X_train,y_train)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values,X_test,show=False)
st.pyplot(plt.gcf())

st.success("Full Research Pipeline Complete")
