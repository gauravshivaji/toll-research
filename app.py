import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
st.title("Toll Traffic Research Dashboard")

# -------------------------------
# Upload Dataset
# -------------------------------

uploaded_file = st.file_uploader("Upload Dataset (Excel or CSV)")

if uploaded_file is None:
    st.stop()

if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# -------------------------------
# Feature Engineering
# -------------------------------

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear
df["Time_Index"] = np.arange(len(df))

# Lag features
for lag in [1,7,14,30]:
    df[f"Lag_{lag}"] = df["Total_Vehicles"].shift(lag)

# Rolling features
for roll in [7,14,30]:
    df[f"Rolling_{roll}"] = df["Total_Vehicles"].rolling(roll).mean()

df = df.dropna()

st.subheader("Basic Statistics")
st.write(df[["Total_Vehicles","Total_Revenue"]].describe())

# -------------------------------
# Weekend & Holiday Effect
# -------------------------------

st.subheader("Weekend Effect Hypothesis Test")

weekend = df[df["Is_Weekend"]==1]["Total_Vehicles"]
weekday = df[df["Is_Weekend"]==0]["Total_Vehicles"]

t_stat, p_val = ttest_ind(weekend, weekday)

st.write("Mean Weekend:", weekend.mean())
st.write("Mean Weekday:", weekday.mean())
st.write("p-value:", p_val)

# -------------------------------
# ADF Test
# -------------------------------

st.subheader("ADF Stationarity Test")

adf_result = adfuller(df["Total_Vehicles"])
st.write("ADF Statistic:", adf_result[0])
st.write("p-value:", adf_result[1])

# -------------------------------
# Seasonal Decomposition
# -------------------------------

st.subheader("Seasonal Decomposition")

decomposition = seasonal_decompose(df["Total_Vehicles"], period=7)
fig = decomposition.plot()
st.pyplot(fig)

# -------------------------------
# Model Comparison
# -------------------------------

st.subheader("Model Training & Comparison")

features = [col for col in df.columns if col not in 
            ["Date","Total_Vehicles","Total_Revenue"]]

X = df[features]
y = df["Total_Vehicles"]

split = int(len(df)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

results = []

# SARIMAX
model_sarimax = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,7))
model_sarimax_fit = model_sarimax.fit(disp=False)
pred_sarimax = model_sarimax_fit.forecast(len(y_test))

# XGBoost
model_xgb = XGBRegressor(n_estimators=300)
model_xgb.fit(X_train,y_train)
pred_xgb = model_xgb.predict(X_test)

# LightGBM
model_lgb = LGBMRegressor()
model_lgb.fit(X_train,y_train)
pred_lgb = model_lgb.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mape = np.mean(np.abs((y_true-y_pred)/y_true))*100
    r2 = r2_score(y_true,y_pred)
    results.append([name,mae,rmse,mape,r2])

evaluate("SARIMAX",y_test,pred_sarimax)
evaluate("XGBoost",y_test,pred_xgb)
evaluate("LightGBM",y_test,pred_lgb)

results_df = pd.DataFrame(results,
        columns=["Model","MAE","RMSE","MAPE","R2"])

st.write(results_df)

# -------------------------------
# Walk Forward Validation
# -------------------------------

st.subheader("Walk Forward Validation")

tscv = TimeSeriesSplit(n_splits=5)
mae_scores = []

for train_idx,test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    model = XGBRegressor()
    model.fit(X_tr,y_tr)
    pred = model.predict(X_te)
    mae_scores.append(mean_absolute_error(y_te,pred))

st.write("Average Walk Forward MAE:", np.mean(mae_scores))

# -------------------------------
# SHAP Explainability
# -------------------------------

st.subheader("SHAP Feature Importance")

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)

fig2 = plt.figure()
shap.summary_plot(shap_values,X_test,show=False)
st.pyplot(fig2)

st.success("Research Pipeline Complete")
