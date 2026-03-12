import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from scipy.stats import ttest_ind
from statsmodels.tsa.stattools import adfuller

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🚦 Toll Traffic Research Dashboard")

# ======================================================
# Upload Dataset
# ======================================================

uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)")

if uploaded_file is None:
    st.stop()

if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# ======================================================
# Feature Engineering
# ======================================================

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear
df["Time_Index"] = np.arange(len(df))

for lag in [1,7,14,30]:
    df[f"Lag_{lag}"] = df["Total_Vehicles"].shift(lag)

for roll in [7,14,30]:
    df[f"Rolling_{roll}"] = df["Total_Vehicles"].rolling(roll).mean()

df = df.dropna().reset_index(drop=True)

st.subheader("Dataset Statistics")
st.write(df[["Total_Vehicles","Total_Revenue"]].describe())

# ======================================================
# Hypothesis Testing
# ======================================================

st.header("Weekend Effect Test")

weekend = df[df["Is_Weekend"]==1]["Total_Vehicles"]
weekday = df[df["Is_Weekend"]==0]["Total_Vehicles"]

t_stat, p_val = ttest_ind(weekend,weekday)

st.write("Mean Weekend:", weekend.mean())
st.write("Mean Weekday:", weekday.mean())
#st.write("p-value:", p_val)

# ======================================================
# ADF Test
# ======================================================

st.header("ADF Stationarity Test")

adf = adfuller(df["Total_Vehicles"])
st.write("ADF Statistic:", adf[0])
st.write("p-value:", adf[1])

# ======================================================
# Feature Matrix
# ======================================================

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

# ======================================================
# Train Models (Cached for Speed)
# ======================================================

@st.cache_resource
def train_models(X_train, y_train):

    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(n_estimators=300),
        "LightGBM": LGBMRegressor()
    }

    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained

trained_models = train_models(X_train, y_train)

# ======================================================
# Model Evaluation
# ======================================================

st.header("Model Comparison")

results = []

for name, model in trained_models.items():

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    r2 = r2_score(y_test,preds)

    results.append([name,mae,rmse,r2])

results_df = pd.DataFrame(results,
        columns=["Model","MAE","RMSE","R2"])

st.write(results_df.sort_values("R2",ascending=False))

# ======================================================
# Walk Forward Validation
# ======================================================

st.header("Walk Forward Validation")

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

# ======================================================
# Regime Detection
# ======================================================

st.header("Traffic Regimes")

regime_features = df[["Total_Vehicles","Lag_1","Lag_7","Is_Weekend","Is_Holiday"]]

scaled = StandardScaler().fit_transform(regime_features)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Regime"] = kmeans.fit_predict(scaled)

plt.figure(figsize=(12,4))
plt.scatter(df["Date"], df["Total_Vehicles"], c=df["Regime"])
st.pyplot(plt.gcf())

# ======================================================
# SHAP Analysis
# ======================================================

st.header("Feature Importance (SHAP)")

xgb_model = trained_models["XGBoost"]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values,X_test,show=False)
st.pyplot(plt.gcf())

# ======================================================
# FUTURE TRAFFIC PREDICTION
# ======================================================

st.header("Future Travel Prediction")

today = datetime.today().date()
max_date = today + timedelta(days=60)

with st.form("prediction_form"):

    travel_date = st.date_input(
        "Select travel date",
        min_value=today,
        max_value=max_date
    )

    submitted = st.form_submit_button("Predict Traffic")

if submitted:

    model = trained_models["Ridge"]

    future_data = {
        "DayOfWeek": travel_date.weekday(),
        "Month": travel_date.month,
        "DayOfYear": travel_date.timetuple().tm_yday,
        "Time_Index": len(df)+1,
        "Lag_1": df["Total_Vehicles"].iloc[-1],
        "Lag_7": df["Total_Vehicles"].iloc[-7],
        "Lag_14": df["Total_Vehicles"].iloc[-14],
        "Lag_30": df["Total_Vehicles"].iloc[-30],
        "Rolling_7": df["Total_Vehicles"].tail(7).mean(),
        "Rolling_14": df["Total_Vehicles"].tail(14).mean(),
        "Rolling_30": df["Total_Vehicles"].tail(30).mean(),
        "Is_Weekend": 1 if travel_date.weekday()>=5 else 0,
        "Is_Holiday": 0
    }

    future_df = pd.DataFrame([future_data])

    # Align columns exactly with training features
    future_df = future_df.reindex(columns=features, fill_value=0)

    prediction = model.predict(future_df)[0]

    avg = df["Total_Vehicles"].mean()

    st.subheader(f"Predicted Traffic: {int(prediction)} vehicles")

    if prediction < avg*0.85:
        st.success("🟢 Low Traffic – Safe to Travel")
    elif prediction < avg*1.1:
        st.warning("🟡 Moderate Traffic – Plan Ahead")
    else:
        st.error("🔴 Heavy Traffic – Avoid Peak Hours")

st.success("Dashboard Ready")
