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
from sklearn.linear_model import LinearRegression

from scipy.stats import ttest_ind
from statsmodels.tsa.stattools import adfuller

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🚦 Toll Traffic & Weather-Integrated Research Dashboard")

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

# CRITICAL FIX: Strip entirely blank formatting rows from the bottom of manual sheets
df = df.dropna(subset=["Date"]).reset_index(drop=True)

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

# DYNAMIC ENGINE: Compute ground-truth historical monthly profiles for runtime prediction lookups
monthly_weather_summary = df.groupby("Month")[["T2M_MAX", "PRECTOTCORR"]].mean().to_dict(orient="index")

st.subheader("Dataset Statistics (With Weather Records Included)")
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

# ======================================================
# Feature Matrix
# ======================================================

leakage_cols = [
    "Car_Count","Bus_Count","LCV_Count","MAV_Count",
    "Car_Revenue","Bus_Revenue","LCV_Revenue","MAV_Revenue"
]

# Weather columns (T2M_MAX, PRECTOTCORR) are automatically included here as features!
features = [col for col in df.columns 
            if col not in ["Date","Total_Vehicles","Total_Revenue"] + leakage_cols]

X = df[features]
y = df["Total_Vehicles"]
y_rev = df["Total_Revenue"]

split = int(len(df)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
y_rev_train, y_rev_test = y_rev.iloc[:split], y_rev.iloc[split:]

# ======================================================
# Train Models (Original)
# ======================================================

@st.cache_resource
def train_models(X_train, y_train):

    models = {
        "LinearRegression": LinearRegression(),
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
# NEW: Train Revenue Model (ADDED ONLY)
# ======================================================

@st.cache_resource
def train_revenue_model(X_train, y_rev_train):
    model = XGBRegressor(n_estimators=300)
    model.fit(X_train, y_rev_train)
    return model

revenue_model = train_revenue_model(X_train, y_rev_train)

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

# Matplotlib Best Practice Refactoring: Using object subplots to avoid figure context leaks
fig, ax = plt.subplots(figsize=(12,4))

colors = ["red","blue","green"]

for i in range(3):
    subset = df[df["Regime"] == i]

    ax.scatter(
        subset["Date"],
        subset["Total_Vehicles"],
        color=colors[i],
        label=f"Regime {i}"
    )

ax.set_xlabel("Date")
ax.set_ylabel("Total Vehicles")
ax.legend(title="Traffic Regime")

st.pyplot(fig)

# ======================================================
# SHAP Analysis
# ======================================================

st.header("Feature Importance (SHAP)")

xgb_model = trained_models["XGBoost"]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots()
shap.summary_plot(shap_values,X_test,show=False)
st.pyplot(fig)

# ======================================================
# FUTURE TRAFFIC + REVENUE PREDICTION
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
    
    # Auto-extract weather properties matching selected month parameter
    weather_defaults = monthly_weather_summary.get(travel_date.month, {"T2M_MAX": 32.0, "PRECTOTCORR": 0.0})

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
        "Is_Holiday": 0,
        "T2M_MAX": weather_defaults["T2M_MAX"],
        "PRECTOTCORR": weather_defaults["PRECTOTCORR"]
    }

    future_df = pd.DataFrame([future_data])
    future_df = future_df.reindex(columns=features, fill_value=0)

    # Traffic Prediction
    prediction = model.predict(future_df)[0]

    st.subheader(f"🚗 Predicted Traffic: {int(prediction)} vehicles")

    avg = df["Total_Vehicles"].mean()

    if prediction < avg*0.85:
        st.success("🟢 Low Traffic – Safe to Travel")
    elif prediction < avg*1.1:
        st.warning("🟡 Moderate Traffic – Plan Ahead")
    else:
        st.error("🔴 Heavy Traffic – Avoid Peak Hours")

    # ================= NEW REVENUE BLOCK =================
    revenue_pred = revenue_model.predict(future_df)[0]

    st.subheader(f"💰 Predicted Revenue: ₹{int(revenue_pred):,}")

    avg_rev = df["Total_Revenue"].mean()

    if revenue_pred < avg_rev*0.85:
        st.info("📉 Low Revenue Day")
    elif revenue_pred < avg_rev*1.1:
        st.warning("📊 Average Revenue Day")
    else:
        st.success("📈 High Revenue Expected")


# ======================================================
# NEW ADDITION: 7-DAY MULTI-DAY FORECAST SECTION
# ======================================================

st.header("🔮 Next 7 Days Continuous Forecast")
st.markdown("Select a start date from the calendar to project traffic and revenue sequentially for the next 7 days.")

# Dynamic base date defaults to day after last log in file if it exists, otherwise today
last_data_date = df["Date"].max().date() if "Date" in df.columns else datetime.today().date()
default_start_date = last_data_date + timedelta(days=1)

with st.form("multi_day_form"):
    start_date = st.date_input(
        "Select Forecast Start Date", 
        value=default_start_date
    )
    multi_submitted = st.form_submit_button("Generate 7-Day Forecast")

if multi_submitted:
    # Use Ridge for consistency with your single prediction segment
    traffic_model = trained_models["Ridge"]
    
    # We maintain a dynamic array list copies of historical vehicles to slide forward
    history_vehicles = list(df["Total_Vehicles"].values)
    forecast_results = []
    
    current_time_index = len(df)
    
    # Generate days sequentially
    for i in range(7):
        target_day = start_date + timedelta(days=i)
        
        # Pull lags progressively updating dynamically from historical and previously generated predictions
        lag1 = history_vehicles[-1]
        lag7 = history_vehicles[-7] if len(history_vehicles) >= 7 else history_vehicles[-1]
        lag14 = history_vehicles[-14] if len(history_vehicles) >= 14 else history_vehicles[-1]
        lag30 = history_vehicles[-30] if len(history_vehicles) >= 30 else history_vehicles[-1]
        
        # Calculate true rolling windows incorporating previous loops
        roll7 = np.mean(history_vehicles[-7:])
        roll14 = np.mean(history_vehicles[-14:])
        roll30 = np.mean(history_vehicles[-30:])
        
        # Extract ground-truth seasonal climate properties for the loop day
        weather_defaults = monthly_weather_summary.get(target_day.month, {"T2M_MAX": 32.0, "PRECTOTCORR": 0.0})
        
        # Package input structure exactly matching existing features matrices
        step_features = {
            "DayOfWeek": target_day.weekday(),
            "Month": target_day.month,
            "DayOfYear": target_day.timetuple().tm_yday,
            "Time_Index": current_time_index + i,
            "Lag_1": lag1,
            "Lag_7": lag7,
            "Lag_14": lag14,
            "Lag_30": lag30,
            "Rolling_7": roll7,
            "Rolling_14": roll14,
            "Rolling_30": roll30,
            "Is_Weekend": 1 if target_day.weekday() >= 5 else 0,
            "Is_Holiday": 0,
            "T2M_MAX": weather_defaults["T2M_MAX"],
            "PRECTOTCORR": weather_defaults["PRECTOTCORR"]
        }
        
        step_df = pd.DataFrame([step_features])
        step_df = step_df.reindex(columns=features, fill_value=0)
        
        # Predict dynamic targets
        pred_traffic = int(max(0, traffic_model.predict(step_df)[0])) # Floor at zero to prevent negative artifacts
        pred_revenue = int(max(0, revenue_model.predict(step_df)[0]))
        
        # Append latest metrics to history array list so the NEXT loop sees it as a rolling lag
        history_vehicles.append(pred_traffic)
        
        # Collect results row item
        forecast_results.append({
            "Date": target_day.strftime('%Y-%m-%d'),
            "Day": target_day.strftime('%A'),
            "Expected Rain": f"{weather_defaults['PRECTOTCORR']:.2f} mm",
            "Max Temp": f"{weather_defaults['T2M_MAX']:.1f} °C",
            "Predicted Traffic (Vehicles)": f"{pred_traffic:,}",
            "Predicted Revenue (₹)": f"₹{pred_revenue:,}"
        })
        
    # Render Output Layout
    out_df = pd.DataFrame(forecast_results)
    st.dataframe(out_df, use_container_width=True, hide_index=True)
    
    # Trend visual metrics chart
    st.subheader("📈 Visual Breakdown Trends")
    fig, ax1 = plt.subplots(figsize=(10, 3.5))
    
    dates_str = [x["Date"] for x in forecast_results]
    raw_traffic = [int(x["Predicted Traffic (Vehicles)"].replace(',', '')) for x in forecast_results]
    raw_revenue = [int(x["Predicted Revenue (₹)"].replace('₹', '').replace(',', '')) for x in forecast_results]
    
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Traffic Count', color=color)
    ax1.plot(dates_str, raw_traffic, color=color, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=15)
    
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Revenue (₹)', color=color)
    ax2.plot(dates_str, raw_revenue, color=color, marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    st.pyplot(fig)

st.success("Dashboard Ready 🚀")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression

# from scipy.stats import ttest_ind
# from statsmodels.tsa.stattools import adfuller

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# import shap

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# st.set_page_config(layout="wide")
# st.title("🚦 Toll Traffic Research Dashboard")

# # ======================================================
# # Upload Dataset
# # ======================================================

# uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)")

# if uploaded_file is None:
#     st.stop()

# if uploaded_file.name.endswith("csv"):
#     df = pd.read_csv(uploaded_file)
# else:
#     df = pd.read_excel(uploaded_file)

# df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
# df = df.sort_values("Date").reset_index(drop=True)

# # ======================================================
# # Feature Engineering
# # ======================================================

# df["DayOfWeek"] = df["Date"].dt.dayofweek
# df["Month"] = df["Date"].dt.month
# df["DayOfYear"] = df["Date"].dt.dayofyear
# df["Time_Index"] = np.arange(len(df))

# for lag in [1,7,14,30]:
#     df[f"Lag_{lag}"] = df["Total_Vehicles"].shift(lag)

# for roll in [7,14,30]:
#     df[f"Rolling_{roll}"] = df["Total_Vehicles"].rolling(roll).mean()

# df = df.dropna().reset_index(drop=True)

# st.subheader("Dataset Statistics")
# st.write(df[["Total_Vehicles","Total_Revenue"]].describe())

# # ======================================================
# # Hypothesis Testing
# # ======================================================

# st.header("Weekend Effect Test")

# weekend = df[df["Is_Weekend"]==1]["Total_Vehicles"]
# weekday = df[df["Is_Weekend"]==0]["Total_Vehicles"]

# t_stat, p_val = ttest_ind(weekend,weekday)

# st.write("Mean Weekend:", weekend.mean())
# st.write("Mean Weekday:", weekday.mean())

# # ======================================================
# # Feature Matrix
# # ======================================================

# leakage_cols = [
#     "Car_Count","Bus_Count","LCV_Count","MAV_Count",
#     "Car_Revenue","Bus_Revenue","LCV_Revenue","MAV_Revenue"
# ]

# features = [col for col in df.columns 
#             if col not in ["Date","Total_Vehicles","Total_Revenue"] + leakage_cols]

# X = df[features]
# y = df["Total_Vehicles"]
# y_rev = df["Total_Revenue"]

# split = int(len(df)*0.8)
# X_train, X_test = X.iloc[:split], X.iloc[split:]
# y_train, y_test = y.iloc[:split], y.iloc[split:]
# y_rev_train, y_rev_test = y_rev.iloc[:split], y_rev.iloc[split:]

# # ======================================================
# # Train Models (Original)
# # ======================================================

# @st.cache_resource
# def train_models(X_train, y_train):

#     models = {
#         "LinearRegression": LinearRegression(),
#         "Ridge": Ridge(),
#         "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
#         "SVR": SVR(),
#         "XGBoost": XGBRegressor(n_estimators=300),
#         "LightGBM": LGBMRegressor()
#     }

#     trained = {}

#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         trained[name] = model

#     return trained

# trained_models = train_models(X_train, y_train)

# # ======================================================
# # NEW: Train Revenue Model (ADDED ONLY)
# # ======================================================

# @st.cache_resource
# def train_revenue_model(X_train, y_rev_train):
#     model = XGBRegressor(n_estimators=300)
#     model.fit(X_train, y_rev_train)
#     return model

# revenue_model = train_revenue_model(X_train, y_rev_train)

# # ======================================================
# # Model Evaluation
# # ======================================================

# st.header("Model Comparison")

# results = []

# for name, model in trained_models.items():

#     preds = model.predict(X_test)

#     mae = mean_absolute_error(y_test,preds)
#     rmse = np.sqrt(mean_squared_error(y_test,preds))
#     r2 = r2_score(y_test,preds)

#     results.append([name,mae,rmse,r2])

# results_df = pd.DataFrame(results,
#         columns=["Model","MAE","RMSE","R2"])

# st.write(results_df.sort_values("R2",ascending=False))

# # ======================================================
# # Walk Forward Validation
# # ======================================================

# st.header("Walk Forward Validation")

# tscv = TimeSeriesSplit(n_splits=5)
# mae_scores = []

# for train_idx,test_idx in tscv.split(X):

#     X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
#     y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

#     model = XGBRegressor()
#     model.fit(X_tr,y_tr)

#     pred = model.predict(X_te)

#     mae_scores.append(mean_absolute_error(y_te,pred))

# st.write("Average Walk Forward MAE:", np.mean(mae_scores))

# # ======================================================
# # Regime Detection
# # ======================================================

# st.header("Traffic Regimes")

# regime_features = df[["Total_Vehicles","Lag_1","Lag_7","Is_Weekend","Is_Holiday"]]

# scaled = StandardScaler().fit_transform(regime_features)

# kmeans = KMeans(n_clusters=3, random_state=42)
# df["Regime"] = kmeans.fit_predict(scaled)

# plt.figure(figsize=(12,4))

# colors = ["red","blue","green"]

# for i in range(3):
#     subset = df[df["Regime"] == i]

#     plt.scatter(
#         subset["Date"],
#         subset["Total_Vehicles"],
#         color=colors[i],
#         label=f"Regime {i}"
#     )

# plt.xlabel("Date")
# plt.ylabel("Total Vehicles")
# plt.legend(title="Traffic Regime")

# st.pyplot(plt.gcf())

# # ======================================================
# # SHAP Analysis
# # ======================================================

# st.header("Feature Importance (SHAP)")

# xgb_model = trained_models["XGBoost"]

# explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(X_test)

# plt.figure()
# shap.summary_plot(shap_values,X_test,show=False)
# st.pyplot(plt.gcf())

# # ======================================================
# # FUTURE TRAFFIC + REVENUE PREDICTION
# # ======================================================

# st.header("Future Travel Prediction")

# today = datetime.today().date()
# max_date = today + timedelta(days=60)

# with st.form("prediction_form"):

#     travel_date = st.date_input(
#         "Select travel date",
#         min_value=today,
#         max_value=max_date
#     )

#     submitted = st.form_submit_button("Predict Traffic")

# if submitted:

#     model = trained_models["Ridge"]

#     future_data = {
#         "DayOfWeek": travel_date.weekday(),
#         "Month": travel_date.month,
#         "DayOfYear": travel_date.timetuple().tm_yday,
#         "Time_Index": len(df)+1,
#         "Lag_1": df["Total_Vehicles"].iloc[-1],
#         "Lag_7": df["Total_Vehicles"].iloc[-7],
#         "Lag_14": df["Total_Vehicles"].iloc[-14],
#         "Lag_30": df["Total_Vehicles"].iloc[-30],
#         "Rolling_7": df["Total_Vehicles"].tail(7).mean(),
#         "Rolling_14": df["Total_Vehicles"].tail(14).mean(),
#         "Rolling_30": df["Total_Vehicles"].tail(30).mean(),
#         "Is_Weekend": 1 if travel_date.weekday()>=5 else 0,
#         "Is_Holiday": 0
#     }

#     future_df = pd.DataFrame([future_data])
#     future_df = future_df.reindex(columns=features, fill_value=0)

#     # Traffic Prediction
#     prediction = model.predict(future_df)[0]

#     st.subheader(f"🚗 Predicted Traffic: {int(prediction)} vehicles")

#     avg = df["Total_Vehicles"].mean()

#     if prediction < avg*0.85:
#         st.success("🟢 Low Traffic – Safe to Travel")
#     elif prediction < avg*1.1:
#         st.warning("🟡 Moderate Traffic – Plan Ahead")
#     else:
#         st.error("🔴 Heavy Traffic – Avoid Peak Hours")

#     # ================= NEW REVENUE BLOCK =================
#     revenue_pred = revenue_model.predict(future_df)[0]

#     st.subheader(f"💰 Predicted Revenue: ₹{int(revenue_pred):,}")

#     avg_rev = df["Total_Revenue"].mean()

#     if revenue_pred < avg_rev*0.85:
#         st.info("📉 Low Revenue Day")
#     elif revenue_pred < avg_rev*1.1:
#         st.warning("📊 Average Revenue Day")
#     else:
#         st.success("📈 High Revenue Expected")


# # ======================================================
# # NEW ADDITION: 7-DAY MULTI-DAY FORECAST SECTION
# # ======================================================

# st.header("🔮 Next 7 Days Continuous Forecast")
# st.markdown("Select a start date from the calendar to project traffic and revenue sequentially for the next 7 days.")

# # Dynamic base date defaults to day after last log in file if it exists, otherwise today
# last_data_date = df["Date"].max().date() if "Date" in df.columns else datetime.today().date()
# default_start_date = last_data_date + timedelta(days=1)

# with st.form("multi_day_form"):
#     start_date = st.date_input(
#         "Select Forecast Start Date", 
#         value=default_start_date
#     )
#     multi_submitted = st.form_submit_button("Generate 7-Day Forecast")

# if multi_submitted:
#     # Use Ridge for consistency with your single prediction segment
#     traffic_model = trained_models["Ridge"]
    
#     # We maintain a dynamic array list copies of historical vehicles to slide forward
#     history_vehicles = list(df["Total_Vehicles"].values)
#     forecast_results = []
    
#     current_time_index = len(df)
    
#     # Generate days sequentially
#     for i in range(7):
#         target_day = start_date + timedelta(days=i)
        
#         # Pull lags progressively updating dynamically from historical and previously generated predictions
#         lag1 = history_vehicles[-1]
#         lag7 = history_vehicles[-7] if len(history_vehicles) >= 7 else history_vehicles[-1]
#         lag14 = history_vehicles[-14] if len(history_vehicles) >= 14 else history_vehicles[-1]
#         lag30 = history_vehicles[-30] if len(history_vehicles) >= 30 else history_vehicles[-1]
        
#         # Calculate true rolling windows incorporating previous loops
#         roll7 = np.mean(history_vehicles[-7:])
#         roll14 = np.mean(history_vehicles[-14:])
#         roll30 = np.mean(history_vehicles[-30:])
        
#         # Package input structure exactly matching existing features matrices
#         step_features = {
#             "DayOfWeek": target_day.weekday(),
#             "Month": target_day.month,
#             "DayOfYear": target_day.timetuple().tm_yday,
#             "Time_Index": current_time_index + i,
#             "Lag_1": lag1,
#             "Lag_7": lag7,
#             "Lag_14": lag14,
#             "Lag_30": lag30,
#             "Rolling_7": roll7,
#             "Rolling_14": roll14,
#             "Rolling_30": roll30,
#             "Is_Weekend": 1 if target_day.weekday() >= 5 else 0,
#             "Is_Holiday": 0
#         }
        
#         step_df = pd.DataFrame([step_features])
#         step_df = step_df.reindex(columns=features, fill_value=0)
        
#         # Predict dynamic targets
#         pred_traffic = int(max(0, traffic_model.predict(step_df)[0])) # Floor at zero to prevent negative artifacts
#         pred_revenue = int(max(0, revenue_model.predict(step_df)[0]))
        
#         # Append latest metrics to history array list so the NEXT loop sees it as a rolling lag
#         history_vehicles.append(pred_traffic)
        
#         # Collect results row item
#         forecast_results.append({
#             "Date": target_day.strftime('%Y-%m-%d'),
#             "Day": target_day.strftime('%A'),
#             "Predicted Traffic (Vehicles)": f"{pred_traffic:,}",
#             "Predicted Revenue (₹)": f"₹{pred_revenue:,}"
#         })
        
#     # Render Output Layout
#     out_df = pd.DataFrame(forecast_results)
#     st.dataframe(out_df, use_container_width=True, hide_index=True)
    
#     # Optional Trend visual metrics chart
#     st.subheader("📈 Visual Breakdown Trends")
#     fig, ax1 = plt.subplots(figsize=(10, 3.5))
    
#     dates_str = [x["Date"] for x in forecast_results]
#     raw_traffic = [int(x["Predicted Traffic (Vehicles)"].replace(',', '')) for x in forecast_results]
#     raw_revenue = [int(x["Predicted Revenue (₹)"].replace('₹', '').replace(',', '')) for x in forecast_results]
    
#     color = 'tab:blue'
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Traffic Count', color=color)
#     ax1.plot(dates_str, raw_traffic, color=color, marker='o', linewidth=2)
#     ax1.tick_params(axis='y', labelcolor=color)
#     plt.xticks(rotation=15)
    
#     ax2 = ax1.twinx()  
#     color = 'tab:green'
#     ax2.set_ylabel('Revenue (₹)', color=color)
#     ax2.plot(dates_str, raw_revenue, color=color, marker='s', linestyle='--', linewidth=2)
#     ax2.tick_params(axis='y', labelcolor=color)
    
#     fig.tight_layout()
#     st.pyplot(fig)

# st.success("Dashboard Ready 🚀")


# st.success("Dashboard Ready 🚀")
