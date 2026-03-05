# ml_engine.py
# Handles all machine learning predictions and forecasting for Veridex
# Uses Random Forest to predict ESG scores and Linear Regression for future trends

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# Raw company features we train on - no E/S/G sub-scores included on purpose
# Using sub-scores would just be adding them up again which isnt real prediction
FEATURE_COLS = [
    "Renewable_Energy_Percentage",
    "Total_Emissions",
    "Emissions_Intensity",
    "Gender_Diversity_Percentage",
    "Board_Diversity_Percentage",
    "Employee_Turnover_Percentage",
    "Training_Hours_Per_Employee",
    "Workplace_Injury_Rate",
    "Independent_Board_Percentage",
    "Audit_Committee_Independence_Percentage",
    "ESG_Linked_Compensation_Yes_No",
    "Carbon_Reduction_Target_Percentage",
    "Sustainability_Investment",
    "Community_Investment",
    "Revenue",
    "Net_Income",
]

TARGET_COL = "Overall_ESG_Score"
def train_random_forest(df):
    # drop rows with missing values in the columns we need
    model_df = df[FEATURE_COLS + [TARGET_COL]].dropna()

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]

    # 80% for training, 20% for testing - standard split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "r2":             round(r2, 3),
        "mae":            round(mae, 2),
        "train_samples":  len(X_train),
        "test_samples":   len(X_test),
    }

    importance_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    # KEY FIX from p1 - return exact column order used during training
    # this is what was causing the ValueError before
    training_feature_order = list(X_train.columns)

    return model, metrics, importance_df, X_test, y_test, training_feature_order
def predict_esg(model, input_dict, feature_cols):
    # builds a single row dataframe from the input values
    row = pd.DataFrame([input_dict])

    # KEY FIX - reindex forces columns to match exact training order
    # this is what was crashing in p1 with the ValueError
    row = row.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(row)[0]
    return round(float(prediction), 1)


def forecast_metric(df_company, metric_col, years_ahead):
    # need at least 2 data points to draw a trend line
    data = df_company[["Year", metric_col]].dropna()
    if len(data) < 2:
        return None

    X = data["Year"].values.reshape(-1, 1)
    y = data[metric_col].values

    model = LinearRegression()
    model.fit(X, y)

    last_year   = int(data["Year"].max())
    future_years = list(range(last_year + 1, last_year + years_ahead + 1))
    future_preds = model.predict(np.array(future_years).reshape(-1, 1))

    # historical portion
    hist = data.copy()
    hist["Type"] = "Historical"
    hist["Year"] = hist["Year"].astype(str)  # FIX - string years = no decimals on chart
    hist = hist.rename(columns={metric_col: "Value"})

    # forecast portion
    forecast = pd.DataFrame({
        "Year":  [str(y) for y in future_years],
        "Value": future_preds,
        "Type":  "Forecast"
    })

    return pd.concat([hist[["Year", "Value", "Type"]], forecast], ignore_index=True)


def get_esg_rating(score):
    # converts a number into a simple label anyone can understand
    if score >= 80: return "Excellent"
    elif score >= 65: return "Good"
    elif score >= 50: return "Fair"
    else: return "Poor"