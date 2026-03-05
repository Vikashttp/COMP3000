# ml_engine.py
# Handles all machine learning logic for the Veridex platform
# Uses Random Forest to predict ESG scores from company financial and ESG data

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


# These are the input features the ML model uses to predict ESG score
# We chose these because they are the most complete columns in our dataset
# and they represent the key drivers of ESG performance in literature

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


def prepare_data(df):
    """
    Clean and prepare the dataset for machine learning.
    
    What we do here:
    - Keep only the columns the model needs
    - Drop any rows where the target (ESG score) is missing
    - Fill remaining missing values with the column average
    - Encode the Sector column into numbers (ML cannot read text)
    """
    working = df.copy()

    # Encode sector as a number so the model can use it
    if "Sector" in working.columns:
        le = LabelEncoder()
        working["Sector_Encoded"] = le.fit_transform(working["Sector"].astype(str))
        feature_cols = FEATURE_COLS + ["Sector_Encoded"]
    else:
        feature_cols = FEATURE_COLS

    # Only keep features that actually exist in the data
    available = [c for c in feature_cols if c in working.columns]

    # Convert all feature columns to numeric, replace errors with NaN
    for col in available:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    working[TARGET_COL] = pd.to_numeric(working[TARGET_COL], errors="coerce")

    # Drop rows where the target ESG score is missing
    working = working.dropna(subset=[TARGET_COL])

    # Fill missing feature values with the column mean
    working[available] = working[available].fillna(working[available].mean())

    X = working[available]
    y = working[TARGET_COL]

    return X, y, available


def train_random_forest(df):
    """
    Train a Random Forest model to predict Overall ESG Score.
    
    Random Forest works by building many decision trees and averaging
    their predictions. It handles non-linear relationships well and
    gives us feature importance scores for transparency.
    
    Returns:
    - model: the trained Random Forest
    - metrics: MAE and R2 score on the test set
    - feature_importance: which features matter most
    - X_test, y_test: the held-out test data for evaluation
    """
    X, y, features = prepare_data(df)

    # Need at least 20 rows to train and test meaningfully
    if len(X) < 20:
        return None, {}, pd.DataFrame(), None, None

    # Split into 80% training and 20% testing
    # random_state=42 means the split is reproducible every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the Random Forest
    # n_estimators=100 means we build 100 decision trees
    # random_state=42 makes results reproducible
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=8,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 3),
        "R2": round(r2, 3),
        "Training samples": len(X_train),
        "Test samples": len(X_test),
    }

    # Feature importance tells us which inputs matter most
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, metrics, importance_df, X_test, y_test


def predict_esg(model, input_data, feature_cols):
    """
    Use the trained model to predict ESG score for a single company.
    Ensures feature columns are in exactly the same order as training.
    """
    row = pd.DataFrame([input_data])

    # Make sure all required columns exist
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)

    # IMPORTANT: reorder columns to exactly match training order
    row = row.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(row)[0]
    return round(float(prediction), 2)


def forecast_metric(df_company, col, steps=3):
    """
    Forecast a metric (like ESG score or Revenue) for future years
    using simple linear regression on historical data.
    
    Why linear regression for forecasting?
    It is explainable, simple, and appropriate for short time series.
    With only 5 years of data, complex models would overfit.
    
    Returns a DataFrame with predicted values for future years.
    """
    data = df_company[["Year", col]].dropna().sort_values("Year")

    # Need at least 3 data points to make a meaningful forecast
    if len(data) < 3:
        return pd.DataFrame()

    X = data["Year"].values.reshape(-1, 1)
    y = data[col].values

    model = LinearRegression()
    model.fit(X, y)

    last_year = int(data["Year"].max())
    future_years = np.array(
        [last_year + i for i in range(1, steps + 1)]
    ).reshape(-1, 1)

    predictions = model.predict(future_years)

    result = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted": predictions.round(2),
        "Type": "Forecast"
    })

    return result