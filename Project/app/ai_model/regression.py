import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# --- CONFIG ---
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "AmesHousing.csv")
MODEL_PATH = os.path.join(BASE_DIR, "custom_model.pkl")


# --- Feature Engineering ---
def add_feature_engineering(df):
    # Normalize column names
    df.columns = [col.strip().lower().replace("-", "_").replace(" ", "_") for col in df.columns]

    # Ensure required columns exist
    required = {"type", "color", "price", "sells", "month"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    # Convert numeric columns
    for col in ["price", "sells", "month"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Create previous_sells if missing
    if "previous_sells" not in df.columns:
        df = df.sort_values(by=['type', 'month'])
        df['previous_sells'] = df.groupby('type')['sells'].shift(1)
        df['previous_sells'] = df.groupby('type')['previous_sells'].transform(lambda x: x.fillna(x.median()))

    # Create season from month
    month_to_season = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    df['season'] = df['month'].map(month_to_season)

    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Interaction feature
    df['price_month_interaction'] = df['price'] * df['month']

    return df


# --- Training Function ---
def train_model_from_csv(csv_path):
    print(f"Loading training data from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = add_feature_engineering(df)

    target = "sells"
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    # Separate categorical & numerical
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ])

    # Model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")

    # Save
    joblib.dump({"model": pipeline, "columns": features}, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model_from_csv(CSV_PATH)
