# predict_ames_nn.py
# Usage: python predict_ames_nn.py

import joblib
import pandas as pd
import numpy as np
import torch
from train_model import FFNN  # your same network architecture

# -------------------------------
# Load preprocessor and model
# -------------------------------
preprocessor = joblib.load("preprocessor.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build dummy row to infer final input dimension
dummy = pd.DataFrame([{c: np.nan for c in preprocessor.feature_names_in_}])
processed_dummy = preprocessor.transform(dummy)
input_dim = processed_dummy.shape[1]

# Reconstruct and load model
model = FFNN(input_dim).to(device)
model.load_state_dict(torch.load("house_model.pt", map_location=device))
model.eval()


# -------------------------------
# Prediction Function
# -------------------------------
def predict_from_dict(feature_dict: dict) -> float:
    """
    Accepts a python dict with original Ames dataset columns.
    Missing values are filled automatically by the preprocessor.
    """

    # Convert dict → DataFrame
    df = pd.DataFrame([feature_dict])

    # Ensure all expected columns exist
    for col in preprocessor.feature_names_in_:
        if col not in df.columns:
            df[col] = np.nan

    # Preprocess
    X = preprocessor.transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy().flatten()[0]

    return float(pred)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    example_input = {
        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotFrontage": 60.0,
        "LotArea": 8450,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Neighborhood": "CollgCr",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemod/Add": 2003,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "GrLivArea": 1710,
        "FullBath": 2,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8,
        "Fireplaces": 0,
        "GarageCars": 2,
        "GarageArea": 548,
        "MoSold": 2,
        "YrSold": 2010,
    }

    pred_price = predict_from_dict(example_input)
    print(f"Predicted Sale Price: ${pred_price:,.2f}")
