

import joblib
import pandas as pd
import numpy as np
import torch
from train_model import FFNN


preprocessor = joblib.load("preprocessor.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dummy = pd.DataFrame([{c: np.nan for c in preprocessor.feature_names_in_}])
processed_dummy = preprocessor.transform(dummy)
input_dim = processed_dummy.shape[1]

model = FFNN(input_dim).to(device)
model.load_state_dict(torch.load("house_model.pt", map_location=device))
model.eval()

def predict_from_dict(feature_dict: dict) -> float:

    df = pd.DataFrame([feature_dict])

    for col in preprocessor.feature_names_in_:
        if col not in df.columns:
            df[col] = np.nan

    X = preprocessor.transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy().flatten()[0]

    return float(pred)



