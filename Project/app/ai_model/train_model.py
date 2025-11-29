import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

# FFNN Model
class FFNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)

# Torch Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    print("Running training for Ames FFNN…")

    # Static directory for saving models/plots
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "ai_model")
    os.makedirs(STATIC_DIR, exist_ok=True)

    CSV_PATH = "AmesHousing.csv"
    df = pd.read_csv(CSV_PATH)

    # Drop unused columns
    df = df.drop(columns=["Order", "PID"], errors="ignore")

    # Target
    y = df["SalePrice"].values.astype(np.float32).reshape(-1, 1)
    X = df.drop(columns=["SalePrice"])

    # Column types
    as_cat = ["MSSubClass"]
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in as_cat]
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist() + as_cat

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Train/test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit preprocessing
    preprocessor.fit(X_train_df)
    X_train = preprocessor.transform(X_train_df).astype(np.float32)
    X_test = preprocessor.transform(X_test_df).astype(np.float32)

    # Save preprocessor
    preprocessor_path = os.path.join(STATIC_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved: {preprocessor_path}")

    # Datasets and loaders
    train_ds = TabularDataset(X_train, y_train)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Model setup
    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNN(input_dim).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    epochs = 60
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(np.mean(batch_losses))

        # Validation
        model.eval()
        batch_val = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                batch_val.append(loss.item())

        val_losses.append(np.mean(batch_val))

        print(f"Epoch {epoch+1}/{epochs}  Train: {train_losses[-1]:.4f}  Val: {val_losses[-1]:.4f}")

    # Save model
    model_path = os.path.join(STATIC_DIR, "house_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    # Save loss plot
    loss_path = os.path.join(STATIC_DIR, "loss_curve.png")
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(loss_path)
    plt.close()

    # Actual vs Predicted
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, device=device)).cpu().numpy().flatten()

    actual_vs_pred_path = os.path.join(STATIC_DIR, "actual_vs_pred.png")
    plt.figure(figsize=(7.5,6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Sale Price")
    plt.savefig(actual_vs_pred_path)
    plt.close()

    print("Training finished. Files saved:")
    print(f"- {preprocessor_path}")
    print(f"- {model_path}")
    print(f"- {loss_path}")
    print(f"- {actual_vs_pred_path}")
