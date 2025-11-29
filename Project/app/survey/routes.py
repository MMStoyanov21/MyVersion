import os
import joblib
import pandas as pd
import numpy as np
import torch
from flask import Blueprint, render_template, request, redirect, flash, current_app
from flask_login import login_required
from .forms import UploadCSVForm
from app.ai_model.train_model import FFNN

survey = Blueprint('survey', __name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts():

    preprocessor_path = os.path.join(current_app.root_path, "ai_model", "preprocessor.pkl")
    model_path = os.path.join(current_app.root_path, "ai_model", "house_model.pt")

    preprocessor = joblib.load(preprocessor_path)

    dummy = pd.DataFrame([{c: np.nan for c in preprocessor.feature_names_in_}])
    processed_dummy = preprocessor.transform(dummy)
    input_dim = processed_dummy.shape[1]

    model = FFNN(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return preprocessor, model


def predict_batch(df: pd.DataFrame, preprocessor, model) -> np.ndarray:

    for col in preprocessor.feature_names_in_:
        if col not in df.columns:
            df[col] = np.nan

    X = preprocessor.transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    return preds


@survey.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    upload_form = UploadCSVForm()

    # Load model & preprocessor inside request context
    preprocessor, model = load_artifacts()

    if request.method == "POST":
        file = request.files.get("csv_file")
        session_file = request.form.get("csv_file_path")

        if file:
            filename = file.filename
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        elif session_file:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], session_file)
        else:
            flash("CSV file is required.", "danger")
            return redirect(request.url)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Error reading CSV: {e}", "danger")
            return redirect(request.url)

        try:
            preds = predict_batch(df, preprocessor, model)
            df['PredictedPrice'] = np.round(preds, 2)
        except Exception as e:
            flash(f"Error during prediction: {e}", "danger")
            return redirect(request.url)

        return render_template(
            "predict_result.html",
            score=int(df['PredictedPrice'].mean()),
            df=df.to_dict(orient='records'),
            product="House Price"
        )

    return render_template("predict.html", upload_form=upload_form)
