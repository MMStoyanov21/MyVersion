import os
import joblib
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, redirect, flash, current_app
from flask_login import login_required
from app.survey.forms import UploadCSVForm

survey = Blueprint('survey', __name__)

SEASON_TO_MONTH = {
    'winter': 1, 'spring': 4, 'summer': 7, 'fall': 10, 'autumn': 10
}


def normalize_columns(df):
    df.columns = [c.strip().lower().replace('-', ' ').replace(' ', '_') for c in df.columns]
    return df


def add_season_features(df):
    SEASON_MAP = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    if 'month' in df.columns:
        df['season'] = df['month'].map(SEASON_MAP)
    return df


def add_cyclical_and_interaction_features(df):
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    if 'price' in df.columns and 'month' in df.columns:
        df['price_month_interaction'] = df['price'] * df['month']
    return df


@survey.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    upload_form = UploadCSVForm()
    if request.method == "POST":
        selected_type = request.form.get("product_type")
        file = request.files.get("csv_file")
        session_file = request.form.get("csv_file_path")

        # --- File handling ---
        if file:
            filename = file.filename
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        elif session_file:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], session_file)
        else:
            flash("CSV file is required.", "danger")
            return redirect(request.url)

        # --- Load CSV ---
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Error reading CSV: {e}", "danger")
            return redirect(request.url)

        df = normalize_columns(df)

        # --- Ensure previous_sells exists ---
        if 'previous_sells' not in df.columns:
            if 'sells' in df.columns:
                df = df.sort_values(by=['type', 'month'] if 'month' in df.columns else ['type'])
                df['previous_sells'] = df.groupby('type')['sells'].shift(1)
                df['previous_sells'] = df.groupby('type')['previous_sells'].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                flash("CSV must contain 'sells' or 'previous_sells' column.", "danger")
                return redirect(request.url)

        # --- Derive month from season if missing ---
        if 'month' not in df.columns and 'season' in df.columns:
            df['month'] = df['season'].map(lambda s: SEASON_TO_MONTH.get(str(s).lower(), np.nan))

        # --- Required columns check ---
        required_cols = ['type', 'color', 'price', 'previous_sells', 'month']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            flash(f"Missing required column(s): {', '.join(missing_cols)}", "danger")
            return redirect(request.url)

        # --- Numeric columns ---
        numeric_cols = ['price', 'previous_sells', 'month']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- Categorical columns ---
        categorical_cols = ['type', 'color', 'season']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().fillna('unknown')

        # --- Feature engineering ---
        df = add_season_features(df)
        df = add_cyclical_and_interaction_features(df)

        # --- Load model ---
        model_file = os.path.join(current_app.root_path, "ai_model", "custom_model.pkl")
        try:
            model_data = joblib.load(model_file)
            model = model_data['model']
            model_columns = model_data['columns']
        except Exception as e:
            flash(f"Error loading model: {e}", "danger")
            return redirect(request.url)

        # --- Ensure all model columns exist ---
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        # --- Order columns as in training ---
        df_for_pred = df[model_columns]

        # --- Make predictions ---
        try:
            preds = np.maximum(0, model.predict(df_for_pred))
            df['predicted_sells'] = np.round(preds, 2)
        except Exception as e:
            flash(f"Error during prediction: {e}", "danger")
            return redirect(request.url)

        # --- Ask user to select product type if not selected ---
        if not selected_type:
            return render_template(
                "predict_choose_product.html",
                product_names=sorted(df['type'].unique()),
                csv_file_path=os.path.basename(filepath)
            )

        # --- Filter for selected product ---
        df_filtered = df[df['type'] == selected_type]

        # --- Render results ---
        return render_template(
            "predict_result.html",
            score=int(round(df_filtered['predicted_sells'].mean())),
            df=df_filtered.to_dict(orient='records'),
            product=selected_type
        )

    return render_template("predict.html", upload_form=upload_form)

