from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]
columns = model_data["columns"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        input_data = {
            "OverallQual": int(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "GarageCars": int(request.form["GarageCars"]),
            "YearBuilt": int(request.form["YearBuilt"]),
            "Neighborhood": request.form["Neighborhood"]
        }

        df = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

        df_scaled = scaler.transform(df_encoded)
        prediction = model.predict(df_scaled)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
