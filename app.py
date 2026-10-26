"""
CredWise - Credit Repayment Risk Predictor
--------------------------------------------
Flask API that loads the trained GradientBoostingClassifier
(creditcard_model.pkl) and the fitted StandardScaler (scaled_model.pkl)
from the ML pipeline in main.py, rebuilds the exact 12-column feature
vector those two objects were fitted on, and returns a Good/Bad
repayment prediction with probabilities.

Developed by Sagar Datti
"""

from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from scipy.stats import yeojohnson

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model artifacts created by the training pipeline
MODEL_PATH = os.path.join(BASE_DIR, "creditcard_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaled model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


# These are the exact feature transformations used by the supplied
# training pipeline for the 12 features expected by the scaler/model.
# The pipeline applies Yeo-Johnson transformation and IQR trimming
# before StandardScaler.
TRANSFORM_PARAMS = {
    "age": (0.6296449147326681, 7.509252484960396, 27.80923511079907),
    "NumberOfOpenCreditLinesAndLoans": (
        0.36619085462775325, 0.251181794301337, 6.334208936324105
    ),
    "NumberRealEstateLoansOrLines": (
        -0.24799246765069444, -1.4425033743411724, 2.4041722905686207
    ),
    "MonthlyIncome_replaced": (
        0.3508267489308711, 19.534620035010562, 91.62381773934442
    ),
}

# Categorical encodings learned from the supplied dataset.
GENDER = {"Female": 0.0, "Male": 1.0}

# OneHotEncoder(drop='first') with alphabetical categories:
# Gender -> Female(reference), Male
# Region -> Central(reference), East, North, South, West
REGION = {
    "Central": [0.0, 0.0, 0.0, 0.0],
    "East":    [1.0, 0.0, 0.0, 0.0],
    "North":   [0.0, 1.0, 0.0, 0.0],
    "South":   [0.0, 0.0, 1.0, 0.0],
    "West":    [0.0, 0.0, 0.0, 1.0],
}

# OrdinalEncoder category order learned from the supplied dataset.
RENTED_OWN = {
    "Ownhouse": 0.0,
    "Rented": 1.0,
}

OCCUPATION = {
    "Non-officer": 0.0,
    "Officer1": 1.0,
    "Officer2": 2.0,
    "Officer3": 3.0,
    "Self_Emp": 4.0,
}

EDUCATION = {
    "Graduate": 0.0,
    "Matric": 1.0,
    "PhD": 2.0,
    "Post-Grad": 3.0,
    "Professional": 4.0,
}


def transform_and_trim(value, feature_name):
    """Apply the same Yeo-Johnson + IQR trimming logic used during training."""
    lmbda, lower, upper = TRANSFORM_PARAMS[feature_name]

    # scipy's yeojohnson() accepts a fixed lambda through lmbda=...
    transformed = float(yeojohnson(np.array([float(value)]), lmbda=lmbda)[0])
    return float(np.clip(transformed, lower, upper))


def build_model_input(form):
    age = float(form["age"])
    open_credit = float(form["open_credit"])
    real_estate = float(form["real_estate"])
    monthly_income = float(form["monthly_income"])

    gender = form["gender"]
    region = form["region"]
    rented_own = form["rented_own"]
    occupation = form["occupation"]
    education = form["education"]

    # Feature order MUST match scaler.feature_names_in_:
    # age, open credit, real estate, monthly income,
    # Gender_Male, Region_East, Region_North, Region_South, Region_West,
    # Rented_OwnHouse_res, Occupation_res, Education_res
    row = [
        transform_and_trim(age, "age"),
        transform_and_trim(
            open_credit, "NumberOfOpenCreditLinesAndLoans"
        ),
        transform_and_trim(
            real_estate, "NumberRealEstateLoansOrLines"
        ),
        transform_and_trim(
            monthly_income, "MonthlyIncome_replaced"
        ),
        GENDER[gender],
        *REGION[region],
        RENTED_OWN[rented_own],
        OCCUPATION[occupation],
        EDUCATION[education],
    ]

    return np.asarray(row, dtype=float).reshape(1, -1)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, probability=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_input = build_model_input(request.form)

        # Scale using the exact StandardScaler saved by the training pipeline.
        scaled_input = scaler.transform(model_input)

        prediction = int(model.predict(scaled_input)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(scaled_input)[0][1]) * 100

        if prediction == 1:
            result = {
                "type": "good",
                "title": "Credit Profile: GOOD",
                "message": "The model classified this application as Good.",
            }
        else:
            result = {
                "type": "bad",
                "title": "Credit Profile: BAD",
                "message": "The model classified this application as Bad.",
            }

        return render_template(
            "index.html",
            result=result,
            probability=probability,
        )

    except Exception as exc:
        result = {
            "type": "error",
            "title": "Prediction Error",
            "message": f"Please check the entered values. Details: {exc}",
        }
        return render_template(
            "index.html",
            result=result,
            probability=None,
        ), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)