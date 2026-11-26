"""
pipeline.py - Backend for the Student Mental Health Score app.

Reproduces the steps from Mental_Health_Score.ipynb (cleaning, country
grouping, preprocessing, Random Forest tuning) and saves the trained
model as a .pkl file. The notebook itself is left untouched.

Usage:
    python pipeline.py            # train and save model/mental_health_model.pkl
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Student Social Media And Mental Health Impact.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "mental_health_model.pkl"

# ---- Feature groups (same as notebook) ----
SKEWED_COLS = ["Study_Hours"]
OTHER_NUMERIC_COLS = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
]
ORDINAL_COL = ["Stress_Level"]
NOMINAL_COLS = [
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country",
]
FEATURE_COLS = SKEWED_COLS + OTHER_NUMERIC_COLS + ORDINAL_COL + NOMINAL_COLS
TARGET = "Mental_Health_Score"
STRESS_ORDER = ["Low", "Medium", "High", "Very High"]


# ---- Data ----
def load_and_clean_data(path=DATA_PATH):
    """Load CSV, drop duplicates, clip negative activity hours, group countries."""
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)

    top_countries = df["Country"].value_counts().index[:10].tolist()
    df["Grouped_country"] = df["Country"].apply(
        lambda c: c if c in top_countries else "Other"
    )
    return df, top_countries


# ---- Model ----
def build_preprocessor():
    skew_pipeline = Pipeline(
        steps=[
            ("log_transform", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
        ]
    )
    plain_numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])
    ordinal_pipeline = Pipeline(
        steps=[("encode", OrdinalEncoder(categories=[STRESS_ORDER]))]
    )
    nominal_pipeline = Pipeline(
        steps=[("encode", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[
            ("Skewed_Pipeline", skew_pipeline, SKEWED_COLS),
            ("Plain_Numeric", plain_numeric_pipeline, OTHER_NUMERIC_COLS),
            ("Ordinal", ordinal_pipeline, ORDINAL_COL),
            ("Normal", nominal_pipeline, NOMINAL_COLS),
        ]
    )


def train_and_save(model_path=MODEL_PATH):
    """Train tuned Random Forest pipeline and save it as a .pkl bundle."""
    df, top_countries = load_and_clean_data()
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("random forest", RandomForestRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "random forest__n_estimators": [100, 200, 300],
        "random forest__max_depth": [5, 10, 15],
        "random forest__min_samples_split": [2, 5, 10],
        "random forest__min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_grid,
        n_iter=15,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    preds = best.predict(X_test)
    metrics = {
        "r2_test": float(r2_score(y_test, preds)),
        "r2_train": float(r2_score(y_train, best.predict(X_train))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "best_params": search.best_params_,
    }

    # Options for the UI dropdowns, taken from the data
    options = {
        "Gender": sorted(df["Gender"].unique().tolist()),
        "Academic_Level": sorted(df["Academic_Level"].unique().tolist()),
        "Most_Used_Platform": sorted(df["Most_Used_Platform"].unique().tolist()),
        "Purpose_Of_Use": sorted(df["Purpose_Of_Use"].unique().tolist()),
        "Grouped_country": sorted(df["Grouped_country"].unique().tolist()),
        "Stress_Level": STRESS_ORDER,
    }
    ranges = {
        c: [float(df[c].min()), float(df[c].max()), float(df[c].median())]
        for c in SKEWED_COLS + OTHER_NUMERIC_COLS
    }

    bundle = {
        "model": best,
        "top_countries": top_countries,
        "options": options,
        "ranges": ranges,
        "metrics": metrics,
        "feature_cols": FEATURE_COLS,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


# ---- Inference ----
def load_bundle(model_path=MODEL_PATH):
    """Load the saved model bundle (trains one first if the .pkl is missing)."""
    model_path = Path(model_path)
    if not model_path.exists():
        return train_and_save(model_path)
    return joblib.load(model_path)


def predict_score(bundle, user_input: dict) -> float:
    """
    Predict Mental_Health_Score for one student.

    user_input needs: Age, Gender, Country, Academic_Level, Most_Used_Platform,
    Purpose_Of_Use, Avg_Daily_Usage_Hours, Daily_Unlocks, Study_Hours,
    Physical_Activity_Hours, Sleep_Hours_Per_Night, Stress_Level
    """
    row = dict(user_input)
    country = row.pop("Country", row.get("Grouped_country", "Other"))
    row["Grouped_country"] = (
        country if country in bundle["top_countries"] else "Other"
    )
    row["Physical_Activity_Hours"] = max(float(row["Physical_Activity_Hours"]), 0.0)

    X = pd.DataFrame([row])[bundle["feature_cols"]]
    pred = float(bundle["model"].predict(X)[0])
    return float(np.clip(pred, 0, 10))


if __name__ == "__main__":
    b = train_and_save()
    print(f"Model saved to: {MODEL_PATH}")
    print(json.dumps(b["metrics"], indent=2))
