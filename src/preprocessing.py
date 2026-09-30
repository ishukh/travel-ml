"""Cleaning, imputation, encoding and feature assembly."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

from src.config import INTERESTS

NUMERIC_FEATURES = ["age", "monthly_income", "budget_per_day", "trip_duration_days",
                    "trips_per_year", "group_size"]
BOOL_FEATURES = ["with_children"]
CATEGORICAL_FEATURES = ["transport_pref", "accommodation_pref", "preferred_season"]


def clean_travelers(df):
    """Fix inconsistent labels, winsorize outliers, impute missing values,
    and derive helper columns."""
    df = df.copy().drop_duplicates(subset="traveler_id")

    # --- inconsistent categorical labels → canonical form
    df["gender"] = (df["gender"].astype(str).str.strip().str.lower()
                    .map({"m": "M", "male": "M", "f": "F", "female": "F"})
                    .fillna("Other"))
    for c in ["transport_pref", "accommodation_pref", "preferred_season"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # --- outliers (income is heavy-tailed) → winsorize at 1st/99th percentile
    lo, hi = df["monthly_income"].quantile([0.01, 0.99])
    df["monthly_income"] = df["monthly_income"].clip(lo, hi)
    df["age"] = df["age"].clip(18, 75)
    df["with_children"] = df["with_children"].astype(int) 

    # --- missing values
    df["age_group"] = pd.cut(df["age"], bins=[17, 24, 34, 44, 54, 200],
                             labels=["18-24", "25-34", "35-44", "45-54", "55+"])
    med = df.groupby("age_group", observed=True)["budget_per_day"].transform("median")
    df["budget_per_day"] = df["budget_per_day"].fillna(med)          # data-driven fallback
    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())
    df["trip_duration_days"] = df["trip_duration_days"].fillna(df["trip_duration_days"].median())
    for c in ["preferred_season", "transport_pref"]:
        df[c] = df[c].fillna(df[c].mode()[0])

    df["trip_duration_days"] = df["trip_duration_days"].clip(2, 30)
    return df.reset_index(drop=True)


def fit_mlb(interests_series):
    """Fit MultiLabelBinarizer with a fixed interest vocabulary (persistable)."""
    mlb = MultiLabelBinarizer(classes=list(INTERESTS))
    mlb.fit(interests_series)
    return mlb


def make_feature_frame(df, mlb):
    """Assemble the model matrix: numeric + boolean + categorical (raw strings,
    encoded inside the pipeline) + multi-hot interests."""
    X = df[NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES].copy()
    X["with_children"] = X["with_children"].astype(int)
    ints = mlb.transform(df["interests"])
    for j, c in enumerate(mlb.classes_):
        X[f"interest_{c}"] = ints[:, j].astype(int)
    return X


def build_preprocessor():
    """Numeric: median-impute + standardize. Categorical: mode-impute + one-hot.
    Interest multi-hot columns pass through (already 0/1)."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer(
        [("num", num, NUMERIC_FEATURES + BOOL_FEATURES),
         ("cat", cat, CATEGORICAL_FEATURES)],
        remainder="passthrough")