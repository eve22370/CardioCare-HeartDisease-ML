from __future__ import annotations


from typing import List, Tuple


import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_CANDIDATES = ["target", "num", "condition", "heart_disease", "HeartDisease"]

CONTINUOUS_CLINICAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:


    out = df.copy()

    out.columns = (

        out.columns.astype(str)

        .str.strip()

        .str.replace(" ", "_", regex=False)

        .str.replace("-", "_", regex=False)

        .str.lower()

    )

    return out


def find_target_column(df: pd.DataFrame) -> str:


    lower_cols = {c.lower(): c for c in df.columns}

    for candidate in TARGET_CANDIDATES:

        if candidate.lower() in lower_cols:

            return lower_cols[candidate.lower()]

    raise ValueError(

        f"Target column not found. Expected one of {TARGET_CANDIDATES}, got {list(df.columns)}"

    )


# Basic cleaning for UCI files
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:


    cleaned = normalize_columns(df)

    cleaned = cleaned.replace(["?", "", " ", "NA", "NaN", "nan", None], np.nan)

    cleaned = cleaned.dropna(axis=1, how="all")

    cleaned = cleaned.drop_duplicates()


    for col in cleaned.columns:

        cleaned[col] = pd.to_numeric(cleaned[col], errors="ignore")


    return cleaned


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:


    target_col = find_target_column(df)

    y_raw = pd.to_numeric(df[target_col], errors="coerce")

    valid_mask = y_raw.notna()

    df = df.loc[valid_mask].copy()

    y_raw = y_raw.loc[valid_mask]

    y = (y_raw.astype(float) > 0).astype(int)


    X = df.drop(columns=[target_col])

    return X, y


def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:


    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return numeric_cols, categorical_cols


# Reusable preprocessing pipeline
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:


    numeric_cols, categorical_cols = get_column_types(X)


    numeric_pipeline = Pipeline(

        steps=[

            ("imputer", SimpleImputer(strategy="median")),

            ("scaler", StandardScaler()),

        ]

    )


    categorical_pipeline = Pipeline(

        steps=[

            ("imputer", SimpleImputer(strategy="most_frequent")),

            ("encoder", OneHotEncoder(handle_unknown="ignore")),

        ]

    )


    return ColumnTransformer(

        transformers=[

            ("numeric", numeric_pipeline, numeric_cols),

            ("categorical", categorical_pipeline, categorical_cols),

        ],

        remainder="drop",

    )


# Clinical input validation
def clinical_range_check(df: pd.DataFrame) -> None:


    checks = {

        "age": (0, 120),

        "trestbps": (0, 300),

        "chol": (0, 600),

        "thalach": (0, 250),

        "oldpeak": (-10, 10),

    }


    lower_cols = {c.lower(): c for c in df.columns}

    for canonical, (lo, hi) in checks.items():

        if canonical in lower_cols:

            col = lower_cols[canonical]

            values = pd.to_numeric(df[col], errors="coerce").dropna()

            if not values.between(lo, hi).all():

                bad = values[~values.between(lo, hi)].head().tolist()

                raise ValueError(

                    f"Column {col} has clinically implausible values. "

                    f"Expected [{lo}, {hi}], examples={bad}"

                )


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:


    cleaned = clean_dataframe(df)

    columns = columns or [c for c in CONTINUOUS_CLINICAL_COLUMNS if c in cleaned.columns]


    rows = []

    for col in columns:

        s = pd.to_numeric(cleaned[col], errors="coerce").dropna()

        if s.empty:

            continue

        q1 = s.quantile(0.25)

        q3 = s.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        n_outliers = int(((s < lower) | (s > upper)).sum())

        rows.append(

            {

                "feature": col,

                "q1": q1,

                "q3": q3,

                "iqr": iqr,

                "lower_bound": lower,

                "upper_bound": upper,

                "n_outliers": n_outliers,

            }

        )


    return pd.DataFrame(rows)
