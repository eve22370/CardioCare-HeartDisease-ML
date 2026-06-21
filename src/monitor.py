from __future__ import annotations


import json

import logging

import sys

from pathlib import Path


import joblib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.stats import ks_2samp

from sklearn.metrics import balanced_accuracy_score


sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.preprocessing import CONTINUOUS_CLINICAL_COLUMNS, clean_dataframe

from src.train import main as train_main


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model.joblib"

METADATA_PATH = ROOT / "models" / "model_metadata.json"

OUTPUT_DIR = ROOT / "outputs"

LOG_DIR = ROOT / "logs"


OUTPUT_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)


logging.basicConfig(

    filename=str(LOG_DIR / "predictions.log"),

    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",

)


def ensure_trained_model() -> None:


    required = [

        MODEL_PATH,

        OUTPUT_DIR / "X_train.csv",

        OUTPUT_DIR / "X_test.csv",

        OUTPUT_DIR / "y_test.csv",

    ]

    if not all(p.exists() for p in required):

        train_main()


def load_monitoring_artifacts():

    ensure_trained_model()

    model = joblib.load(MODEL_PATH)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")

    X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

    y_test = pd.read_csv(OUTPUT_DIR / "y_test.csv").iloc[:, 0]

    return model, metadata, clean_dataframe(X_train), clean_dataframe(X_test), y_test


# Simulate input drift
def create_drifted_copy(X_test: pd.DataFrame) -> pd.DataFrame:


    drifted = X_test.copy()

    rng = np.random.default_rng(42)


    if "chol" in drifted.columns:

        drifted["chol"] = pd.to_numeric(drifted["chol"], errors="coerce") + 30

        drifted["chol"] = drifted["chol"] + rng.normal(0, 20, size=len(drifted))


    if "trestbps" in drifted.columns:

        drifted["trestbps"] = pd.to_numeric(drifted["trestbps"], errors="coerce") + 10


    if "oldpeak" in drifted.columns:

        drifted["oldpeak"] = pd.to_numeric(drifted["oldpeak"], errors="coerce") + 0.5


    return drifted


# KS test for continuous features
def ks_drift_report(X_train: pd.DataFrame, X_drifted: pd.DataFrame) -> pd.DataFrame:


    rows = []

    for col in CONTINUOUS_CLINICAL_COLUMNS:

        if col not in X_train.columns or col not in X_drifted.columns:

            continue


        train_values = pd.to_numeric(X_train[col], errors="coerce").dropna()

        drift_values = pd.to_numeric(X_drifted[col], errors="coerce").dropna()


        if train_values.empty or drift_values.empty:

            continue


        stat, p_value = ks_2samp(train_values, drift_values)

        rows.append(

            {

                "feature": col,

                "ks_statistic": float(stat),

                "p_value": float(p_value),

                "drift_flag": bool(p_value < 0.05),

            }

        )


    report = pd.DataFrame(rows)

    report.to_csv(OUTPUT_DIR / "drift_ks_report.csv", index=False)

    return report


def log_batch_predictions(model, metadata: dict, X: pd.DataFrame, y_true, label: str) -> np.ndarray:


    y_pred = model.predict(X)

    for idx, pred in enumerate(y_pred):

        true_value = y_true.iloc[idx] if hasattr(y_true, "iloc") and idx < len(y_true) else None

        logging.info(

            "label=%s model_version=%s input_shape=%s index=%s prediction=%s true=%s",

            label,

            metadata.get("model_name", "unknown"),

            X.shape,

            idx,

            int(pred),

            None if true_value is None else int(true_value),

        )

    return y_pred


# Synthetic monitoring chart
def make_timeseries_plot(original_score: float, drifted_score: float) -> None:


    dates = pd.date_range("2026-01-01", periods=6, freq="MS")

    values = np.linspace(original_score, drifted_score, len(dates))

    values += np.array([0.0, 0.01, -0.005, -0.01, -0.015, 0.0])


    plt.figure()

    plt.plot(dates, values, marker="o")

    plt.title("Balanced Accuracy Over Time")

    plt.xlabel("Month")

    plt.ylabel("Balanced Accuracy")

    plt.ylim(0, 1)

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "metric_timeseries.png", dpi=150)

    plt.close()


def main() -> None:

    model, metadata, X_train, X_test, y_test = load_monitoring_artifacts()


    X_drifted = create_drifted_copy(X_test)

    X_drifted.to_csv(OUTPUT_DIR / "X_test_drifted.csv", index=False)


    original_pred = log_batch_predictions(model, metadata, X_test, y_test, "original")

    drifted_pred = log_batch_predictions(model, metadata, X_drifted, y_test, "drifted")


    original_bal_acc = balanced_accuracy_score(y_test, original_pred)

    drifted_bal_acc = balanced_accuracy_score(y_test, drifted_pred)


    ks_report = ks_drift_report(X_train, X_drifted)

    performance_report = pd.DataFrame(

        [

            {"dataset": "original_test", "balanced_accuracy": original_bal_acc},

            {"dataset": "drifted_test", "balanced_accuracy": drifted_bal_acc},

        ]

    )

    performance_report.to_csv(OUTPUT_DIR / "drift_performance_report.csv", index=False)


    make_timeseries_plot(original_bal_acc, drifted_bal_acc)


    print("KS drift report")

    print(ks_report)

    print("\nPerformance report")

    print(performance_report)

    print("\nSaved logs to logs/predictions.log")


if __name__ == "__main__":

    main()
