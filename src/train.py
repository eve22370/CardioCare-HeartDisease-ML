from __future__ import annotations


import json

import os

import sys

import warnings

from pathlib import Path


import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow

import mlflow.sklearn

import numpy as np

import pandas as pd

from scipy import sparse

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    ConfusionMatrixDisplay,

    balanced_accuracy_score,

    confusion_matrix,

    f1_score,

    precision_score,

    recall_score,

)

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier


                                                    

# Script execution path setup
sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.preprocessing import (

    build_preprocessor,

    clean_dataframe,

    clinical_range_check,

    iqr_outlier_summary,

    split_features_target,

)


RANDOM_STATE = 42

TEST_SIZE = 0.2


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

MODEL_DIR = ROOT / "models"

OUTPUT_DIR = ROOT / "outputs"

LOG_DIR = ROOT / "logs"


RAW_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR.mkdir(parents=True, exist_ok=True)


DATA_PATH = RAW_DIR / "heart.csv"


UCI_COLUMNS = [

    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",

    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"

]


PROCESSED_FILES = [

    "processed.cleveland.data",

    "processed.hungarian.data",

    "processed.switzerland.data",

    "processed.va.data",

]


def load_processed_uci_file(path: Path, source_name: str) -> pd.DataFrame:


    df = pd.read_csv(

        path,

        header=None,

        names=UCI_COLUMNS,

        na_values=["?", " ?"],

        skipinitialspace=True,

    )

    df["source"] = source_name

    return df


MODEL_PATH = MODEL_DIR / "model.joblib"

METADATA_PATH = MODEL_DIR / "model_metadata.json"


# Load UCI processed datasets
def load_heart_dataset() -> pd.DataFrame:


    if DATA_PATH.exists():

        return pd.read_csv(DATA_PATH)


    frames = []

    used_files = []

    for filename in PROCESSED_FILES:

        path = RAW_DIR / filename

        if path.exists():

            frames.append(load_processed_uci_file(path, filename))

            used_files.append(filename)


    if frames:

        df = pd.concat(frames, ignore_index=True)

        combined_path = RAW_DIR / "heart_combined.csv"

        df.to_csv(combined_path, index=False)

        (OUTPUT_DIR / "dataset_sources.txt").write_text(

            "\n".join(used_files), encoding="utf-8"

        )

        return df


    try:

        from ucimlrepo import fetch_ucirepo


        heart = fetch_ucirepo(id=45)

        X = heart.data.features.copy()

        y = heart.data.targets.copy()

        if isinstance(y, pd.DataFrame):

            y = y.iloc[:, 0]

        df = X.copy()

        df["target"] = y.values

        df.to_csv(DATA_PATH, index=False)

        return df

    except Exception as exc:

        warnings.warn(f"ucimlrepo download failed: {exc}. Trying CSV fallback.")


    url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"

    try:

        df = pd.read_csv(url)

        df.to_csv(DATA_PATH, index=False)

        return df

    except Exception as exc:

        raise RuntimeError(

            "Could not load dataset. Put UCI processed files in data/raw/ "

            "or provide data/raw/heart.csv with target or num column."

        ) from exc


# Clean data and create binary target
def prepare_data() -> tuple[pd.DataFrame, pd.Series, dict]:


    df_raw = load_heart_dataset()

    df = clean_dataframe(df_raw)


    target_info = {

        "dataset": "UCI Heart Disease processed combined dataset",

        "binary_target_rule": "num > 0 -> 1, num == 0 -> 0",

    }

    X, y = split_features_target(df)


                                                                                    

                                                             

    if "source" in X.columns:

        target_info["source_distribution"] = X["source"].value_counts().to_dict()

        X = X.drop(columns=["source"])


                              

    df.head().to_csv(OUTPUT_DIR / "head.csv", index=False)

    df.describe(include="all").to_csv(OUTPUT_DIR / "describe.csv")

    df.isna().sum().to_csv(OUTPUT_DIR / "missing_values.csv", header=["missing_count"])

    y.value_counts(normalize=True).rename("proportion").to_csv(

        OUTPUT_DIR / "target_distribution.csv"

    )

    iqr_outlier_summary(df).to_csv(OUTPUT_DIR / "iqr_outlier_summary.csv", index=False)


                                                             

                                                                                             

    try:

        clinical_range_check(X)

        target_info["clinical_range_check"] = "passed"

    except ValueError as exc:

        target_info["clinical_range_check"] = f"warning: {exc}"


    return X, y, target_info


# Build comparable model pipelines
def build_model_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:


    preprocessor = build_preprocessor(X)


    selector_estimator = RandomForestClassifier(

        n_estimators=100,

        random_state=RANDOM_STATE,

        class_weight="balanced",

        n_jobs=-1,

    )


    feature_selector = SelectFromModel(

        estimator=selector_estimator,

        threshold="median",

    )


    return {

        "logistic_regression": Pipeline(

            steps=[

                ("preprocessor", preprocessor),

                ("feature_selection", feature_selector),

                (

                    "model",

                    LogisticRegression(

                        max_iter=1000,

                        class_weight="balanced",

                        random_state=RANDOM_STATE,

                    ),

                ),

            ]

        ),

        "svc": Pipeline(

            steps=[

                ("preprocessor", preprocessor),

                ("feature_selection", feature_selector),

                (

                    "model",

                    SVC(

                        probability=True,

                        class_weight="balanced",

                        random_state=RANDOM_STATE,

                    ),

                ),

            ]

        ),

        "random_forest": Pipeline(

            steps=[

                ("preprocessor", preprocessor),

                ("feature_selection", feature_selector),

                (

                    "model",

                    RandomForestClassifier(

                        n_estimators=300,

                        max_depth=None,

                        min_samples_leaf=2,

                        class_weight="balanced",

                        random_state=RANDOM_STATE,

                        n_jobs=-1,

                    ),

                ),

            ]

        ),

        "knn": Pipeline(

            steps=[

                ("preprocessor", preprocessor),

                ("feature_selection", feature_selector),

                ("model", KNeighborsClassifier(n_neighbors=7)),

            ]

        ),

    }


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:


    return {

        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "precision": precision_score(y_true, y_pred, zero_division=0),

        "recall": recall_score(y_true, y_pred, zero_division=0),

        "f1": f1_score(y_true, y_pred, zero_division=0),

    }


def log_confusion_matrix(y_true, y_pred, name: str) -> Path:


    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])

    disp.plot(values_format="d")

    plt.title(f"Confusion Matrix - {name}")

    path = OUTPUT_DIR / f"confusion_matrix_{name}.png"

    plt.tight_layout()

    plt.savefig(path, dpi=150)

    plt.close()

    return path


def get_selected_feature_names(pipeline: Pipeline, X: pd.DataFrame) -> list[str]:


    try:

        preprocessor = pipeline.named_steps["preprocessor"]

        selector = pipeline.named_steps["feature_selection"]

        names = preprocessor.get_feature_names_out()

        mask = selector.get_support()

        return [str(n) for n, keep in zip(names, mask) if keep]

    except Exception:

        return []


# Hyperparameter tuning for recall priority
def tune_random_forest(X_train, y_train, X_template: pd.DataFrame) -> GridSearchCV:


    preprocessor = build_preprocessor(X_template)

    selector = SelectFromModel(

        RandomForestClassifier(

            n_estimators=100,

            random_state=RANDOM_STATE,

            class_weight="balanced",

            n_jobs=-1,

        ),

        threshold="median",

    )

    pipeline = Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("feature_selection", selector),

            (

                "model",

                RandomForestClassifier(

                    class_weight="balanced",

                    random_state=RANDOM_STATE,

                    n_jobs=-1,

                ),

            ),

        ]

    )


    param_grid = {

        "model__n_estimators": [200, 400],

        "model__max_depth": [None, 4, 8],

        "model__min_samples_leaf": [1, 2, 4],

    }


    grid = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        scoring="recall",                               

        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),

        n_jobs=-1,

        refit=True,

    )

    grid.fit(X_train, y_train)

    return grid


def ensure_sample_input(X_test: pd.DataFrame) -> None:


    sample = X_test.head(5).copy()

    DATA_DIR.mkdir(exist_ok=True)

    sample.to_csv(DATA_DIR / "sample_input.csv", index=False)


def main() -> None:

    np.random.seed(RANDOM_STATE)

    mlflow.set_experiment("CardioCare")


    X, y, data_notes = prepare_data()


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y,

    )


                                                

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)

    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)

    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    ensure_sample_input(X_test)


    pipelines = build_model_pipelines(X_train)


    scoring = {

        "balanced_accuracy": "balanced_accuracy",

        "precision": "precision",

        "recall": "recall",

        "f1": "f1",

    }


    results = []

    best_name = None

    best_model = None

    best_recall = -1.0

    best_balanced_accuracy = -1.0


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


    for name, pipeline in pipelines.items():

        with mlflow.start_run(run_name=name):

            mlflow.set_tag("model_family", name)

            mlflow.log_param("random_state", RANDOM_STATE)

            mlflow.log_param("test_size", TEST_SIZE)


            cv_scores = cross_validate(

                pipeline,

                X_train,

                y_train,

                cv=cv,

                scoring=scoring,

                n_jobs=-1,

                error_score="raise",

            )


            for metric in scoring:

                mlflow.log_metric(f"cv_{metric}_mean", float(np.mean(cv_scores[f"test_{metric}"])))

                mlflow.log_metric(f"cv_{metric}_std", float(np.std(cv_scores[f"test_{metric}"])))


            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            test_metrics = compute_metrics(y_test, y_pred)


            for key, value in test_metrics.items():

                mlflow.log_metric(f"test_{key}", float(value))


            cm = confusion_matrix(y_test, y_pred)

            mlflow.log_dict({"confusion_matrix": cm.tolist()}, f"confusion_matrix_{name}.json")

            cm_path = log_confusion_matrix(y_test, y_pred, name)

            mlflow.log_artifact(str(cm_path))


            selected_features = get_selected_feature_names(pipeline, X_train)

            feature_path = OUTPUT_DIR / f"selected_features_{name}.txt"

            feature_path.write_text("\n".join(selected_features), encoding="utf-8")

            mlflow.log_artifact(str(feature_path))


            mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{name}")


            row = {

                "model": name,

                **{f"cv_{m}_mean": float(np.mean(cv_scores[f"test_{m}"])) for m in scoring},

                **{f"test_{k}": float(v) for k, v in test_metrics.items()},

            }

            results.append(row)


                                                                                  

            if (

                test_metrics["recall"] > best_recall

                or (

                    np.isclose(test_metrics["recall"], best_recall)

                    and test_metrics["balanced_accuracy"] > best_balanced_accuracy

                )

            ):

                best_name = name

                best_model = pipeline

                best_recall = test_metrics["recall"]

                best_balanced_accuracy = test_metrics["balanced_accuracy"]


                                                      

    with mlflow.start_run(run_name="random_forest_grid_search"):

        mlflow.set_tag("model_family", "random_forest")

        mlflow.set_tag("run_type", "grid_search")


        grid = tune_random_forest(X_train, y_train, X_train)

        tuned_model = grid.best_estimator_

        y_pred = tuned_model.predict(X_test)

        tuned_metrics = compute_metrics(y_test, y_pred)


        mlflow.log_params(grid.best_params_)

        mlflow.log_metric("best_cv_recall", float(grid.best_score_))

        for key, value in tuned_metrics.items():

            mlflow.log_metric(f"test_{key}", float(value))


        cm_path = log_confusion_matrix(y_test, y_pred, "random_forest_tuned")

        mlflow.log_artifact(str(cm_path))

        mlflow.sklearn.log_model(tuned_model, artifact_path="model_random_forest_tuned")


        results.append(

            {

                "model": "random_forest_tuned",

                "cv_recall_mean": float(grid.best_score_),

                **{f"test_{k}": float(v) for k, v in tuned_metrics.items()},

            }

        )


        if (

            tuned_metrics["recall"] > best_recall

            or (

                np.isclose(tuned_metrics["recall"], best_recall)

                and tuned_metrics["balanced_accuracy"] > best_balanced_accuracy

            )

        ):

            best_name = "random_forest_tuned"

            best_model = tuned_model

            best_recall = tuned_metrics["recall"]

            best_balanced_accuracy = tuned_metrics["balanced_accuracy"]


    comparison = pd.DataFrame(results).sort_values(

        by=["test_recall", "test_balanced_accuracy"],

        ascending=False,

    )

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)


    selected_features = get_selected_feature_names(best_model, X_train)

    (OUTPUT_DIR / "selected_features.txt").write_text(

        "\n".join(selected_features), encoding="utf-8"

    )


    # Save final model
    joblib.dump(best_model, MODEL_PATH)

    metadata = {

        "model_name": best_name,

        "random_state": RANDOM_STATE,

        "test_size": TEST_SIZE,

        "selection_policy": "maximize recall first to reduce false negatives; tie-break by balanced accuracy",

        "best_recall": float(best_recall),

        "best_balanced_accuracy": float(best_balanced_accuracy),

        "selected_features_count": len(selected_features),

        "data_notes": data_notes,

    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


                            

    final_pred = best_model.predict(X_test)

    final_cm_path = log_confusion_matrix(y_test, final_pred, "final_model")

    print(f"Training complete. Best model: {best_name}")

    print(comparison)

    print(f"Saved model to: {MODEL_PATH}")

    print(f"Saved metadata to: {METADATA_PATH}")

    print(f"Saved final confusion matrix to: {final_cm_path}")


if __name__ == "__main__":

    main()
