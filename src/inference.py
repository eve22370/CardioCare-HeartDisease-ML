from __future__ import annotations


import argparse

import json

import logging

import sys

from datetime import datetime

from pathlib import Path


import joblib

import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.preprocessing import clean_dataframe, clinical_range_check


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model.joblib"

METADATA_PATH = ROOT / "models" / "model_metadata.json"

LOG_DIR = ROOT / "logs"

OUTPUT_DIR = ROOT / "outputs"


LOG_DIR.mkdir(exist_ok=True)

OUTPUT_DIR.mkdir(exist_ok=True)


logging.basicConfig(

    filename=str(LOG_DIR / "predictions.log"),

    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",

)


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."

        )

    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:

    if METADATA_PATH.exists():

        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return {"model_name": "unknown"}


# Load model and run batch inference
def predict(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:

    model = load_model()

    metadata = load_metadata()


    raw = pd.read_csv(input_path)

    X = clean_dataframe(raw)


                                         

    for target_col in ["target", "num", "condition", "heart_disease"]:

        if target_col in X.columns:

            X = X.drop(columns=[target_col])


    clinical_range_check(X)


    preds = model.predict(X)

    if hasattr(model, "predict_proba"):

        proba = model.predict_proba(X)

        disease_proba = proba[:, 1]

    else:

        disease_proba = [None] * len(preds)


    out = pd.DataFrame(

        {

            "prediction": preds,

            "probability_heart_disease": disease_proba,

        }

    )


    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)


    # Inference logging
    logging.info(

        "model_version=%s input_shape=%s predictions=%s",

        metadata.get("model_name", "unknown"),

        X.shape,

        preds.tolist(),

    )


    return out


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default=str(ROOT / "data" / "sample_input.csv"))

    parser.add_argument("--output", default=str(OUTPUT_DIR / "predictions.csv"))

    args = parser.parse_args()


    result = predict(args.input, args.output)

    print(result)


if __name__ == "__main__":

    main()
