from __future__ import annotations


import unittest


import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline


from src.preprocessing import build_preprocessor, clinical_range_check


# Required unit tests
class TestCardioCarePipeline(unittest.TestCase):

    def setUp(self):

        self.X = pd.DataFrame(

            {

                "age": [45, 50, 60, 35, 70, 55, 48, 62],

                "trestbps": [120, 130, 140, 110, 150, 135, 125, 145],

                "chol": [210, 240, 280, 180, 300, 260, 220, 290],

                "thalach": [160, 150, 135, 175, 120, 145, 155, 130],

                "oldpeak": [0.0, 1.2, 2.3, 0.1, 3.0, 1.5, 0.8, 2.2],

                "sex": [1, 1, 1, 0, 1, 0, 1, 0],

            }

        )

        self.y = pd.Series([0, 0, 1, 0, 1, 1, 0, 1])


        self.pipeline = Pipeline(

            steps=[

                ("preprocessor", build_preprocessor(self.X)),

                (

                    "model",

                    RandomForestClassifier(

                        n_estimators=50,

                        random_state=42,

                        class_weight="balanced",

                    ),

                ),

            ]

        )

        self.pipeline.fit(self.X, self.y)


    def test_prediction_output_shape_matches_input_shape(self):

        preds = self.pipeline.predict(self.X)

        self.assertEqual(preds.shape[0], self.X.shape[0])


    def test_prediction_probabilities_are_valid(self):

        proba = self.pipeline.predict_proba(self.X)

        self.assertTrue(np.all(proba >= 0))

        self.assertTrue(np.all(proba <= 1))

        row_sums = proba.sum(axis=1)

        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))


    def test_clinical_input_range_validation(self):

        clinical_range_check(self.X)


        invalid = self.X.copy()

        invalid.loc[0, "chol"] = 999

        with self.assertRaises(ValueError):

            clinical_range_check(invalid)


    def test_pipeline_is_deterministic_with_fixed_seed(self):

        pred1 = self.pipeline.predict(self.X)


        pipeline2 = Pipeline(

            steps=[

                ("preprocessor", build_preprocessor(self.X)),

                (

                    "model",

                    RandomForestClassifier(

                        n_estimators=50,

                        random_state=42,

                        class_weight="balanced",

                    ),

                ),

            ]

        )

        pipeline2.fit(self.X, self.y)

        pred2 = pipeline2.predict(self.X)


        self.assertTrue(np.array_equal(pred1, pred2))


if __name__ == "__main__":

    unittest.main()
