import unittest

import pandas as pd

from pcos_utils import SYMPTOM_COLUMNS, prepare_prediction_features, split_features_target


class PCOSUtilsTests(unittest.TestCase):
    def test_split_features_target_is_multiclass(self):
        frame = pd.DataFrame({
            "PCOS (Y/N)": [0, 1, 1],
            "Age (yrs)": [20, 25, 30],
            "Sl. No": [1, 2, 3],
            **{
                name: [0, 0, 1] for name in SYMPTOM_COLUMNS
            },
        })
        features, target = split_features_target(frame)
        self.assertNotIn("Sl. No", features.columns)
        self.assertEqual(target.tolist(), [0, 1, 2])

    def test_rejects_missing_target(self):
        with self.assertRaises(ValueError):
            split_features_target(pd.DataFrame({"Age (yrs)": [20]}))

    def test_aligns_prediction_features(self):
        frame = pd.DataFrame({"A": [1], "Extra": [2]})
        features, missing, extra = prepare_prediction_features(frame, ["A", "B"])
        self.assertEqual(features.columns.tolist(), ["A", "B"])
        self.assertEqual(missing, ["B"])
        self.assertEqual(extra, ["Extra"])


if __name__ == "__main__":
    unittest.main()
