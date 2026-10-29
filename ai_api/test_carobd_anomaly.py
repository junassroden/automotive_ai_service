import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from scripts.carobd_anomaly import (
    FEATURE_COLUMNS,
    build_pipeline,
    chronological_group_split,
    load_pipeline,
    validate_feature_frame,
)


class CarObdAnomalyTests(SimpleTestCase):
    def test_missing_features_are_rejected(self):
        frame = pd.DataFrame({FEATURE_COLUMNS[0]: [1000.0]})

        with self.assertRaisesRegex(ValueError, "Required anomaly features are missing"):
            validate_feature_frame(frame)

    def test_invalid_and_infinite_values_become_missing_for_pipeline_imputation(self):
        row = {name: 1.0 for name in FEATURE_COLUMNS}
        row[FEATURE_COLUMNS[0]] = "not-a-number"
        row[FEATURE_COLUMNS[1]] = np.inf
        valid_row = {name: 2.0 for name in FEATURE_COLUMNS}

        validated = validate_feature_frame(pd.DataFrame([row, valid_row]))

        self.assertTrue(pd.isna(validated.iloc[0][FEATURE_COLUMNS[0]]))
        self.assertTrue(pd.isna(validated.iloc[0][FEATURE_COLUMNS[1]]))

    def test_feature_order_is_explicit(self):
        row = {name: index for index, name in enumerate(reversed(FEATURE_COLUMNS))}
        validated = validate_feature_frame(pd.DataFrame([row]))

        self.assertEqual(list(validated.columns), FEATURE_COLUMNS)

    def test_pipeline_joblib_round_trip_validates_local_artifact_structure(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            rng.normal(size=(40, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        model = build_pipeline(42).fit(frame)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pipeline.joblib"
            import joblib

            joblib.dump(model, path)
            loaded = load_pipeline(path)

        self.assertEqual(list(loaded.named_steps), ["imputer", "isolation_forest"])
        self.assertTrue(np.isfinite(loaded.score_samples(frame)).all())

    def test_fixed_seed_training_is_reproducible(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(
            rng.normal(size=(80, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )

        first = build_pipeline(42).fit(frame)
        second = build_pipeline(42).fit(frame)

        np.testing.assert_array_equal(
            first.score_samples(frame),
            second.score_samples(frame),
        )

    def test_chronological_split_keeps_each_session_tail_in_evaluation(self):
        frame = pd.DataFrame(
            [
                {
                    "source_session": session,
                    "within_session_order": index,
                    "ENGINE_RUN_TINE ()": float(index // 2),
                }
                for session in ("idle1", "drive1")
                for index in range(10)
            ]
        )

        training, evaluation, splits = chronological_group_split(frame, 0.7)

        self.assertEqual(len(training), 12)
        self.assertEqual(len(evaluation), 8)
        self.assertEqual(set(training["source_session"]), {"idle1", "drive1"})
        self.assertEqual(set(evaluation["source_session"]), {"idle1", "drive1"})
        for session in splits:
            self.assertLessEqual(
                splits[session]["train_last_run_time"],
                splits[session]["evaluation_first_run_time"],
            )
            self.assertLess(
                splits[session]["train_last_run_time"],
                splits[session]["cutoff_run_time"],
            )

    def test_csv_importer_normalizes_trailing_delimiter_and_short_rows(self):
        from scripts.carobd_anomaly import _read_source_csv

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "idle1.csv"
            header = list(FEATURE_COLUMNS) + ["ENGINE_RUN_TINE ()"]
            values = [str(index) for index in range(len(header))]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerow(values + [""])
                writer.writerow(values[:4])

            frame, stats, _ = _read_source_csv(path)

        self.assertEqual(frame.shape, (2, len(header)))
        self.assertEqual(stats["trailing_empty_fields_removed"], 1)
        self.assertEqual(stats["short_rows_padded"], 1)
