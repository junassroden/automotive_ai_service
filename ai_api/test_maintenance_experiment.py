import tempfile
from pathlib import Path

import pandas as pd
from django.test import SimpleTestCase

from scripts.maintenance_experiment import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_cell_records,
    read_prepared_data,
)


class BatteryRulPreparationTests(SimpleTestCase):
    def test_target_ends_at_first_valid_documented_threshold_crossing(self):
        records = [
            {"capacity_ah": 1.8, "ambient_temperature_c": 24},
            {"capacity_ah": 1.6, "ambient_temperature_c": 24},
            {"capacity_ah": 0, "ambient_temperature_c": 24},
            {"capacity_ah": 1.39, "ambient_temperature_c": 24},
            {"capacity_ah": 1.2, "ambient_temperature_c": 24},
        ]

        rows, invalid_count, reason = prepare_cell_records("B0005", records, 1.4)

        self.assertIsNone(reason)
        self.assertEqual(invalid_count, 1)
        self.assertEqual([row[TARGET_COLUMN] for row in rows], [3, 2, 0])
        self.assertEqual([row["discharge_cycle_index"] for row in rows], [0, 1, 3])
        self.assertEqual(set(rows[0]).intersection(FEATURE_COLUMNS), set(FEATURE_COLUMNS))

    def test_does_not_invent_threshold_crossing_when_missing(self):
        rows, _, reason = prepare_cell_records(
            "B0006",
            [
                {"capacity_ah": 1.9, "ambient_temperature_c": 24},
                {"capacity_ah": 1.6, "ambient_temperature_c": 24},
            ],
        )

        self.assertEqual(rows, [])
        self.assertIn("no observed crossing", reason)

    def test_rejects_cells_that_begin_at_or_below_threshold(self):
        rows, _, reason = prepare_cell_records(
            "B0007",
            [
                {"capacity_ah": 1.4, "ambient_temperature_c": 24},
                {"capacity_ah": 1.2, "ambient_temperature_c": 24},
            ],
        )

        self.assertEqual(rows, [])
        self.assertIn("initial valid capacity", reason)

    def test_prepared_data_rejects_missing_model_features(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prepared.csv"
            pd.DataFrame(
                {
                    "cell_id": ["B0005"],
                    "remaining_discharge_cycles": [2],
                    "capacity_ah": [1.5],
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                read_prepared_data(path)

    def test_feature_order_is_explicit_and_independent_of_csv_column_order(self):
        frame = pd.DataFrame(
            {
                "cell_id": ["B0005"],
                "remaining_discharge_cycles": [2],
                "ambient_temperature_c": [24],
                "capacity_ah": [1.5],
                "discharge_cycle_index": [10],
            }
        )

        model_input = frame[FEATURE_COLUMNS]

        self.assertEqual(list(model_input.columns), FEATURE_COLUMNS)
        self.assertEqual(model_input.iloc[0].tolist(), [10, 1.5, 24])
