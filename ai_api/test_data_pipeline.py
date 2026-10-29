import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from django.test import SimpleTestCase

from data_pipeline.cli import (
    build_inventory,
    prepare_enginefaultdb,
    safe_extract_archive,
    train_grouped_classifier,
)
from data_pipeline.status import record_training_status


class DatasetInventoryTests(SimpleTestCase):
    def test_failed_candidate_status_does_not_replace_accepted_registry_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            status_path = root / "training_status.json"
            registry_path = root / "model_registry.json"
            accepted = {
                "status": "evaluated_accepted",
                "eligible_for_inference": True,
                "artifact_path": "models/accepted.joblib",
            }

            with patch("data_pipeline.status.TRAINING_STATUS_PATH", status_path):
                with patch("data_pipeline.status.MODEL_REGISTRY_PATH", registry_path):
                    with patch("data_pipeline.status.SERVICE_ROOT", root):
                        record_training_status("maintenance_model", accepted)
                        record_training_status(
                            "maintenance_model",
                            {
                                "status": "failed",
                                "eligible_for_inference": False,
                                "reason": "candidate training failed",
                            },
                        )

            registry = json.loads(
                registry_path.read_text(encoding="utf-8")
            )
            status = json.loads(
                status_path.read_text(encoding="utf-8")
            )

        self.assertEqual(registry["maintenance_model"], accepted)
        self.assertEqual(
            status["models"]["maintenance_model"]["status"],
            "failed",
        )

    def test_inventory_summarizes_csv_without_exposing_cell_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "raw"
            root.mkdir()
            (root / "records.csv").write_text(
                "vehicle_id,value\nv1,1\nv1,1\n",
                encoding="utf-8",
            )

            inventory = build_inventory(root)

        self.assertEqual(inventory["files_inspected"], 1)
        self.assertEqual(inventory["files"][0]["rows"], 2)
        self.assertEqual(inventory["files"][0]["duplicate_rows"], 1)
        self.assertNotIn("v1", str(inventory))

    def test_inventory_reports_zero_byte_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "empty.cbf").write_bytes(b"")

            inventory = build_inventory(root)

        self.assertEqual(inventory["files"][0]["inspection_status"], "empty")

    def test_inventory_reports_a_corrupt_archive_instead_of_skipping_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "broken.zip").write_bytes(b"not a zip archive")

            inventory = build_inventory(root)

        self.assertEqual(inventory["files"][0]["inspection_status"], "error")
        self.assertIn("BadZipFile", inventory["files"][0]["issue"])


class CapabilitiesEndpointTests(SimpleTestCase):
    def test_capabilities_does_not_claim_unverified_predictive_models(self):
        response = self.client.get("/api/capabilities/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence_backed_predictive_models"], [])
        self.assertEqual(
            payload["maintenance_model"]["eligible_for_inference"],
            False,
        )
        self.assertEqual(payload["training_readiness"]["engine_fault_classification"], "blocked")


class SafeArchiveTests(SimpleTestCase):
    def test_safe_extract_writes_inside_destination(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "safe.zip"
            destination = root / "extracted"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("nested/data.txt", "safe")

            extracted = safe_extract_archive(archive, destination)

            self.assertEqual(len(extracted), 1)
            self.assertEqual(
                (destination / "nested" / "data.txt").read_text(encoding="utf-8"),
                "safe",
            )

    def test_safe_extract_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../outside.txt", "unsafe")

            with self.assertRaisesRegex(ValueError, "Unsafe archive member path"):
                safe_extract_archive(archive, root / "extracted")
            self.assertFalse((root / "extracted").exists())

    def test_safe_extract_rejects_corruption_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "broken.zip"
            archive.write_bytes(b"not a zip archive")

            with self.assertRaises(zipfile.BadZipFile):
                safe_extract_archive(archive, root / "extracted")
            self.assertFalse((root / "extracted").exists())

    def test_safe_extract_does_not_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "safe.zip"
            destination = root / "extracted"
            destination.mkdir()
            existing = destination / "keep.txt"
            existing.write_text("keep", encoding="utf-8")
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("new.txt", "new")

            with self.assertRaisesRegex(ValueError, "destination must be empty"):
                safe_extract_archive(archive, destination)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")


class PreparationAndTrainingGateTests(SimpleTestCase):
    def test_preparation_normalizes_columns_and_removes_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = (
                root
                / "EngineFaultDB-main"
                / "EngineFaultDB-main"
                / "EngineFaultDB_Final.csv"
            )
            source.parent.mkdir(parents=True)
            row = {
                "Fault": 0,
                "MAP": 3.5,
                "TPS": 1.8,
                "Force": 7.4,
                "Power": 5.2,
                "RPM": 1190,
                "Consumption L/H": 3.0,
                "Consumption L/100KM": 11.7,
                "Speed": 24.9,
                "CO": 0.46,
                "HC": 196.0,
                "CO2": 14.3,
                "O2": 1.08,
                "Lambda": 1.04,
                "AFR": 15.3,
            }
            pd.DataFrame([row, row]).to_csv(source, index=False)

            metadata = prepare_enginefaultdb(root, root / "processed")
            prepared = pd.read_csv(
                root / "processed" / "enginefaultdb" / "enginefaultdb_normalized.csv"
            )

        self.assertEqual(metadata["rows_after"], 1)
        self.assertEqual(metadata["exact_duplicates_removed"], 1)
        self.assertIn("fault_class", prepared.columns)
        self.assertNotIn("Fault", prepared.columns)

    def test_training_refuses_missing_group_identifier(self):
        frame = pd.DataFrame({"value": [1, 2, 3, 4], "target": [0, 0, 1, 1]})

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "row-random splitting is intentionally disabled"):
                train_grouped_classifier(
                    frame,
                    "target",
                    "vehicle_id",
                    temporary_directory,
                )

    def test_training_uses_disjoint_groups_and_saves_metrics(self):
        frame = pd.DataFrame(
            {
                "value": list(range(20)),
                "category": ["a", "b"] * 10,
                "target": [0, 1] * 10,
                "trip_id": [f"trip-{index // 2}" for index in range(20)],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = train_grouped_classifier(
                frame,
                "target",
                "trip_id",
                temporary_directory,
            )

            self.assertEqual(result["metrics"]["group_overlap"], 0)
            self.assertTrue((Path(temporary_directory) / "evaluation.json").is_file())
            self.assertTrue(
                (Path(temporary_directory) / "grouped_classifier.joblib").is_file()
            )
            saved_model = joblib.load(
                Path(temporary_directory) / "grouped_classifier.joblib"
            )
            self.assertEqual(
                len(saved_model.predict(frame.loc[:0, ["value", "category"]])),
                1,
            )

    def test_training_rejects_suspected_target_leakage(self):
        frame = pd.DataFrame(
            {
                "value": list(range(20)),
                "repair_result": [0, 1] * 10,
                "target": [0, 1] * 10,
                "trip_id": [f"trip-{index // 2}" for index in range(20)],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "target-leakage"):
                train_grouped_classifier(
                    frame,
                    "target",
                    "trip_id",
                    temporary_directory,
                )
