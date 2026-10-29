import json
import hashlib
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, Client, override_settings
from django.conf import settings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .chatbot_service import (
    answer_chat,
    answer_database_context,
    answer_rental_question,
)
from .prediction_service import (
    MaintenancePredictionsUnavailableError,
    predict_vehicle,
)
from .automotive_knowledge import (
    format_knowledge_answer,
    retrieve_knowledge,
)


class HealthEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.old_key = settings.AI_SERVICE_API_KEY
        settings.AI_SERVICE_API_KEY = ""

    def tearDown(self):
        settings.AI_SERVICE_API_KEY = self.old_key

    def test_health_endpoint_returns_service_name(self):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["service"], "custom-automotive-ai")
        self.assertFalse(payload["external_ai_provider"])
        self.assertFalse(payload["maintenance_predictions_enabled"])
        self.assertFalse(payload["maintenance_model_evidence_verified"])
        self.assertEqual(payload["maintenance_model_status"], "blocked")


class PredictionEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.old_key = settings.AI_SERVICE_API_KEY
        settings.AI_SERVICE_API_KEY = ""

    def tearDown(self):
        settings.AI_SERVICE_API_KEY = self.old_key

    def test_prediction_endpoint_reports_unavailable_by_default(self):
        response = self.client.post(
            "/api/predict/",
            data=json.dumps({"vehicle": {"id": 1}}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertEqual(response.json()["code"], "model_not_eligible")

    def test_fleet_endpoint_reports_unavailable_by_default(self):
        response = self.client.post(
            "/api/fleet/analyze/",
            data=json.dumps({"vehicles": [{"id": 1}]}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertEqual(response.json()["code"], "model_not_eligible")

    def test_prediction_service_blocks_direct_unverified_model_use(self):
        with self.assertRaises(MaintenancePredictionsUnavailableError):
            predict_vehicle({"vehicle_type": "car"})

    def test_registered_accepted_compatible_model_can_predict(self):
        classifier = Pipeline([("classifier", LogisticRegression())])
        classifier.fit(
            pd.DataFrame({"mileage": [0, 1, 2, 3]}),
            [0, 0, 1, 1],
        )
        artifact = {
            "models": {"brake_pads": classifier},
            "feature_columns": ["mileage"],
            "target_columns": ["brake_pads"],
            "thresholds": {"brake_pads": 0.5},
            "constant_targets": {},
        }

        with tempfile.TemporaryDirectory(dir=settings.BASE_DIR) as directory:
            root = Path(directory)
            artifact_path = root / "maintenance.joblib"
            registry_path = root / "registry.json"
            evaluation_path = root / "evaluation.json"
            import joblib

            joblib.dump(artifact, artifact_path)
            evaluation_path.write_text(
                json.dumps({
                    "status": "evaluated_accepted",
                    "acceptance_criteria": {
                        "acceptance_criteria_passed": True,
                    },
                    "metrics": {
                        "balanced_accuracy": 0.8,
                        "baseline_balanced_accuracy": 0.5,
                        "macro_f1": 0.8,
                        "baseline_macro_f1": 0.5,
                    },
                    "group_split": {
                        "train_group_count": 4,
                        "test_group_count": 2,
                        "group_overlap": 0,
                    },
                    "probability_validation": {
                        "status": "calibrated",
                        "method": "test fixture calibration",
                        "brier_score": 0.18,
                        "validation_records": 10,
                        "independent_groups": 2,
                    },
                }),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps({
                    "maintenance_model": {
                        "status": "evaluated_accepted",
                        "eligible_for_inference": True,
                        "artifact_path": str(artifact_path),
                        "artifact_sha256": hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest(),
                        "evaluation_report_path": str(evaluation_path),
                        "evaluation_report_sha256": hashlib.sha256(
                            evaluation_path.read_bytes()
                        ).hexdigest(),
                        "feature_columns": ["mileage"],
                        "evaluation": {"acceptance_criteria_passed": True},
                    },
                }),
                encoding="utf-8",
            )

            with override_settings(
                AI_MAINTENANCE_MODEL_REGISTRY_PATH=registry_path
            ):
                result = predict_vehicle({"mileage": 2})

        self.assertEqual(result["status"], "success")
        self.assertIn("brake_pads", [item["component"] for item in result["all_categories"]])

    def test_registered_artifact_with_invalid_structure_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=settings.BASE_DIR) as directory:
            root = Path(directory)
            artifact_path = root / "maintenance.joblib"
            registry_path = root / "registry.json"
            import joblib

            joblib.dump({"not_a_maintenance_model": True}, artifact_path)
            report_path = root / "evaluation.json"
            report_path.write_text(
                json.dumps({
                    "status": "evaluated_accepted",
                    "acceptance_criteria": {
                        "acceptance_criteria_passed": True,
                    },
                    "metrics": {
                        "balanced_accuracy": 0.8,
                        "baseline_balanced_accuracy": 0.5,
                        "macro_f1": 0.8,
                        "baseline_macro_f1": 0.5,
                    },
                    "group_split": {
                        "train_group_count": 4,
                        "test_group_count": 2,
                        "group_overlap": 0,
                    },
                    "probability_validation": {
                        "status": "calibrated",
                        "method": "test fixture calibration",
                        "brier_score": 0.18,
                        "validation_records": 10,
                        "independent_groups": 2,
                    },
                }),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps({
                    "maintenance_model": {
                        "status": "evaluated_accepted",
                        "eligible_for_inference": True,
                        "artifact_path": str(artifact_path),
                        "artifact_sha256": hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest(),
                        "evaluation_report_path": str(report_path),
                        "evaluation_report_sha256": hashlib.sha256(
                            report_path.read_bytes()
                        ).hexdigest(),
                        "feature_columns": ["mileage"],
                        "evaluation": {"acceptance_criteria_passed": True},
                    },
                }),
                encoding="utf-8",
            )

            with override_settings(
                AI_MAINTENANCE_MODEL_REGISTRY_PATH=registry_path
            ):
                with self.assertRaisesRegex(
                    MaintenancePredictionsUnavailableError,
                    "unsupported structure",
                ):
                    predict_vehicle({"mileage": 2})


class ChatEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.old_key = settings.AI_SERVICE_API_KEY
        settings.AI_SERVICE_API_KEY = ""

    def tearDown(self):
        settings.AI_SERVICE_API_KEY = self.old_key

    @patch("ai_api.views.answer_chat")
    def test_chat_endpoint_forwards_rental_context(self, mock_answer):
        mock_answer.return_value = {
            "status": "success",
            "intent": "rental_pricing",
            "confidence": 0.9,
            "answer": "Daily rate: PHP 1,500.",
            "data": None,
        }
        rental_context = {
            "currency": "PHP",
            "vehicles": [{"make": "Toyota", "model": "Vios"}],
        }
        response = self.client.post(
            "/api/chat/",
            data=json.dumps({
                "message": "How much is a Toyota Vios?",
                "rental_context": rental_context,
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Daily rate: PHP 1,500.")
        mock_answer.assert_called_once_with(
            message="How much is a Toyota Vios?",
            vehicle=None,
            vehicles=None,
            rental_context=rental_context,
            diagnostic_context=None,
        )

    def test_chat_endpoint_rejects_non_object_rental_context(self):
        response = self.client.post(
            "/api/chat/",
            data=json.dumps({"message": "What cars are available?", "rental_context": []}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    @patch("ai_api.views.answer_chat")
    def test_chat_endpoint_passes_admin_diagnostic_context(self, mock_answer):
        mock_answer.return_value = {
            "status": "success",
            "intent": "brakes",
            "confidence": 0.8,
            "answer": "General brake guidance.",
            "data": None,
        }
        diagnostic_context = {
            "selected_component": "brakes",
            "vehicle": {"make": "Toyota", "model": "Vios"},
            "maintenance_history": [],
        }
        response = self.client.post(
            "/api/chat/",
            data=json.dumps({
                "message": "Why are the brakes making noise?",
                "rental_context": {"diagnostic_context": diagnostic_context},
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        mock_answer.assert_called_once_with(
            message="Why are the brakes making noise?",
            vehicle=None,
            vehicles=None,
            rental_context={"diagnostic_context": diagnostic_context},
            diagnostic_context=diagnostic_context,
        )

    def test_chat_endpoint_rejects_non_object_diagnostic_context(self):
        response = self.client.post(
            "/api/chat/",
            data=json.dumps({
                "message": "Why are the brakes making noise?",
                "rental_context": {"diagnostic_context": []},
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


class RentalChatContextTests(TestCase):
    def test_rental_price_uses_supplied_vehicle_records(self):
        answer = answer_rental_question(
            "rental_pricing",
            "How much does a Toyota Vios cost?",
            {
                "currency": "PHP",
                "vehicles": [
                    {"make": "Toyota", "model": "Vios", "daily_rental_price": 1500}
                ],
            },
        )
        self.assertIn("Toyota Vios: PHP 1,500 per day", answer)

    def test_public_availability_does_not_expose_internal_status(self):
        answer = answer_rental_question(
            "car_availability",
            "What cars are available?",
            {
                "vehicle_status_authorized": False,
                "vehicles": [{"make": "Toyota", "model": "Vios", "status": "available"}],
            },
        )
        self.assertIn("not include availability information", answer)
        self.assertNotIn("Toyota Vios", answer)

    def test_authorized_availability_uses_current_status_and_warns_about_dates(self):
        answer = answer_rental_question(
            "car_availability",
            "What cars are available?",
            {
                "vehicle_status_authorized": True,
                "vehicles": [
                    {"make": "Toyota", "model": "Vios", "status": "available"},
                    {"make": "Honda", "model": "City", "status": "maintenance"},
                ],
            },
        )
        self.assertIn("Toyota Vios", answer)
        self.assertNotIn("Honda City", answer)
        self.assertIn("specific dates", answer)

    def test_date_checked_availability_uses_authoritative_boolean(self):
        answer = answer_rental_question(
            "car_availability",
            "Is anything available?",
            {
                "vehicle_status_authorized": True,
                "availability_period": {
                    "status": "checked",
                    "start_date": "2026-07-10",
                    "end_date": "2026-07-13",
                },
                "vehicles": [
                    {"make": "Toyota", "model": "Vios", "is_available": True},
                    {"make": "Honda", "model": "City", "is_available": False},
                ],
            },
        )
        self.assertIn("Toyota Vios", answer)
        self.assertNotIn("Honda City", answer)
        self.assertIn("return date exclusive", answer)
        self.assertNotIn("provide pickup and return dates", answer)

    def test_availability_asks_for_missing_period_details(self):
        answer = answer_rental_question(
            "car_availability",
            "Is anything available?",
            {
                "availability_period": {
                    "status": "needs_details",
                    "missing": "rental duration or return date",
                },
                "vehicles": [],
            },
        )
        self.assertIn("rental duration or return date", answer)

    def test_price_quote_calculates_only_a_daily_rate_estimate(self):
        answer = answer_rental_question(
            "rental_pricing",
            "How much is a Toyota Vios for 3 days?",
            {
                "currency": "PHP",
                "rental_duration_days": 3,
                "vehicles": [
                    {
                        "make": "Toyota",
                        "model": "Vios",
                        "daily_rental_price": 1500,
                        "security_deposit": 5000,
                    }
                ],
            },
        )
        self.assertIn("PHP 1,500 per day", answer)
        self.assertIn("PHP 4,500.00 base-rate estimate for 3 days", answer)
        self.assertIn("listed security deposit PHP 5,000.00", answer)
        self.assertIn("confirm fees", answer)

    def test_price_query_for_unknown_vehicle_does_not_fall_back_to_other_cars(self):
        answer = answer_rental_question(
            "rental_pricing",
            "How much is a Lamborghini?",
            {
                "currency": "PHP",
                "vehicles": [
                    {"make": "Toyota", "model": "Vios", "daily_rental_price": 1500}
                ],
            },
        )
        self.assertIn("could not match that vehicle", answer)
        self.assertNotIn("Toyota Vios", answer)


class DatabaseChatContextTests(TestCase):
    def test_safety_critical_knowledge_precedes_inventory_context(self):
        class FakePipeline:
            classes_ = ["car_availability"]

            def predict_proba(self, messages):
                return np.array([[1.0]])

        knowledge = {"document": {"safety_critical": True}}
        with patch("ai_api.chatbot_service.load_chatbot", return_value=(FakePipeline(), {})):
            with patch("ai_api.chatbot_service.retrieve_knowledge", return_value=knowledge):
                with patch(
                    "ai_api.chatbot_service._knowledge_response",
                    return_value={"status": "success", "answer": "Stop driving and seek inspection."},
                ) as mock_safety:
                    result = answer_chat(
                        "Is the car available if its brakes failed?",
                        rental_context={
                            "database_context": {
                                "kind": "vehicle_inventory",
                                "purpose": "availability",
                            },
                            "vehicles": [],
                        },
                    )

        self.assertIn("Stop driving", result["answer"])
        mock_safety.assert_called_once()

    def test_database_answer_precedes_low_confidence_general_classifier_fallback(self):
        class FakePipeline:
            classes_ = ["greeting", "unknown"]

            def predict_proba(self, messages):
                return np.array([[0.51, 0.49]])

        with patch("ai_api.chatbot_service.load_chatbot", return_value=(FakePipeline(), {})):
            with patch("ai_api.chatbot_service.retrieve_knowledge", return_value=None):
                result = answer_chat(
                    "What is my booking status?",
                    rental_context={
                        "currency": "PHP",
                        "database_context": {
                            "kind": "customer_bookings",
                            "booking_data": {
                                "total_count": 0,
                                "reservations": [],
                            },
                        },
                    },
                )

        self.assertIn("no reservation records", result["answer"])
        self.assertEqual(result["intent"], "greeting")
        self.assertEqual(result["data"]["data_source"], "reservations belonging to the authenticated account")

    def test_customer_booking_answer_contains_only_supplied_reservation_facts(self):
        answer, data = answer_database_context(
            {
                "kind": "customer_bookings",
                "booking_data": {
                    "total_count": 1,
                    "reservations": [
                        {
                            "id": 42,
                            "make": "Toyota",
                            "model": "Vios",
                            "start_date": "2026-07-10",
                            "end_date": "2026-07-12",
                            "rental_days": 2,
                            "status": "confirmed",
                            "total_amount": 3000,
                        }
                    ],
                },
            },
            "What is my booking status?",
            {"currency": "PHP"},
        )
        self.assertIn("Reservation #42", answer)
        self.assertIn("confirmed", answer)
        self.assertIn("PHP 3,000.00 stored total", answer)
        self.assertEqual(data["reservation_ids"], [42])

    def test_admin_maintenance_answer_separates_legacy_logs_from_verified_evidence(self):
        answer, data = answer_database_context(
            {
                "kind": "admin_maintenance",
                "records": {
                    "legacy_service_records": [
                        {
                            "maintenanceID": 3,
                            "vehicleID": 7,
                            "make": "Honda",
                            "model": "City",
                            "maintenanceType": "Oil change",
                            "maintenanceDate": "2026-06-01",
                            "maintenanceStatus": "Completed",
                            "description": "Routine service",
                        }
                    ],
                    "verified_evidence_records": [
                        {
                            "id": 9,
                            "make": "Honda",
                            "model": "City",
                            "record_type": "mechanic_confirmed_fault",
                            "recorded_at": "2026-06-02 10:00:00",
                            "confirmed_cause": "Worn belt",
                        }
                    ],
                    "verified_evidence_table_available": True,
                },
            },
            "Show maintenance history",
            {},
        )
        self.assertIn("Legacy service log #3", answer)
        self.assertIn("not a verified fault label", answer)
        self.assertIn("Verified evidence record #9", answer)
        self.assertIn("confirmed cause: Worn belt", answer)
        self.assertEqual(data["legacy_record_ids"], [3])
        self.assertEqual(data["verified_record_ids"], [9])


class AutomotiveKnowledgeTests(TestCase):
    def test_overheating_retrieval_assembles_relevant_passages_and_limits(self):
        question = "What could cause an engine to overheat?"
        result = retrieve_knowledge(question)
        self.assertEqual(result["document"]["id"], "engine_overheating")
        self.assertGreaterEqual(len(result["passages"]), 4)

        answer = format_knowledge_answer(result, query=question)
        for expected in (
            "Coolant loss",
            "cooling fan",
            "thermostat",
            "coolant-pump",
            "Workshop diagnostic approach:",
            "Never remove a radiator cap",
            "Vehicle-specific differences:",
            "Source limitation:",
        ):
            self.assertIn(expected.casefold(), answer.casefold())

    def test_knowledge_scores_and_source_references_are_not_overstated(self):
        class FixedPipeline:
            classes_ = ["engine_cooling"]

            def predict_proba(self, _messages):
                return np.array([[0.1846]])

        with patch("ai_api.chatbot_service.load_chatbot") as load_model:
            load_model.return_value = (FixedPipeline(), {})
            result = answer_chat("What could cause an engine to overheat?")

        self.assertEqual(result["confidence"], 0.1846)
        self.assertIn("not calibrated", result["data"]["intent_confidence_interpretation"])
        self.assertIn("not a probability", result["data"]["retrieval_score_interpretation"])
        self.assertEqual(result["data"]["source_count"], 0)
        self.assertEqual(result["data"]["source_references_fetched"], 0)
        self.assertIn("Source limitation:", result["answer"])

    def test_high_voltage_questions_only_give_safe_general_limits(self):
        result = retrieve_knowledge("Can I open the high-voltage battery pack?")
        self.assertEqual(result["document"]["id"], "hybrid_ev_high_voltage_safety")
        answer = format_knowledge_answer(result)
        self.assertIn("Do not touch", answer)
        self.assertIn("does not provide an isolation procedure", answer)

    def test_out_of_domain_and_unsupported_specific_questions_abstain(self):
        for question in (
            "How do I bake sourdough bread?",
            "What is the weather forecast for Manila tomorrow?",
            "Explain the proof of Fermat's Last Theorem.",
            "What does my manufacturer's service bulletin say for this VIN?",
        ):
            self.assertIsNone(retrieve_knowledge(question), question)

    def test_novel_automotive_questions_retrieve_expected_topics(self):
        cases = [
            ("Why does my car shake when I accelerate?", "clutch_drivetrain_vibration"),
            ("What happens if I keep driving with worn brake pads?", "brake_wear"),
            ("How can I tell if my alternator is failing?", "battery_charging_starting"),
            ("Why does the engine temperature rise when I am stuck in traffic?", "engine_overheating"),
            ("When should I change my engine oil?", "oil_change_schedule"),
            ("What could cause the transmission to slip?", "transmission_slip"),
            ("Which parts of this rental vehicle should be inspected first?", "vehicle_health_data_limits"),
            ("Why did the predictive maintenance model flag this vehicle?", "maintenance_model_explanation"),
            ("What does a high maintenance risk mean?", "maintenance_model_explanation"),
            ("Which vehicles in the fleet need attention first?", "fleet_maintenance_prioritization"),
            ("What information do you need to assess my vehicle?", "vehicle_health_data_limits"),
            ("Can you explain the difference between a symptom and a confirmed mechanical fault?", "symptom_vs_fault"),
            ("What maintenance should be completed before a rental vehicle is released?", "pre_rental_inspection"),
            ("I smell gasoline near the car", "fuel_ignition_air"),
            ("The red oil pressure light came on", "oil_pressure_leak"),
            ("The ABS warning light is on", "brake_warning_abs"),
        ]
        correct = 0
        for question, expected_id in cases:
            result = retrieve_knowledge(question)
            self.assertIsNotNone(result, question)
            self.assertEqual(result["document"]["id"], expected_id, question)
            correct += 1

        self.assertEqual(correct / len(cases), 1.0)

    def test_unrelated_question_does_not_retrieve_automotive_advice(self):
        self.assertIsNone(retrieve_knowledge("How do I make pancakes?"))

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_vehicle_inspection_question_uses_knowledge_without_model_inputs(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["vehicle_prediction"]

            def predict_proba(self, _messages):
                return np.array([[1.0]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        result = answer_chat(
            "Which parts of this rental vehicle should be inspected first?"
        )
        self.assertEqual(
            result["data"]["knowledge_id"],
            "vehicle_health_data_limits",
        )
        self.assertIn("A low model score cannot establish", result["answer"])

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_fleet_question_without_records_does_not_claim_analysis(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["fleet_analysis"]

            def predict_proba(self, _messages):
                return np.array([[1.0]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        result = answer_chat(
            "Which vehicles in the fleet need attention first?"
        )
        self.assertEqual(
            result["data"]["fleet_analysis_status"],
            "not_run",
        )
        self.assertIn("no authorized maintenance-model records", result["answer"])

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_vehicle_prediction_chat_reports_unavailable_by_default(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["vehicle_prediction"]

            def predict_proba(self, _messages):
                return np.array([[1.0]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        result = answer_chat(
            "Predict maintenance for this vehicle",
            vehicle={
                "vehicle_type": "car",
                "brand": "Toyota",
                "model": "Vios",
                "engine_type": "gasoline",
                "make_year": 2021,
                "region": "Mimaropa",
                "mileage_range": "50000-60000",
                "mileage": 55000,
            },
        )

        self.assertEqual(result["data"]["prediction_status"], "unavailable")
        self.assertIn("unavailable", result["answer"])
        self.assertNotIn("score", result["data"])

    @patch("ai_api.chatbot_service.analyze_fleet")
    @patch("ai_api.chatbot_service.load_chatbot")
    def test_fleet_chat_with_records_reports_unavailable_by_default(
        self,
        mock_load_chatbot,
        mock_analyze_fleet,
    ):
        class FixedPipeline:
            classes_ = ["fleet_analysis"]

            def predict_proba(self, _messages):
                return np.array([[1.0]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        result = answer_chat(
            "Which vehicles in the fleet need attention first?",
            vehicles=[{"id": 1}],
        )

        self.assertEqual(
            result["data"]["fleet_analysis_status"],
            "unavailable",
        )
        self.assertIn("not eligible", result["answer"])
        mock_analyze_fleet.assert_not_called()

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_low_confidence_unrelated_query_is_not_answered_as_rental(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["booking_reservation"]

            def predict_proba(self, _messages):
                return np.array([[0.379]])

        mock_load_chatbot.return_value = (
            FixedPipeline(),
            {"booking_reservation": {"answer": "Book a vehicle."}},
        )
        result = answer_chat("How do I make pancakes?")
        self.assertEqual(result["intent"], "unknown")
        self.assertNotIn("reservation", result["answer"].lower())

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_unrelated_capability_query_is_not_claimed_as_supported(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["capabilities"]

            def predict_proba(self, _messages):
                return np.array([[0.9]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        result = answer_chat("Can you help me with my homework?")
        self.assertEqual(result["intent"], "unknown")

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_out_of_domain_query_is_rejected_even_with_high_class_score(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["cooling"]

            def predict_proba(self, _messages):
                return np.array([[0.99]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        result = answer_chat("What is the weather today?")
        self.assertEqual(result["intent"], "unknown")

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_supported_capability_question_still_gets_an_answer(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["capabilities"]

            def predict_proba(self, _messages):
                return np.array([[0.9]])

        mock_load_chatbot.return_value = (
            FixedPipeline(),
            {"capabilities": {"answer": "I can help with rental and automotive questions."}},
        )
        result = answer_chat("What can you do?")
        self.assertEqual(result["intent"], "capabilities")

    def test_component_function_question_returns_component_roles(self):
        result = retrieve_knowledge("What does an alternator do?")
        answer = format_knowledge_answer(
            result["document"],
            query="What does an alternator do?",
        )
        self.assertIn("Component roles:", answer)
        self.assertIn("The alternator supplies electrical power", answer)
        self.assertNotIn("Possible causes:", answer)

    @patch("ai_api.chatbot_service.load_chatbot")
    def test_safety_critical_advice_precedes_rental_feature_routing(
        self,
        mock_load_chatbot,
    ):
        class FixedPipeline:
            classes_ = ["vehicle_features"]

            def predict_proba(self, _messages):
                return np.array([[0.9]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        result = answer_chat(
            "I smell gasoline near the car",
            rental_context={"vehicles": []},
        )
        self.assertEqual(result["data"]["knowledge_id"], "fuel_ignition_air")
        self.assertIn("fire risk", result["answer"])
        self.assertNotIn("vehicle listing", result["answer"])

    def test_knowledge_answer_includes_caution_and_references(self):
        result = retrieve_knowledge("What happens if I keep driving with worn brake pads?")
        answer = format_knowledge_answer(result["document"])
        self.assertIn("Urgency:", answer)
        self.assertIn("References:", answer)
        self.assertIn("not a confirmed diagnosis", answer)

    @patch("ai_api.chatbot_service.load_maintenance_artifact")
    @patch("ai_api.chatbot_service.load_chatbot")
    def test_diagnostic_prediction_reports_missing_model_features(
        self,
        mock_load_chatbot,
        mock_load_artifact,
    ):
        class FixedPipeline:
            classes_ = ["vehicle_prediction"]

            def predict_proba(self, _messages):
                return np.array([[1.0]])

        mock_load_chatbot.return_value = (FixedPipeline(), {})
        mock_load_artifact.return_value = {
            "feature_columns": [
                "vehicle_type", "brand", "model", "engine_type",
                "make_year", "region", "mileage_range", "mileage",
            ]
        }
        result = answer_chat(
            "Predict maintenance for this vehicle",
            diagnostic_context={
                "vehicle": {
                    "make": "Toyota",
                    "model": "Vios",
                    "year": 2022,
                    "current_mileage_km": 42000,
                }
            },
        )
        self.assertEqual(result["data"]["prediction_status"], "not_run")
        self.assertIn("vehicle_type", result["answer"])
        self.assertIn("region", result["answer"])
        self.assertIn("mileage_range", result["answer"])

    @patch("ai_api.chatbot_service.predict_vehicle")
    @patch("ai_api.chatbot_service.load_maintenance_artifact")
    @patch("ai_api.chatbot_service.load_chatbot")
    def test_complete_vehicle_features_are_sent_to_maintenance_model(
        self,
        mock_load_chatbot,
        mock_load_artifact,
        mock_predict,
    ):
        class FixedPipeline:
            classes_ = ["vehicle_prediction"]

            def predict_proba(self, _messages):
                return np.array([[1.0]])

        feature_columns = [
            "vehicle_type", "brand", "model", "engine_type",
            "make_year", "region", "mileage_range", "mileage",
        ]
        vehicle = {
            "vehicle_type": "car",
            "brand": "Toyota",
            "model": "Vios",
            "engine_type": "gasoline",
            "make_year": 2021,
            "region": "Mimaropa",
            "mileage_range": "50000-60000",
            "mileage": 55000,
        }
        mock_load_chatbot.return_value = (FixedPipeline(), {})
        mock_load_artifact.return_value = {"feature_columns": feature_columns}
        mock_predict.return_value = {
            "recommendations": [],
            "summary": {"trained_categories": 12},
        }
        result = answer_chat(
            "Predict maintenance for this vehicle",
            vehicle=vehicle,
        )
        mock_predict.assert_called_once_with(vehicle)
        self.assertIn("did not flag any trained", result["answer"])
