import json
import os
import tempfile
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
TRAINING_STATUS_PATH = SERVICE_ROOT / "data" / "training" / "training_status.json"
MODEL_REGISTRY_PATH = SERVICE_ROOT / "models" / "model_registry.json"


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def record_training_status(model_name, entry):
    status = _read_json(TRAINING_STATUS_PATH, {"schema_version": 1, "models": {}})
    if not isinstance(status, dict) or not isinstance(status.get("models"), dict):
        models = {}
    else:
        models = status["models"]
    models[model_name] = entry
    status = {"schema_version": 1, "models": models}
    _write_json_atomic(TRAINING_STATUS_PATH, status)

    registry = _read_json(MODEL_REGISTRY_PATH, {"schema_version": 1})
    if not isinstance(registry, dict):
        registry = {"schema_version": 1}
    registry["schema_version"] = 1
    if model_name == "maintenance_model":
        current_model = registry.get("maintenance_model")
        current_is_accepted = (
            isinstance(current_model, dict)
            and current_model.get("status") == "evaluated_accepted"
            and current_model.get("eligible_for_inference") is True
        )
        new_is_accepted = (
            entry.get("status") == "evaluated_accepted"
            and entry.get("eligible_for_inference") is True
        )
        if not current_is_accepted or new_is_accepted:
            registry["maintenance_model"] = entry
        _write_json_atomic(MODEL_REGISTRY_PATH, registry)
        _write_json_atomic(
            SERVICE_ROOT / "models" / "evidence_backed_status.json",
            registry.get("maintenance_model", entry),
        )
    else:
        _write_json_atomic(MODEL_REGISTRY_PATH, registry)


def read_training_status():
    return _read_json(TRAINING_STATUS_PATH, {"schema_version": 1, "models": {}})
