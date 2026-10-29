import json
import sys
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from scripts.carobd_anomaly import (
    EXPERIMENT_ROOT,
    EXPERIMENTS_ROOT,
    prepare_dataset,
    resolve_within,
    write_json,
)


def prepare(experiment_dir=EXPERIMENT_ROOT):
    experiment_dir = resolve_within(experiment_dir, EXPERIMENTS_ROOT)
    if experiment_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing experiment: {experiment_dir}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    prepared_path = experiment_dir / "prepared_carobd.csv"
    _, provenance = prepare_dataset(output_path=prepared_path)
    write_json(experiment_dir / "dataset_provenance.json", provenance)
    print(
        f"Prepared {provenance['preprocessed_record_count']} unique rows from "
        f"{provenance['file_count']} CSV sessions."
    )
    print(json.dumps(provenance["statistics"], indent=2))
    return provenance


if __name__ == "__main__":
    prepare()

