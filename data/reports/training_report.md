# Training and evaluation report

## Selected task

A separately isolated NASA laboratory Li-ion battery-cell RUL experiment. The target is discharge cycles remaining until first positive measured capacity at or below 1.4 Ah. This does not predict future failure in a car, motorcycle, pickup, van, minivan, or fleet.

## Dataset and split

- Prepared rows: 331
- Training rows / cells: 206 / 2 (B0006, B0018)
- Test rows / cells: 125 / 1 (B0005)
- Split: GroupShuffleSplit with random seed 42; cell overlap: 0.
- Features: discharge_cycle_index, capacity_ah, ambient_temperature_c
- Preprocessing: median imputation fitted only on training groups.

## Holdout results

| Model | MAE (cycles) | RMSE (cycles) | R-squared |
|---|---:|---:|---:|
| Random forest | 14.1948 | 15.7995 | 0.8082760613059384 |
| Training-median baseline | 32.2160 | 37.7227 | -0.0929339477726574 |

MAE/RMSE are measured on held-out cell rows. Per-cell results and hashes are stored in `evaluation.json`. Only one cell is held out, so these metrics are highly uncertain and do not establish population generalization; they are an experimental result, not production acceptance.

## Artifacts

- Pipeline: `trained_models\experiments\nasa_battery_cell_rul_1p4ah\pipeline.joblib`
- Artifact SHA-256: `809fbea43a2775c16d4cd096a967d608c7820ad38397cbd9fdc14e149cc3b7c6`
- Dataset SHA-256: `bc0f54bd03a6717b9b190e23279d88bed171755b1945e824eb8ff5dd28e1fb04`
- Split manifest, feature schema, training configuration, dataset provenance, metrics, and model card are stored beside the pipeline under `trained_models/experiments/nasa_battery_cell_rul_1p4ah`.

## Exclusions, safety, and limits

Only the selected cohort with a documented 1.4 Ah end-of-life threshold and cells with a valid measured crossing are used. Nonpositive capacities are treated as invalid measurements, not end-of-life labels. Other battery cohorts are excluded when their cutoff is different/unknown, the cell starts below the threshold, or a valid crossing is absent. There is no vehicle-specific battery pack data, no fleet labels, and no validated future component-failure target. The experiment is not connected to the Django service; production safeguards and the legacy model remain unchanged.

## Reproduction

From `automotive_ai_service/` with the project virtual environment active:

```powershell
python scripts\audit_datasets.py
python scripts\prepare_maintenance_data.py
python scripts\train_maintenance_model.py
python scripts\evaluate_maintenance_model.py
```

For a separate repeat run, use paths that do not already exist, for example:

```powershell
python scripts\prepare_maintenance_data.py --output data\processed\nasa_battery_rul_repeat
python scripts\train_maintenance_model.py --input data\processed\nasa_battery_rul_repeat\discharge_cycles.csv --output trained_models\experiments\nasa_battery_cell_rul_repeat
python scripts\evaluate_maintenance_model.py --experiment-dir trained_models\experiments\nasa_battery_cell_rul_repeat
```

The scripts refuse to overwrite existing prepared data or experiment artifacts.
```
