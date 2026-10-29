# Raw dataset inventory

- Files inspected: **237**
- Top-level directories: **14**
- Content inspection errors: **0**

Raw files were only read. Archives were checked for unsafe paths and corruption; nothing was extracted or modified. Code and notebooks were not executed. SHA-256 hashes were calculated for byte-level duplicate checks. The JSON companion contains the complete per-file manifest and schemas.

## Collection overview

| Top-level collection | Files | Total size (MiB) |
|---|---:|---:|
| `10.35097-1130` | 8 | 11.01 |
| `5. Battery Data Set` | 48 | 199.96 |
| `Automotive_Diagnostics-main` | 94 | 166.56 |
| `Chinese Vehicle Fleet Fault Association Dataset` | 5 | 23.08 |
| `EngineFaultDB-main` | 3 | 5.13 |
| `Experimental Energy Characterization` | 8 | 1.58 |
| `OBD-Dataset-main` | 10 | 12.16 |
| `[root files]` | 11 | 209.14 |
| `ai4i+2020+predictive+maintenance+dataset` | 1 | 0.50 |
| `levin-openData-master` | 13 | 0.56 |
| `navarra-050926-tanda1` | 5 | 25.59 |
| `navarra-050926-tanda1-raw` | 5 | 1.89 |
| `navarra-050926-tandas2-4` | 5 | 46.55 |
| `navarra-050926-tandas2-4-raw` | 7 | 3.50 |
| `navarra_telemetry-raw` | 14 | 6.21 |

## Byte-identical files

- SHA-256 `050c377e85a2c41165f2e2e0d22bf9575385dd8c835a713c27edb65af6bb25e8`: `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0028.mat`, `5. Battery Data Set/3. BatteryAgingARC_25-44/B0028.mat`
- SHA-256 `38894747a9c79d719e95a2e6c42be18017e29c522b760056b680a65105d1a15f`: `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0025.mat`, `5. Battery Data Set/3. BatteryAgingARC_25-44/B0025.mat`
- SHA-256 `388a449c60d00fea9a976a232813376b5dc64026b026ec25f959a9dfba5ed695`: `navarra-050926-tandas2-4/raw_manifest.json`, `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw_manifest.json`
- SHA-256 `4a62b68a633d79c53a6fd8893e8ea42dcf2b9a8a3e907b1b9861661f04f21517`: `Automotive_Diagnostics-main/Automotive_Diagnostics-main/.ipynb_checkpoints/Dashboard-checkpoint.ipynb`, `Automotive_Diagnostics-main/Automotive_Diagnostics-main/.ipynb_checkpoints/Dataset_Overview-checkpoint.ipynb`, `Automotive_Diagnostics-main/Automotive_Diagnostics-main/.ipynb_checkpoints/Machine_Learning_Application-checkpoint.ipynb`, `Automotive_Diagnostics-main/Automotive_Diagnostics-main/.ipynb_checkpoints/Visualisation-checkpoint.ipynb`
- SHA-256 `57bbfaac92684314304e4b49b7055458c6a82c03f3634e7fee502427a576e079`: `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/README.txt`, `5. Battery Data Set/3. BatteryAgingARC_25-44/README_25_26_27_28.txt`
- SHA-256 `8e8d980a95e81d27f2db40ef86f324af4a82fb8df521e14d9818bb78ff95e08d`: `navarra_telemetry-raw/navarra_telemetry/raw_manifest.json`, `raw_manifest.json`
- SHA-256 `afaa836462a82831a7274c6ee7c4a209f7303de4e31dade935f9edb6d19c1bab`: `ingest_mbl.py`, `navarra-050926-tanda1/ingest_mbl.py`, `navarra-050926-tandas2-4/ingest_mbl.py`
- SHA-256 `ca5eee0b33d8070e3a6c254696444a0c65a12f40587fcfc8db858640b2e6e36c`: `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0026.mat`, `5. Battery Data Set/3. BatteryAgingARC_25-44/B0026.mat`
- SHA-256 `de2b4045ef20a6900885e169f3540664bb637f5f94b41fb4333c31d759237209`: `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0027.mat`, `5. Battery Data Set/3. BatteryAgingARC_25-44/B0027.mat`
- SHA-256 `eb7cb6fa400d19c004b4bb757372ad6e45c83f0237e12adc6be9c45937eca655`: `navarra-050926-tanda1/raw_manifest.json`, `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw_manifest.json`

## Tabular data and structured files

### `10.35097-1130/data/dataset/OBD-II-Dataset.zip`

- Size: 11,538,523 bytes; inspection: `inspected`.
- Archive members: 82; encrypted: 0; unsafe paths: 0; corrupt member: `None`.

### `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat`

- Size: 15,956,874 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0005", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0005", "cycle_count": 616, "operation_counts": {"charge": 170, "discharge": 168, "impedance": 278}}]`.

### `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0006.mat`

- Size: 16,028,159 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0006", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0006", "cycle_count": 616, "operation_counts": {"charge": 170, "discharge": 168, "impedance": 278}}]`.

### `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0007.mat`

- Size: 16,049,460 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0007", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0007", "cycle_count": 616, "operation_counts": {"charge": 170, "discharge": 168, "impedance": 278}}]`.

### `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0018.mat`

- Size: 8,502,218 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0018", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0018", "cycle_count": 319, "operation_counts": {"charge": 134, "impedance": 53, "discharge": 132}}]`.

### `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0025.mat`

- Size: 2,753,821 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0025", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0025", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0026.mat`

- Size: 2,764,789 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0026", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0026", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0027.mat`

- Size: 2,783,781 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0027", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0027", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0028.mat`

- Size: 2,820,196 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0028", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0028", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0025.mat`

- Size: 2,753,821 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0025", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0025", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0026.mat`

- Size: 2,764,789 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0026", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0026", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0027.mat`

- Size: 2,783,781 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0027", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0027", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0028.mat`

- Size: 2,820,196 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0028", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0028", "cycle_count": 80, "operation_counts": {"impedance": 21, "charge": 31, "discharge": 28}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0029.mat`

- Size: 2,879,840 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0029", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0029", "cycle_count": 97, "operation_counts": {"impedance": 17, "discharge": 40, "charge": 40}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0030.mat`

- Size: 2,880,995 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0030", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0030", "cycle_count": 97, "operation_counts": {"impedance": 17, "discharge": 40, "charge": 40}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0031.mat`

- Size: 2,882,510 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0031", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0031", "cycle_count": 97, "operation_counts": {"impedance": 17, "discharge": 40, "charge": 40}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0032.mat`

- Size: 2,918,592 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0032", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0032", "cycle_count": 97, "operation_counts": {"impedance": 17, "discharge": 40, "charge": 40}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0033.mat`

- Size: 11,057,919 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0033", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0033", "cycle_count": 486, "operation_counts": {"discharge": 197, "impedance": 92, "charge": 197}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0034.mat`

- Size: 11,091,182 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0034", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0034", "cycle_count": 486, "operation_counts": {"discharge": 197, "impedance": 92, "charge": 197}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0036.mat`

- Size: 11,404,549 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0036", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0036", "cycle_count": 486, "operation_counts": {"discharge": 197, "impedance": 92, "charge": 197}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0038.mat`

- Size: 3,697,900 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0038", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0038", "cycle_count": 122, "operation_counts": {"discharge": 47, "impedance": 28, "charge": 47}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0039.mat`

- Size: 3,727,683 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0039", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0039", "cycle_count": 122, "operation_counts": {"discharge": 47, "impedance": 28, "charge": 47}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0040.mat`

- Size: 3,749,942 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0040", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0040", "cycle_count": 122, "operation_counts": {"discharge": 47, "impedance": 28, "charge": 47}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0041.mat`

- Size: 3,733,558 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0041", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0041", "cycle_count": 163, "operation_counts": {"discharge": 67, "charge": 67, "impedance": 29}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0042.mat`

- Size: 7,499,364 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0042", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0042", "cycle_count": 275, "operation_counts": {"charge": 113, "impedance": 50, "discharge": 112}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0043.mat`

- Size: 7,483,891 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0043", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0043", "cycle_count": 275, "operation_counts": {"charge": 113, "impedance": 50, "discharge": 112}}]`.

### `5. Battery Data Set/3. BatteryAgingARC_25-44/B0044.mat`

- Size: 7,511,915 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0044", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0044", "cycle_count": 275, "operation_counts": {"charge": 113, "impedance": 50, "discharge": 112}}]`.

### `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0045.mat`

- Size: 3,599,285 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0045", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0045", "cycle_count": 184, "operation_counts": {"discharge": 72, "impedance": 40, "charge": 72}}]`.

### `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0046.mat`

- Size: 3,642,687 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0046", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0046", "cycle_count": 184, "operation_counts": {"discharge": 72, "impedance": 40, "charge": 72}}]`.

### `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0047.mat`

- Size: 3,715,274 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0047", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0047", "cycle_count": 184, "operation_counts": {"discharge": 72, "impedance": 40, "charge": 72}}]`.

### `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0048.mat`

- Size: 3,752,042 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0048", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0048", "cycle_count": 184, "operation_counts": {"discharge": 72, "impedance": 40, "charge": 72}}]`.

### `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0049.mat`

- Size: 2,183,593 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0049", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0049", "cycle_count": 62, "operation_counts": {"discharge": 25, "impedance": 12, "charge": 25}}]`.

### `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0050.mat`

- Size: 2,164,473 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0050", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0050", "cycle_count": 62, "operation_counts": {"discharge": 25, "impedance": 12, "charge": 25}}]`.

### `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0051.mat`

- Size: 2,213,520 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0051", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0051", "cycle_count": 62, "operation_counts": {"discharge": 25, "impedance": 12, "charge": 25}}]`.

### `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0052.mat`

- Size: 2,157,035 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0052", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0052", "cycle_count": 62, "operation_counts": {"discharge": 25, "impedance": 12, "charge": 25}}]`.

### `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0053.mat`

- Size: 3,347,771 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0053", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0053", "cycle_count": 137, "operation_counts": {"discharge": 56, "impedance": 26, "charge": 55}}]`.

### `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0054.mat`

- Size: 7,089,384 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0054", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0054", "cycle_count": 253, "operation_counts": {"discharge": 103, "impedance": 48, "charge": 102}}]`.

### `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0055.mat`

- Size: 7,199,776 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0055", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0055", "cycle_count": 252, "operation_counts": {"discharge": 102, "impedance": 48, "charge": 102}}]`.

### `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0056.mat`

- Size: 7,279,819 bytes; inspection: `inspected`.
- MATLAB variables: `[{"name": "B0056", "shape": [1, 1], "dtype": "struct"}]`.
- Structure summary: `[{"variable": "B0056", "cycle_count": 252, "operation_counts": {"discharge": 102, "impedance": 48, "charge": 102}}]`.

### `ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv`

- Size: 522,048 bytes; inspection: `inspected`.
- Rows: 10,000; columns: 14; duplicate rows: 0.
- Schema: `UDI` (int64), `Product ID` (str), `Type` (str), `Air temperature [K]` (float64), `Process temperature [K]` (float64), `Rotational speed [rpm]` (int64), `Torque [Nm]` (float64), `Tool wear [min]` (int64), `Machine failure` (int64), `TWF` (int64), `HDF` (int64), `PWF` (int64), `OSF` (int64), `RNF` (int64).
- Missing values: `{}`.
- Observed target-like values: `{"Machine failure": {"0": 9661, "1": 339}}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-05_Seat_Leon_RT_S_Stau.csv`

- Size: 3,025,723 bytes; inspection: `inspected`.
- Rows: 46,349; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-05_Seat_Leon_S_KA_Normal.csv`

- Size: 1,994,373 bytes; inspection: `inspected`.
- Rows: 30,817; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_KA_Normal.csv`

- Size: 1,141,943 bytes; inspection: `inspected`.
- Rows: 18,111; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_RT_Normal.csv`

- Size: 3,707,593 bytes; inspection: `inspected`.
- Rows: 57,249; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-07_Seat_Leon_RT_S_Normal.csv`

- Size: 2,457,763 bytes; inspection: `inspected`.
- Rows: 38,080; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-07_Seat_Leon_S_RT_Normal.csv`

- Size: 1,774,743 bytes; inspection: `inspected`.
- Rows: 27,306; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-10_Seat_Leon_KA_KA_Stau.csv`

- Size: 913,203 bytes; inspection: `inspected`.
- Rows: 14,570; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_KA_Stau.csv`

- Size: 1,144,629 bytes; inspection: `inspected`.
- Rows: 17,934; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_S_Normal.csv`

- Size: 2,651,123 bytes; inspection: `inspected`.
- Rows: 41,050; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-11_Seat_Leon_S_RT_Frei.csv`

- Size: 1,806,293 bytes; inspection: `inspected`.
- Rows: 28,114; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-12_Seat_Leon_RT_S_Normal.csv`

- Size: 2,780,003 bytes; inspection: `inspected`.
- Rows: 43,252; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-12_Seat_Leon_S_RT_Normal.csv`

- Size: 2,083,083 bytes; inspection: `inspected`.
- Rows: 32,373; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-13_Seat_Leon_KA_KA_Normal.csv`

- Size: 710,309 bytes; inspection: `inspected`.
- Rows: 11,216; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-13_Seat_Leon_RT_KA_Stau_Messfehler.csv`

- Size: 3,893,466 bytes; inspection: `inspected`.
- Rows: 59,634; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-14_Seat_Leon_KA_KA_Frei.csv`

- Size: 445,777 bytes; inspection: `inspected`.
- Rows: 6,829; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-14_Seat_Leon_KA_RT_Normal.csv`

- Size: 3,482,396 bytes; inspection: `inspected`.
- Rows: 54,037; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-24_Seat_Leon_RT_KA_Normal.csv`

- Size: 3,557,097 bytes; inspection: `inspected`.
- Rows: 54,866; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-26_Seat_Leon_RT_S_Stau.csv`

- Size: 3,037,603 bytes; inspection: `inspected`.
- Rows: 47,350; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-26_Seat_Leon_S_KA_Normal.csv`

- Size: 1,395,291 bytes; inspection: `inspected`.
- Rows: 21,618; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-27_Seat_Leon_KA_KA_2_Normal.csv`

- Size: 1,070,443 bytes; inspection: `inspected`.
- Rows: 16,961; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-27_Seat_Leon_KA_KA_Normal.csv`

- Size: 875,486 bytes; inspection: `inspected`.
- Rows: 13,583; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-28_Seat_Leon_KA_KA_Normal.csv`

- Size: 882,573 bytes; inspection: `inspected`.
- Rows: 13,917; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-28_Seat_Leon_KA_RT_Stau.csv`

- Size: 4,778,525 bytes; inspection: `inspected`.
- Rows: 73,200; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-31_Seat_Leon_KA_KA_Frei.csv`

- Size: 678,123 bytes; inspection: `inspected`.
- Rows: 10,637; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-31_Seat_Leon_RT_KA_Normal.csv`

- Size: 3,705,053 bytes; inspection: `inspected`.
- Rows: 57,331; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-01_Seat_Leon_KA_KA_Frei.csv`

- Size: 919,123 bytes; inspection: `inspected`.
- Rows: 14,510; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-01_Seat_Leon_KA_RT_Normal.csv`

- Size: 3,080,613 bytes; inspection: `inspected`.
- Rows: 47,797; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-02_Seat_Leon_RT_S_Normal.csv`

- Size: 2,478,403 bytes; inspection: `inspected`.
- Rows: 38,524; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-03_Seat_Leon_S_RT_Normal.csv`

- Size: 1,650,933 bytes; inspection: `inspected`.
- Rows: 25,540; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-04_Seat_Leon_S_RT_Normal.csv`

- Size: 2,124,483 bytes; inspection: `inspected`.
- Rows: 32,903; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-05_Seat_Leon_S_CW_Normal.csv`

- Size: 2,300,822 bytes; inspection: `inspected`.
- Rows: 35,469; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-07_Seat_Leon_KA_RT_Frei.csv`

- Size: 1,777,613 bytes; inspection: `inspected`.
- Rows: 27,371; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-07_Seat_Leon_RT_KA_Frei_Vollbremsung.csv`

- Size: 2,571,068 bytes; inspection: `inspected`.
- Rows: 39,110; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-08_Seat_Leon_RT_S_Normal.csv`

- Size: 1,583,563 bytes; inspection: `inspected`.
- Rows: 24,535; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-08_Seat_Leon_S_RT_Normal.csv`

- Size: 1,608,456 bytes; inspection: `inspected`.
- Rows: 24,442; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-09_Seat_Leon_RT_S_Normal.csv`

- Size: 2,527,776 bytes; inspection: `inspected`.
- Rows: 38,662; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-09_Seat_Leon_S_RT_Normal.csv`

- Size: 1,814,723 bytes; inspection: `inspected`.
- Rows: 28,057; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-10_Seat_Leon_RT_S_Normal.csv`

- Size: 1,890,423 bytes; inspection: `inspected`.
- Rows: 29,475; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-10_Seat_Leon_S_RT_Frei.csv`

- Size: 1,298,230 bytes; inspection: `inspected`.
- Rows: 19,645; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [Â°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [Â°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [Â°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00c2\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00c2\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-14_Seat_Leon_RT_S_Normal.csv`

- Size: 1,994,768 bytes; inspection: `inspected`.
- Rows: 30,973; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-14_Seat_Leon_S_RT_Normal.csv`

- Size: 1,639,852 bytes; inspection: `inspected`.
- Rows: 26,102; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-16_Seat_Leon_S_RT_Normal.csv`

- Size: 1,787,513 bytes; inspection: `inspected`.
- Rows: 28,040; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-17_Seat_Leon_BB_RT_Normal_Glatteis.csv`

- Size: 2,608,673 bytes; inspection: `inspected`.
- Rows: 41,611; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-17_Seat_Leon_RT_BB_Frei.csv`

- Size: 1,859,107 bytes; inspection: `inspected`.
- Rows: 29,341; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-18_Seat_Leon_RT_KA_Stau.csv`

- Size: 3,420,280 bytes; inspection: `inspected`.
- Rows: 53,843; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-20_Seat_Leon_KA_KA_Frei.csv`

- Size: 558,019 bytes; inspection: `inspected`.
- Rows: 8,903; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-21_Seat_Leon_BB_RT_Normal.csv`

- Size: 1,532,515 bytes; inspection: `inspected`.
- Rows: 24,129; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-21_Seat_Leon_KA_BB_Normal.csv`

- Size: 2,754,687 bytes; inspection: `inspected`.
- Rows: 43,590; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-21_Seat_Leon_KA_KA_Normal.csv`

- Size: 978,597 bytes; inspection: `inspected`.
- Rows: 15,372; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-22_Seat_Leon_RT_S_Normal.csv`

- Size: 2,355,112 bytes; inspection: `inspected`.
- Rows: 36,251; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_RT_RT_Frei_Beschleunigung.csv`

- Size: 640,928 bytes; inspection: `inspected`.
- Rows: 10,176; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_RT_S_Normal.csv`

- Size: 2,073,216 bytes; inspection: `inspected`.
- Rows: 31,958; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_S_RT_Normal.csv`

- Size: 1,297,500 bytes; inspection: `inspected`.
- Rows: 20,413; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 17, "Accelerator Pedal Position E [%]": 8}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_S_S_Stau.csv`

- Size: 1,395,239 bytes; inspection: `inspected`.
- Rows: 22,521; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-26_Seat_Leon_KA_KA_Frei.csv`

- Size: 701,165 bytes; inspection: `inspected`.
- Rows: 10,980; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-26_Seat_Leon_RT_KA_Normal.csv`

- Size: 3,679,574 bytes; inspection: `inspected`.
- Rows: 56,676; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (float64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (int64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Engine Coolant Temperature [\u00b0C]": 7, "Intake Manifold Absolute Pressure [kPa]": 8, "Engine RPM [RPM]": 9, "Intake Air Temperature [\u00b0C]": 1, "Air Flow Rate from Mass Flow Sensor [g/s]": 2, "Absolute Throttle Position [%]": 3, "Ambient Air Temperature [\u00b0C]": 4, "Accelerator Pedal Position D [%]": 5, "Accelerator Pedal Position E [%]": 6}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-26_Seat_Leon_RT_RT_Frei.csv`

- Size: 762,416 bytes; inspection: `inspected`.
- Rows: 11,534; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-27_Seat_Leon_KA_KA_Normal.csv`

- Size: 1,112,347 bytes; inspection: `inspected`.
- Rows: 17,405; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-27_Seat_Leon_KA_RT_Normal.csv`

- Size: 3,066,467 bytes; inspection: `inspected`.
- Rows: 47,345; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-28_Seat_Leon_RT_S_Normal.csv`

- Size: 2,449,725 bytes; inspection: `inspected`.
- Rows: 37,385; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-28_Seat_Leon_S_RT_Normal.csv`

- Size: 2,198,627 bytes; inspection: `inspected`.
- Rows: 34,380; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-01_Seat_Leon_RT_S_Normal.csv`

- Size: 2,078,237 bytes; inspection: `inspected`.
- Rows: 32,097; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-01_Seat_Leon_S_RT_Normal.csv`

- Size: 3,457,740 bytes; inspection: `inspected`.
- Rows: 54,784; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-02_Seat_Leon_RT_S_Normal.csv`

- Size: 2,220,587 bytes; inspection: `inspected`.
- Rows: 34,648; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-06_Seat_Leon_KA_RT_Normal.csv`

- Size: 2,935,662 bytes; inspection: `inspected`.
- Rows: 46,002; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-07_Seat_Leon_RT_S_Normal.csv`

- Size: 2,158,592 bytes; inspection: `inspected`.
- Rows: 34,082; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-07_Seat_Leon_S_RT_Normal.csv`

- Size: 2,190,266 bytes; inspection: `inspected`.
- Rows: 34,710; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-08_Seat_Leon_RT_S_Normal.csv`

- Size: 2,335,964 bytes; inspection: `inspected`.
- Rows: 37,067; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-19_Seat_Leon_KA_KA_Frei.csv`

- Size: 708,379 bytes; inspection: `inspected`.
- Rows: 11,307; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-19_Seat_Leon_RT_KA_Normal.csv`

- Size: 4,472,762 bytes; inspection: `inspected`.
- Rows: 70,522; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-20_Seat_Leon_KA_KA_Normal.csv`

- Size: 1,196,414 bytes; inspection: `inspected`.
- Rows: 19,070; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-21_Seat_Leon_KA_RT_Normal.csv`

- Size: 3,256,548 bytes; inspection: `inspected`.
- Rows: 51,566; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-22_Seat_Leon_RT_S_Normal.csv`

- Size: 2,354,002 bytes; inspection: `inspected`.
- Rows: 37,082; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-22_Seat_Leon_S_RT_Normal.csv`

- Size: 1,739,824 bytes; inspection: `inspected`.
- Rows: 27,579; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-23_Seat_Leon_RT_S_Normal.csv`

- Size: 2,369,708 bytes; inspection: `inspected`.
- Rows: 37,556; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-23_Seat_Leon_S_RT_Stau.csv`

- Size: 2,434,035 bytes; inspection: `inspected`.
- Rows: 38,509; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-26_Seat_Leon_RT_S_Normal.csv`

- Size: 2,124,791 bytes; inspection: `inspected`.
- Rows: 33,563; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-26_Seat_Leon_S_RT_Stau.csv`

- Size: 1,233,024 bytes; inspection: `inspected`.
- Rows: 19,510; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-29_Seat_Leon_KA_RT_Stau.csv`

- Size: 5,503,423 bytes; inspection: `inspected`.
- Rows: 86,654; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-04-23_Seat_Leon_KA_KA_Frei.csv`

- Size: 629,469 bytes; inspection: `inspected`.
- Rows: 9,701; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-04-23_Seat_Leon_RT_KA_Normal.csv`

- Size: 4,904,757 bytes; inspection: `inspected`.
- Rows: 76,463; columns: 11; duplicate rows: 0.
- Schema: `Time` (str), `Engine Coolant Temperature [°C]` (int64), `Intake Manifold Absolute Pressure [kPa]` (float64), `Engine RPM [RPM]` (float64), `Vehicle Speed Sensor [km/h]` (float64), `Intake Air Temperature [°C]` (float64), `Air Flow Rate from Mass Flow Sensor [g/s]` (float64), `Absolute Throttle Position [%]` (float64), `Ambient Air Temperature [°C]` (float64), `Accelerator Pedal Position D [%]` (float64), `Accelerator Pedal Position E [%]` (float64).
- Missing values: `{"Intake Manifold Absolute Pressure [kPa]": 1, "Engine RPM [RPM]": 2, "Vehicle Speed Sensor [km/h]": 3, "Intake Air Temperature [\u00b0C]": 4, "Air Flow Rate from Mass Flow Sensor [g/s]": 5, "Absolute Throttle Position [%]": 6, "Ambient Air Temperature [\u00b0C]": 7, "Accelerator Pedal Position D [%]": 8, "Accelerator Pedal Position E [%]": 9}`.

### `Chinese Vehicle Fleet Fault Association Dataset/Chinese Vehicle Fleet Fault Association Dataset/fault_events.csv`

- Size: 8,952,818 bytes; inspection: `inspected`.
- Rows: 29,431; columns: 33; duplicate rows: 0.
- Schema: `event_id` (str), `vehicle_id` (str), `model_id` (str), `vehicle_type` (str), `fuel_type` (str), `fault_code` (str), `fault_system` (str), `fault_category` (str), `fault_name` (str), `fault_description_raw` (str), `occurrence_date` (str), `mileage_at_fault` (int64), `severity` (int64), `difficulty` (int64), `common_mileage` (int64), `typical_season` (str), `required_tool` (str), `is_compound` (int64), `related_faults` (str), `vehicle_age_at_fault` (float64), `temperature_celsius` (float64), `pressure_kpa` (float64), `engine_rpm` (float64), `temp_trend_1h` (float64), `temp_trend_24h` (float64), `pressure_trend_1h` (float64), `rpm_variance` (float64), `mileage_ratio` (float64), `age_mileage_interaction` (float64), `seasonal_risk_score` (float64), `system_degradation_index` (float64), `owner_type` (str), `primary_route` (str).
- Missing values: `{"related_faults": 28098, "temperature_celsius": 1447, "pressure_kpa": 1447, "engine_rpm": 1447}`.
- Observed target-like values: `{"vehicle_type": {"\u8f7b\u578b\u8d27\u8f66": 7587, "\u91cd\u578b\u5361\u8f66": 7174, "\u5927\u578b\u5ba2\u8f66": 4319, "\u4e2d\u578b\u5361\u8f66": 3558, "\u5fae\u578b\u5ba2\u8f66": 2032, "\u4e13\u7528\u4f5c\u4e1a\u8f66": 1689, "\u4e58\u7528\u8f66": 1625, "\u7535\u52a8\u8d27\u8f66": 1447}, "fault_code": {"F46": 889, "F37": 881, "F40": 879, "F33": 878, "F44": 878, "F07": 866, "F35": 862, "F36": 858, "F34": 858, "F39": 856, "F48": 850, "F32": 848, "F10": 846, "F15": 845, "F43": 844, "F17": 843, "F41": 841, "F23": 841, "F42": 840, "F24": 839, "F31": 838, "F02": 833, "F47": 832, "F11": 831, "F06": 829, "F29": 829, "F20": 827, "F03": 824, "F38": 821, "F45": 818, "F01": 816, "F50": 808, "F08": 801, "F05": 792, "F49": 790}, "fault_category": {"\u8fdb\u6c14\u7cfb\u7edf": 1747, "\u71c3\u6cb9\u7cfb\u7edf": 1680, "\u542f\u52a8\u7cfb\u7edf": 1660, "\u51b7\u5374\u7cfb\u7edf": 1640, "\u4f20\u52a8\u8f74": 881, "\u51cf\u9707\u7cfb\u7edf": 879, "\u5a31\u4e50\u7cfb\u7edf": 878, "\u538b\u7f29\u673a": 878, "\u79bb\u5408\u5668\u7cfb\u7edf": 866, "\u53d1\u52a8\u673a\u672c\u4f53": 858, "\u7535\u5b50\u5236\u52a8": 856, "\u9897\u7c92\u6355\u96c6\u5668": 850, "\u6db2\u538b\u7cfb\u7edf": 846, "\u52a9\u529b\u7cfb\u7edf": 845, "\u7535\u6e90\u7cfb\u7edf": 843, "\u8f6e\u80ce": 841, "\u5236\u51b7\u7cfb\u7edf": 841, "\u5145\u7535\u7cfb\u7edf": 840, "\u540e\u5904\u7406\u7cfb\u7edf": 839, "\u591a\u7cfb\u7edf": 838, "\u6020\u901f\u7cfb\u7edf": 833, "\u5e9f\u6c14\u518d\u5faa\u73af": 832, "\u5e73\u8861\u7cfb\u7edf": 831, "\u6da6\u6ed1\u7cfb\u7edf": 829, "\u9a7b\u8f66\u7cfb\u7edf": 829, "\u7f51\u7edc\u7cfb\u7edf": 827, "\u71c3\u70e7\u7cfb\u7edf": 824, "\u5dee\u901f\u5668": 821, "\u8f66\u7a97\u7cfb\u7edf": 808, "\u53d8\u901f\u7bb1\u7cfb\u7edf": 801, "\u88ab\u52a8\u5b89\u5168": 790}}`.

### `Chinese Vehicle Fleet Fault Association Dataset/Chinese Vehicle Fleet Fault Association Dataset/knowledge_graph_triples.csv`

- Size: 38,229 bytes; inspection: `inspected`.
- Rows: 565; columns: 5; duplicate rows: 0.
- Schema: `subject` (str), `predicate` (str), `object` (str), `source` (str), `confidence` (float64).
- Missing values: `{}`.

### `Chinese Vehicle Fleet Fault Association Dataset/Chinese Vehicle Fleet Fault Association Dataset/ner_annotations.csv`

- Size: 15,023,433 bytes; inspection: `inspected`.
- Rows: 29,431; columns: 6; duplicate rows: 0.
- Schema: `id` (str), `text` (str), `entities` (str), `n_entities` (int64), `fault_code` (str), `source` (str).
- Missing values: `{}`.
- Observed target-like values: `{"fault_code": {"F46": 889, "F37": 881, "F40": 879, "F33": 878, "F44": 878, "F07": 866, "F35": 862, "F36": 858, "F34": 858, "F39": 856, "F48": 850, "F32": 848, "F10": 846, "F15": 845, "F43": 844, "F17": 843, "F41": 841, "F23": 841, "F42": 840, "F24": 839, "F31": 838, "F02": 833, "F47": 832, "F11": 831, "F06": 829, "F29": 829, "F20": 827, "F03": 824, "F38": 821, "F45": 818, "F01": 816, "F50": 808, "F08": 801, "F05": 792, "F49": 790}}`.

### `Chinese Vehicle Fleet Fault Association Dataset/Chinese Vehicle Fleet Fault Association Dataset/vehicle_fleet.csv`

- Size: 186,673 bytes; inspection: `inspected`.
- Rows: 936; columns: 20; duplicate rows: 0.
- Schema: `vehicle_id` (str), `model_id` (str), `vehicle_name` (str), `manufacturer` (str), `vehicle_type` (str), `fuel_type` (str), `purchase_date` (str), `current_mileage` (int64), `annual_mileage` (float64), `vehicle_status` (str), `owner_type` (str), `primary_scenario` (str), `insurance_type` (str), `insurance_expiry` (str), `insurance_company` (str), `last_maintenance_date` (str), `next_maintenance_mileage` (int64), `purchase_year` (int64), `entry_year` (int64), `exit_year` (float64).
- Missing values: `{"exit_year": 769}`.
- Observed target-like values: `{"vehicle_type": {"\u8f7b\u578b\u8d27\u8f66": 245, "\u91cd\u578b\u5361\u8f66": 223, "\u5927\u578b\u5ba2\u8f66": 131, "\u4e2d\u578b\u5361\u8f66": 117, "\u5fae\u578b\u5ba2\u8f66": 68, "\u4e13\u7528\u4f5c\u4e1a\u8f66": 53, "\u4e58\u7528\u8f66": 52, "\u7535\u52a8\u8d27\u8f66": 47}}`.

### `EngineFaultDB-main/EngineFaultDB-main/EngineFaultDB_Final.csv`

- Size: 5,334,236 bytes; inspection: `inspected`.
- Rows: 55,999; columns: 15; duplicate rows: 1.
- Schema: `Fault` (int64), `MAP` (float64), `TPS` (float64), `Force` (float64), `Power` (float64), `RPM` (float64), `Consumption L/H` (float64), `Consumption L/100KM` (float64), `Speed` (float64), `CO` (float64), `HC` (float64), `CO2` (float64), `O2` (float64), `Lambda` (float64), `AFR` (float64).
- Missing values: `{}`.
- Observed target-like values: `{"Fault": {"0": 16000, "2": 15000, "3": 14001, "1": 10998}}`.

### `Experimental Energy Characterization/NEDC EXTRAURBAN 1.csv`

- Size: 103,146 bytes; inspection: `inspected`.
- Rows: 2,106; columns: 14; duplicate rows: 0.
- Schema: `Temperatura` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str).
- Missing values: `{"Unnamed: 8": 2106, "Unnamed: 9": 2099, "Unnamed: 10": 2105, "Unnamed: 11": 2100, "Unnamed: 12": 2100, "Unnamed: 13": 2105}`.

### `Experimental Energy Characterization/NEDC EXTRAURBAN 2.csv`

- Size: 106,051 bytes; inspection: `inspected`.
- Rows: 2,164; columns: 14; duplicate rows: 0.
- Schema: `Temperatura` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str).
- Missing values: `{"Unnamed: 8": 2164, "Unnamed: 9": 2157, "Unnamed: 10": 2163, "Unnamed: 11": 2158, "Unnamed: 12": 2158, "Unnamed: 13": 2163}`.

### `Experimental Energy Characterization/NEDC URBAN 1.csv`

- Size: 232,711 bytes; inspection: `inspected`.
- Rows: 3,028; columns: 41; duplicate rows: 0.
- Schema: `Temperatura` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str), `Unnamed: 14` (float64), `Unnamed: 15` (float64), `Unnamed: 16` (float64), `Unnamed: 17` (float64), `Unnamed: 18` (float64), `Unnamed: 19` (float64), `Unnamed: 20` (float64), `Unnamed: 21` (float64), `Unnamed: 22` (float64), `Unnamed: 23` (float64), `Unnamed: 24` (float64), `Unnamed: 25` (float64), `Unnamed: 26` (float64), `Unnamed: 27` (float64), `Unnamed: 28` (float64), `Unnamed: 29` (float64), `Unnamed: 30` (float64), `Unnamed: 31` (float64), `Unnamed: 32` (float64), `Unnamed: 33` (float64), `Unnamed: 34` (float64), `Unnamed: 35` (float64), `Unnamed: 36` (float64), `Unnamed: 37` (float64), `Unnamed: 38` (float64), `Unnamed: 39` (float64), `Unnamed: 40` (str).
- Missing values: `{"Unnamed: 8": 3028, "Unnamed: 9": 3020, "Unnamed: 10": 3027, "Unnamed: 11": 3022, "Unnamed: 12": 3022, "Unnamed: 13": 3027, "Unnamed: 14": 3028, "Unnamed: 15": 3028, "Unnamed: 16": 3028, "Unnamed: 17": 3028, "Unnamed: 18": 3028, "Unnamed: 19": 3028, "Unnamed: 20": 3028, "Unnamed: 21": 3028, "Unnamed: 22": 3028, "Unnamed: 23": 3028, "Unnamed: 24": 3028, "Unnamed: 25": 3028, "Unnamed: 26": 3028, "Unnamed: 27": 3028, "Unnamed: 28": 3028, "Unnamed: 29": 3028, "Unnamed: 30": 3028, "Unnamed: 31": 3028, "Unnamed: 32": 3028, "Unnamed: 33": 3028, "Unnamed: 34": 3028, "Unnamed: 35": 3028, "Unnamed: 36": 3028, "Unnamed: 37": 3028, "Unnamed: 38": 3028, "Unnamed: 39": 3028, "Unnamed: 40": 3027}`.

### `Experimental Energy Characterization/NEDC URBAN 2.csv`

- Size: 159,679 bytes; inspection: `inspected`.
- Rows: 3,229; columns: 14; duplicate rows: 0.
- Schema: `Temperatura` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str).
- Missing values: `{"Unnamed: 8": 3229, "Unnamed: 9": 3222, "Unnamed: 10": 3228, "Unnamed: 11": 3223, "Unnamed: 12": 3223, "Unnamed: 13": 3228}`.

### `Experimental Energy Characterization/WLTP EXTRAURBAN 1.csv`

- Size: 230,295 bytes; inspection: `inspected`.
- Rows: 4,584; columns: 14; duplicate rows: 0.
- Schema: `Temperatura` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str).
- Missing values: `{"Unnamed: 8": 4584, "Unnamed: 9": 4577, "Unnamed: 10": 4583, "Unnamed: 11": 4578, "Unnamed: 12": 4578, "Unnamed: 13": 4583}`.

### `Experimental Energy Characterization/WLTP EXTRAURBAN 2.csv`

- Size: 226,671 bytes; inspection: `inspected`.
- Rows: 4,530; columns: 14; duplicate rows: 0.
- Schema: `Temperatura ©` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str).
- Missing values: `{"Unnamed: 8": 4530, "Unnamed: 9": 4523, "Unnamed: 10": 4529, "Unnamed: 11": 4524, "Unnamed: 12": 4524, "Unnamed: 13": 4529}`.

### `Experimental Energy Characterization/WLTP URBAN 1.csv`

- Size: 280,822 bytes; inspection: `inspected`.
- Rows: 5,544; columns: 14; duplicate rows: 0.
- Schema: `Temperatura` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str).
- Missing values: `{"Unnamed: 8": 5544, "Unnamed: 9": 5524, "Unnamed: 10": 5543, "Unnamed: 11": 5539, "Unnamed: 12": 5539, "Unnamed: 13": 5543}`.

### `Experimental Energy Characterization/WLTP URBAN 2.csv`

- Size: 316,972 bytes; inspection: `inspected`.
- Rows: 6,233; columns: 14; duplicate rows: 0.
- Schema: `Temperatura` (str), ` RPM` (int64), ` Ángulo` (str), `Tensión` (str), `I. de corriente` (str), ` Aceleración` (int64), ` Tiempo(ms)` (int64), `Tiempo (s)` (str), `Unnamed: 8` (float64), `Unnamed: 9` (str), `Unnamed: 10` (str), `Unnamed: 11` (str), `Unnamed: 12` (str), `Unnamed: 13` (str).
- Missing values: `{"Unnamed: 8": 6233, "Unnamed: 9": 6213, "Unnamed: 10": 6232, "Unnamed: 11": 6227, "Unnamed: 12": 6227, "Unnamed: 13": 6232}`.

### `main_df_standardized.parquet`

- Size: 103,584,585 bytes; inspection: `inspected`.
- Rows: 1,572,243; columns: 21; duplicate rows: 136.
- Schema: `TempEGR` (double), `HPCInPrs` (double), `EngRPM` (double), `ExPrs` (double), `CACOutTemp` (double), `MAF` (double), `EngTq` (double), `O2concEx` (double), `TempCoEGR` (double), `ExTemp` (double), `InThrPos` (double), `NOxconcEx` (double), `PrsEGR` (double), `HPCOutTemp` (double), `InTemp` (double), `LPTOutTemp` (double), `MAP` (double), `EGR_position_desired` (double), `drive_cycle` (string), `fault_type` (string), `flag` (double).
- Missing values: `{"flag": 1572210}`.
- Observed target-like values: `{"fault_type": {"Healthy": 368565, "MAFLowDrift_80": 133742, "MAFLowDrift_90": 133742, "MAPLowDrift_70": 133742, "MAPLowDrift_80": 133742, "EGVClog_20": 133742, "EGVClog_10": 133742, "MAFLowDrift_70": 133742, "MAPLowDrift_90": 133742, "EGVClog_30": 133742}, "flag": {"nan": 1572210, "1.0": 33}}`.

### `metadata.json`

- Size: 43,323 bytes; inspection: `inspected`.

### `navarra-050926-tanda1/metadata.json`

- Size: 37,430 bytes; inspection: `inspected`.

### `navarra-050926-tanda1/raw_manifest.json`

- Size: 631 bytes; inspection: `inspected`.

### `navarra-050926-tanda1/samples.csv/samples.csv`

- Size: 10,828,813 bytes; inspection: `inspected`.
- Rows: 19,565; columns: 99; duplicate rows: 0.
- Schema: `session` (str), `src_file` (str), `boot_id` (int64), `ts` (float64), `actuators_diag` (int64), `adu_beacon` (int64), `adu_gain_loss_invert` (float64), `adu_gps_heading` (float64), `adu_gps_height` (float64), `adu_gps_height_2d` (float64), `adu_gps_lat_2d` (int64), `adu_gps_lon_2d` (int64), `adu_gps_num_sat` (int64), `adu_gps_speed` (float64), `adu_gps_status` (int64), `adu_gps_valid_2d` (int64), `adu_track_gain_loss` (float64), `adu_track_lap` (float64), `adu_track_lap_time` (float64), `ats` (float64), `aux_lights` (int64), `ax` (float64), `ay` (float64), `az` (float64), `bap_mbar` (float64), `bike_lean_angle` (float64), `bike_vert_acc` (float64), `board_temp` (float64), `brake_front` (float64), `brake_rear` (float64), `buttons` (int64), `cd` (float64), `clutch_slip` (float64), `cut_function` (int64), `cut_level` (int64), `demand` (float64), `dropoff_state` (int64), `dtps_percent_sec` (float64), `ecu_runtime_sec` (float64), `eng_state` (int64), `engine_functions` (int64), `engine_revolutions` (float64), `fuel_consumed_l` (float64), `fuel_pressure_bar` (float64), `gas` (float64), `gear` (int64), `gear_down_state` (int64), `gear_up_state` (int64), `gear_voltage` (float64), `gx` (float64), `gy` (float64), `gz` (float64), `ign_correction_deg` (float64), `imu_ok` (float64), `inj_corr_cyl1` (float64), `inj_corr_cyl2` (float64), `inj_corr_cyl3` (float64), `inj_corr_cyl4` (float64), `inj_correction_lambda` (float64), `lambda` (float64), `lambda_flags` (int64), `lambda_temp` (float64), `lap_info` (int64), `lap_time` (float64), `lines` (float64), `loadcell_n` (float64), `manufacturer_crc` (int64), `map` (float64), `oil_pressure_bar` (float64), `ots` (float64), `outputs` (int64), `rbw_state` (int64), `rpm` (int64), `sensors_diag` (int64), `shift_light` (int64), `speed_front` (float64), `speed_front_hz` (float64), `speed_front_raw_kmh` (float64), `speed_rear` (float64), `speed_rear_hz` (float64), `speed_rear_raw_kmh` (float64), `speed_vehicle` (float64), `stepper_position` (float64), `stepper_state` (int64), `strategy_functions` (int64), `susp_front` (float64), `susp_rear` (float64), `team_crc` (int64), `torque_rear` (float64), `tps` (float64), `transmission_ratio` (float64), `trumpet_pwm_percent` (float64), `vbat` (float64), `virtual_rpm` (float64), `vref1_v` (float64), `vref2_v` (float64), `workmode_id` (int64), `workmode_mask` (int64), `wts` (float64).
- Missing values: `{"ax": 15, "ay": 15, "az": 15, "cd": 15, "gx": 15, "gy": 15, "gz": 15, "imu_ok": 15, "lines": 15}`.

### `navarra-050926-tanda1/updates.csv/updates.csv`

- Size: 15,933,002 bytes; inspection: `inspected`.
- Rows: 314,402; columns: 6; duplicate rows: 0.
- Schema: `boot_id` (int64), `ts` (float64), `src_file` (str), `channel` (str), `value` (float64), `origin` (str).
- Missing values: `{}`.

### `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw_manifest.json`

- Size: 631 bytes; inspection: `inspected`.

### `navarra-050926-tandas2-4/metadata.json`

- Size: 39,141 bytes; inspection: `inspected`.

### `navarra-050926-tandas2-4/raw_manifest.json`

- Size: 908 bytes; inspection: `inspected`.

### `navarra-050926-tandas2-4/samples.csv/samples.csv`

- Size: 18,233,551 bytes; inspection: `inspected`.
- Rows: 31,709; columns: 99; duplicate rows: 0.
- Schema: `session` (str), `src_file` (str), `boot_id` (int64), `ts` (float64), `actuators_diag` (int64), `adu_beacon` (int64), `adu_gain_loss_invert` (float64), `adu_gps_heading` (float64), `adu_gps_height` (float64), `adu_gps_height_2d` (float64), `adu_gps_lat_2d` (int64), `adu_gps_lon_2d` (int64), `adu_gps_num_sat` (int64), `adu_gps_speed` (float64), `adu_gps_status` (int64), `adu_gps_valid_2d` (int64), `adu_track_gain_loss` (float64), `adu_track_lap` (float64), `adu_track_lap_time` (float64), `ats` (float64), `aux_lights` (int64), `ax` (float64), `ay` (float64), `az` (float64), `bap_mbar` (float64), `bike_lean_angle` (float64), `bike_vert_acc` (float64), `board_temp` (float64), `brake_front` (float64), `brake_rear` (float64), `buttons` (int64), `cd` (float64), `clutch_slip` (float64), `cut_function` (int64), `cut_level` (int64), `demand` (float64), `dropoff_state` (int64), `dtps_percent_sec` (float64), `ecu_runtime_sec` (float64), `eng_state` (int64), `engine_functions` (int64), `engine_revolutions` (float64), `fuel_consumed_l` (float64), `fuel_pressure_bar` (float64), `gas` (float64), `gear` (int64), `gear_down_state` (int64), `gear_up_state` (int64), `gear_voltage` (float64), `gx` (float64), `gy` (float64), `gz` (float64), `ign_correction_deg` (float64), `imu_ok` (float64), `inj_corr_cyl1` (float64), `inj_corr_cyl2` (float64), `inj_corr_cyl3` (float64), `inj_corr_cyl4` (float64), `inj_correction_lambda` (float64), `lambda` (float64), `lambda_flags` (int64), `lambda_temp` (float64), `lap_info` (int64), `lap_time` (float64), `lines` (float64), `loadcell_n` (float64), `manufacturer_crc` (int64), `map` (float64), `oil_pressure_bar` (float64), `ots` (float64), `outputs` (int64), `rbw_state` (int64), `rpm` (int64), `sensors_diag` (int64), `shift_light` (int64), `speed_front` (float64), `speed_front_hz` (float64), `speed_front_raw_kmh` (float64), `speed_rear` (float64), `speed_rear_hz` (float64), `speed_rear_raw_kmh` (float64), `speed_vehicle` (float64), `stepper_position` (float64), `stepper_state` (int64), `strategy_functions` (int64), `susp_front` (float64), `susp_rear` (float64), `team_crc` (int64), `torque_rear` (float64), `tps` (float64), `transmission_ratio` (float64), `trumpet_pwm_percent` (float64), `vbat` (float64), `virtual_rpm` (float64), `vref1_v` (float64), `vref2_v` (float64), `workmode_id` (int64), `workmode_mask` (int64), `wts` (float64).
- Missing values: `{"ax": 25, "ay": 25, "az": 25, "cd": 25, "gx": 25, "gy": 25, "gz": 25, "imu_ok": 25, "lines": 25}`.

### `navarra-050926-tandas2-4/updates.csv/updates.csv`

- Size: 30,502,012 bytes; inspection: `inspected`.
- Rows: 601,953; columns: 6; duplicate rows: 0.
- Schema: `boot_id` (int64), `ts` (float64), `src_file` (str), `channel` (str), `value` (float64), `origin` (str).
- Missing values: `{}`.

### `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw_manifest.json`

- Size: 908 bytes; inspection: `inspected`.

### `navarra_telemetry-raw/navarra_telemetry/raw_manifest.json`

- Size: 1,839 bytes; inspection: `inspected`.

### `OBD-Dataset-main/OBD-Dataset-main/MASTER_TEST_X.xlsx`

- Size: 2,531,527 bytes; inspection: `inspected`.
- Sheets: `Test` (33,838 rows, 26 columns, 0 missing cells, 26,280 duplicate rows).
- `Test` target-like counts: `{"Mode": {"0": 12308, "1": 12235, "2": 9295}}`.

### `OBD-Dataset-main/OBD-Dataset-main/MASTER_TRAIN_X.xlsx`

- Size: 8,995,072 bytes; inspection: `inspected`.
- Sheets: `Train` (118,477 rows, 26 columns, 0 missing cells, 90,657 duplicate rows).
- `Train` target-like counts: `{"Mode": {"1": 48922, "2": 37171, "0": 32384}}`.

### `raw_data.csv`

- Size: 1,301,801 bytes; inspection: `inspected`.
- Rows: 19,535; columns: 7; duplicate rows: 0.
- Schema: `Engine rpm` (int64), `Lub oil pressure` (float64), `Fuel pressure` (float64), `Coolant pressure` (float64), `lub oil temp` (float64), `Coolant temp` (float64), `Engine Condition` (int64).
- Missing values: `{}`.
- Observed target-like values: `{"Engine Condition": {"1": 12317, "0": 7218}}`.

### `raw_manifest.json`

- Size: 1,839 bytes; inspection: `inspected`.

### `samples.csv`

- Size: 72,049,527 bytes; inspection: `inspected`.
- Rows: 134,984; columns: 99; duplicate rows: 0.
- Schema: `session` (str), `src_file` (str), `boot_id` (int64), `ts` (float64), `actuators_diag` (int64), `adu_beacon` (int64), `adu_gain_loss_invert` (float64), `adu_gps_heading` (float64), `adu_gps_height` (float64), `adu_gps_height_2d` (float64), `adu_gps_lat_2d` (int64), `adu_gps_lon_2d` (int64), `adu_gps_num_sat` (int64), `adu_gps_speed` (float64), `adu_gps_status` (int64), `adu_gps_valid_2d` (int64), `adu_track_gain_loss` (float64), `adu_track_lap` (float64), `adu_track_lap_time` (float64), `ats` (float64), `aux_lights` (int64), `ax` (int64), `ay` (int64), `az` (int64), `bap_mbar` (float64), `bike_lean_angle` (float64), `bike_vert_acc` (float64), `board_temp` (float64), `brake_front` (float64), `brake_rear` (float64), `buttons` (int64), `cd` (int64), `clutch_slip` (float64), `cut_function` (int64), `cut_level` (int64), `demand` (float64), `dropoff_state` (int64), `dtps_percent_sec` (float64), `ecu_runtime_sec` (float64), `eng_state` (int64), `engine_functions` (int64), `engine_revolutions` (float64), `fuel_consumed_l` (float64), `fuel_pressure_bar` (float64), `gas` (float64), `gear` (int64), `gear_down_state` (int64), `gear_up_state` (int64), `gear_voltage` (float64), `gx` (int64), `gy` (int64), `gz` (int64), `ign_correction_deg` (float64), `imu_ok` (int64), `inj_corr_cyl1` (float64), `inj_corr_cyl2` (float64), `inj_corr_cyl3` (float64), `inj_corr_cyl4` (float64), `inj_correction_lambda` (float64), `lambda` (float64), `lambda_flags` (int64), `lambda_temp` (float64), `lap_info` (int64), `lap_time` (float64), `lines` (int64), `loadcell_n` (float64), `manufacturer_crc` (int64), `map` (float64), `oil_pressure_bar` (float64), `ots` (float64), `outputs` (int64), `rbw_state` (int64), `rpm` (int64), `sensors_diag` (int64), `shift_light` (int64), `speed_front` (float64), `speed_front_hz` (float64), `speed_front_raw_kmh` (float64), `speed_rear` (float64), `speed_rear_hz` (float64), `speed_rear_raw_kmh` (float64), `speed_vehicle` (float64), `stepper_position` (float64), `stepper_state` (int64), `strategy_functions` (int64), `susp_front` (float64), `susp_rear` (float64), `team_crc` (int64), `torque_rear` (float64), `tps` (float64), `transmission_ratio` (float64), `trumpet_pwm_percent` (float64), `vbat` (float64), `virtual_rpm` (float64), `vref1_v` (float64), `vref2_v` (float64), `workmode_id` (int64), `workmode_mask` (int64), `wts` (float64).
- Missing values: `{}`.

### `test.csv`

- Size: 256,533 bytes; inspection: `inspected`.
- Rows: 3,907; columns: 7; duplicate rows: 0.
- Schema: `engine_rpm` (int64), `lub_oil_pressure` (float64), `fuel_pressure` (float64), `coolant_pressure` (float64), `lub_oil_temp` (float64), `coolant_temp` (float64), `engine_condition` (int64).
- Missing values: `{}`.
- Observed target-like values: `{"engine_condition": {"1": 2463, "0": 1444}}`.

### `train.csv`

- Size: 1,025,834 bytes; inspection: `inspected`.
- Rows: 15,628; columns: 7; duplicate rows: 0.
- Schema: `engine_rpm` (int64), `lub_oil_pressure` (float64), `fuel_pressure` (float64), `coolant_pressure` (float64), `lub_oil_temp` (float64), `coolant_temp` (float64), `engine_condition` (int64).
- Missing values: `{}`.
- Observed target-like values: `{"engine_condition": {"1": 9854, "0": 5774}}`.

### `updates.csv`

- Size: 40,897,628 bytes; inspection: `inspected`.
- Rows: 821,492; columns: 6; duplicate rows: 0.
- Schema: `boot_id` (int64), `ts` (float64), `src_file` (str), `channel` (str), `value` (float64), `origin` (str).
- Missing values: `{}`.

### `Vehicle Maintenance- Service Records.csv`

- Size: 97,707 bytes; inspection: `inspected`.
- Rows: 1,139; columns: 24; duplicate rows: 0.
- Schema: `slno` (int64), `vehicle_type` (str), `brand` (str), `model` (str), `engine_type` (str), `make_year` (int64), `region` (str), `mileage_range` (int64), `mileage` (int64), `oil_filter` (int64), `engine_oil` (int64), `washer_plug_drain` (int64), `dust_and_pollen_filter` (int64), `whell_alignment_and_balancing` (int64), `air_clean_filter` (int64), `fuel_filter` (int64), `spark_plug` (int64), `brake_fluid` (int64), `brake_and_clutch_oil` (int64), `transmission_fluid` (int64), `brake_pads` (int64), `clutch` (int64), `coolant` (int64), `cost` (int64).
- Missing values: `{}`.
- Observed target-like values: `{"vehicle_type": {"car": 1139}}`.

## Files requiring format-specific follow-up

- `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-banner.webp` (10,760 bytes): No content reader is configured for .webp.
- `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00004.MBL` (72,977 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00005.MBL` (104,047 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00006.MBL` (86,020 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00010.MBL` (1,715,722 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00000.MBL` (1,223,611 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00001.MBL` (245,673 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00002.MBL` (655,707 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00003.MBL` (487,362 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00004.MBL` (1,031,187 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00005.MBL` (27,069 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00000.MBL` (8,323 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00001.MBL` (25,028 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00002.MBL` (50,883 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00003.MBL` (356,226 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00004.MBL` (17,900 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00005.MBL` (1,182,434 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00006.MBL` (3,236,574 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00007.MBL` (484,854 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00008.MBL` (395 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00009.MBL` (1,634 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00010.MBL` (0 bytes): File is zero bytes.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00011.MBL` (556,191 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `navarra_telemetry-raw/navarra_telemetry/raw/LOG00012.MBL` (586,641 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
- `OBD-Dataset-main/OBD-Dataset-main/FILE.cbf` (1,194,568 bytes): Binary telemetry/diagnostic format was not decoded; inspect the accompanying dataset documentation and decoder requirements.
