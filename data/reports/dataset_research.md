# Dataset research and source verification

Research checked on 2026-10-29. Facts below distinguish external source records from
claims found only in the local download. Source pages were retrieved where possible;
the accompanying local documents remain the authority for the exact bytes present in
`data/raw/`.

## Identified sources

| Collection | Verified information and local evidence | Source, license, and use decision |
|---|---|---|
| Automotive OBD-II Dataset (KIT) | The local RADAR metadata identifies Marc Weber, DOI `10.35097/1130`, ten OBD-II signals, KIWI 3 / OBD Auto Doctor acquisition, production year 2018, and CC BY 4.0. The archive has 82 ZIP members, including 81 CSVs; all 81 extracted CSVs are byte-identical to their archive members. They contain 2,693,824 samples with one vehicle model in filenames (Seat Leon), timestamps and operating signals, but no confirmed failure/repair labels. The local package identifies normal, free-flow and traffic conditions, not component faults. | [KIT RADAR dataset record](https://radar.kit.edu/radar/en/dataset/bCtGxdTklQlfQcAq) · [DOI](https://doi.org/10.35097/1130) · [local descriptive metadata](../raw/10.35097-1130/data/descriptive-md/dataset.desc_md.xml). The retrieved RADAR page exposed general service terms, not the complete item metadata; the detailed field and license claims are from the supplied XML. No supervised failure prediction is supported. |
| EngineFaultDB | The local CSV has 55,999 rows, 14 numeric measurements and a four-valued `Fault` field; there are no missing cells and one exact duplicate row. The repository documentation describes controlled laboratory operation of a C14NE spark-ignition engine, simulated operating/fault states, and no per-run or vehicle identifier. Its README gives counts for numeric labels 0–3 but does not map them to specific fault mechanisms. The local copy has a GPL-3.0 license file. | [Source repository](https://github.com/Leo-Thomas/EngineFaultDB) · [paper DOI](https://doi.org/10.1109/ACCESS.2023.3331316) · [local README](../raw/EngineFaultDB-main/EngineFaultDB-main/README.md). GitHub repository documentation was retrieved; the IEEE article could not be retrieved by the available page fetch. Prepared for analysis, but withheld from training because a group-disjoint holdout and verified fault-name mapping are unavailable. |
| NASA Li-ion Battery Aging | NASA's data page describes 18650 cells tested on a prognostics testbed under charge, discharge and impedance profiles, with cycle-level capacity and a 30% capacity-fade end-of-life criterion. The local collection contains 38 MAT files and 7,885 cycle records when duplicate copies are counted. The four B0025–B0028 files are byte-identical copies across two folders, leaving 34 distinct cell files. These are laboratory cells, not installed vehicle batteries. | [NASA dataset page](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) · [local battery README](../raw/5.%20Battery%20Data%20Set/1.%20BatteryAgingARC-FY08Q4/README.txt). The NASA page verifies the experimental scope and supports battery-cell aging/RUL research only. No vehicle-battery model was trained. The local bundle has no separate license statement; confirm reuse terms before redistribution. |
| AI4I 2020 Predictive Maintenance | The UCI record identifies 10,000 synthetic industrial time-series rows and 14 fields. The local file has no missing values or exact duplicates and contains 339 positive `Machine failure` records. It describes machine-tool failure modes, not road vehicles. | [UCI dataset record](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) · [DOI](https://doi.org/10.24432/C5HS5C). UCI identifies the data as synthetic and CC BY 4.0. Excluded from vehicle training. |
| LEVIN vehicle telematics | The repository describes an approximately four-month sample from about 30 four-wheelers, with OBD at 1 Hz and accelerometer data at 25 Hz. The local folder contains documentation and notebooks but no raw CSV/SQLite/ZIP measurement export. Notebooks were inventoried, not executed. | [LEVIN repository](https://github.com/YunSolutions/levin-openData) · [local README](../raw/levin-openData-master/README.md). The README states the dataset is CC BY-NC-SA; the checked-in repository `LICENSE` is MIT for repository code and must not be mistaken for the data license. No local measurements were available to prepare or train. |

## Local claims that remain unverified or inconsistent

| Collection | Local contents and problems | Decision |
|---|---|---|
| `OBD-Dataset-main` (diesel OBD) | The README claims eight diesel vehicles, manually recorded DTCs and CC BY 4.0, but no separate license file or source download URL is included. Two workbooks contain 118,477 training and 33,838 test rows, 26 columns, no missing cells, and respectively 90,657 and 26,280 exact duplicate rows. The workbooks have no vehicle/trip/run ID. `FILE.cbf` is proprietary/opaque and was not decoded. The README cites a conference dated 2026-11-25/26, which is in the future relative to this audit date. A GitHub search for the quoted dataset title and cited paper title returned no repository result. | Source, label collection process and split-by-vehicle status are not verified. Do not treat the README's claimed labels or license as proven; no model training. |
| Chinese Vehicle Fleet Fault Association Dataset (CVFFAD) | Local README says events and NER data are generated and states CC BY-NC-SA 4.0, but the README's DOI is the placeholder `10.5281/zenodo.XXXXXXX`. The README refers to a separate `LICENSE`, `.zenodo.json` and data-availability statement; those files are absent from this five-file folder. Four CSVs contain 29,431 fault events, 936 fleet rows, 29,431 NER rows and 565 knowledge-graph triples. Fault-event data has 32,439 missing cells (mostly `related_faults`) and no exact duplicate rows. Eight vehicle types are present in generated records; they do not establish real fleet coverage. | Provenance, license and field validity cannot be independently confirmed. Do not use for commercial training or claim vehicle-type coverage. |
| Navarra motorcycle telemetry | Local notes describe a Kawasaki Ninja 400 track bike and provide decoded samples, updates, metadata and custom MBL decoder source. The root sample exports contain 134,984, 19,565 and 31,709 rows across three sessions. The raw folders contain 23 MBL files, one zero-byte `LOG00010.MBL`; 95 channels are listed in local metadata. Examples include RPM, water temperature, oil pressure, battery voltage, front brake pressure and front/rear suspension travel. There are no confirmed failures, repairs, maintenance outcomes or RUL labels. The cited Zenodo record describes a separate 2026 campaign with native TXT logs, rather than verifying these local MBL files; the linked article page returned HTTP 403. | Local files support sensor-measurement analysis only. Origin/license claims in the local README are not independently verified; raw MBL was not decoded by executing its bundled code. No prediction model or failure inference. |
| Experimental Energy Characterization | Eight semicolon-delimited CSV files have 31,418 rows total, NEDC/WLTP names, 14–41 columns and many empty cells. Files contain Spanish-labelled telemetry such as temperature, RPM, voltage, current, acceleration and time. No README, source URL, vehicle identity, labels or license were found. | Provenance and reuse terms unknown. Keep separate; no training or fleet claims. |
| Unnamed root `main_df_standardized.parquet` | 1,572,243 rows and 21 columns, including 15 `drive_cycle` values and 10 `fault_type` values. There are 136 exact duplicate rows and 1,572,210 missing values in `flag`. No README, origin, units, vehicle/run identity or license accompanies it. | Labels look potentially useful but their semantics and source cannot be established. Excluded from training. |
| Unnamed root `raw_data.csv`, `train.csv`, `test.csv` | `raw_data.csv` has 19,535 rows, six measurements and a two-value `Engine Condition` label. The 15,628-row train and 3,907-row test files are an exact partition of the raw table (verified by row comparison); no missing cells or exact duplicates. The tables have no units, vehicle/trip ID, source or license. | Keep the provided split documented, but do not treat it as automotive fleet evidence or train a vehicle model without provenance and group keys. |
| `Vehicle Maintenance- Service Records.csv` | 1,139 rows, 24 columns, all rows marked `car`, two makes and four models. There are 14 service-related binary fields and one cost field; `oil_filter` and `engine_oil` are always 1. There are no missing values or exact duplicate rows, but no source, license, vehicle identifier, service dates or repair-outcome semantics. `slno` is unique but is not documented as a vehicle key. | Labels cannot be validated as future maintenance targets or grouped by vehicle. Do not train. |

## Inspection limitations

- The raw collection contains 237 files in 14 top-level directories plus a root-file
  bundle. The inventory includes per-file schemas, row counts, missing values,
  within-table exact duplicates, archive checks, SHA-256 hashes and read errors in
  [`dataset_inventory.json`](dataset_inventory.json).
- The table in [`dataset_inventory.md`](dataset_inventory.md) records every file and
  every unsupported format. Archives were inspected for member names and corruption;
  no raw archive was extracted or altered.
- The single CBF file is opaque/proprietary. The 23 MBL files are a custom format; a
  zero-byte log has no content to decode. A decoder is present in the raw collection
  but was not executed. One WebP banner was classified by file type, not decoded.
- Python scripts were inventoried by line count and notebooks by cell count; downloaded
  scripts/notebooks were not executed. Relevant dataset READMEs and selected scripts
  were reviewed for schemas, claims and decoder context.
- The original dataset collection contains byte-identical duplicates: 81 extracted
  KIT CSVs match all 81 CSV members in the KIT ZIP; B0025–B0028 MAT files repeat in
  two battery folders. Duplicate hashes are included in the machine-readable report.

## References

1. Karlsruhe Institute of Technology, *Automotive OBD-II Dataset*, DOI
   [10.35097/1130](https://doi.org/10.35097/1130).
2. Vergara et al., *EngineFaultDB: A Novel Dataset for Automotive Engine Fault
   Classification and Baseline Results*, IEEE Access (2023), DOI
   [10.1109/ACCESS.2023.3331316](https://doi.org/10.1109/ACCESS.2023.3331316).
3. NASA Ames Prognostics Center of Excellence, [Li-ion Battery Aging Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets).
4. UCI Machine Learning Repository, *AI4I 2020 Predictive Maintenance Dataset*,
   DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
5. Yuñ Solutions, [LEVIN-Open Data](https://github.com/YunSolutions/levin-openData).
