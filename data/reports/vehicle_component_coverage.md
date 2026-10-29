# Vehicle and component coverage matrix

This matrix describes evidence present in the local files, not a deployed prediction
capability. No cell is classified as directly supported for predictive maintenance:
the available data do not establish externally validated, group-disjoint failure or
repair prediction for the listed fleets. "General diagnostic assistance only" refers
to the existing local knowledge retrieval, not a trained model.

**Status key**

- **Directly supported** — verified labels and inputs directly cover the stated task.
- **Partially supported** — some relevant measurements or labels exist, but key
  outcomes, vehicle diversity, provenance, or component evidence is missing.
- **General diagnostic assistance only** — the knowledge base can provide cautious
  general information; no suitable vehicle-specific data supports the task.
- **Unsupported** — no relevant evidence is available.

| Component / maintenance area | Passenger cars | Motorcycles | Pickup trucks | Vans | Minivans |
|---|---|---|---|---|---|
| Engine health and engine faults | Partially supported | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Engine temperature and overheating | Partially supported | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Oil condition, pressure, lubrication | General diagnostic assistance only | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Battery health and electrical systems | General diagnostic assistance only | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Charging system / alternator | General diagnostic assistance only | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Cooling system | Partially supported | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Transmission and drivetrain | General diagnostic assistance only | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Brake system and brake wear | General diagnostic assistance only | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Tires and tire wear | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Suspension and steering | General diagnostic assistance only | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Fuel system | Partially supported | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Sensors and diagnostic trouble codes | Partially supported | Partially supported | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Mileage-based maintenance | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Maintenance and repair history | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |
| Component failure risk | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |
| Maintenance scheduling | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only | General diagnostic assistance only |
| Remaining useful life | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |

## Evidence and limits

- **Passenger cars:** the KIT OBD-II files describe one Seat Leon and record ten
  engine/driving signals, but have no confirmed fault, repair or maintenance labels.
  EngineFaultDB adds controlled laboratory engine measurements, but covers one
  C14NE engine and its four numeric class meanings are not documented. The claimed
  diesel DTC workbook lacks verified source/vehicle IDs and has heavily duplicated
  rows. These sources justify limited sensor/engine analysis, not fleet predictions.
- **Motorcycles:** the local decoded telemetry includes RPM, water temperature,
  oil pressure, battery voltage, fuel pressure, front brake pressure, gear/clutch
  signals and suspension travel. Rear brake pressure is absent/dead, fuel pressure
  is static, and no faults or maintenance outcomes are labelled. The local source
  claim is not independently verified by the linked record. It supports observations
  of these channels only, not failure risk or component wear.
- **Pickup trucks, vans and minivans:** the only apparent broad-vehicle coverage is in
  CVFFAD, whose local README says its records are generated and whose DOI/license
  materials are incomplete. It is not accepted as real fleet evidence. No
  category-specific predictive coverage is claimed.
- **Battery RUL:** NASA data can support research on laboratory 18650 cells under
  controlled cycles. It does not represent a vehicle battery pack, alternator,
  vehicle environment or fleet repair outcome and therefore does not change the
  vehicle-category coverage.
- A partial sensor match does not imply a diagnostic threshold, validated fault
  detector, failure probability, service interval or RUL estimate. See
  [`dataset_research.md`](dataset_research.md) for provenance and license findings.
