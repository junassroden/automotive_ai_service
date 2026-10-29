# Kawasaki Ninja 400 SSP300 — raw telemetry, Circuito de Navarra, two acquisition days, one rider

Raw on-vehicle telemetry recorded with an open-architecture STM32H723 datalogger
(firmware `h723-revC2 v3.5`, MBL binary log format, schema 1) during track stints at the
Circuito de Navarra on two separate days, same rider, same machine, same ECU calibration.

- **Vehicle**: Kawasaki Ninja 400 (model year 2018, 399 cm³ parallel twin), prepared to
  World Supersport 300 class specification, Mektronic-family racing ECU. The same machine
  as the Talavera deposit <https://doi.org/10.5281/zenodo.21872051>.
- **Venue**: Circuito de Navarra, Los Arcos, Navarra, Spain. The venue is a folder label;
  the ECU-forwarded GPS latitude channel (42.5646° N) is consistent with it.
- **Rider**: the depositor. The dataset contains no personal data of any third party.
- **Days**: (a) one morning in mid-August 2026, one stint; (b) 5 September 2026, four
  stints. Neither date comes from the files: every log header carries `utc_unix = 0`
  (the logger had no RTC fix). Day (a) is dated from the depositor's own record; day (b)
  from the archive name and the file-system timestamps of the copied files. Nothing in the
  logs orders the two days, and no analysis in this record may depend on a calendar
  instant the files do not contain.

## Contents

| Path | What it is |
|---|---|
| `navarra_telemetry/` | day (a), mid-August 2026: 13 raw files (one is zero bytes), 13 704.6 s logged, 134 984 records |
| `navarra-050926-tanda1/` | day (b), stint 1: 4 raw files, 1 848.0 s logged, 19 565 records |
| `navarra-050926-tandas2-4/` | day (b), stints 2–4 in one folder (stint boundaries are not marked in the files): 6 raw files, 3 170.6 s logged, 31 709 records |
| `*/raw/LOG*.MBL` | raw log files, byte-for-byte as written on the SD card |
| `*/raw_manifest.json` | SHA-256 and size of every raw file |
| `*/metadata.json` | header fields, per-channel liveness inventory (live / dead / static / silent), integrity audit, known issues |
| `*/samples.csv.gz` | convenience decode: full reconstructed state at every record (forward-filled; see warning) |
| `*/updates.csv.gz` | true update stream: one row per actually-logged field change, no fill |
| `*/ingest_mbl.py` | the decoder that produced the CSVs from `raw/`; identical in the three folders |

Upload archives: one `<session>-raw.zip` per session (raw files and manifest only) and
`telemetry-navarra-processed.zip` (metadata, CSV decodes, decoder and this README).
`upload-manifest.json` lists size and MD5 of each archive.

## Format

MBL binary, schema 1. Record framing `AA55 | type | len | payload | crc16-ccitt`. Record
types: `0xB0` header, `0xB1` ECU keyframe, `0xB2` ECU delta, `0xB3` ADU keyframe,
`0xB4` ADU delta, `0xB5` status. Every 30 s a keyframe re-asserts the full ECU state;
between keyframes only changed fields are written. `ingest_mbl.py` is the reference
decoder and documents the field registry (order is fixed by the firmware; names are not
in the file).

**Time base.** `ts` is the logger tick counter in seconds. It is board uptime: it restarts
at power-up, not per file, and consecutive files may or may not share a power cycle. It
must never be differenced across files. `boot_id` is assigned per file and incremented
again at any backwards jump above 1 s, so continuity segments are never wrongly joined,
only shortened. Group by `boot_id` for anything that needs continuity.

**One file per engine cycle**, not per power cycle: a file opens at `rpm ≥ 500` sustained
1 s and closes at `rpm < 300` sustained 15 s.

**Forward-fill warning.** `samples.csv.gz` propagates every channel to every record for
convenience, so sparse channels look dense. Timing-sensitive analyses must use
`updates.csv.gz`.

**Silent channels.** Some channels never broadcast a change and are known only from the
30 s keyframe re-assertions. A delta-only reader reports them as absent; they are listed
as `silent_channels` in each `metadata.json`.

## Integrity

| Session | CRC failures | resync bytes | truncated tail | ts regressions |
|---|---|---|---|---|
| `navarra_telemetry` | 0 | 0 | 0 | 0 |
| `navarra-050926-tanda1` | 0 | 17 (`LOG00004.MBL`) | 0 | 1 (`LOG00004.MBL`, split into a new `boot_id`) |
| `navarra-050926-tandas2-4` | 0 | 0 | 0 | 0 |

## Channels

95 live channels per session: Mektronic-family ECU fields over CAN (engine speed, gear,
throttle, manifold pressure, wheel speeds, front brake pressure, suspension travel, lean
angle, vertical acceleration, lambda, temperatures, cut function and cut level, launch and
traction-control state words), ECU-forwarded GPS (`adu_gps_*`, latitude and longitude as
degrees × 1e7 with satellite count and validity), and datalogger status. Full per-channel
inventory with update counts, ranges and liveness in each `metadata.json`.

Channels present in the CAN dictionary but **never non-zero in any Navarra session**
(sensor absent or not wired): `brake_rear`, `torque_rear`, `ots`, `lap_time`, `lap_info`,
`rbw_state`, `gear_down_state`, `aux_lights`, `trumpet_pwm_percent`, `adu_gps_heading`.

The `cut_function` channel is a manufacturer code: 0 none, 1 key-off, 3 over-run fuel
cut, 4 upshift, 5 upshift shake, 6 engine-speed limit, 7 downshift, 8 downshift shake,
11 road-speed limit, 12 engine braking, 13 anti-jerk. No code in this dictionary denotes a
traction-control cut; whether the traction controller intervened cannot be read from this
channel alone.

## Known issues, preserved in the raw files

- `utc_unix = 0` in every header: no calendar date and no wall-clock reference in the
  data.
- **Front brake pressure saturates at 106.27 bar** on day (b): 265 of 4 003 updates in
  stint 1 and 314 of 7 676 in stints 2–4 sit at that ceiling. Values at or above 106 bar are
  right-censored, not measured peaks. Day (a) never reaches the ceiling (maximum 55.8 bar).
- The GPS longitude channel (`adu_gps_lon_2d`) carries out-of-range values in every
  Navarra session and must not be used without inspection; latitude is consistent with the
  venue.
- `navarra_telemetry/raw/LOG00010.MBL` is zero bytes: the engine-cycle open triggered and
  power was lost before the header was flushed.
- `adu_gps_heading` is declared as a float in the delta registry and as `int16` in the
  keyframe struct; it is dead in every session, so the ambiguity is not exercised.
- No IMU raw channels are present in the MBL stream of this firmware version; only the
  `imu_ok` status word is logged.

## Scope and limitations

- One rider, one machine, one venue, two days. Between-rider and between-venue
  generalisation is not supported by this record alone; combined with the Talavera deposit
  it gives the same rider and machine at two permanent circuits.
- Day (a) was recorded on a surface damp in two corners after overnight rain, racing line
  dry, about 26 °C (depositor's statement). No surface or tyre label exists for day (b).
- No loss-of-grip event is labelled in either day.
- Stint boundaries within `navarra-050926-tandas2-4` are not marked.
- Final drive ratio, kerb mass as instrumented, exact ECU designation and firmware, tyre
  make and compound, and ambient and track temperatures on day (b) are not recorded.

## Licence and citation

CC BY 4.0. If you use this dataset, cite the dataset DOI and the platform description
article: A. García Cuadra, F. Santos Olalla, A. Brunete González, "Design and Experimental
Evaluation of an Open-Architecture Multi-Sensor Telemetry System for Real-Time Motorcycle
Dynamics Acquisition", *Electronics* 15(12):2604, 2026. DOI 10.3390/electronics15122604.
