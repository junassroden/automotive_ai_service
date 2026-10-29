# carOBD sensor anomaly experiment

This research-only pipeline scores unusual patterns in ten numeric OBD-II telemetry fields from the locally documented 2014 Toyota Etios data. It does not identify confirmed mechanical faults, component failures, maintenance needs, or future failures. No verified anomaly/fault labels are available.

The split targets 70% training per source CSV in non-decreasing engine-run-time/source order; rows sharing the boundary run-time value stay together, so the exact fraction may differ. The later tail is held out with no run-time value overlap. Median imputation is fitted on training records only. File mode and session IDs are descriptive metadata, not model inputs. Units are not supplied in local headers. The data represents one vehicle and cannot establish generalization to any other vehicle.

Anomaly flags are threshold-relative statistical outliers. They may reflect operating-mode changes, sensor update behavior, missingness, or unusual-but-normal conditions. They must not be interpreted as diagnostic findings or used for maintenance decisions. Threshold sensitivity is reported rather than presenting an arbitrary flag rate as accuracy.
