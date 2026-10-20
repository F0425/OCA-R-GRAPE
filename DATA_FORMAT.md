# Input interface

Original records are confidential and are not included. Use an authorised CSV.
The model accepts a voyage-segment table, not raw AIS trajectories. Time-weighted
AIS/environment aggregation and MRV matching described in the paper occurred
upstream; this release starts from the integrated segment table.

## Numerical imputation features

| Group | Fields |
| --- | --- |
| MRV / Transport | cargo_volume, total_teu, heavy_ctns, total_distance |
| AIS | voyage_duration_h, ais_distance_nm, ais_sog_sailing_time_mean_kn, ais_sailing_time_ratio, ais_draft_time_mean_m |
| Ship | ship_age_years, ship_deadweight_t, ship_teu_capacity, ship_design_draught_m, ship_service_speed_kn, main_power_kw |
| Wind | wind_speed_time_mean_ms, headwind_mean_ms, crosswind_abs_mean_ms |
| WindRatio | headwind_time_ratio, crosswind_time_ratio |
| Wave | wave_height_time_mean_m, wave_height_p90_m |
| Current | current_speed_time_mean_ms, headcurrent_mean_ms, crosscurrent_abs_mean_ms |
| CurrentRatio | headcurrent_time_ratio, crosscurrent_time_ratio |

Units follow the names: distance in nautical miles, duration in hours, speed in
knots, draught/wave height in metres, wind/current components in m/s, ship age in
years, deadweight in tonnes, power in kW, capacities in TEU and ratios in [0,1].
Cargo volume and heavy-container count must use the definitions and units of the
authorised source dataset; do not infer a conversion from the field name alone.

## Metadata and target

- `vessel_code`: vessel identifier, also accepted from `imo` or `mmsi`.
- `end_port_code`: destination-port identifier; use destination only, not an OD pair.
- `fuel_total`: segment fuel consumption in tonnes, used only for downstream
  regression and not as a numerical imputation feature or training label.

The loader supports the existing MRV aliases such as `primary_mrv_cargo_volume`
and `primary_mrv_end_port_code`. Canonical column names above are recommended.
The paper's reference input has complete numerical features and fuel labels;
the artificial masks provide evaluation ground truth. Missing/unavailable
columns are warned about and skipped for compatibility, but such a run is not
the paper's 27-feature experiment. Positive reported distances and finite fuel
labels are required by the existing cleaning procedure.

Preserve input order. Both relation rules select nearby positions within each
metadata group. Supplying time columns does not change this rule. The loader
retains a constant missing-category column for the absent `end_is_eu` field to
preserve downstream compatibility; see README. Do not include varying values in
that field when reproducing the paper protocol.
