# Aurora Trace Schema

This document describes the structure of Aurora trace files and the meaning of the main trace fields.

Related docs:

- [Command Reference](command_reference.md)
- [Bridge Workflows](bridge_workflows.md)
- [Engineering Notes](engineering_notes.md)
- [Scenario Guide](scenario_guide.md)

## Purpose

Aurora traces are the portable time-series artifacts used for:

- engineering assessment
- replay in Mission Planner
- bridge conversion
- SITL-oriented workflows
- offline diagnostics and plotting

The trace writer itself is simple and lives in [`src/aurora_vtol/allocator/trace.py`](../src/aurora_vtol/allocator/trace.py):

```text
{
  "meta": {...},
  "hist": {...}
}
```

The exact `meta` and `hist` contents depend on the scenario, but the root shape is stable.

## Root structure

### `meta`

Static or summary information about the run.

Typical contents include:

- scenario version and type
- mission or maneuver parameters
- simulation parameters
- limits and plant parameters
- arrival or final-state summaries

### `hist`

Time-series history arrays.

Each key is typically a list with one value per simulation sample. For array-valued telemetry, each time sample may itself contain a list.

Examples:

- scalar history: `x`, `y`, `speed`, `battery_v`
- vectorized history: `fan_thrust_16`, `alpha_deg_32`
- categorical or label-like history: `mission_phase`

## Stable top-level contract

The top-level contract is currently:

- `meta` is a JSON object
- `hist` is a JSON object
- traces are written as UTF-8 JSON
- `save_trace_json(...)` writes exactly one file containing both blocks

That part should be treated as stable for downstream tooling.

## Common `meta` sections

Different runners populate different `meta` sections, but the following are common or representative:

- `version`
- `mission`
- `guidance`
- `planner`
- `obstacles`
- `sim`
- `limits`
- `plenum`
- `power`
- `arrival`
- `final`

A representative coordinate-mission trace currently includes those sections in [`runs/trace_coordinates_power.json`](../runs/trace_coordinates_power.json).

## Common `hist` fields

The exact history set varies by scenario. Downstream consumers should treat unknown fields as optional and should not assume every trace has every key.

### Core kinematics

These are the most stable fields across many scenarios:

- `t`
- `x`
- `y`
- `z`
- `vx`
- `vy`
- `vz`
- `yaw_deg`
- `yaw_rate_deg_s`
- `speed`

### Controller and force tracking

Representative fields:

- `fx_cmd`
- `fy_cmd`
- `fz_cmd`
- `fx_ach`
- `fy_ach`
- `fz_ach`
- `fx_err`
- `fy_err`
- `fz_err`
- `fxy_budget_n`
- `goal_force_scale`
- `guard_scale`

### Vane and tangential data

Representative fields:

- `alpha_cmd_deg_32`
- `alpha_deg_32`
- `alpha_cmd_rms`
- `alpha_deg_rms`
- `ft_tan_32`
- `ft_tan_rms`

### Fan and thermal data

Representative fields:

- `fan_cmd_16`
- `fan_actual_16`
- `fan_thrust_16`
- `fan_response_pct`
- `fan_temp_c_16`
- `fan_temp_max_c`
- `fan_temp_mean_c`
- `fan_thermal_scale_16`

### Power and supply data

Representative fields:

- `battery_v`
- `battery_a`
- `battery_soc`
- `power_w`
- `energy_wh`
- `power_margin_kw`
- `continuous_power_ratio`
- `continuous_power_pct`
- `thrust_scale_pct`
- `thermal_scale_pct`
- `budget_ratio`

### Burst-power data

Some current plant and maneuver flows expose burst-aware telemetry indirectly in assessment summaries rather than as a guaranteed trace contract. When present in trace-adjacent analysis or summary artifacts, it may include concepts such as:

- burst reserve minimum percentage
- burst clip time
- burst active time

These fields should currently be treated as engineering-analysis outputs, not as guaranteed trace keys.

### Mission and routing data

Representative fields:

- `mission_phase`
- `route_goal_idx`
- `subgoal_x_m`
- `subgoal_y_m`
- `dist_to_goal`
- `dist_to_subgoal`
- `nearest_obstacle_m`
- `avoid_fx`
- `avoid_fy`
- `target_z_m`

## Array-valued telemetry

Aurora traces frequently include grouped array-valued histories.

Typical sizes:

- `16` fan-group values per sample
- `32` vane or segment values per sample

Examples:

- `fan_cmd_16`
- `fan_thrust_16`
- `fan_temp_c_16`
- `alpha_deg_32`
- `alpha_cmd_deg_32`
- `ft_tan_32`

Downstream tools should not hard-code only scalar histories.

## Naming conventions

Common conventions in the trace files:

- positions end in `_m`
- velocities end in `_mps`
- forces end in `_n`
- moments end in `_nm`
- angles often use `_deg`
- percentages often use `_pct` or ratio-like scalar names

Older traces may still contain some shorter legacy names such as `x`, `vx`, or `speed`, so consumers should be tolerant.

## Scenario-specific variability

Aurora does not yet have a frozen, versioned trace schema for every scenario family. In practice:

- maneuver traces may contain step or snap timing fields
- mission traces may contain route, obstacle, and planner fields
- engineering-heavy traces may contain additional power, thermal, or guard telemetry

This is why the safest downstream rule is:

- rely on the root `meta` / `hist` contract
- treat many individual keys as optional
- tolerate additional keys without failure

## Recommended consumer behavior

If you are building tooling against Aurora traces:

1. always check for the existence of a key before assuming it exists
2. tolerate extra keys
3. use `meta.version` and the scenario-specific `meta` sections to understand provenance
4. avoid assuming every trace family has identical telemetry density
5. keep array-valued fields separate from scalar plotting logic

## Truth boundary

Aurora traces are software artifacts that capture modeled behavior. They are extremely useful for engineering review and bridge workflows, but they should not be mistaken for proof of hardware truth or certified flight data.
