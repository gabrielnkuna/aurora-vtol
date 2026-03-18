# Aurora Command Reference

This document is the user-facing guide to the Aurora CLI. It focuses on what each command is for, when to use it, and the most useful examples.

See also:

- [Architecture](architecture.md)
- [Module Reference](module_reference.md)
- [Bridge Workflows](bridge_workflows.md)
- [Engineering Notes](engineering_notes.md)
- [Trace Schema](trace_schema.md)
- [Scenario Guide](scenario_guide.md)

## Command structure

Aurora currently exposes two main command groups:

```text
aurora-vtol alloc ...
aurora-vtol bridge ...
```

Use `alloc` for control, maneuver, mission, and engineering workflows.
Use `bridge` for replay, export, Mission Planner, and SITL workflows.

## `aurora-vtol alloc` commands

### `aurora-vtol alloc assess`

Assess a trace against engineering criteria.

Use this when you want a structured verdict on maneuver or mission quality instead of only inspecting plots or replay.

Example:

```bash
uv run aurora-vtol alloc assess \
  --trace runs/trace_coordinates_medium.json \
  --format text \
  --out runs/trace_coordinates_medium_assessment.txt
```

### `aurora-vtol alloc demo`

Run a basic allocator demo.

Use this for quick radial or yaw-aware allocation sanity checks.

Example:

```bash
uv run aurora-vtol alloc demo --version v2 --dir-deg 90 --fxy 3000 --mz-nm 0
```

### `aurora-vtol alloc step`

Run a step-style directional transition.

Use this for simple direction-change response studies without the full snap-stop profile.

Example:

```bash
uv run aurora-vtol alloc step \
  --dir-a-deg 0 \
  --dir-b-deg 180 \
  --fxy 3000 \
  --step-time-s 3 \
  --total-s 8 \
  --trace-out runs/trace_step.json
```

### `aurora-vtol alloc coordinates`

Run a coordinate mission with optional presets, obstacles, and power-aware tuning.

Use this for mission-scale studies rather than isolated maneuvers.

Examples:

```bash
uv run aurora-vtol alloc coordinates \
  --preset medium \
  --trace-out runs/trace_coordinates_medium.json
```

```bash
uv run aurora-vtol alloc coordinates \
  --preset medium \
  --power-safe \
  --continuous-power-kw 130 \
  --trace-out runs/trace_coordinates_power_safe.json
```

### `aurora-vtol alloc effectiveness-report`

Materialize the current nominal effectiveness table and write a compact assumptions report.

Use this when you want to inspect the hardware-facing authority assumptions directly instead of inferring them from source code and raw JSON assets.

Example:

```bash
uv run aurora-vtol alloc effectiveness-report \
  --out-dir runs/effectiveness_report_geometry_seed
```

This writes a small artifact pack, including:

- `summary.json`
- `summary.md`
- `materialized_table.json`
- `source_spec.json` when the source is a geometry seed

### `aurora-vtol alloc effectiveness-compare`

Compare a candidate effectiveness source against the current baseline.

Use this when a future CAD-, CFD-, or bench-derived candidate arrives and you want a compact diff instead of a manual JSON inspection.

Example:

```bash
uv run aurora-vtol alloc effectiveness-compare   --candidate-table data/effectiveness/aurora_ring32_provisional_v1.json   --out-dir runs/effectiveness_compare_geometry_vs_provisional
```

This writes a comparison pack, including:

- `summary.json`
- `summary.md`
- `baseline_table.json`
- `candidate_table.json`

### `aurora-vtol alloc power-sweep`

Sweep plant assumptions such as continuous power, hover power, voltage, or resistance.

Use this to build propulsion and battery requirement tables.

Example:

```bash
uv run aurora-vtol alloc power-sweep \
  --preset medium \
  --continuous-power-kw 125 \
  --continuous-power-kw 130 \
  --continuous-power-kw 135 \
  --out runs/power_requirements_medium.json \
  --requirements-out runs/power_requirements_medium.md
```

### `aurora-vtol alloc sizing-pack`

Generate a multi-preset sizing study.

Use this when you want one requirements pack across `medium`, `long`, and `very-long` mission presets.

Example:

```bash
uv run aurora-vtol alloc sizing-pack \
  --continuous-power-kw 125 \
  --continuous-power-kw 130 \
  --continuous-power-kw 135 \
  --out-dir runs/power_sizing_pack
```

### `aurora-vtol alloc fault-envelope`

Search fault cases and rank the worst ones.

Use this to discover which actuator or plenum faults are most limiting for a mission preset.

Example:

```bash
uv run aurora-vtol alloc fault-envelope \
  --preset medium \
  --continuous-power-kw 125 \
  --out-dir runs/fault_envelope_medium
```

### `aurora-vtol alloc fault-threshold`

Find the minimum viable continuous power for selected fault cases.

Use this after fault-envelope runs when you want a threshold study on specific worst cases.

Example:

```bash
uv run aurora-vtol alloc fault-threshold \
  --summary runs/fault_envelope_medium/summary.json \
  --case dead-fan-12-x0 \
  --continuous-power-kw 120 \
  --continuous-power-kw 125 \
  --continuous-power-kw 130 \
  --out-dir runs/fault_threshold_medium
```

### `aurora-vtol alloc fault-threshold-pack`

Run the same threshold workflow across multiple envelope summaries.

Use this for multi-preset worst-case threshold summaries.

Example:

```bash
uv run aurora-vtol alloc fault-threshold-pack \
  --summary runs/fault_envelope_medium/summary.json \
  --summary runs/fault_envelope_long/summary.json \
  --summary runs/fault_envelope_very_long/summary.json \
  --continuous-power-kw 125 \
  --continuous-power-kw 130 \
  --out-dir runs/fault_threshold_pack
```

### `aurora-vtol alloc repel`

Run the repel-field scenario.

Use this when studying obstacle-like field avoidance or response timing.

Example:

```bash
uv run aurora-vtol alloc repel \
  --ox 30 --oy 0 \
  --radius-m 30 \
  --k 120 \
  --trace-out runs/trace_repel.json
```

### `aurora-vtol alloc step-redirect`

Run the UFO-style redirect profile that reorients the motion vector without forcing a full stop.

Use this for 45, 90, 135, and 180 degree redirect studies when you want fixed yaw and smoother vectoring.

Examples:

```bash
uv run aurora-vtol alloc step-redirect \
  --dir-b-deg 90 \
  --maneuver-safe \
  --trace-out runs/trace_step_redirect_90.json
```

```bash
uv run aurora-vtol alloc step-redirect \
  --eco \
  --dir-b-deg 90 \
  --maneuver-safe \
  --trace-out runs/trace_step_redirect_90_eco.json
```

### `aurora-vtol alloc step-snap`

Run the snap-stop profile.

Use this for hard brake, reversal, and snap-stop studies. Use `--eco` when you want the lower-power variant.

Examples:

```bash
uv run aurora-vtol alloc step-snap \
  --dir-a-deg 0 \
  --dir-b-deg 180 \
  --maneuver-safe \
  --trace-out runs/trace_step_snap_180.json
```

```bash
uv run aurora-vtol alloc step-snap \
  --eco \
  --dir-a-deg 0 \
  --dir-b-deg 180 \
  --maneuver-safe \
  --trace-out runs/trace_step_snap_180_eco.json
```

### `aurora-vtol alloc maneuver-pack`

Run a batch of maneuver cases and produce summary artifacts.

Use this for comparing fault cases, angles, and maneuver profiles in one report.

Examples:

```bash
uv run aurora-vtol alloc maneuver-pack \
  --profile step-redirect \
  --maneuver-safe \
  --out-dir runs/maneuver_pack_redirect_demo
```

```bash
uv run aurora-vtol alloc maneuver-pack \
  --profile step-snap-eco \
  --maneuver-safe \
  --out-dir runs/maneuver_pack_step_snap_eco_demo
```

## `aurora-vtol bridge` commands

### `aurora-vtol bridge inspect`

Inspect a bridge JSONL script and optionally generate an SVG preview.

Example:

```bash
uv run aurora-vtol bridge inspect \
  --script runs/bridge_coordinates_long.jsonl \
  --svg-out runs/bridge_coordinates_long.svg
```

### `aurora-vtol bridge mp-graphs`

Generate helper presets for Mission Planner graph selection.

Use this when replaying traces into Mission Planner and you want grouped engineering telemetry channels.

Example:

```bash
uv run aurora-vtol bridge mp-graphs \
  --trace runs/trace_coordinates_power.json \
  --format text
```

### `aurora-vtol bridge trace`

Convert a trace into bridge-ready outputs or stream it as live setpoints.

Examples:

```bash
uv run aurora-vtol bridge trace \
  --trace runs/trace_coordinates_medium.json \
  --jsonl-out runs/bridge_coordinates_medium.jsonl
```

```bash
uv run aurora-vtol bridge trace \
  --trace runs/trace_coordinates_medium.json \
  --mavlink-out tcp:127.0.0.1:5771 \
  --no-dry-run
```

### `aurora-vtol bridge mp`

Export a Mission Planner or QGroundControl waypoint mission.

Use this for GUI-driven AUTO-mode experiments rather than Aurora's continuous setpoint path.

Example:

```bash
uv run aurora-vtol bridge mp \
  --trace runs/trace_coordinates_long.json \
  --out runs/mission_coordinates_long.waypoints \
  --home-lat -26.2041 \
  --home-lon 28.0473
```

### `aurora-vtol bridge sitl`

Execute a trace or script against ArduPilot SITL.

Use this when you want the autopilot in the loop and Mission Planner watching SITL instead of a synthetic replay.

Example:

```bash
uv run aurora-vtol bridge sitl \
  --trace runs/trace_coordinates_long.json \
  --connect tcp:127.0.0.1:5760 \
  --execute \
  --takeoff-alt-m 2
```

## Direct Mission Planner replay

Aurora also supports a direct replay path outside the `aurora-vtol bridge` CLI surface.

Use [`scripts/trace_to_tlog.py`](../scripts/trace_to_tlog.py) when you want Mission Planner to observe a synthetic Aurora replay vehicle directly.

Example:

```bash
python scripts/trace_to_tlog.py \
  --trace runs/trace_coordinates_medium.json \
  --out listen:127.0.0.1:5770 \
  --home-lat -26.2041 \
  --home-lon 28.0473 \
  --pre-roll-s 8 \
  --repeat 5
```

## Practical command split

Use this rule of thumb:

- `aurora-vtol alloc ...` for generating, tuning, and assessing motion
- `aurora-vtol bridge ...` for exporting or streaming that motion into external tools
- `scripts/trace_to_tlog.py` for direct Mission Planner replay

## Notes on truth boundaries

The command surface is powerful, but the engineering meaning still matters:

- replay commands help visualize traces, but they do not prove hardware behavior
- bridge and SITL commands help validate interfaces, not actuator truth
- mission export commands are useful, but they are not equivalent to Aurora's continuous control path

Keep those boundaries explicit when using the CLI for demonstrations or engineering claims.
