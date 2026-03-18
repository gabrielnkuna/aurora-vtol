# Aurora Bridge Workflows

Aurora's bridge layer converts simulated traces and planned motion into artifacts or live streams that external flight tools can consume. In the current repo, this functionality lives primarily in [`src/aurora_vtol/bridge.py`](../src/aurora_vtol/bridge.py) and is exposed through the `aurora-vtol bridge ...` CLI commands in [`src/aurora_vtol/cli.py`](../src/aurora_vtol/cli.py).

The bridge is not just export plumbing. It is a control-path component that must preserve:

- units
- timing
- phase labels
- safety assumptions
- target IDs
- mode, arm, takeoff, and landing orchestration when talking to SITL

Related docs:

- [Setup and First Run](setup.md)
- [Architecture](architecture.md)
- [Interface Control Document](icd.md)
- [Engineering Review Checklist](engineering_review_checklist.md)
- [Trace Schema](trace_schema.md)
- [Command Reference](command_reference.md)

## Main bridge paths

Aurora currently supports four practical bridge paths:

1. JSONL setpoint scripts
2. Live MAVLink setpoint streaming
3. Mission Planner or QGroundControl waypoint export
4. Direct Mission Planner replay through `scripts/trace_to_tlog.py`

There is also a SITL-specific execution path that adds heartbeat, mode switching, arming, optional takeoff, optional landing mode, and optional disarming.

## Core bridge data model

### BridgeSetpoint

The bridge layer standardizes per-sample setpoints using a `BridgeSetpoint` dataclass with fields such as:

- `time_s`
- `north_m`
- `east_m`
- `down_m`
- `v_north_mps`
- `v_east_mps`
- `v_down_mps`
- `yaw_deg`
- `phase`

This is the canonical bridge-time representation inside the module.

### Coordinate conversion

The trace loader reads internal history values such as `x`, `y`, `z`, `vx`, `vy`, and `vz`, then converts them into local NED semantics:

- internal `x` -> bridge `east_m`
- internal `y` -> bridge `north_m`
- internal `z` (up) -> bridge `down_m = -up`
- internal `vx` -> bridge `v_east_mps`
- internal `vy` -> bridge `v_north_mps`
- internal `vz` (up) -> bridge `v_down_mps = -v_up`

This mapping is one of the most important bridge truth boundaries in the repo.

## Workflow 1: Build a JSONL setpoint script from a trace

### CLI

```bash
uv run aurora-vtol bridge trace --trace runs/trace_repel.json --jsonl-out runs/bridge_repel.jsonl
```

### What happens

1. The trace loader reads the trace JSON.
2. The bridge builds a time-ordered setpoint script in memory.
3. The script can be rate-limited to a chosen output frequency.
4. A final hold sample can be appended with `--hold-final-s`.
5. The script is written as one JSON record per line.

### Useful options

- `--rate-hz`
- `--hold-final-s`
- `--dry-run` and `--no-dry-run`
- `--jsonl-out`

### When to use this

Use this when you want a deterministic replay artifact that can be inspected, versioned, validated, or reused later.

## Workflow 2: Inspect a bridge script before replaying or streaming it

### CLI

```bash
uv run aurora-vtol bridge inspect --script runs/bridge_repel.jsonl --svg-out runs/bridge_repel.svg
```

### What happens

1. The script is reconstructed from JSONL.
2. Validation and summary checks are performed.
3. An SVG preview can be generated.
4. The command can optionally fail if the script has issues.

### What to check

- sample count and duration
- phase counts
- path continuity
- start and final state sanity
- whether the script is suitable for the downstream consumer

## Workflow 3: Stream a trace as live MAVLink setpoints

### CLI

```bash
uv run aurora-vtol bridge trace --trace runs/trace_repel.json --mavlink-out tcp:127.0.0.1:5771 --no-dry-run
```

### What happens

1. The trace is converted into a bridge script in memory.
2. The module opens a MAVLink connection.
3. It sends periodic GCS-style heartbeats.
4. For each sample, it sends `SET_POSITION_TARGET_LOCAL_NED` with position, velocity, and yaw.

### Important options

- `--target-system`
- `--target-component`
- `--source-system`
- `--source-component`
- `--heartbeat-hz`
- `--realtime` and `--no-realtime`

### Best use case

Use this for live bridge testing against a MAVLink consumer or router.

## Workflow 4: Export a waypoint mission for Mission Planner or QGroundControl

### CLI

```bash
uv run aurora-vtol bridge mp --trace runs/trace_coordinates_long.json --out runs/mission_coordinates_long.waypoints --home-lat -26.2041 --home-lon 28.0473
```

### What happens

1. The route intent is sampled into a mission-like path.
2. The exporter writes a QGC WPL style waypoint file.
3. Mission Planner can load and upload it in `AUTO` mode.

### Important caveat

This is useful for GUI-driven path experiments, but it is not the same thing as streaming the full Aurora control path. The autopilot will still fly it using its own waypoint behavior.

## Workflow 5: Direct Mission Planner replay from a trace

### CLI

```bash
python scripts/trace_to_tlog.py --trace runs/trace_coordinates_medium.json --out listen:127.0.0.1:5770 --home-lat -26.2041 --home-lon 28.0473 --pre-roll-s 8 --repeat 5
```

### What happens

1. `scripts/trace_to_tlog.py` loads the Aurora trace.
2. It synthesizes an MP-friendly MAVLink replay vehicle.
3. Mission Planner connects to `tcp://127.0.0.1:5770`.
4. The script sends state, status, home, battery, and engineering telemetry.

### When to use this

Use this when you want to visualize the trace directly in Mission Planner without an autopilot in the loop.

## Workflow 6: Execute against ArduPilot SITL

### CLI

```bash
uv run aurora-vtol bridge sitl --trace runs/trace_coordinates_long.json --connect tcp:127.0.0.1:5760 --execute --takeoff-alt-m 2
```

### What happens

1. The bridge connects to SITL.
2. It waits for heartbeat and discovers target IDs.
3. It can change mode, arm, take off, stream setpoints, and optionally land.
4. Mission Planner can watch the same SITL separately.

### Best use case

Use this when you want to test the bridge and setpoint path against an autopilot stack rather than a synthetic replay.

## Bridge truth boundaries

Aurora is strongest when these boundaries are kept explicit:

- trace replay is visualization truth, not hardware truth
- bridge streaming is control-path truth, not actuator truth
- Mission Planner `AUTO` missions are convenient, but they are not equivalent to Aurora's continuous setpoint path
- SITL integration is valuable, but SITL dynamics are not the same as the Aurora plant model

Those boundaries matter because bridge behavior is part of the engineering story, not just a convenience layer.
