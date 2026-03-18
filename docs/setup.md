# Aurora Setup and First Run

This repository is not a pitch deck. It is a working engineering platform you can clone, run, inspect, and replay into external tools.

This guide gets you from:

1. clone
2. CLI working
3. Mission Planner replay working
4. optional ArduPilot SITL bridge working

Related docs:

- [README](../README.md)
- [Command Reference](command_reference.md)
- [Bridge Workflows](bridge_workflows.md)
- [Trace Schema](trace_schema.md)
- [Limitations](limitations.md)

## Recommended host setup

Aurora works best when you treat the stack as three layers:

- `aurora-vtol` CLI and simulation: Linux, macOS, or WSL
- Mission Planner replay: usually Windows
- ArduPilot SITL: Linux or WSL

Recommended Windows layout:

- run `aurora-vtol` inside WSL
- run Mission Planner on Windows
- optionally run ArduPilot SITL in the same WSL environment

That is the most practical setup for the current repo.

## Prerequisites

### Required for the CLI

- `git`
- Python `>=3.10`
- [`uv`](https://docs.astral.sh/uv/)

Aurora's package metadata currently requires Python `>=3.10` in [pyproject.toml](../pyproject.toml).

### Required for Mission Planner replay and bridge streaming

- `pymavlink`

The repo installs the CLI dependencies with `uv sync`, but Mission Planner replay scripts and MAVLink bridge paths also need `pymavlink`.

### Required for Mission Planner itself

- Mission Planner on Windows

Official install docs:

- [Installing Mission Planner](https://ardupilot.org/planner/docs/mission-planner-installation.html)
- [Mission Planner Simulation](https://ardupilot.org/planner/docs/mission-planner-simulation.html)

### Required for ArduPilot SITL

Aurora does not vendor ArduPilot. Install and run SITL separately using the official ArduPilot docs:

- [Setting up the Build Environment on Windows 11 using WSL](https://ardupilot.org/dev/docs/building-setup-windows11.html)
- [Setting up the Build Environment on Windows 10/11 using WSL1 or WSL2](https://ardupilot.org/dev/docs/building-setup-windows10.html)
- [Setting up the Build Environment (Linux/Ubuntu)](https://ardupilot.org/dev/docs/building-setup-linux.html)
- [Setting up SITL on Linux](https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html)

## 1. Clone and install Aurora

```bash
git clone https://github.com/gabrielnkuna/aurora-vtol.git
cd aurora-vtol
uv venv
uv sync
uv run aurora-vtol --help
```

If you want Mission Planner replay, `bridge trace --mavlink-out`, or `bridge sitl`, install `pymavlink` into the same environment:

```bash
uv pip install pymavlink
```

## 2. Run a first CLI trace

Generate a trace artifact first. For example:

```bash
uv run aurora-vtol alloc coordinates --preset medium --power-safe --trace-out runs/trace_coordinates_medium.json
```

Then assess it:

```bash
uv run aurora-vtol alloc assess --trace runs/trace_coordinates_medium.json --format text
```

If those work, the core CLI path is installed correctly.

## 3. Replay a trace in Mission Planner

First generate or reuse a trace:

```bash
uv run aurora-vtol alloc repel --ox 30 --oy 0 --radius-m 30 --k 120 --trace-out runs/trace_repel.json
```

Then convert it into a Mission Planner-friendly MAVLink replay stream:

```bash
python scripts/trace_to_tlog.py --trace runs/trace_repel.json --out listen:127.0.0.1:5770 --home-lat -26.2041 --home-lon 28.0473 --pre-roll-s 8 --repeat 5
```

In Mission Planner:

1. choose `TCP`
2. host `127.0.0.1`
3. port `5770`
4. connect

This path is for visualization and telemetry replay. It is not an autopilot-in-the-loop run.

## 4. Run the bridge path directly

If `pymavlink` is installed, you can also build bridge artifacts or stream setpoints without Mission Planner replay scripts.

Build a JSONL bridge script:

```bash
uv run aurora-vtol bridge trace --trace runs/trace_coordinates_medium.json --jsonl-out runs/bridge_coordinates_medium.jsonl
```

Inspect it:

```bash
uv run aurora-vtol bridge inspect --script runs/bridge_coordinates_medium.jsonl --svg-out runs/bridge_coordinates_medium.svg
```

## 5. Run against ArduPilot SITL

After you have installed ArduPilot SITL separately, start a normal SITL instance using the official `sim_vehicle.py` flow from the ArduPilot docs. A typical developer example is:

```bash
cd ~/ardupilot/ArduCopter
sim_vehicle.py --console --map
```

Then, from the Aurora repo, stream a trace into that SITL instance:

```bash
uv run aurora-vtol bridge sitl --trace runs/trace_coordinates_medium.json --connect tcp:127.0.0.1:5760 --execute --takeoff-alt-m 2
```

Use this when you want the autopilot in the loop. Keep the truth boundary clear:

- Mission Planner replay validates replay and telemetry presentation
- `bridge sitl` validates bridge and setpoint plumbing against an autopilot stack
- neither one proves final Aurora vehicle mechanics or actuator truth

## Common pitfalls

### `pymavlink` import errors

If `trace_to_tlog.py`, `bridge trace --mavlink-out`, or `bridge sitl` complain about MAVLink imports, install:

```bash
uv pip install pymavlink
```

### `VIRTUAL_ENV` does not match `.venv`

If `uv` warns that another virtual environment is active, either:

- deactivate the other environment and use the repo's `.venv`, or
- rerun with `uv run --active ...` if you intentionally want the active environment

### Mission Planner connects but nothing useful appears

Check:

- the replay script is still running
- the host and port match exactly
- you used a valid trace JSON
- your `--home-lat` and `--home-lon` values are sensible for map display

## Minimum proof path

If you want the smallest path that supports the claim that this repo is a working engineering platform, do these three steps:

```bash
uv sync
uv run aurora-vtol alloc coordinates --preset medium --trace-out runs/trace_coordinates_medium.json
uv run aurora-vtol alloc assess --trace runs/trace_coordinates_medium.json --format text
```

Then add one replay step:

```bash
python scripts/trace_to_tlog.py --trace runs/trace_coordinates_medium.json --out listen:127.0.0.1:5770 --home-lat -26.2041 --home-lon 28.0473
```

That gives a reproducible path from source code to CLI output to external visualization.
