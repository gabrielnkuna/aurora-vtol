# Aurora VTOL

Aurora is a software-first control, maneuver, and simulation stack for a 16-fan / 32-vane ring vehicle. The repository studies how guidance, vehicle control, control allocation, actuator dynamics, power limits, fault handling, and bridge tooling work together for a UFO-style VTOL concept.

The repository is not a pitch deck. It is a working engineering platform you can clone, run, replay into Mission Planner or SITL-oriented paths, and scrutinize from source.

This project is best understood as a flight-control and validation stack, not yet a complete aircraft implementation. It is designed to answer questions like:

- Can the craft generate lateral force while holding yaw?
- How does the allocator behave under actuator faults or degraded plenum sectors?
- How quickly can the vehicle redirect, repel, brake, and snap-stop?
- How do power, thermal, and topology constraints affect achievable motion?
- How can traces be replayed into Mission Planner or streamed into SITL-oriented workflows?

## What the repo currently includes

- A typed interface layer in [`src/aurora_vtol/icd.py`](src/aurora_vtol/icd.py)
- A vehicle controller layer in [`src/aurora_vtol/vehicle_controller.py`](src/aurora_vtol/vehicle_controller.py)
- A topology layer in [`src/aurora_vtol/topology.py`](src/aurora_vtol/topology.py)
- A fault-aware allocator core in [`src/aurora_vtol/allocator/allocate.py`](src/aurora_vtol/allocator/allocate.py)
- Dynamics and fault models in [`src/aurora_vtol/allocator/dynamics.py`](src/aurora_vtol/allocator/dynamics.py) and [`src/aurora_vtol/allocator/faults.py`](src/aurora_vtol/allocator/faults.py)
- Mission, plant, and power simulation in [`src/aurora_vtol/allocator/sim.py`](src/aurora_vtol/allocator/sim.py)
- Engineering assessment and tuning helpers in [`src/aurora_vtol/allocator/engineering.py`](src/aurora_vtol/allocator/engineering.py)
- Bridge, Mission Planner, and SITL tooling in [`src/aurora_vtol/bridge.py`](src/aurora_vtol/bridge.py) and [`scripts/trace_to_tlog.py`](scripts/trace_to_tlog.py)

## High-level architecture

```text
Mission / Guidance
        |
        v
Vehicle Controller
        |
        v
Desired Wrench (Fx, Fy, Fz, Mz)
        |
        v
Allocator
        |
        v
Fan / Vane / Tangential Commands
        |
        v
Actuator Dynamics + Faults + Plenum / Power Limits
        |
        v
Plant / Mission / Engineering Assessment
        |
        v
Trace Export + Replay + Bridge / SITL
```

## Repository layout

```text
src/aurora_vtol/
  cli.py
  icd.py
  topology.py
  effectiveness.py
  effectiveness_workflows.py
  vehicle_controller.py
  bridge.py

  allocator/
    model.py
    allocate.py
    dynamics.py
    faults.py
    field.py
    sim.py
    metrics.py
    response.py
    trace.py
    engineering.py

scripts/
  trace_to_tlog.py
```

## Quick start

### Setup

```bash
uv venv
uv sync
uv run aurora-vtol --help
```

If you want Mission Planner replay, direct MAVLink bridge streaming, or `bridge sitl`, also install:

```bash
uv pip install pymavlink
```

For the full clone-to-CLI-to-Mission-Planner/SITL path, see [Setup and First Run](docs/setup.md).

### Core allocator and maneuver demos

```bash
uv run aurora-vtol alloc demo --version v2 --dir-deg 90 --fxy 3000 --mz-nm 0
uv run aurora-vtol alloc step --dir-a-deg 0 --dir-b-deg 180 --fxy 3000 --step-time-s 3 --total-s 8
uv run aurora-vtol alloc repel --ox 30 --oy 0 --radius-m 30 --k 120 --trace-out runs/trace_repel.json
uv run aurora-vtol alloc step-redirect --dir-b-deg 90 --maneuver-safe --trace-out runs/trace_step_redirect_90.json
uv run aurora-vtol alloc step-snap --dir-b-deg 180 --maneuver-safe --eco --trace-out runs/trace_step_snap_180_eco.json
```

### Mission and engineering workflows

```bash
uv run aurora-vtol alloc coordinates --preset medium --power-safe --trace-out runs/trace_coordinates_medium.json
uv run aurora-vtol alloc assess --trace runs/trace_coordinates_medium.json --format text
uv run aurora-vtol alloc effectiveness-report --out-dir runs/effectiveness_report_geometry_seed
uv run aurora-vtol alloc effectiveness-template --out-dir runs/effectiveness_candidate_template
uv run aurora-vtol alloc effectiveness-validate --candidate-spec data/effectiveness_specs/aurora_ring32_candidate_template_v1.json --out-dir runs/effectiveness_validate_candidate_template
uv run aurora-vtol alloc effectiveness-adoption --candidate-spec data/effectiveness_specs/aurora_ring32_candidate_template_v1.json --candidate-note docs/effectiveness_candidate_provenance_template.md --out-dir runs/effectiveness_adoption_candidate_template
uv run aurora-vtol alloc effectiveness-promote --candidate-spec data/effectiveness_specs/aurora_ring32_candidate_template_v1.json --candidate-note docs/effectiveness_candidate_provenance_template.md --out-dir runs/effectiveness_promotion_candidate_template
uv run aurora-vtol alloc effectiveness-switch --promotion-manifest runs/effectiveness_promotion_reviewed_candidate/promotion_manifest.json --out-dir runs/effectiveness_switch_reviewed_candidate
uv run aurora-vtol alloc effectiveness-rollback --switch-manifest runs/effectiveness_switch_reviewed_candidate/switch_manifest.json --out-dir runs/effectiveness_rollback_reviewed_candidate
uv run aurora-vtol alloc effectiveness-compare --candidate-table data/effectiveness/aurora_ring32_provisional_v1.json --out-dir runs/effectiveness_compare_geometry_vs_provisional
uv run aurora-vtol alloc power-sweep --preset medium --continuous-power-kw 125 --continuous-power-kw 130
uv run aurora-vtol alloc maneuver-pack --profile step-snap-eco --maneuver-safe --out-dir runs/maneuver_pack_step_snap_eco_demo
```

### Bridge and replay workflows

```bash
uv run aurora-vtol bridge inspect --script runs/bridge_coordinates_long.jsonl --svg-out runs/bridge_coordinates_long.svg
uv run aurora-vtol bridge trace --trace runs/trace_coordinates_medium.json --jsonl-out runs/bridge_coordinates_medium.jsonl
uv run aurora-vtol bridge sitl --help
python scripts/trace_to_tlog.py --trace runs/trace_coordinates_medium.json --out listen:127.0.0.1:5770 --home-lat -26.2041 --home-lon 28.0473
```

## Documentation

- [Setup and First Run](docs/setup.md)
- [Architecture](docs/architecture.md)
- [Module Reference](docs/module_reference.md)
- [Engineering Notes](docs/engineering_notes.md)
- [Current-State Repo Review](docs/repo_review_current_state.md)
- [Command Reference](docs/command_reference.md)
- [Topology](docs/topology.md)
- [Effectiveness Map](docs/effectiveness_map.md)
- [Effectiveness Candidate Provenance Template](docs/effectiveness_candidate_provenance_template.md)
- [Fault Model](docs/fault_model.md)
- [Trace Schema](docs/trace_schema.md)
- [Scenario Guide](docs/scenario_guide.md)
- [Limitations](docs/limitations.md)
- [Interface Control Document](docs/icd.md)
- [Bridge Workflows](docs/bridge_workflows.md)
- [Hardware Control Architecture](docs/hardware_control_architecture.md)
- [Engineering Review Checklist](docs/engineering_review_checklist.md)

## What this repo proves today

The current codebase supports these software claims:

- 16-fan / 32-vane control is a meaningful software architecture, not just a diagram.
- The controller and allocator can generate holonomic-style lateral maneuvers while holding yaw.
- Maneuver feasibility can be studied under actuator, topology, power, thermal, and fault limits.
- Degraded operation can be explored before hardware exists.
- Traces can be exported into replay, Mission Planner, bridge, and SITL-oriented workflows.

## What it does not yet prove

This repository does not yet validate:

- final mechanical packaging
- final aerodynamic efficiency
- exact duct and plenum geometry
- exact ESC, actuator, and wiring layout
- certified avionics and safety architecture
- final CAD-accurate hardware implementation

## Recommended next engineering steps

1. Freeze the hardware-facing fan-to-vane and plenum topology assumptions.
2. Add geometry-aware effectiveness maps derived from the real vehicle layout.
3. Formalize estimator and actuator-health publisher interfaces against the ICD.
4. Separate central flight-computer logic from local actuator-node ownership.
5. Define a CAD-facing control requirements interface.
6. Add more hardware-oriented validation around timing, buses, and actuator feedback.
