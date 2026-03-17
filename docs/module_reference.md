# Aurora Module Reference

This document summarizes the responsibilities of the main Aurora modules as they exist today.

See also:

- [Architecture](architecture.md)
- [Interface Control Document](icd.md)
- [Engineering Notes](engineering_notes.md)
- [Command Reference](command_reference.md)
- [Topology](topology.md)
- [Effectiveness Map](effectiveness_map.md)
- [Fault Model](fault_model.md)
- [Trace Schema](trace_schema.md)

## Top-level modules

### `cli.py`

Command-line entry point for Aurora workflows. For user-facing examples and workflow guidance, see [Command Reference](command_reference.md).

Responsibilities:

- expose allocator, mission, engineering, and bridge commands
- parse CLI inputs into module calls
- provide a consistent surface for demos, assessments, and replay workflows

Representative command groups:

- `aurora-vtol alloc demo`
- `aurora-vtol alloc step`
- `aurora-vtol alloc coordinates`
- `aurora-vtol alloc repel`
- `aurora-vtol alloc step-redirect`
- `aurora-vtol alloc step-snap`
- `aurora-vtol alloc maneuver-pack`
- `aurora-vtol bridge inspect`
- `aurora-vtol bridge trace`
- `aurora-vtol bridge mp`
- `aurora-vtol bridge sitl`

### `icd.py`

Typed software contracts between layers.

Important role:

- defines the canonical objects passed between guidance, controller, allocator, and bridge-adjacent layers
- reduces coupling between modules
- prepares the project for more formal interface versioning later

Typical concepts:

- estimated state
- guidance targets
- actuator health
- desired wrench
- allocator command

### `topology.py`

Vehicle topology and structural ownership mapping.

Responsibilities:

- represent the 16-fan / 32-segment concept
- map fan-group and plenum-sector degradation onto segment authority
- centralize topology assumptions that would otherwise become implicit

### `effectiveness.py`

Nominal authority-table loading and smoothing helpers.

Responsibilities:

- load versioned nominal effectiveness-table assets and geometry seed specs
- provide a stable software interface for hardware-facing authority maps
- separate provisional nominal authority data from allocator math

### `vehicle_controller.py`

Controller layer between guidance and allocation.

Responsibilities:

- convert position, velocity, or maneuver intent into desired wrench commands
- keep allocator focused on distribution, not mission decisions
- respect lateral budget and redirect / braking priorities

Representative behavior areas:

- position tracking
- redirect velocity tracking
- directional-force command generation
- snap-stop braking and reverse phases

### `bridge.py`

Trace export and external-system bridge tooling.

Responsibilities:

- load and inspect trace files
- convert traces into bridge or setpoint scripts
- export JSONL and related replay-oriented formats
- support MAVLink, Mission Planner, and SITL workflows

## Allocator package

### `allocator/model.py`

Core vehicle and ring math.

Responsibilities:

- encode geometric assumptions behind segment-based force production
- provide the low-level math used by allocation and achieved-force evaluation

### `allocator/allocate.py`

Main control-allocation logic.

Responsibilities:

- solve radial allocation
- add yaw and tangential authority
- apply topology-aware effectiveness scaling
- handle constrained or faulted authority

Conceptual variants:

- baseline radial allocation
- yaw-aware or swirl-augmented allocation

### `allocator/dynamics.py`

Actuator and plenum dynamics.

Responsibilities:

- rate-limit vane and tangential commands
- model plenum lag
- shape achieved behavior over time rather than assuming instantaneous response

### `allocator/faults.py`

Fault injection and degraded-behavior helpers.

Representative fault types:

- stuck vane
- slow vane
- weak vane
- biased vane
- dead fan group
- degraded plenum sector

### `allocator/field.py`

Field-based force shaping.

Responsibilities:

- repel-field logic
- spatial force generation around obstacles or field centers
- maneuver shaping inputs used by higher-level tests

### `allocator/sim_runtime.py`

Shared simulation runtime helpers.

Responsibilities:

- define `SimParams` and `SimState`
- propagate vehicle state
- rate-limit XY commands and clip lateral force
- append common engineering telemetry
- translate between segment and fan-group views when needed

### `allocator/coordinate_mission.py`

Coordinate mission orchestration.

Responsibilities:

- run routed coordinate missions
- coordinate the route, allocator, actuator, and power loops for mission execution
- record mission telemetry and outputs

### `allocator/coordinate_support.py`

Pure coordinate-mission support helpers.

Responsibilities:

- initialize coordinate mission defaults, topology/effectiveness, and planner state
- advance routed subgoals and resolve active route geometry
- apply budget-aware speed and altitude shaping
- manage arrival/hold state and phase classification
- build packaged mission outputs

### `allocator/coordinate_execution.py`

Coordinate mission execution helpers.

Responsibilities:

- apply allocation, faults, actuator limits, plenum lag, and power effects for one mission step
- evaluate achieved forces and propagate the vehicle state
- append per-step mission telemetry and engineering history

### `allocator/sim.py`

Compatibility fa?ade for the coordinate mission entry point and shared sim types.

### `allocator/maneuver_scenarios.py`

Maneuver scenario orchestration.

Responsibilities:

- run demo, step, step-snap, step-redirect, and repel scenarios
- shape maneuver commands and phase logic around the shared execution path
- record maneuver-specific telemetry and summaries

### `allocator/maneuver_execution.py`

Shared maneuver execution helpers.

Responsibilities:

- apply allocation, faults, actuator limits, plenum lag, and power effects for one maneuver step
- evaluate achieved forces and propagate vehicle state during stateful maneuvers
- keep stateful maneuver execution logic consistent across step, snap, redirect, and repel paths

### `allocator/maneuver_support.py`

Shared maneuver setup and shaping helpers.

Responsibilities:

- initialize common stateful maneuver defaults and topology/effectiveness state
- provide reusable turn-geometry helpers for redirect and snap maneuvers
- centralize small maneuver math helpers that should not live inside individual scenarios

### `allocator/power_system.py`

Electrical, burst-power, thermal, and motion-budget modeling.

Responsibilities:

- define `PowerSystemParams` and `PowerSystemState`
- estimate battery, burst-power, and thermal limitations
- compute motion guard budgets from power/fault state
- apply supply, spool, burst, and thermal effects to thrust

Representative scenario families:

- demo runs
- step response
- step redirect
- step snap
- repel
- coordinate missions

### `allocator/metrics.py`

Derived motion and coupling metrics.

Responsibilities:

- compute quality measures such as track / yaw coupling
- quantify maneuver behavior for engineering review

### `allocator/response.py`

Step-response and maneuver timing metrics.

Responsibilities:

- compute stop time, reversal time, alignment time, and related metrics
- provide repeatable evaluation logic for directional transitions

### `allocator/trace.py`

Trace serialization utilities.

Responsibilities:

- write time-series outputs for replay, dashboards, or bridge conversion
- normalize scenario outputs into portable artifacts

### `allocator/engineering.py`

Engineering assessment and tuning helpers.

Responsibilities:

- evaluate traces against engineering criteria
- summarize scenario quality
- support parameter tuning for mission or maneuver behavior
- distinguish continuous-power stress from bounded burst-power use

## Scripts and supporting paths

### `scripts/trace_to_tlog.py`

Direct replay path for Mission Planner.

Responsibilities:

- convert Aurora traces into MAVLink telemetry and state replay
- synthesize takeoff, flight, landing, and grounded hold behavior for visualization
- expose engineering telemetry into Mission Planner-friendly channels

### `runs/`

Working artifact directory.

Typical contents:

- exported traces
- assessment reports
- sweep reports
- bridge scripts
- maneuver pack summaries
- Mission Planner waypoint files

## Cross-module flow

A typical Aurora execution path looks like this:

1. A CLI command selects a scenario or workflow.
2. Guidance or maneuver logic defines intent.
3. The vehicle controller converts intent into a desired wrench.
4. The allocator distributes commands across the ring segments.
5. Dynamics and faults shape achieved response.
6. Simulation propagates the vehicle state.
7. Metrics and engineering helpers score the outcome.
8. Trace, replay, or bridge tooling exports the result.

## Current strengths and current simplifications

### Strong today

- explicit controller seam between guidance and allocation
- typed software contracts
- topology-aware fault modeling
- mission-scale simulation, not just isolated maneuvers
- replay, bridge, and SITL tooling as first-class engineering paths

### Still simplified

- topology is explicit but still low-fidelity
- controller focus is strongest in planar motion and yaw-hold rather than full 6-DoF control
- bridge truth is not hardware truth
- plant behavior is richer than before, but still not CFD or certification-grade simulation
