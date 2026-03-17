# Aurora Architecture

## 1. Purpose

Aurora is a software-first control architecture for a 16-fan / 32-vane ring vehicle. The current repository models the flight-control, maneuver-validation, bridge, and simulation side of the concept rather than a final aircraft.

At the repo level, the architecture is built around this layered flow:

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
Actuator Commands
        |
        v
Dynamics / Faults / Plenum / Power Limits
        |
        v
Plant / Mission / Engineering Assessment
        |
        v
Trace / Replay / Bridge / SITL
```

Companion docs:

- [Module Reference](module_reference.md)
- [Interface Control Document](icd.md)
- [Bridge Workflows](bridge_workflows.md)
- [Command Reference](command_reference.md)
- [Topology](topology.md)
- [Effectiveness Map](effectiveness_map.md)
- [Fault Model](fault_model.md)
- [Hardware Control Architecture](hardware_control_architecture.md)

## 2. Layer descriptions

### 2.1 Mission / Guidance

This layer decides what the craft should do.

Typical outputs:

- desired position
- desired velocity
- desired acceleration or force scale
- desired yaw-hold or redirect state
- maneuver mode such as hover, redirect, repel, snap-stop, or landing

Representative guidance behaviors show up in the repo through commands such as `step`, `repel`, `step-redirect`, `step-snap`, and `coordinates`.

### 2.2 Vehicle Controller

This layer converts mission intent into a desired generalized wrench.

Typical outputs:

- `Fx`
- `Fy`
- `Fz`
- `Mz`

The controller decides things like:

- move sideways without yawing
- brake hard while staying aligned
- reverse into a new direction
- respect available lateral authority

This layer keeps mission logic out of the allocator. The current implementation is in [`src/aurora_vtol/vehicle_controller.py`](../src/aurora_vtol/vehicle_controller.py).

### 2.3 Allocator

The allocator distributes requested force and moment across the ring actuators.

Inputs:

- desired wrench
- actuator health
- topology effectiveness
- power and thermal limits

Outputs:

- vane angle commands
- tangential or yaw-related commands
- fan-group or segment-level authority distribution

This remains the mathematical core of Aurora and lives primarily in [`src/aurora_vtol/allocator/allocate.py`](../src/aurora_vtol/allocator/allocate.py).

### 2.4 Actuator Dynamics and Faults

This layer models the gap between ideal commands and achieved behavior.

Included behaviors:

- vane rate limits
- tangential effort rate limits
- plenum lag
- stuck, slow, weak, and biased vane faults
- dead fan groups
- degraded plenum sectors

This is where Aurora becomes more physically honest than a perfect instantaneous solver. See [`src/aurora_vtol/allocator/dynamics.py`](../src/aurora_vtol/allocator/dynamics.py) and [`src/aurora_vtol/allocator/faults.py`](../src/aurora_vtol/allocator/faults.py).

### 2.5 Plant / Mission / Power Simulation

This layer estimates what happens when allocator outputs are applied.

Representative responsibilities:

- state propagation
- obstacle-aware mission shaping
- battery, burst-power, and thermal scaling
- maneuver scoring and engineering telemetry
- trace export

The main implementation is split between [`src/aurora_vtol/allocator/sim_runtime.py`](../src/aurora_vtol/allocator/sim_runtime.py) for shared runtime state and stepping helpers, [`src/aurora_vtol/allocator/coordinate_mission.py`](../src/aurora_vtol/allocator/coordinate_mission.py) for coordinate mission flow, [`src/aurora_vtol/allocator/coordinate_support.py`](../src/aurora_vtol/allocator/coordinate_support.py) for route and mission-support helpers, [`src/aurora_vtol/allocator/coordinate_execution.py`](../src/aurora_vtol/allocator/coordinate_execution.py) for mission-step actuation and achieved-force evaluation, [`src/aurora_vtol/allocator/maneuver_scenarios.py`](../src/aurora_vtol/allocator/maneuver_scenarios.py) for maneuver scenario loops, [`src/aurora_vtol/allocator/maneuver_execution.py`](../src/aurora_vtol/allocator/maneuver_execution.py) for shared stateful maneuver execution, [`src/aurora_vtol/allocator/maneuver_support.py`](../src/aurora_vtol/allocator/maneuver_support.py) for shared maneuver bootstrap, control-context, history, and turn-shaping helpers, [`src/aurora_vtol/allocator/power_system.py`](../src/aurora_vtol/allocator/power_system.py) for the electrical/burst/thermal model, and [`src/aurora_vtol/allocator/engineering.py`](../src/aurora_vtol/allocator/engineering.py) for assessment.

### 2.6 Bridge / Replay / SITL

This layer converts simulation traces into external workflows.

Use cases:

- inspect and summarize traces
- export bridge-ready JSONL scripts
- visualize scripts
- stream setpoints over MAVLink
- integrate with Mission Planner and ArduPilot SITL
- replay traces directly into Mission Planner using `scripts/trace_to_tlog.py`

The bridge path is in [`src/aurora_vtol/bridge.py`](../src/aurora_vtol/bridge.py). The direct Mission Planner replay path is in [`scripts/trace_to_tlog.py`](../scripts/trace_to_tlog.py).

## 3. Main data flow

The intended software flow now looks like this:

```text
guidance target or maneuver
        |
        v
vehicle controller
        |
        v
desired wrench
        |
        v
allocator
        |
        v
segment thrust + vane angles + tangential commands
        |
        v
actuator dynamics + faults + plenum and power limits
        |
        v
plant / mission / assessment
        |
        v
telemetry + trace export + bridge output
```

That split is now reflected in the repo structure through [`src/aurora_vtol/icd.py`](../src/aurora_vtol/icd.py), [`src/aurora_vtol/vehicle_controller.py`](../src/aurora_vtol/vehicle_controller.py), [`src/aurora_vtol/topology.py`](../src/aurora_vtol/topology.py), and [`src/aurora_vtol/effectiveness.py`](../src/aurora_vtol/effectiveness.py).

## 4. Topology model

Aurora currently uses an explicit ring topology built around:

- 16 fan groups
- 32 vane / segment commands
- plenum-sector effectiveness scaling

The current topology model is a strong intermediate step between pure abstraction and full hardware fidelity. It provides a place to encode:

- fan-to-segment influence
- segment-level authority reduction
- plenum-sector degradation
- grouped fault behavior

The current default mapping is defined in [`src/aurora_vtol/topology.py`](../src/aurora_vtol/topology.py). It is explicit, but still simple. It should be treated as the current engineering assumption, not frozen hardware truth.

## 5. Control ownership for future hardware

If Aurora becomes a hardware vehicle, the recommended ownership split is:

### Central flight computer

- mission and guidance
- state estimation
- fault management
- vehicle control
- allocation
- safety supervision

### Fan control electronics

- ESC interface
- RPM, current, and temperature feedback
- local watchdogs

### Vane actuator electronics

- local position loops
- jam detection
- calibration and homing

### Power system

- BMS
- bus monitoring
- thermal protection
- contactors and power zoning

This means the vanes are not autonomous in the system sense. They are commanded centrally and closed locally.

## 6. What the architecture proves vs does not prove

### Supported by the repo today

- allocator-driven lateral maneuvering is meaningful in software
- yaw-aware force distribution can be simulated
- faults and degraded authority can be studied
- power-aware and burst-aware maneuver behavior can be approximated
- trace replay, bridge, and SITL workflows can be exercised

### Not yet validated

- exact aerodynamic performance
- final duct or plenum geometry
- final mechanical packaging
- final electrical zoning and wiring
- certification-grade safety behavior

## 7. Engineering guardrails

Aurora should be presented honestly as a software and systems-engineering prototype. Mechanics, actuator truth, and bridge truth outrank visuals.

For the review standard we should use on every major change, see [Engineering Review Checklist](engineering_review_checklist.md).
