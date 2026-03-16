# Aurora Interface Control Document (ICD)

This document defines the main software handoff objects used across the Aurora flight stack. In the current repo, these contracts live in [`src/aurora_vtol/icd.py`](../src/aurora_vtol/icd.py) and formalize how guidance, controller, allocator, and bridge-adjacent layers exchange state and intent.

The goal of this ICD is to make each layer independently testable while keeping units, ownership, and truth boundaries explicit.

Related docs:

- [Architecture](architecture.md)
- [Module Reference](module_reference.md)
- [Engineering Notes](engineering_notes.md)

## Design intent

Aurora is structured around a layered control path:

1. state estimation and health assessment
2. guidance target generation
3. vehicle controller
4. allocator
5. actuator execution, simulation, and bridge output

The dataclasses in `icd.py` are the narrow interfaces between those layers.

## Units and conventions

Unless otherwise noted:

- position uses meters (`_m`)
- velocity uses meters per second (`_mps`)
- yaw uses degrees in the ICD objects (`_deg`)
- forces use newtons (`_n`)
- moments use newton-meters (`_nm`)
- vane angles use radians in allocator commands (`_rad`)
- battery state of charge uses percent (`_pct`)

Coordinate intent in the current software is primarily planar XY motion with an explicit yaw-hold concept. The bridge layer converts the internal trace convention into MAVLink local NED setpoints when exporting or streaming.

## 1. EstimatedVehicleState

### Purpose

Published by the estimation or plant side. Consumed by the vehicle controller and any mission logic that needs the current state.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `x_m` | `float` | Current X position in meters |
| `y_m` | `float` | Current Y position in meters |
| `z_m` | `float` | Current Z position in meters |
| `vx_mps` | `float` | Current X velocity |
| `vy_mps` | `float` | Current Y velocity |
| `vz_mps` | `float` | Current Z velocity |
| `yaw_deg` | `float` | Current yaw angle in degrees |
| `yaw_rate_deg_s` | `float` | Current yaw rate in degrees per second |
| `battery_soc_pct` | `float | None` | Battery state of charge if available |
| `bus_voltage_v` | `float | None` | DC bus voltage if available |
| `continuous_power_ratio` | `float | None` | Ratio against continuous power limit |
| `thermal_scale_pct` | `float | None` | Thermal derating scale in percent |
| `fault_available_scale` | `float | None` | Effective remaining actuator availability |
| `fault_asymmetry_pct` | `float | None` | Sector authority asymmetry indicator |

### Ownership notes

This object should represent the best available estimate, not a raw sensor packet. In simulation, it may come directly from the plant model. In future hardware, it should come from a fused estimator.

## 2. GuidanceTarget

### Purpose

Represents a position-tracking guidance objective.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `goal_x_m` | `float` | Target X position |
| `goal_y_m` | `float` | Target Y position |
| `goal_z_m` | `float` | Target Z position |
| `desired_vx_mps` | `float` | Desired X velocity |
| `desired_vy_mps` | `float` | Desired Y velocity |
| `desired_vz_mps` | `float` | Desired Z velocity |
| `hold_yaw_deg` | `float` | Desired yaw hold angle |
| `mode` | `str` | Guidance mode label, default `track-position` |
| `force_scale` | `float` | Scalar to soften or harden controller output |

### Typical producer and consumer

Produced by mission or guidance logic. Consumed by `track_xy_position(...)` in [`src/aurora_vtol/vehicle_controller.py`](../src/aurora_vtol/vehicle_controller.py).

## 3. RedirectTarget

### Purpose

Represents a velocity-redirection style guidance command rather than a point-tracking command.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `desired_ux` | `float` | Desired unit direction X component |
| `desired_uy` | `float` | Desired unit direction Y component |
| `desired_speed_mps` | `float` | Desired along-track speed |
| `reference_speed_mps` | `float` | Reference speed used for normalization |
| `cross_gain` | `float` | Gain on cross-track correction |
| `hold_yaw_deg` | `float` | Desired yaw hold |
| `phase` | `str` | Maneuver phase label |
| `force_scale` | `float` | Scalar on controller output |

### Typical producer and consumer

Produced by redirect-style maneuver logic. Consumed by `track_redirect_velocity(...)` in [`src/aurora_vtol/vehicle_controller.py`](../src/aurora_vtol/vehicle_controller.py).

## 4. ActuatorHealthState

### Purpose

Represents the controller-facing actuator budget and health summary.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `lateral_budget_n` | `float` | Available lateral force budget |
| `guard_scale` | `float` | Generic guard scale on aggressive behavior |
| `response_scale` | `float` | Effective actuator response scale |
| `continuous_power_ratio` | `float` | Current continuous-power usage ratio |
| `thermal_scale_pct` | `float` | Thermal derating scale in percent |
| `supply_scale_pct` | `float` | Supply availability or scaling percentage |
| `fault_available_scale` | `float` | Remaining overall actuator availability |
| `fault_asymmetry_pct` | `float` | Current asymmetry indicator |

### Design intent

This object lets the controller react to power, thermal, or fault pressure before the allocator clips late.

## 5. DesiredWrench

### Purpose

Represents the controller output into the allocator.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fx_n` | `float` | Requested X force |
| `fy_n` | `float` | Requested Y force |
| `fz_n` | `float` | Requested Z force |
| `mz_nm` | `float` | Requested yaw moment |
| `source` | `str` | Provenance string for debugging and telemetry |

### Ownership notes

The controller owns this object. The allocator should consume it and decide how to realize it, not reinterpret mission intent.

## 6. AllocatorCommand

### Purpose

Represents the allocator output into the actuator or plant side.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fan_thrust_cmd_n_16` | `tuple[float, ...]` | Commanded fan thrust per fan group |
| `vane_angle_cmd_rad_32` | `tuple[float, ...]` | Commanded vane angle per segment |
| `tangential_cmd_n_32` | `tuple[float, ...]` | Tangential or yaw-related effort per segment |
| `note` | `str` | Optional note for diagnostics |

## Ownership summary

The intended software ownership is:

```text
Estimator or plant publishes EstimatedVehicleState
Guidance publishes GuidanceTarget or RedirectTarget
Controller publishes DesiredWrench
Allocator publishes AllocatorCommand
Dynamics, simulation, replay, or bridge consume AllocatorCommand
```

Keeping these boundaries explicit is one of the key steps toward a more modular Aurora stack.
