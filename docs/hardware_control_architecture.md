# Aurora Hardware Control Architecture

This document maps the current allocator software to a plausible real vehicle
control architecture for the Project Aurora concept.

It is intentionally split into two views:

- "Modeled now": what the codebase actually represents today.
- "Recommended hardware architecture": what a real vehicle would likely need.

This is not a final mechanical design or certified avionics architecture.

## Engineering Guardrails

These points are intentional constraints on how Aurora should be developed and
judged:

- Mechanics win over visuals. A maneuver is only credible if the actuator,
  power, thermal, and fault models can support it, not just because a replay or
  dashboard looks convincing.
- Bridge layers are first-class engineering components. The Mission Planner
  replay path, SITL bridge, and future vehicle bridge must preserve units,
  timing, health, limits, and failsafe behavior. They are not "just plumbing."
- Software-first does not mean physics-last. It is acceptable to build the
  guidance, allocator, and controller stack before hardware exists, but the
  software must stay anchored to mechanical and electrical reality.
- Modeled now vs real later must stay explicit. If something is synthetic,
  assumed, or concept art, it should be labeled that way rather than presented
  as validated hardware truth.
- Honesty is part of the architecture. If a result is limited by continuous
  power, actuator authority, structural assumptions, or bridge behavior, that is
  the engineering result, not a presentation problem to hide.


See also [engineering_review_checklist.md](engineering_review_checklist.md) for the review standard we should apply to control, plant, and bridge changes.

## Modeled Now

The current software stack models the vehicle at the force-allocation level.

- `guidance` chooses the maneuver or mission-level motion objective.
- `allocator` converts requested body forces and yaw moment into ring actuator
  commands.
- `actuator dynamics` apply rate limits, lag, and faults to the commands.
- `plant/power` estimates what motion and electrical loading result.

Relevant code:

- [model.py](../src/aurora_vtol/allocator/model.py)
- [allocate.py](../src/aurora_vtol/allocator/allocate.py)
- [dynamics.py](../src/aurora_vtol/allocator/dynamics.py)
- [sim.py](../src/aurora_vtol/allocator/sim.py)

### Flaps / Vector Vanes

The software explicitly models `32` ring segments / vector vanes.

- The allocator accepts a requested body force and yaw moment.
- It computes vane-angle commands per segment.
- Those commands are then rate-limited and faulted in actuator dynamics.

In practical terms, the flaps are controlled by the allocator. There is not a
separate high-fidelity vane controller model yet.

Current software responsibilities:

- `allocator`: computes desired vane angles and thrust distribution
- `dynamics`: applies rate limits, lag, weak/slow/stuck/biased flap behavior
- `sim`: evaluates resulting maneuver quality, power, and fault tolerance

### Fans

The concept being developed is `16` ducted fans feeding a ring / plenum system.

What is represented today:

- force/thrust distribution behavior
- grouped fan effects in the power and fault model
- dead-fan and degraded fan scenarios
- spool response and power limits

What is not yet fully represented:

- exact duct geometry
- exact motor/ESC wiring
- detailed fan CFD or plenum pressure distribution
- exact mapping from each physical fan to local vane effectiveness

### Sensors and Avionics

The current repo uses sensor concepts in the guidance and replay sense, not as
full embedded software modules.

The codebase assumes a real craft would need:

- IMU
- GNSS / RTK GNSS
- barometric altitude
- obstacle sensing
- local position sensing for precision hold

But exact avionics placement, buses, update rates, and redundancy management
are not yet fully implemented as embedded subsystems.

## Recommended Hardware Control Architecture

For a real vehicle, the clean control stack is:

1. Mission / Guidance
2. Vehicle Controller
3. Allocator
4. Actuator Controllers
5. Power Management
6. State Estimation / Fault Management

### 1. Mission / Guidance

Inputs:

- destination or maneuver command
- obstacle map
- mission mode
- geofence / safety constraints

Outputs:

- desired position / velocity / acceleration
- maneuver state such as hover, redirect, snap-stop, landing

### 2. Vehicle Controller

Inputs:

- estimated state
- guidance target

Outputs:

- desired generalized forces and moments:
  - `Fx`
  - `Fy`
  - `Fz`
  - `Mz`

This is the layer that says, "move right without yawing" or "stop lateral
motion while holding heading."

### 3. Allocator

Inputs:

- desired `Fx/Fy/Fz/Mz`
- actuator availability and health
- power / thermal limits

Outputs:

- `16` fan thrust commands or grouped fan requests
- `32` vane angle commands

This is the software layer already closest to what exists in the repo today.

### 4. Actuator Controllers

This is not fully built yet in the repo, but a real craft would likely need:

- fan ESC control path for all `16` fans
- vane servo / actuator controller path for all `32` vanes
- local feedback on actual vane angle
- local feedback on fan RPM / current / temperature

Recommended structure:

- central flight computer computes allocator outputs
- distributed actuator nodes handle local timing and health
- each node reports actual actuator state back upstream

### 5. Power Management

A real Aurora-style craft needs power management as a first-class controller,
not as telemetry only.

Needed functions:

- battery pack monitoring
- bus voltage and current sensing
- continuous and peak power enforcement
- thermal derating
- safe degraded operation after faults

The repo now has an early software model for this, but not a full BMS or power
electronics implementation.

### 6. State Estimation and Fault Management

A real craft would need:

- fused state estimate from IMU, GNSS, barometer, and local sensors
- actuator health estimation
- fault isolation for fans, vanes, and plenum sectors
- degraded-mode control laws

This is the layer that would tell the allocator, for example:

- fan group 3 is unavailable
- vane 8 is slow and biased
- plenum sector 1 has reduced authority

## Recommended Physical Control Ownership

If this concept becomes hardware, the clean ownership is:

- flight computer:
  guidance, state estimation, vehicle control, allocation, safety
- fan control electronics:
  motor commutation, RPM/current/temperature feedback
- vane actuator electronics:
  position loop for each vane, actuator health, jam detection
- power system:
  BMS, contactors, DC-DC, bus monitoring, thermal protection

That means the flaps are not "autonomous" on their own. They are commanded by
the central controller and closed locally by their actuator drives.

## How Accurate the Concept Images Are

The images are best treated as concept architecture art, not validated
engineering drawings.

### Layer 1: Underside

Directionally consistent with the model:

- annular exhaust concept
- `32` vector vanes
- downward flow and vectoring idea

Not yet validated by the repo:

- exact vane hinge geometry
- exact actuator packaging
- exact slit gap dimensions
- exact landing gear integration

### Layer 2: Fan Ring

Directionally consistent with the model:

- `16` fans
- plenum / ring-fed concept
- structural outer ring

Not yet validated by the repo:

- exact fan diameter and spacing
- exact duct cross-sections
- exact ESC placement
- exact plenum equalization behavior

### Layer 3: Main Hull and Battery Bay

Mostly concept only.

The repo does model battery voltage, power draw, and continuous / peak limits,
but it does not validate:

- exact cell layout
- exact `55 kWh` pack sizing
- exact `280 kg` mass claim
- exact bus bar and converter placement

### Layer 4: Avionics Bay

This is partly aligned with what a real vehicle would need, but still concept
art rather than validated layout.

Plausible:

- redundant IMUs
- GNSS / RTK
- barometric redundancy
- lidar / optical flow / thermal sensing

Not yet validated:

- exact board count
- exact processor partitioning
- exact update rates and buses
- exact mounting and isolation

### Layer 5: Upper Dome

Mostly industrial design / packaging concept at this stage.

The repo does not yet validate:

- exact air intake geometry
- debris screen pressure loss
- sensor dome placement
- exact exterior shell construction

## What the Repo Currently Proves

The repo currently supports these high-level claims:

- `16` fan / `32` vane control is a meaningful software concept
- the allocator can produce holonomic-style lateral maneuvers while holding yaw
- maneuver and mission feasibility can be studied under power and fault limits
- degraded cases can be assessed in software before hardware exists

It does not yet prove:

- final mechanical packaging
- final aerodynamic efficiency
- final structural layout
- final sensor / avionics physical design
- certification-grade safety architecture

## Next Engineering Steps

To turn the concept art into a stronger engineering baseline, the next steps are:

1. Define a hard fan-to-vane topology.
   State exactly which fans feed which plenum sectors and vane sectors.

2. Define a real actuator architecture.
   Choose vane actuator type, gearing, position sensing, bus, and update rate.

3. Define avionics partitioning.
   Decide what runs on the central flight computer versus distributed nodes.

4. Define electrical zones.
   Partition battery, power distribution, ESC groups, and fault containment.

5. Add geometry-aware effectiveness maps.
   Replace generic ring assumptions with sector-by-sector actuator authority.

6. Add a CAD-facing interface.
   Turn the control model assumptions into explicit mechanical requirements.

## Short Answer

If asked, "what controls the flaps?" the answer is:

- in software today: the allocator and actuator dynamics model
- in hardware later: the flight controller commands them, local actuator
  electronics execute them

If asked, "are the pictures accurate?" the answer is:

- accurate as concept intent
- not yet accurate as validated engineering truth

