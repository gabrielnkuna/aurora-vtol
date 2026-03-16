# Aurora Fault Model

This document describes the currently supported degraded modes in Aurora and how they are applied in the software stack.

Related docs:

- [Architecture](architecture.md)
- [Topology](topology.md)
- [Interface Control Document](icd.md)
- [Engineering Notes](engineering_notes.md)

## Purpose

Aurora treats faults as first-class engineering inputs, not as afterthoughts. The current fault model is designed to answer questions like:

- what happens if one vane jams or becomes weak?
- what happens if one fan group loses authority?
- what happens if one plenum sector degrades?
- can the controller and allocator degrade gracefully?

The main implementation lives in [`src/aurora_vtol/allocator/faults.py`](../src/aurora_vtol/allocator/faults.py), with topology effects in [`src/aurora_vtol/topology.py`](../src/aurora_vtol/topology.py).

## Core fault object

Aurora currently represents fault conditions with `FaultSpec`.

Current fields:

- `stuck_flap_idx`
- `stuck_flap_alpha_deg`
- `slow_flap_idx`
- `slow_flap_rate_scale`
- `weak_flap_idx`
- `weak_flap_scale`
- `biased_flap_idx`
- `biased_flap_offset_deg`
- `dead_fan_group`
- `dead_fan_scale`
- `plenum_sector_idx`
- `plenum_sector_scale`

Not every workflow uses every field, but this is the current supported degraded-mode vocabulary.

## Supported fault families

### 1. Stuck flap

A vane or flap is forced to a fixed angle.

Current fields:

- `stuck_flap_idx`
- `stuck_flap_alpha_deg`

Effect:

- the selected segment is clamped to the requested stuck angle
- both command-side and achieved-side paths can reflect the fault

### 2. Slow flap

A vane still moves, but more slowly than nominal.

Current fields:

- `slow_flap_idx`
- `slow_flap_rate_scale`

Effect:

- actuator dynamics for the selected vane are rate-limited more aggressively
- this is mainly a dynamics-path fault rather than a pure allocator-path fault

### 3. Weak flap

A vane produces less achieved effect than requested.

Current fields:

- `weak_flap_idx`
- `weak_flap_scale`

Effect:

- command-side compensation can ask for more on the affected segment
- achieved-side behavior still reflects reduced authority

### 4. Biased flap

A vane has a position or effectiveness offset.

Current fields:

- `biased_flap_idx`
- `biased_flap_offset_deg`

Effect:

- command-side and achieved-side paths treat the offset differently so the controller can try to compensate while the plant still carries the bias

### 5. Dead fan group

A fan group loses some or all thrust authority.

Current fields:

- `dead_fan_group`
- `dead_fan_scale`

Effect:

- the topology layer reduces the authority of the owned segments most strongly and applies a smaller local angular spillover to neighboring segments on the default 32-segment Aurora ring
- this currently works through segment effectiveness scaling, not through a detailed motor or ESC failure model

### 6. Degraded plenum sector

A plenum sector loses some of its local effectiveness.

Current fields:

- `plenum_sector_idx`
- `plenum_sector_scale`

Effect:

- the topology layer reduces authority on the affected sector most strongly and applies a smaller local spillover to adjacent sectors on the default 32-segment Aurora ring
- this represents local flow degradation, not full fluid-dynamic truth

## Command-side vs achieved-side behavior

Aurora deliberately distinguishes between two kinds of fault application.

### Command-side application

Used when the controller or allocator should be allowed to compensate.

Example:

- for weak or biased flaps, the command-side path can request a modified command so the achieved behavior is closer to the target

Relevant function:

- `apply_command_faults_to_alpha(...)`

### Achieved-side application

Used when the plant should reflect the degraded real behavior.

Examples:

- weak flap reduces achieved alpha
- biased flap adds achieved offset
- stuck flap clamps the achieved angle
- fan or plenum degradation reduces achieved thrust authority

Relevant functions:

- `apply_faults_to_alpha(...)`
- `apply_faults_to_thrust(...)`

That split is important. It keeps Aurora from hiding faults by only changing the achieved state after the allocator has already assumed nominal authority.

## Topology interaction

Dead-fan and plenum faults are topology-aware.

That means:

- `dead_fan_group` reduces the owned segments most strongly, with limited angular spillover to nearby segments on the default 32-segment Aurora ring
- `plenum_sector_idx` reduces the owned sector most strongly, with limited local spillover to nearby sectors on the default 32-segment Aurora ring

This behavior comes from the topology layer rather than ad hoc fault math scattered across the simulator.

## What the current model does well

The current fault model is good for:

- grouped authority reduction studies
- degraded maneuver comparisons
- fault-envelope and threshold workflows
- controller and allocator robustness work

## What the current model does not yet include

The current model does not yet cover:

- backlash
- deadband
- hysteresis
- multi-node electrical failures beyond grouped scaling
- detailed ESC faults
- detailed servo current or thermal faults
- structural damage or geometry changes
- CFD-accurate plenum cross-coupling

So the fault model is already useful and honest, but it is still a control-oriented fault model, not a full hardware reliability model.

## Practical use in the CLI

Faults are exercised through allocator commands such as:

- `aurora-vtol alloc repel`
- `aurora-vtol alloc step`
- `aurora-vtol alloc step-redirect`
- `aurora-vtol alloc step-snap`
- `aurora-vtol alloc maneuver-pack`
- `aurora-vtol alloc fault-envelope`
- `aurora-vtol alloc fault-threshold`

Representative examples:

```bash
uv run aurora-vtol alloc step-redirect --fault-case biased-flap-0 --dir-b-deg 90
uv run aurora-vtol alloc step-snap --fault-case dead-fan-0 --dir-b-deg 180
uv run aurora-vtol alloc fault-envelope --preset medium --out-dir runs/fault_envelope_medium
```

## Truth boundary

Aurora's fault model is a software and control-engineering fault model. It is not yet a full FMEA, hardware reliability model, or certification-grade safety case.
