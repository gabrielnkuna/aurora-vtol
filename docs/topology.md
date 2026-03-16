# Aurora Topology

This document describes the current hardware-facing topology assumptions encoded in Aurora.

Related docs:

- [Architecture](architecture.md)
- [Module Reference](module_reference.md)
- [Fault Model](fault_model.md)
- [Trace Schema](trace_schema.md)

## Purpose

Aurora models the vehicle as a segmented ring with explicit fan-group and plenum-sector ownership. The topology layer exists so that actuator coupling, grouped degradation, and authority scaling are not hidden inside the allocator or simulation loops.

The implementation lives in [`src/aurora_gates/topology.py`](../src/aurora_gates/topology.py).

## Current topology objects

### `RingActuatorTopology`

Represents the structural mapping between segments, fan groups, and plenum sectors.

Current fields:

- `segment_count`
- `fan_to_segments`
- `plenum_to_segments`

Key helpers:

- `fan_segments(fan_index)`
- `plenum_segments(plenum_index)`
- `segment_values_to_fan_means(values)`
- `distribute_fan_means_to_segments(fan_mean_n, segment_targets_n)`
- `effectiveness_map(fault)`
- `segment_effectiveness_scales(fault)`

### `RingEffectivenessMap`

Represents the resolved segment-level effectiveness after topology and fault effects are combined.

Current fields:

- `fan_index_by_segment`
- `plenum_index_by_segment`
- `fan_scale_by_segment`
- `plenum_scale_by_segment`
- derived `segment_scale`

## Current default Aurora mapping

Aurora's default ring assumes:

- `32` control segments around the ring
- `16` fan groups
- `32` plenum sectors

### Fan mapping

The current default mapping is a simple even-pair ownership model:

```text
fan 00 -> segments 00, 01
fan 01 -> segments 02, 03
fan 02 -> segments 04, 05
...
fan 15 -> segments 30, 31
```

In code, this is `AURORA_FAN_TO_SEGMENTS_32`.

### Plenum mapping

The current plenum mapping is one plenum sector per control segment:

```text
plenum 00 -> segment 00
plenum 01 -> segment 01
...
plenum 31 -> segment 31
```

In code, this is `AURORA_PLENUM_TO_SEGMENTS_32`.

## Fallback topology

If a simulation or trace uses a segment count other than `32`, Aurora falls back to `even_pairs(segment_count)`, which:

- groups segments into fan pairs in order
- assigns one plenum sector per segment

That is useful for generic ring experiments, but it should be treated as a modeling convenience, not as Aurora hardware truth.

## How topology affects the rest of the stack

Topology is used in three important ways.

### 1. Fault ownership

Dead-fan and plenum faults do not reduce the whole ring uniformly. They reduce only the segments owned by the affected fan group or plenum sector.

### 2. Command aggregation

Many plant and telemetry paths carry segment-level values but summarize fan behavior at the 16-group level. `segment_values_to_fan_means(...)` is the explicit conversion point.

### 3. Redistribution

Some plant paths start from fan-group means and then redistribute them back to segments using the current segment targets. `distribute_fan_means_to_segments(...)` is the current implementation for that step.

## Current engineering assumptions

The topology layer is explicit, which is good, but it is still simple. Current assumptions include:

- each fan group influences exactly two segments
- each plenum sector influences exactly one segment
- there is no cross-sector mixing model in `topology.py`
- no geometry-derived effectiveness matrix exists yet
- the mapping is ordinal, not yet CAD-derived

This means the topology layer is strong enough for grouped authority studies and fault ownership, but it is not yet a high-fidelity aerodynamic or ducting model.

## What is already useful about the topology layer

Even with its current simplicity, the topology layer gives Aurora three real engineering advantages:

- grouped faults are explicit instead of implicit
- allocator and simulation code share one ownership model
- hardware-facing assumptions now have a stable place in the repo

That makes it much easier to evolve toward a more physical fan-to-vane or plenum-effectiveness map later.

## What should evolve next

The next topology upgrades should be:

1. replace the simple pair mapping with hardware-derived ownership where needed
2. add geometry-aware effectiveness weights instead of pure ownership scalars
3. distinguish structural ownership from aerodynamic influence
4. encode any real plenum cross-coupling explicitly rather than assuming one-sector isolation
5. freeze the hardware-facing mapping once the mechanical design is mature enough

## Truth boundary

The current topology is an engineering model. It is not yet proof of final packaging, final duct routing, or final aerodynamic coupling.
