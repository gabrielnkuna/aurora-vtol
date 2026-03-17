# Aurora Effectiveness Map

This document defines Aurora's current nominal effectiveness-table interface.

Related docs:

- [Topology](topology.md)
- [Fault Model](fault_model.md)
- [Architecture](architecture.md)
- [Module Reference](module_reference.md)
- [Limitations](limitations.md)

## Purpose

Aurora now has an explicit nominal effectiveness-table layer between topology assumptions and allocator force evaluation.

The goal is simple: nominal authority should come from a named, versioned data structure rather than from hidden math inside the allocator.

That keeps the project honest in two ways:

- the allocator can be driven by a hardware-facing table
- provisional assumptions stay visible until they are replaced by CAD, CFD, or measured data

The implementation lives in [`src/aurora_vtol/effectiveness.py`](../src/aurora_vtol/effectiveness.py).

## Current default asset

The current default Aurora source asset is:

- [`data/effectiveness_specs/aurora_ring32_geometry_seed_v1.json`](../data/effectiveness_specs/aurora_ring32_geometry_seed_v1.json)

Aurora builds the nominal effectiveness table from that geometry-seeded spec at runtime.

It is explicitly provisional.

Its provenance is:

- seeded from explicit geometric assumptions about fan footprint centers, spans, and spillover width
- not derived from CAD
- not derived from CFD
- not derived from measured hardware data

## Schema

Aurora now uses two related structures.

### Geometry seed spec

Current fields:

- `schema_version`
- `spec_name`
- `segment_count`
- `fan_center_deg`
- `fan_half_span_deg`
- `fan_sigma_deg`
- `plenum_center_deg`
- `plenum_half_span_deg`
- `plenum_sigma_deg`
- `axial_scale_by_segment`
- `radial_scale_by_segment`
- `tangential_scale_by_segment`
- `provenance`

### Generated nominal effectiveness table

Current fields:

- `schema_version`
- `table_name`
- `segment_count`
- `fan_count`
- `plenum_count`
- `fan_segment_weights`
- `plenum_segment_weights`
- `axial_scale_by_segment`
- `radial_scale_by_segment`
- `tangential_scale_by_segment`
- `provenance`

## What the table controls today

### Fan-to-segment coupling

The geometry seed spec defines fan footprint centers, half-spans, and Gaussian spillover widths. Aurora converts that into `fan_segment_weights`, which define how segment-level commanded values are aggregated to fan-group means and smoothed back onto the ring for nominal authority evaluation.

### Component effectiveness

Aurora currently distinguishes three nominal component channels:

- axial effectiveness
- radial effectiveness
- tangential effectiveness

Those are represented as per-segment scales:

- `axial_scale_by_segment`
- `radial_scale_by_segment`
- `tangential_scale_by_segment`

In the current provisional geometry-seeded table, all three are unity. That means the current default still changes nominal coupling structure more than per-segment component gain.

### Plenum coupling placeholder

`plenum_segment_weights` is included now so the schema already has a place for future nominal plenum-distribution assumptions.

In the current provisional table, it is effectively identity ownership.

## CLI workflow

Aurora now has a dedicated CLI workflow for inspecting the current nominal authority assumptions as artifacts instead of only code:

```bash
uv run aurora-vtol alloc effectiveness-report \
  --out-dir runs/effectiveness_report_geometry_seed
```

That writes:

- `summary.json`
- `summary.md`
- `materialized_table.json`
- `source_spec.json` when the source is a geometry seed

This workflow is meant to make the current mechanics-facing assumptions reviewable. It does not turn the current provisional geometry seed into validated hardware truth.

## How the allocator uses it today

The allocator now uses the explicit table for nominal force evaluation in [`src/aurora_vtol/allocator/allocate.py`](../src/aurora_vtol/allocator/allocate.py).

Current use:

- axial nominal force smoothing
- radial nominal force smoothing
- tangential nominal per-segment effectiveness

Faulted authority still goes through the topology and fault-effectiveness path. The nominal table does not replace the fault model; it provides the nominal authority source that the fault model modifies later.

## Fallback behavior

For segment counts other than the default Aurora `32`-segment ring, Aurora falls back to a seeded table built from the current topology helper instead of the geometry spec asset.

That fallback is useful for generic experiments, but it is modeling convenience, not hardware truth.

## Current engineering meaning

The effectiveness-table layer is a bridge step.

It is more explicit than a hidden nominal smoother, and stronger than a pure topology-seeded table, but it is still not final physics.

Today it proves:

- the allocator can consume a versioned hardware-facing authority table
- nominal authority assumptions can be separated from fault logic
- geometric assumptions can live on disk instead of being hidden inside allocator code
- future CAD or bench data has a clean place to land in the software stack

It does not yet prove:

- the numbers in the current default JSON are physically validated
- the current table matches real duct, vane, or plenum geometry
- the current table captures full 3D flow behavior

## Recommended next upgrades

1. replace the provisional geometry-seeded spec with a CAD-, CFD-, or test-derived spec
2. distinguish fan-driven axial authority from vane-driven radial authority more physically
3. make plenum nominal coupling explicit instead of identity-only
4. add version comparison and reporting so traces state which nominal table or seed spec was used
5. import future CAD, CFD, or bench-derived coefficients without changing allocator interfaces

## Truth boundary

The current effectiveness map is an engineering interface and a geometry-seeded nominal authority model.

It is not yet a validated aerodynamic truth source.
