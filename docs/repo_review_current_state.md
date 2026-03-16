# Aurora VTOL Current-State Repo Review

This note is the corrected current-state repo review for Aurora VTOL. It supersedes older migration-era reviews that described the project as partially renamed or inconsistently presented.

## Summary

Aurora VTOL is now presented consistently as a software-first VTOL control, maneuver, and simulation stack.

The repo is currently strongest in these areas:

- control allocation and maneuver generation
- topology-aware actuator modeling
- power-, burst-, and fault-aware plant simulation
- scenario generation, assessment, and sizing workflows
- replay, bridge, Mission Planner, and SITL-oriented tooling
- documentation of truth boundaries and engineering guardrails

The repo is not yet strongest in these areas:

- CAD-validated geometry
- CFD-validated plenum and duct behavior
- bench-validated actuator authority tables
- final electrical and wiring architecture
- certification-grade flight safety design

## What is now clearly true

The repo identity is aligned:

- project name: `aurora-vtol`
- Python package: `src/aurora_vtol`
- console command: `aurora-vtol`
- GitHub repo presentation: Aurora VTOL

The docs set is visible and coherent through the normal repo tree, not just through raw-file links.

The documentation now covers the main engineering seams:

- architecture
- module responsibilities
- command workflows
- topology
- effectiveness map assumptions
- fault model
- trace schema
- scenario guide
- limitations
- ICD
- bridge workflows
- hardware control architecture
- engineering review checklist

The repo structure also now matches that story. The visible source tree uses `src/aurora_vtol`, and the CLI and examples consistently use `aurora-vtol`.

## Engineering improvements that matter

The project story is no longer just "allocator demos." It now reads as a layered control and validation stack:

`guidance -> vehicle controller -> desired wrench -> allocator -> actuator dynamics -> plant/power -> trace/bridge/SITL`

The topology/effectiveness path is more mature than before:

- explicit `16` fan / `32` vane / `32` plenum framing
- topology-local smoothing and spillover behavior
- a geometry-seeded nominal effectiveness table interface
- clearer separation between allocator math and hardware-facing assumptions

The command surface is also materially broader than an early allocator-only repo. It now includes:

- maneuver generation
- mission and coordinate runs
- engineering assessment
- power sweeps and sizing packs
- fault envelopes and threshold studies
- maneuver packs
- bridge export, inspect, MP, and SITL workflows

## What remains honestly provisional

The repo still stops short of validated hardware truth.

The most important remaining boundaries are:

- nominal effectiveness is geometry-seeded and provisional, not CAD/CFD/bench validated
- burst-power behavior is a bounded plant model, not a hardware-certified power envelope
- bridge and replay paths validate interfaces and workflows, not airworthiness
- SITL integration validates software/control plumbing, not final Aurora flight mechanics

That is the right honest posture for the project at this stage.

## What is no longer true from older reviews

These older criticisms are no longer accurate:

- the repo is no longer split between `aurora_gates` and `aurora_vtol`
- the README no longer presents the old allocator-only framing
- the docs tree is no longer limited to only a couple of visible files
- the pushed tree no longer contains `Zone.Identifier` artifacts
- `pyproject.toml` no longer uses the old allocator-only package description

## Current verdict

Aurora VTOL now looks like a consolidated software engineering project rather than a half-migrated allocator prototype.

The strongest improvement is not cosmetic branding by itself. It is that the repo presentation, source layout, CLI surface, and engineering documentation now tell the same story.

The main remaining work is no longer repo consolidation. It is replacing provisional modeled authority with measured or geometry-derived authority and continuing to push timing, power, actuator, and bridge truth forward.

## Best next engineering moves

1. Replace the provisional geometry-seeded effectiveness asset with CAD-, CFD-, or bench-derived coefficients.
2. Continue formalizing estimator and actuator-health interfaces against the ICD.
3. Push more hardware-facing timing, bus, and actuator-node assumptions into explicit modeled interfaces.
4. Keep replay and bridge workflows aligned with engineering truth, not just usability.
5. Add validation artifacts that compare modeled authority assumptions against future hardware evidence.
