# Aurora Limitations

This document states the main engineering limits of what Aurora currently proves and what it does not.

Related docs:

- [Engineering Notes](engineering_notes.md)
- [Architecture](architecture.md)
- [Scenario Guide](scenario_guide.md)
- [Hardware Control Architecture](hardware_control_architecture.md)

## Purpose

Aurora has grown into a strong software and systems-engineering prototype. That makes it valuable, but it also makes overclaiming easier if the boundaries are not written down clearly.

This page is the explicit truth-boundary companion to the rest of the documentation set.

## What Aurora is strong at today

Aurora is credible today as a repo for:

- control allocation studies
- maneuver logic and controller shaping
- topology-aware grouped degradation
- fault robustness comparisons
- power, thermal, and bounded burst-power tradeoff studies
- trace export, replay, bridge, and SITL workflow validation
- interface and architecture work for a future flight stack

## What Aurora does not yet validate

Aurora does not yet validate:

- final aerodynamic efficiency
- CFD-accurate plenum behavior
- exact duct geometry or flow coupling
- final structural packaging
- exact ESC, wiring, and bus architecture
- actuator current draw and hardware timing at embedded rates
- certification-grade safety behavior
- final CAD-accurate mechanical implementation

## Important truth boundaries

## 1. Allocator truth is not hardware truth

The allocator may find a mathematically valid command distribution. That does not by itself prove:

- the hardware can package it
- the ducts can deliver the assumed flow
- the structure can tolerate the loads
- the power electronics can sustain the demanded dynamics

## 2. Plant truth is modeled truth

Aurora's plant is much richer than a toy simulator, but it is still a model.

It includes useful approximations for:

- actuator lag
- plenum lag
- grouped faults
- battery and thermal effects
- bounded burst-power behavior

It does not yet include:

- CFD-grade flow interaction
- verified structural dynamics
- full 6-DoF aircraft mechanics
- certified estimator or avionics behavior

## 3. Replay truth is visualization truth

Mission Planner replay through `scripts/trace_to_tlog.py` is useful for visualizing traces and engineering telemetry, but it is not the same thing as a real autopilot or real vehicle.

It should be treated as:

- operator-visualization support
- telemetry debugging support
- replay support

It should not be treated as proof that the vehicle will fly identically in hardware.

## 4. Bridge truth is interface truth

Bridge and SITL tooling are valuable because they validate:

- timing assumptions
- coordinate conversion
- command-path shape
- MAVLink integration behavior

They do not yet prove actuator truth, aero truth, or structural truth.

## 5. Burst-power truth is conditional truth

Some current maneuver passes, especially in `step-snap-eco`, depend on the modeled burst-power envelope.

That is acceptable engineeringly because the model is explicit and bounded, but it comes with a condition:

- those passes are only as credible as the burst-power assumptions in the plant model

If the real hardware cannot safely deliver that burst reserve, the maneuver status would need to be re-evaluated.

## Current known simplifications

Important current simplifications include:

- topology is explicit but still simple
- default 32-segment Aurora fan ownership is pair-based structurally, but nominal aggregation and redistribution are now locally smoothed
- plenum ownership is one-sector-per-segment in the current model
- many traces use scenario-specific fields rather than a frozen versioned schema per family
- the controller is strongest in planar lateral motion and yaw-hold rather than full aircraft control
- fault modeling is control-oriented, not full reliability engineering

## Practical review rule

When reviewing a new result, ask which truth level it belongs to:

- controller truth
- allocator truth
- plant-model truth
- replay truth
- bridge truth
- hardware truth

If a claim is being made at a higher truth level than the evidence supports, it should be rewritten or downgraded.

## What would reduce these limitations next

The most valuable next upgrades would be:

1. geometry-derived effectiveness maps
2. more hardware-facing topology assumptions
3. richer actuator timing and local-node models
4. more explicit electrical and power-distribution assumptions
5. better separation of modeled behavior vs validated hardware-facing assumptions
6. eventual coupling to CAD, test rigs, or measured hardware data

## Bottom line

Aurora is already a strong software engineering and control-engineering platform. It should be presented that way confidently.

It should not yet be presented as proof of final hardware mechanics, final aerodynamics, or final certified flight behavior.
