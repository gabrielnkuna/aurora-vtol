# Aurora Scenario Guide

This document explains the main scenario families in Aurora, what each one is for, and which metrics matter most when reviewing results.

Related docs:

- [Command Reference](command_reference.md)
- [Engineering Notes](engineering_notes.md)
- [Trace Schema](trace_schema.md)
- [Limitations](limitations.md)

## Purpose

Aurora includes both maneuver-scale and mission-scale scenarios. They are not interchangeable.

Use maneuver scenarios when you want to study fast directional behavior, actuator stress, and fault robustness in a controlled setting.

Use mission scenarios when you want to study route planning, obstacle handling, arrival quality, and sustained power behavior over longer durations.

## Scenario families

### 1. `demo`

Purpose:

- allocator sanity checks
- quick directional force checks
- fast regression testing

Best for:

- checking that the core radial or yaw-aware allocator still behaves sensibly
- quick smoke tests after control or topology changes

What to watch:

- net direction of commanded force
- yaw moment behavior if tangential authority is involved
- obvious regressions in commanded segment patterns

### 2. `step`

Purpose:

- simple direction-change response
- clean baseline before more complex maneuvers

Best for:

- checking response timing without full snap-stop logic
- comparing how faults affect basic redirection behavior

What to watch:

- reversal time
- alignment time
- yaw-hold error
- flap peak usage
- fan tracking error

### 3. `step-redirect`

Purpose:

- UFO-style vector redirect without forcing a literal stop
- fixed-yaw lateral motion studies

Best for:

- 45 degree or 90 degree redirect studies
- evaluating whether the vehicle can change motion direction smoothly while holding yaw
- lower-drama vectoring studies compared with snap-stop

Current status note:

- this family is now strongest for smooth fixed-yaw redirection
- in the latest representative redirect pack, weak and biased flap cases are already passing, while nominal and slow-flap cases remain near the sustained-power threshold in some runs

Representative outputs to watch:

- align time
- yaw hold error
- flap peak percent
- fan tracking percent
- power p95 percent

When to prefer it over `step-snap`:

- when the behavior you want is `redirect the velocity vector` rather than `brake hard, stop, and reverse`

### 4. `step-snap`

Purpose:

- hard brake and revector behavior
- snap-stop, reversal, and aggressive redirect testing

Best for:

- 90, 135, and 180 degree redirection studies where sharp response matters
- testing the limit behavior of power guard, fault guard, and controller shaping
- seeing whether the plant can support short maneuver bursts without clipping

Current status note:

- the `step-snap-eco` family is now the stronger review path for the current plant model
- the current representative pack in [`runs/maneuver_pack_step_snap_eco_demo/summary.md`](../runs/maneuver_pack_step_snap_eco_demo/summary.md) is passing across all tested cases because the bounded burst-power model keeps raw overloads inside the modeled burst reserve with zero clipping time

Most important metrics:

- stop time
- reversal time
- align time
- yaw hold error
- raw sustained power p95
- burst reserve minimum percent
- burst clip time

Important interpretation rule:

- a pass here does not mean raw power stayed below continuous limit at all times
- it means the run stayed inside the modeled burst-power envelope without clipping or exhausting reserve

### 5. `repel`

Purpose:

- field-based obstacle or exclusion-zone response
- timing studies for avoidance onset and recede behavior

Best for:

- validating repel-field shaping
- visual replay in Mission Planner
- studying how faults change a simple reactive avoidance behavior

What to watch:

- response latency
- recede latency
- distance from the obstacle center or radius boundary
- actuator usage during the field interaction

### 6. `coordinates`

Purpose:

- mission-scale route following with optional obstacle handling
- arrival and sustained-power studies

Best for:

- checking whether a route is feasible under power, thermal, and fault limits
- studying obstacle bypass behavior
- generating traces for bridge or Mission Planner workflows

Representative current status:

- a representative medium mission assessment in [`runs/trace_coordinates_power_assessment.txt`](../runs/trace_coordinates_power_assessment.txt) currently comes back `caution` because continuous power demand is slightly above the configured continuous limit for a sustained portion of the run
- this is a good example of why mission assessment and maneuver assessment need to be interpreted differently

What to watch:

- arrival time
- final goal error
- minimum obstacle clearance
- continuous power p95 and peak
- guard-active time
- supply and thermal margins

## Review strategy by scenario type

### Maneuver scenarios

Use these when the engineering question is mostly about:

- agility
- vectoring quality
- fault robustness during transients
- yaw-hold under fast directional change
- whether burst-power support is enough for short maneuvers

### Mission scenarios

Use these when the engineering question is mostly about:

- sustained power
- arrival quality
- route planning
- obstacle bypass
- overall feasibility across time rather than a single transient

## Recommended artifact set

For serious reviews, keep these together:

- trace JSON
- text assessment or maneuver-pack summary
- Markdown summary artifact
- any relevant Mission Planner replay or bridge script only as supporting visualization

## Truth boundary

Scenario outputs are modeling and engineering-review artifacts. They are useful for comparing controller, allocator, fault, and power behavior, but they are not proof of final airworthiness or hardware mechanics.
