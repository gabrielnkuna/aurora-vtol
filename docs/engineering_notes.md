# Aurora Engineering Notes

These notes are the working engineering stance for the Aurora repo. They are meant to keep the project honest as the software stack grows.

Related docs:

- [Architecture](architecture.md)
- [Interface Control Document](icd.md)
- [Bridge Workflows](bridge_workflows.md)
- [Current-State Repo Review](repo_review_current_state.md)
- [Hardware Control Architecture](hardware_control_architecture.md)
- [Engineering Review Checklist](engineering_review_checklist.md)
- [Scenario Guide](scenario_guide.md)
- [Limitations](limitations.md)

## 1. Current engineering stance

Aurora is strong as a software architecture, allocator, maneuver-validation, and bridge prototype. It is not yet a full aero-mechanical design.

The code is most credible when used to discuss:

- control allocation behavior
- maneuver feasibility
- fault tolerance and degraded authority
- timing and response trade-offs
- bridge, replay, and SITL workflows
- power, burst-power, and thermal constraints in the plant model

It is less credible when used to claim:

- exact lift efficiency
- final packing dimensions
- final battery mass and geometry
- CFD-accurate plenum behavior
- certifiable flight safety architecture

## 2. Assumptions that should stay explicit

Important assumptions to keep visible in docs, traces, and code comments:

- the craft is modeled as a ring vehicle with segmented control authority
- segment effectiveness can be reduced by topology, fan faults, or plenum faults
- vehicle control outputs generalized force and moment requests
- dynamics are simplified but intentionally non-instantaneous
- bridge traces are abstractions of commanded behavior, not proof of airworthiness
- burst-power allowances are a modeled plant feature and must remain bounded, visible, and testable

## 3. Documentation discipline

For every major feature change, document:

- what changed in control logic
- what changed in assumptions
- what commands expose the feature
- what metrics should improve or worsen
- whether the change is simulation-only or intended to reflect hardware reality

If a change makes replay look better without improving engineering truth, that should be stated explicitly.

## 4. Validation discipline

Before calling a control or plant change an improvement, check:

- baseline demos still work
- maneuver metrics do not regress unexpectedly
- fault-envelope behavior remains explainable
- power-limited cases degrade smoothly
- burst-power use stays within the modeled reserve and clipping rules
- trace export remains replay and bridge compatible
- Mission Planner or SITL behavior still matches the intended truth boundary

## 5. Suggested future docs

As the project grows, add or extend:

- `docs/scenario_guide.md` for maneuver test intent and expected metrics
- `docs/limitations.md` for explicit boundaries between modeled and validated behavior
- deeper hardware-oriented supplements to `docs/topology.md`, `docs/fault_model.md`, and `docs/trace_schema.md` as the design matures
- `docs/command_reference.md` for a user-facing CLI guide

## 6. Current engineering priorities

The next software-engineering work should continue to prioritize:

1. freezing hardware-facing topology assumptions
2. adding geometry-aware effectiveness maps
3. formalizing estimator and health publisher interfaces against the ICD
4. separating central control logic from local actuator-node ownership
5. strengthening bridge truth and timing truth
6. keeping mechanics, power, and actuator behavior ahead of visuals
