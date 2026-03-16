# Aurora Engineering Review Checklist

Use this checklist when reviewing a new control law, allocator change, plant-model
change, bridge feature, replay feature, or maneuver profile.

The goal is to keep Aurora honest:
- software-first
- but never mechanics-last
- and never bridge-truth-last

## 1. Change Framing

Confirm the change is described clearly.

- What changed?
- Why was it changed?
- Which layer is affected?
  - guidance
  - vehicle controller
  - allocator
  - actuator dynamics
  - plant / power
  - bridge / replay / SITL
- Is the change nominal-only, fault-related, or both?
- Is it a real model change, or just a visualization / operator-view change?

## 2. Truth Boundary

Make the modeled-vs-assumed boundary explicit.

- What is newly modeled?
- What is still synthetic or assumed?
- What is still concept art / conceptual architecture only?
- Does the change risk making an unvalidated assumption look "real"?
- Are we clear about whether the result is:
  - simulated truth
  - bridge / replay behavior
  - UI-only behavior

## 3. Mechanics and Actuation

Check that the change respects mechanical and actuator reality.

- Does it stay within vane angle limits?
- Does it stay within vane rate / lag limits?
- Does it stay within fan response / spool limits?
- Does it respect degraded authority under faults?
- Does it rely on unrealistic instantaneous response?
- Does it hide a mechanical or actuator bottleneck behind tuning?

## 4. Power and Thermal Reality

Check whether the maneuver is actually supportable.

- Continuous power p95 within target?
- Peak power within allowed envelope?
- Voltage sag acceptable?
- Thermal headroom acceptable?
- Supply / thrust derating acceptable?
- If a case is still over budget, is that stated plainly?

## 5. Fault Robustness

Check whether the change behaves honestly under degraded hardware.

- Nominal case checked?
- Single-fault cases checked?
- Multi-fault cases checked where relevant?
- Dead-fan, vane fault, and plenum reduction behavior still make sense?
- Did the change improve one fault family while hurting another?
- Are any new recoveries / rescues physically plausible?

## 6. Bridge and Timing Integrity

Bridge work is engineering, not plumbing.

- Are units explicit and correct?
- Are timing semantics explicit and correct?
- Are rate limits preserved across the bridge?
- Are health / fault / limit signals preserved?
- Are failsafe assumptions explicit?
- Does replay or SITL behavior differ materially from the plant result?
- If so, is that difference documented and understood?

## 7. Maneuver Quality

Check whether the vehicle behavior actually improved.

For maneuver work, review at least:
- stop time
- reversal time
- alignment time
- yaw hold error
- XY tracking RMS
- flap usage
- fan tracking
- continuous power p95

For mission work, review at least:
- arrival quality
- obstacle clearance
- final error
- sustained power
- degraded-case behavior

## 8. Evidence

A change is not complete without evidence.

- Compile / lint check run?
- Smoke trace generated?
- Summary artifact generated?
- Faulted case checked if relevant?
- If the result is still caution/risk, is that visible in the report?
- Are the referenced output files saved in `runs/`?

## 9. Review Outcome

Classify the change honestly.

- `model improvement`
- `control improvement`
- `bridge / interface improvement`
- `operator-view improvement only`
- `documentation / architecture clarification`

And record the actual engineering outcome:

- `pass`
- `caution`
- `risk`
- `unknown / not verified`

## 10. Red Flags

Stop and re-check if any of these are true.

- The visuals look much better but plant metrics did not improve.
- The bridge makes a maneuver look better than the underlying simulation.
- A change fixes nominal behavior but silently weakens fault behavior.
- A result is presented as hardware truth when it is only replay truth.
- A controller improvement depends on unrealistically slow or fast actuators.
- Continuous power is still exceeded but the narrative sounds successful.

## Suggested Close-Out Format

Use this format when summarizing meaningful engineering changes:

1. What changed in code.
2. What improved in engineering terms.
3. What is still limited.
4. Whether the result is pass / caution / risk.
5. Which artifacts prove it.

## Current Aurora Reminder

For this project specifically, keep these in mind:

- Mechanics outrank visuals.
- Power, thermal, and actuator limits are part of the control problem.
- Replay and SITL bridges must not be confused with validated vehicle behavior.
- The allocator is important, but it is only one layer of the vehicle.
- Good software structure is valuable only if it stays tied to physical truth.
