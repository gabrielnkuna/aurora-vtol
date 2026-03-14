# Aurora Allocator V1–V4 (VS Code / uv)

This project contains the Project Aurora "Allocator" prototypes up to V4:
- V1: cosine flap distribution (translation)
- V2: adds tangential swirl ring (yaw-moment trim channel)
- V3: actuator rate limits + plenum lag + step-response metrics
- V4: repel-from-point field + fault injection + JSON trace export for dashboard replay

See [docs/hardware_control_architecture.md](/home/gabriel/projects/aurora-allocator-v4/docs/hardware_control_architecture.md) for the current software-to-hardware control mapping of the 16-fan / 32-vane concept.

## Setup (WSL recommended)
```bash
uv venv
uv sync
uv run aurora --help
```

## Run demos
### V2 demo (translation + optional yaw moment)
```bash
uv run aurora alloc demo --version v2 --dir-deg 90 --fxy 3000 --mz-nm 0
```

### V3 step test (direction change without yaw)
```bash
uv run aurora alloc step --dir-a-deg 0 --dir-b-deg 180 --fxy 3000 --step-time-s 3 --total-s 8
```

### V4 repel test + export replay trace
```bash
uv run aurora alloc repel --ox 30 --oy 0 --radius-m 30 --k 120 --trace-out runs/trace_repel.json
```

Fault examples:
```bash
uv run aurora alloc repel --stuck-flap-idx 7 --stuck-flap-alpha-deg 10 --trace-out runs/trace_fault_stuckflap.json
uv run aurora alloc repel --dead-fan-group 3 --dead-fan-scale 0 --trace-out runs/trace_fault_deadfan.json
uv run aurora alloc repel --plenum-sector-idx 12 --plenum-sector-scale 0.7 --trace-out runs/trace_fault_plenum.json
```

### Snappier configurations (new options)
You can now adjust actuator and plenum dynamics directly from the CLI.  These
help reduce latency and make the response feel more "instant":

```bash
# default values shown here for reference
--alpha-rate-deg-s 200.0     # flap rate limit (deg/s)
--plenum-tau-s 0.12          # plenum lag time constant (s)
```

### Step‑snap maneuver (V3)
A new `step-snap` command performs the three‑phase gate D maneuver: build up
motion in direction A, aggressively brake opposite velocity, and then snap into
direction B. The headline metrics focus on how quickly the craft stops and
reverses during the snap-stop phase.

```bash
uv run aurora alloc step-snap \
  --dir-a-deg 0 --dir-b-deg 180 \
  --fxy 2500 \
  --step-time-s 3 \
  --snap-stop-s 0.7 \
  --brake-gain 1.6 \
  --alpha-rate-deg-s 500 \
  --plenum-tau-s 0.05 \
  --total-s 9
```

Key metrics to watch:
* `t_to_speed_below_thr_s` – time until craft almost stops during snap
* `snap_stop_distance_m` – how far it slides before stopping
* `t_reversal_s` & `t90_dir_s` – reversal and realignment times
* `peak_speed_mps` and yaw/track coupling remain available for context.


Example of a faster repel demo:
```bash
uv run aurora alloc repel --ox 40 --oy 0 --radius-m 30 --k 250 --init-vx 2 \
    --total-s 12 --alpha-rate-deg-s 350 --plenum-tau-s 0.08 \
    --trace-out runs/trace_repel_demo_fast.json
```

### Hard-wall entry (optional patch)
If you prefer the field to feel like a **solid wall**, modify `repel_force_xy`
in `src/aurora_gates/allocator/field.py` so the force is non-zero immediately
upon crossing the radius. For example:

```python
pen = field.radius_m - d
# Add a "kick" as soon as you enter the radius
kick = 0.35 * field.fxy_max_n  # 35% of max immediately at boundary (tunable)
mag = clamp(kick + field.k_n_per_m * pen, 0.0, field.fxy_max_n)
```

Rerunning with a stiffer field:
```bash
uv run aurora alloc repel --ox 40 --oy 0 --radius-m 30 --k 600 --fxy-max 8000 \
    --init-vx 2 --total-s 12 --alpha-rate-deg-s 350 --plenum-tau-s 0.08 \
    --trace-out runs/trace_repel_demo_wall.json
```
should produce a recede latency well under a second, giving a very abrupt
rebound.


## Output
- JSON results print to stdout
- Traces export to `runs/*.json` (replay in your dashboard)

### Headline metrics
The `repel` command now includes a couple of extra fields in the headline:

* `enter_radius_time_s` – when the craft first crossed into the  field radius
* `response_time_s` – (legacy) first time speed changed by ≥0.3 m/s
* `response_latency_s` – difference between enter and the above
* `recede_time_s` – first moment radial velocity becomes positive (moving away)
* `recede_latency_s` – latency based on the more physical “recede” metric

The `recede_*` numbers give a more accurate “start of repel” indicator compared to
the speed‑difference rule, and are the ones you’ll want to plot on the dashboard.

In addition to the headline metrics, trace JSONs now record per‑segment data:
* `hist.alpha_deg_32[t]` – list of 32 flap angles (degrees) at time step `t`
* `hist.ft_tan_32[t]` – list of tangential efforts per segment (N)
* `hist.fan_thrust_16[t]` – 16‑pair averaged fan thrusts (existing)

These arrays enable ring‑visualization of flap deflections or 16‑fan thrust
patterns in both repel and step‑snap traces.



