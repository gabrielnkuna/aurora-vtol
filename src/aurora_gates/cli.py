from __future__ import annotations
import json
import typer

from aurora_gates.allocator.metrics import yaw_track_coupling_mean_abs
from aurora_gates.allocator.sim import run_demo, run_step_test_v3, run_repel_test_v4, run_step_snap_v3
from aurora_gates.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_gates.allocator.field import RepelField
from aurora_gates.allocator.faults import FaultSpec

app = typer.Typer(help="Project Aurora Allocator V1–V4 CLI")
alloc_app = typer.Typer(help="Allocator demos")
app.add_typer(alloc_app, name="alloc")


@alloc_app.command("demo")
def alloc_demo(
    dir_deg: float = typer.Option(90.0, "--dir-deg", help="Direction of travel in BODY frame degrees (0=+X)"),
    fxy_n: float = typer.Option(3000.0, "--fxy", help="Desired lateral force magnitude (N)"),
    duration_s: float = typer.Option(6.0, "--duration-s", help="Sim duration (s)"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg", help="Yaw hold (deg)"),
    version: str = typer.Option("v2", "--version", help="v1 or v2"),
    mz_nm: float = typer.Option(0.0, "--mz-nm", help="Desired yaw moment (N*m). 0 = near-zero yaw trim"),
):
    hist = run_demo(
        dir_deg=dir_deg,
        fxy_n=fxy_n,
        duration_s=duration_s,
        yaw_hold_deg=yaw_hold_deg,
        mz_nm=mz_nm,
        version=version,
    )
    coupling = yaw_track_coupling_mean_abs(hist)
    out = {
        "allocator_version": version,
        "dir_deg": dir_deg,
        "fxy_n": fxy_n,
        "mz_nm_cmd": mz_nm,
        "duration_s": duration_s,
        "yaw_hold_deg": yaw_hold_deg,
        "yaw_track_coupling_mean_abs_deg": coupling,
        "final": {
            "x_m": hist["x"][-1],
            "y_m": hist["y"][-1],
            "vx_mps": hist["vx"][-1],
            "vy_mps": hist["vy"][-1],
            "alpha_deg_rms": hist["alpha_deg_rms"][-1],
            "ft_tan_rms": hist.get("ft_tan_rms", [0])[-1],
            "mz_est_nm": hist["mz_est"][-1],
        },
    }
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("step")
def alloc_step(
    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg", help="Direction before step (deg)"),
    dir_b_deg: float = typer.Option(180.0, "--dir-b-deg", help="Direction after step (deg)"),
    fxy_n: float = typer.Option(3000.0, "--fxy", help="Lateral force magnitude (N)"),
    step_time_s: float = typer.Option(3.0, "--step-time-s", help="Time of direction change (s)"),
    total_s: float = typer.Option(8.0, "--total-s", help="Total sim time (s)"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),
    mz_nm: float = typer.Option(0.0, "--mz-nm", help="Desired yaw moment (N*m). 0 = near-zero yaw trim"),
    alpha_rate_deg_s: float = typer.Option(200.0, "--alpha-rate-deg-s"),
    plenum_tau_s: float = typer.Option(0.12, "--plenum-tau-s"),
):
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    out, _hist = run_step_test_v3(
        dir_deg_a=dir_a_deg,
        dir_deg_b=dir_b_deg,
        fxy_n=fxy_n,
        step_time_s=step_time_s,
        total_s=total_s,
        yaw_hold_deg=yaw_hold_deg,
        mz_nm=mz_nm,
        lim=lim,
        pl=pl,
    )
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("repel")
def alloc_repel(
    obstacle_x_m: float = typer.Option(30.0, "--ox"),
    obstacle_y_m: float = typer.Option(0.0, "--oy"),
    total_s: float = typer.Option(12.0, "--total-s"),
    init_vx_mps: float = typer.Option(1.0, "--init-vx"),
    init_vy_mps: float = typer.Option(0.0, "--init-vy"),
    radius_m: float = typer.Option(30.0, "--radius-m"),
    k_n_per_m: float = typer.Option(120.0, "--k"),
    fxy_max_n: float = typer.Option(4000.0, "--fxy-max"),
    stuck_flap_idx: int = typer.Option(-1, "--stuck-flap-idx"),
    stuck_flap_alpha_deg: float = typer.Option(0.0, "--stuck-flap-alpha-deg"),
    dead_fan_group: int = typer.Option(-1, "--dead-fan-group"),
    dead_fan_scale: float = typer.Option(0.0, "--dead-fan-scale"),
    plenum_sector_idx: int = typer.Option(-1, "--plenum-sector-idx"),
    plenum_sector_scale: float = typer.Option(0.7, "--plenum-sector-scale"),
    alpha_rate_deg_s: float = typer.Option(200.0, "--alpha-rate-deg-s", help="Actuator flap rate (deg/s)"),
    plenum_tau_s: float = typer.Option(0.12, "--plenum-tau-s", help="Plenum lag time constant (s)"),
    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path (for dashboard replay)"),
):
    field = RepelField(k_n_per_m=k_n_per_m, radius_m=radius_m, fxy_max_n=fxy_max_n)
    fault = FaultSpec(
        stuck_flap_idx=(None if stuck_flap_idx < 0 else stuck_flap_idx),
        stuck_flap_alpha_deg=stuck_flap_alpha_deg,
        dead_fan_group=(None if dead_fan_group < 0 else dead_fan_group),
        dead_fan_scale=dead_fan_scale,
        plenum_sector_idx=(None if plenum_sector_idx < 0 else plenum_sector_idx),
        plenum_sector_scale=plenum_sector_scale,
    )
    # build actuator limits / plenum objects from CLI options
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)
    out, _hist = run_repel_test_v4(
        obstacle_x_m=obstacle_x_m,
        obstacle_y_m=obstacle_y_m,
        total_s=total_s,
        initial_vx_mps=init_vx_mps,
        initial_vy_mps=init_vy_mps,
        field=field,
        fault=fault,
        lim=lim,
        pl=pl,
        trace_out=(trace_out if trace_out else None),
    )
    typer.echo(json.dumps(out, indent=2))


@alloc_app.command("step-snap")
def alloc_step_snap(
    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg"),
    dir_b_deg: float = typer.Option(180.0, "--dir-b-deg"),
    fxy_n: float = typer.Option(3000.0, "--fxy"),
    step_time_s: float = typer.Option(3.0, "--step-time-s"),
    snap_stop_s: float = typer.Option(0.8, "--snap-stop-s", help="Duration of snap-stop brake phase"),
    total_s: float = typer.Option(9.0, "--total-s"),
    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),
    mz_nm: float = typer.Option(0.0, "--mz-nm"),
    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),
    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),
    brake_gain: float = typer.Option(1.2, "--brake-gain", help="Multiplier on fxy during snap-stop"),
    speed_stop_thr_mps: float = typer.Option(0.2, "--stop-thr-mps", help="Speed threshold considered 'stopped'"),
):
    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)
    pl = PlenumModel(tau_s=plenum_tau_s)

    out, _hist = run_step_snap_v3(
        dir_deg_a=dir_a_deg,
        dir_deg_b=dir_b_deg,
        fxy_n=fxy_n,
        step_time_s=step_time_s,
        snap_stop_s=snap_stop_s,
        total_s=total_s,
        yaw_hold_deg=yaw_hold_deg,
        mz_nm=mz_nm,
        lim=lim,
        pl=pl,
        brake_gain=brake_gain,
        speed_stop_thr_mps=speed_stop_thr_mps,
    )
    typer.echo(json.dumps(out, indent=2))
