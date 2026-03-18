from __future__ import annotations

import csv

import json


from pathlib import Path

from typing import Optional

import numpy as np

import typer



from aurora_vtol.allocator.metrics import yaw_track_coupling_mean_abs

from aurora_vtol.allocator.engineering import assess_trace, render_assessment, tune_coordinate_mission

from aurora_vtol.bridge import build_qgc_wpl_mission, build_setpoint_script, inspect_script, load_script_jsonl, load_script_jsonl_metadata, load_trace_json, script_metadata_from_trace_meta, stream_script_ardupilot_sitl, stream_script_mavlink, summarize_qgc_wpl_mission, summarize_script, summarize_setpoints, write_qgc_wpl_mission, write_script_jsonl, write_script_svg

from aurora_vtol.allocator.mission_planning import MissionObstacle
from aurora_vtol.allocator.power_system import PowerSystemParams
from aurora_vtol.allocator.maneuver_scenarios import run_demo, run_step_redirect_v3, run_step_test_v3, run_repel_test_v4, run_step_snap_v3
from aurora_vtol.allocator.coordinate_mission import run_coordinate_mission_v5

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel

from aurora_vtol.allocator.field import RepelField

from aurora_vtol.allocator.faults import FaultSpec

from aurora_vtol.allocator.model import RingGeometry

from aurora_vtol.allocator.trace import save_trace_json

from aurora_vtol.reporting import infer_requirements_table_format, render_fault_envelope_table, render_fault_threshold_pack_table, render_fault_threshold_table, render_preset_summary_table, summarize_power_params

from aurora_vtol.study_workflows import build_coordinate_mission_kwargs, build_power_params, build_power_sweep_report_for_fault, write_power_sweep_outputs

from aurora_vtol.fault_workflows import build_fault_spec, resolve_fault_case, run_fault_envelope_report, run_fault_threshold_pack_report, run_fault_threshold_report, select_fault_cases, summarize_fault_case
from aurora_vtol.maneuver_analysis import render_maneuver_pack_markdown, tune_maneuver_profile
from aurora_vtol.effectiveness_workflows import (
    build_effectiveness_comparison_report,
    build_effectiveness_report,
    write_effectiveness_candidate_template_outputs,
    write_effectiveness_comparison_outputs,
    write_effectiveness_report_outputs,
)



app = typer.Typer(help="Aurora VTOL control and simulation CLI")

alloc_app = typer.Typer(help="Allocator demos")

bridge_app = typer.Typer(help="Bridge planned traces into dry-run or live setpoint streams")

app.add_typer(alloc_app, name="alloc")

app.add_typer(bridge_app, name="bridge")



COORDINATE_PRESETS = {

    "medium": {

        "dest_x_m": 120.0,

        "dest_y_m": 40.0,

        "dest_z_m": 0.0,

        "total_s": 45.0,

        "cruise_alt_m": 18.0,

        "max_speed_mps": 8.0,

        "arrival_radius_m": 2.0,

        "slow_radius_m": 22.0,

        "descent_radius_m": 28.0,

        "hold_s": 2.0,

        "fxy_max_n": 3600.0,

        "pos_k_n_per_m": 100.0,

        "vel_k_n_per_mps": 1000.0,

        "obstacles": ["60,18,14,180,650,28"],

    },

    "long": {

        "dest_x_m": 180.0,

        "dest_y_m": 60.0,

        "dest_z_m": 0.0,

        "total_s": 70.0,

        "cruise_alt_m": 25.0,

        "max_speed_mps": 10.0,

        "arrival_radius_m": 2.0,

        "slow_radius_m": 28.0,

        "descent_radius_m": 35.0,

        "hold_s": 2.0,

        "fxy_max_n": 3800.0,

        "pos_k_n_per_m": 90.0,

        "vel_k_n_per_mps": 1100.0,

        "obstacles": ["95,25,18,180,700,35"],

    },

    "very-long": {

        "dest_x_m": 300.0,

        "dest_y_m": 120.0,

        "dest_z_m": 0.0,

        "total_s": 110.0,

        "cruise_alt_m": 35.0,

        "max_speed_mps": 12.0,

        "arrival_radius_m": 3.0,

        "slow_radius_m": 40.0,

        "descent_radius_m": 50.0,

        "hold_s": 3.0,

        "fxy_max_n": 4000.0,

        "pos_k_n_per_m": 80.0,

        "vel_k_n_per_mps": 1200.0,

        "obstacles": ["170,70,25,180,700,45"],

    },

}







def resolve_coordinate_preset_context(preset: str) -> tuple[dict, list[MissionObstacle]]:
    if preset not in COORDINATE_PRESETS:
        raise typer.BadParameter('--preset must be one of: medium, long, very-long')
    preset_cfg = COORDINATE_PRESETS[preset]
    obstacles = parse_mission_obstacles(list(preset_cfg.get('obstacles', [])))
    return preset_cfg, obstacles

MANEUVER_DEFAULT_ANGLES = [45.0, 90.0, 135.0, 180.0]

MP_GRAPH_PRESETS = {

    "overview": {

        "description": "Core mission and energy channels for quick health checks.",

        "channels": ["speed", "goal_m", "obs_m", "bat_v", "soc_pct", "pwr_kw", "thr_pct"],

    },

    "battery-power": {

        "description": "Battery, power draw, remaining energy, and thrust derating.",

        "channels": ["bat_v", "bat_a", "soc_pct", "pwr_kw", "eng_wh", "thr_pct"],

    },

    "thermal-power": {

        "description": "Continuous power pressure, thermal headroom, and fan response.",

        "channels": ["cont_pct", "pwr_marg", "therm_pct", "temp_max", "temp_avg", "rsp_pct"],

    },

    "fan-tracking": {

        "description": "Commanded versus achieved fan thrust summaries.",

        "channels": ["cmd_avg", "fan_avg", "cmd_stdp", "fan_stdp", "cmd_maxp", "fan_maxp"],

    },

    "force-tracking": {

        "description": "Requested versus achieved vehicle force in XYZ.",

        "channels": ["fx_cmd", "fx_ach", "fx_err", "fy_cmd", "fy_ach", "fy_err", "fz_cmd", "fz_ach", "fz_err"],

    },

    "flap-tracking": {

        "description": "Commanded versus achieved flap summary and sampled segments.",

        "channels": ["a_cmd", "a_rms", "c0", "a0", "c2", "a2", "c4", "a4", "c6", "a6"],

    },

}



for fan_start in range(0, 16, 8):

    fan_stop = fan_start + 7

    MP_GRAPH_PRESETS[f"fan-actual-{fan_start}-{fan_stop}"] = {

        "description": f"Actual thrust for fans {fan_start}-{fan_stop}.",

        "channels": [f"f{idx:02d}" for idx in range(fan_start, fan_stop + 1)],

    }

    MP_GRAPH_PRESETS[f"fan-command-{fan_start}-{fan_stop}"] = {

        "description": f"Commanded thrust for fans {fan_start}-{fan_stop}.",

        "channels": [f"fc{idx:02d}" for idx in range(fan_start, fan_stop + 1)],

    }



for flap_start in range(0, 32, 8):

    flap_stop = flap_start + 7

    MP_GRAPH_PRESETS[f"flap-actual-{flap_start}-{flap_stop}"] = {

        "description": f"Actual flap angles for segments {flap_start}-{flap_stop}.",

        "channels": [f"a{idx:02d}" for idx in range(flap_start, flap_stop + 1)],

    }

    MP_GRAPH_PRESETS[f"flap-command-{flap_start}-{flap_stop}"] = {

        "description": f"Commanded flap angles for segments {flap_start}-{flap_stop}.",

        "channels": [f"c{idx:02d}" for idx in range(flap_start, flap_stop + 1)],

    }



MP_GRAPH_CHANNEL_SOURCES = {

    "speed": "speed",

    "goal_m": "dist_to_goal",

    "obs_m": "nearest_obstacle_m",

    "bat_v": "battery_v",

    "bat_a": "battery_a",

    "soc_pct": "battery_soc",

    "pwr_kw": "power_w",

    "eng_wh": "energy_wh",

    "thr_pct": "thrust_scale_pct",

    "cont_pct": "continuous_power_pct",

    "pwr_marg": "power_margin_kw",

    "therm_pct": "thermal_scale_pct",

    "temp_max": "fan_temp_max_c",

    "temp_avg": "fan_temp_mean_c",

    "rsp_pct": "fan_response_pct",

    "cmd_avg": "fan_cmd_16",

    "cmd_stdp": "fan_cmd_16",

    "cmd_maxp": "fan_cmd_16",

    "fan_avg": "fan_thrust_16",

    "fan_stdp": "fan_thrust_16",

    "fan_maxp": "fan_thrust_16",

    "fx_cmd": "fx_cmd",

    "fy_cmd": "fy_cmd",

    "fz_cmd": "fz_cmd",

    "fx_ach": "fx_ach",

    "fy_ach": "fy_ach",

    "fz_ach": "fz_ach",

    "fx_err": "fx_err",

    "fy_err": "fy_err",

    "fz_err": "fz_err",

    "a_cmd": "alpha_cmd_rms",

    "a_rms": "alpha_deg_rms",

    "c0": "alpha_cmd_deg_32",

    "c2": "alpha_cmd_deg_32",

    "c4": "alpha_cmd_deg_32",

    "c6": "alpha_cmd_deg_32",

    "a0": "alpha_deg_32",

    "a2": "alpha_deg_32",

    "a4": "alpha_deg_32",

    "a6": "alpha_deg_32",

}



for fan_idx in range(16):

    MP_GRAPH_CHANNEL_SOURCES[f"f{fan_idx:02d}"] = "fan_thrust_16"

    MP_GRAPH_CHANNEL_SOURCES[f"fc{fan_idx:02d}"] = "fan_cmd_16"



for flap_idx in range(32):

    MP_GRAPH_CHANNEL_SOURCES[f"a{flap_idx:02d}"] = "alpha_deg_32"

    MP_GRAPH_CHANNEL_SOURCES[f"c{flap_idx:02d}"] = "alpha_cmd_deg_32"







def mp_graph_channel_available(hist: dict, channel: str) -> bool:

    source = MP_GRAPH_CHANNEL_SOURCES.get(channel)

    if source is None:

        return False

    values = hist.get(source, [])

    return bool(values)





def mp_graph_ui_label(channel: str) -> str:

    return f"MAV_{channel.upper()}"





def build_mp_graph_preset_report(hist: dict | None = None, preset_name: str | None = None) -> dict:

    selected = MP_GRAPH_PRESETS if preset_name is None else {preset_name: MP_GRAPH_PRESETS[preset_name]}

    report = {}

    for name, spec in selected.items():

        channels = list(spec["channels"])

        entry = {

            "description": spec["description"],

            "channels": channels,

            "paste": ", ".join(channels),

        }

        if hist is not None:

            available = [channel for channel in channels if mp_graph_channel_available(hist, channel)]

            missing = [channel for channel in channels if channel not in available]

            entry["available"] = available

            entry["missing"] = missing

        report[name] = entry

    return report





def render_mp_graph_preset_report(report: dict, fmt: str) -> str:

    if fmt == "json":

        return json.dumps(report, indent=2)

    if fmt == "paste":

        lines = [entry["paste"] for entry in report.values()]

        return "\n".join(lines) + "\n"

    if fmt == "mp-ui":

        lines = [f"{name}: {', '.join(mp_graph_ui_label(channel) for channel in entry['channels'])}" for name, entry in report.items()]

        return "\n".join(lines) + "\n"



    lines: list[str] = []

    if fmt == "md":

        lines.append("# Mission Planner Graph Presets")

    else:

        lines.append("Mission Planner Graph Presets")

    for name, entry in report.items():

        if fmt == "md":

            lines.append("")

            lines.append(f"## {name}")

            lines.append(entry["description"])

            lines.append("")

            lines.append(f"Paste: `{entry['paste']}`")

            if "available" in entry:

                lines.append("")

                lines.append(f"Available: `{', '.join(entry['available'])}`" if entry["available"] else "Available: none")

                if entry["missing"]:

                    lines.append(f"Missing: `{', '.join(entry['missing'])}`")

        else:

            lines.append("")

            lines.append(f"[{name}]")

            lines.append(f"description = {entry['description']}")

            lines.append(f"paste = {entry['paste']}")

            if "available" in entry:

                lines.append(f"available = {', '.join(entry['available'])}")

                lines.append(f"missing = {', '.join(entry['missing'])}")

    return "\n".join(lines) + "\n"





@alloc_app.command("assess")

def alloc_assess(

    trace: str = typer.Option(..., "--trace", help="Trace JSON path to assess"),

    out: str = typer.Option("", "--out", help="Optional output path for the rendered assessment"),

    format: str = typer.Option("text", "--format", help="Output format: json, text, or md"),

):

    fmt = format.lower().strip()

    if fmt not in {"json", "text", "md"}:

        raise typer.BadParameter("--format must be one of: json, text, md")

    meta, hist = load_trace_json(trace)

    report = assess_trace(meta, hist)

    rendered = render_assessment(report, fmt)

    if out:

        Path(out).write_text(rendered, encoding="utf-8")

    typer.echo(rendered)





@bridge_app.command("inspect")

def bridge_inspect(

    script: str = typer.Option(..., "--script", help="Bridge JSONL setpoint script path"),

    svg_out: str = typer.Option("", "--svg-out", help="Optional SVG top-down path output"),

    fail_on_issues: bool = typer.Option(False, "--fail-on-issues/--no-fail-on-issues", help="Exit non-zero if validation finds issues"),

):

    metadata = load_script_jsonl_metadata(script)

    setpoints = load_script_jsonl(script)

    report = inspect_script(setpoints, metadata=metadata)

    if svg_out:

        write_script_svg(svg_out, setpoints, title=Path(script).name)

        report["svg_out"] = svg_out

    typer.echo(json.dumps(report, indent=2))

    if fail_on_issues and not report.get("ok", False):

        raise typer.Exit(code=1)





@bridge_app.command("mp-graphs")

def bridge_mp_graphs(

    trace: str = typer.Option("", "--trace", help="Optional trace JSON path to validate which graph channels are available"),

    preset: str = typer.Option("", "--preset", help="Optional preset name to output by itself"),

    out: str = typer.Option("", "--out", help="Optional output path for the rendered preset list"),

    format: str = typer.Option("json", "--format", help="Output format: json, text, or md"),

):

    fmt = format.lower().strip()

    if fmt not in {"json", "text", "md", "paste", "mp-ui"}:

        raise typer.BadParameter("--format must be one of: json, text, md, paste, mp-ui")

    preset_name = preset.strip() or None

    if preset_name is not None and preset_name not in MP_GRAPH_PRESETS:

        raise typer.BadParameter(f"unknown preset: {preset_name}")



    hist = None

    trace_path = trace.strip()

    if trace_path:

        _meta, hist = load_trace_json(trace_path)



    preset_report = build_mp_graph_preset_report(hist=hist, preset_name=preset_name)

    report = {

        "trace": (trace_path or None),

        "presets": preset_report,

    }

    rendered = render_mp_graph_preset_report(preset_report, fmt)

    if out:

        Path(out).write_text(rendered, encoding="utf-8")

    typer.echo(rendered)





@bridge_app.command("trace")

def bridge_trace(

    trace: str = typer.Option(..., "--trace", help="Planned trace JSON path"),

    jsonl_out: str = typer.Option("", "--jsonl-out", help="Optional JSONL setpoint script output path"),

    mavlink_out: str = typer.Option("", "--mavlink-out", help="Optional MAVLink connection string for live setpoint streaming"),

    hold_final_s: float = typer.Option(2.0, "--hold-final-s", help="How long to hold the final setpoint after the trace ends (s)"),

    rate_hz: float = typer.Option(20.0, "--rate-hz", help="Bridge output rate in Hz; use <=0 for native trace rate"),

    realtime: bool = typer.Option(True, "--realtime/--no-realtime", help="When streaming MAVLink, honor trace timing or send as fast as possible"),

    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Print a bridge summary even when outputs are requested"),

    target_system: int = typer.Option(1, "--target-system", help="Target MAVLink system id for live setpoint streaming"),

    target_component: int = typer.Option(1, "--target-component", help="Target MAVLink component id for live setpoint streaming"),

    source_system: int = typer.Option(245, "--source-system", help="Source MAVLink system id for the bridge"),

    source_component: int = typer.Option(190, "--source-component", help="Source MAVLink component id for the bridge"),

    heartbeat_hz: float = typer.Option(1.0, "--heartbeat-hz", help="Heartbeat rate while streaming MAVLink setpoints"),

):

    meta, hist = load_trace_json(trace)

    script = build_setpoint_script(

        meta,

        hist,

        hold_final_s=hold_final_s,

        rate_hz=(None if rate_hz <= 0.0 else rate_hz),

    )

    summary = summarize_script(meta, script)



    if jsonl_out:

        write_script_jsonl(jsonl_out, script, metadata=script_metadata_from_trace_meta(meta))

        summary["jsonl_out"] = jsonl_out



    if dry_run or (not jsonl_out and not mavlink_out):

        typer.echo(json.dumps(summary, indent=2))



    if mavlink_out:

        stream_script_mavlink(

            script,

            out=mavlink_out,

            target_system=target_system,

            target_component=target_component,

            source_system=source_system,

            source_component=source_component,

            heartbeat_hz=heartbeat_hz,

            realtime=realtime,

        )







@bridge_app.command("mp")

def bridge_mp(

    trace: str = typer.Option(..., "--trace", help="Planned trace JSON path"),

    out: str = typer.Option(..., "--out", help="Mission Planner mission file output path (.waypoints/.txt)"),

    home_lat: float = typer.Option(..., "--home-lat", help="Home latitude in degrees"),

    home_lon: float = typer.Option(..., "--home-lon", help="Home longitude in degrees"),

    home_alt_m: float = typer.Option(0.0, "--home-alt-m", help="Home altitude reference in meters"),

    takeoff_alt_m: float = typer.Option(0.0, "--takeoff-alt-m", help="Takeoff altitude for the MP mission. 0 uses the trace cruise altitude"),

    cruise_alt_m: float = typer.Option(0.0, "--cruise-alt-m", help="Cruise altitude override for mission waypoints"),

    waypoint_spacing_m: float = typer.Option(10.0, "--waypoint-spacing-m", help="Route spacing when sampling mission waypoints from the trace"),

    trace_route: bool = typer.Option(False, "--trace-route/--planner-route", help="Use trace-sampled waypoints instead of the default interpolated planner route"),

    spline: bool = typer.Option(True, "--spline/--no-spline", help="Use SPLINE_WAYPOINT for the route to reduce stop-and-go behavior in AUTO"),

    speed_mps: float = typer.Option(8.0, "--speed-mps", help="Mission speed command in m/s for AUTO"),

    acceptance_radius_m: float = typer.Option(4.0, "--acceptance-radius-m", help="Waypoint acceptance radius in meters for the MP mission"),

    approach_distance_m: float = typer.Option(18.0, "--approach-distance-m", help="Distance before destination to insert a lower approach waypoint"),

    descent_alt_m: float = typer.Option(0.0, "--descent-alt-m", help="Final approach altitude before LAND. 0 uses an automatic descent altitude"),

    land: bool = typer.Option(True, "--land/--no-land", help="Append a LAND command at the destination"),

):

    meta, hist = load_trace_json(trace)

    mission_items = build_qgc_wpl_mission(

        meta,

        hist,

        home_lat_deg=home_lat,

        home_lon_deg=home_lon,

        home_alt_m=home_alt_m,

        takeoff_alt_m=takeoff_alt_m,

        cruise_alt_m=cruise_alt_m,

        waypoint_spacing_m=waypoint_spacing_m,

        land=land,

        prefer_trace_points=trace_route,

        approach_distance_m=approach_distance_m,

        descent_alt_m=descent_alt_m,

        use_spline=spline,

        speed_mps=speed_mps,

        acceptance_radius_m=acceptance_radius_m,

    )

    write_qgc_wpl_mission(out, mission_items)

    typer.echo(json.dumps(summarize_qgc_wpl_mission(mission_items, out_path=out, home_lat_deg=home_lat, home_lon_deg=home_lon), indent=2))





@bridge_app.command("sitl")

def bridge_sitl(

    trace: str = typer.Option("", "--trace", help="Planned trace JSON path to convert into setpoints"),

    script: str = typer.Option("", "--script", help="Existing bridge JSONL setpoint script path"),

    jsonl_out: str = typer.Option("", "--jsonl-out", help="Optional JSONL output when building from --trace"),

    connect: str = typer.Option("tcp:127.0.0.1:5760", "--connect", help="ArduPilot SITL MAVLink connection string"),

    hold_final_s: float = typer.Option(2.0, "--hold-final-s", help="How long to hold the final setpoint after the trace ends (s)"),

    rate_hz: float = typer.Option(20.0, "--rate-hz", help="Bridge output rate in Hz; use <=0 for native trace rate"),

    realtime: bool = typer.Option(True, "--realtime/--no-realtime", help="Honor trace timing or send setpoints as fast as possible"),

    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Print the resolved SITL bridge plan before executing"),

    execute: bool = typer.Option(False, "--execute/--no-execute", help="Actually connect to SITL after printing the resolved plan"),

    target_system: int = typer.Option(0, "--target-system", help="Target MAVLink system id. 0 = auto from heartbeat"),

    target_component: int = typer.Option(0, "--target-component", help="Target MAVLink component id. 0 = auto from heartbeat"),

    source_system: int = typer.Option(245, "--source-system", help="Source MAVLink system id for the bridge"),

    source_component: int = typer.Option(190, "--source-component", help="Source MAVLink component id for the bridge"),

    heartbeat_hz: float = typer.Option(1.0, "--heartbeat-hz", help="Heartbeat rate while streaming MAVLink setpoints"),

    wait_heartbeat_s: float = typer.Option(30.0, "--wait-heartbeat-s", help="How long to wait for the SITL heartbeat"),

    mode: str = typer.Option("GUIDED", "--mode", help="ArduPilot mode to enter before streaming; empty skips the mode change"),

    mode_timeout_s: float = typer.Option(10.0, "--mode-timeout-s", help="How long to wait for the requested mode"),

    arm: bool = typer.Option(True, "--arm/--no-arm", help="Arm SITL before streaming the setpoint script"),

    arm_timeout_s: float = typer.Option(15.0, "--arm-timeout-s", help="How long to wait for arming"),

    takeoff_alt_m: float = typer.Option(0.0, "--takeoff-alt-m", help="Optional GUIDED takeoff altitude before streaming; 0 disables takeoff"),

    takeoff_timeout_s: float = typer.Option(45.0, "--takeoff-timeout-s", help="How long to wait for the takeoff altitude"),

    takeoff_strict: bool = typer.Option(False, "--takeoff-strict/--no-takeoff-strict", help="Fail if SITL does not confirm climbing after the takeoff command"),

    land_mode: str = typer.Option("", "--land-mode", help="Optional mode to switch to after the script finishes, e.g. LAND"),

    land_timeout_s: float = typer.Option(20.0, "--land-timeout-s", help="How long to wait for the post-run mode change"),

    disarm: bool = typer.Option(False, "--disarm/--no-disarm", help="Disarm after the run finishes"),

    disarm_timeout_s: float = typer.Option(15.0, "--disarm-timeout-s", help="How long to wait for disarming"),

):

    if bool(trace) == bool(script):

        raise typer.BadParameter("provide exactly one of --trace or --script")

    if script and jsonl_out:

        raise typer.BadParameter("--jsonl-out can only be used with --trace")



    if trace:

        meta, hist = load_trace_json(trace)

        setpoints = build_setpoint_script(

            meta,

            hist,

            hold_final_s=hold_final_s,

            rate_hz=(None if rate_hz <= 0.0 else rate_hz),

        )

        summary = summarize_script(meta, setpoints)

        summary["input"] = {"trace": trace}

        if jsonl_out:

            write_script_jsonl(jsonl_out, setpoints, metadata=script_metadata_from_trace_meta(meta))

            summary["jsonl_out"] = jsonl_out

    else:

        script_metadata = load_script_jsonl_metadata(script)

        setpoints = load_script_jsonl(script)

        summary = summarize_setpoints(setpoints, metadata=script_metadata)

        summary["input"] = {"script": script}



    summary["sitl"] = {

        "connect": connect,

        "target_system": (target_system or "auto"),

        "target_component": (target_component or "auto"),

        "source_system": source_system,

        "source_component": source_component,

        "mode": mode or None,

        "arm": arm,

        "takeoff_alt_m": takeoff_alt_m,

        "takeoff_strict": takeoff_strict,

        "land_mode": land_mode or None,

        "disarm": disarm,

        "realtime": realtime,

        "execute": execute,

    }



    if dry_run or not execute:

        typer.echo(json.dumps(summary, indent=2))

    if not execute:

        return



    result = stream_script_ardupilot_sitl(

        setpoints,

        connect=connect,

        target_system=target_system,

        target_component=target_component,

        source_system=source_system,

        source_component=source_component,

        heartbeat_hz=heartbeat_hz,

        realtime=realtime,

        wait_heartbeat_s=wait_heartbeat_s,

        mode=mode,

        mode_timeout_s=mode_timeout_s,

        arm=arm,

        arm_timeout_s=arm_timeout_s,

        takeoff_alt_m=takeoff_alt_m,

        takeoff_timeout_s=takeoff_timeout_s,

        takeoff_strict=takeoff_strict,

        land_mode=land_mode,

        land_timeout_s=land_timeout_s,

        disarm=disarm,

        disarm_timeout_s=disarm_timeout_s,

    )

    typer.echo(json.dumps({"sitl_result": result}, indent=2))



def parse_mission_obstacles(specs: list[str]) -> list[MissionObstacle]:

    obstacles: list[MissionObstacle] = []

    for spec in specs:

        parts = [p.strip() for p in spec.split(",") if p.strip()]

        if len(parts) < 3 or len(parts) > 6:

            raise typer.BadParameter("--obstacle must be x,y,radius[,k[,swirl[,influence]]]")

        try:

            x_m = float(parts[0])

            y_m = float(parts[1])

            radius_m = float(parts[2])

            k_n_per_m = float(parts[3]) if len(parts) >= 4 else 180.0

            swirl_n = float(parts[4]) if len(parts) >= 5 else 900.0

            influence_m = float(parts[5]) if len(parts) >= 6 else max(20.0, 1.5 * radius_m)

        except ValueError as exc:

            raise typer.BadParameter(f"invalid --obstacle '{spec}'") from exc

        obstacles.append(MissionObstacle(x_m=x_m, y_m=y_m, radius_m=radius_m, k_n_per_m=k_n_per_m, swirl_n=swirl_n, influence_m=influence_m))

    return obstacles





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

    topology = default_ring_topology(32)

    effectiveness = effectiveness_table_for_topology(topology)

    out = {

        "allocator_version": version,

        "dir_deg": dir_deg,

        "fxy_n": fxy_n,

        "mz_nm_cmd": mz_nm,

        "duration_s": duration_s,

        "yaw_hold_deg": yaw_hold_deg,

        "yaw_track_coupling_mean_abs_deg": coupling,

        "hardware_assumptions": {

            "topology": summarize_topology(topology),

            "effectiveness": summarize_effectiveness_table(effectiveness),

        },

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

    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or slow-flap-0"),

    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),

    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),

    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),

    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),

    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),

    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),

):

    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)

    fault = build_fault_spec(normalized_fault_case)

    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)

    pl = PlenumModel(tau_s=plenum_tau_s)

    base_kwargs = {

        'dir_a_deg': dir_a_deg,

        'dir_b_deg': dir_b_deg,

        'fxy_n': fxy_n,

        'step_time_s': step_time_s,

        'total_s': total_s,

        'yaw_hold_deg': yaw_hold_deg,

        'mz_nm': mz_nm,

        'lim': lim,

        'pl': pl,

        'fault': fault,

    }

    if maneuver_safe:

        out, _hist, assessment, tuning = tune_maneuver_profile(

            'step',

            base_kwargs,

            target_continuous_power_p95_pct=power_target_pct,

            target_flap_peak_pct=flap_target_pct,

            target_fan_tracking_pct=fan_target_pct,

            target_xy_tracking_rms_n=xy_target_n,

            target_yaw_coupling_deg=yaw_target_deg,

            min_aggressiveness=tune_min_aggressiveness,

            max_iterations=tune_iterations,

        )

        out['maneuver_tuning'] = tuning

    else:

        out, _hist = _run_maneuver_profile('step', **base_kwargs)

        assessment = assess_maneuver_result(out, _hist, 'step')

    out['fault'] = summarize_fault_case(normalized_fault_case, fault)

    out['maneuver_assessment'] = assessment

    if trace_out:

        save_trace_json(trace_out, meta=out, hist=_hist)

    typer.echo(json.dumps(out, indent=2))





@alloc_app.command("coordinates")

def alloc_coordinates(

    preset: str = typer.Option("", "--preset", help="Mission preset: medium, long, or very-long"),

    dest_x_m: Optional[float] = typer.Option(None, "--dest-x", help="Destination east offset from home (m)"),

    dest_y_m: Optional[float] = typer.Option(None, "--dest-y", help="Destination north offset from home (m)"),

    dest_z_m: float = typer.Option(0.0, "--dest-z", help="Destination landing altitude relative to home (m)"),

    start_x_m: float = typer.Option(0.0, "--start-x", help="Start east offset from home (m)"),

    start_y_m: float = typer.Option(0.0, "--start-y", help="Start north offset from home (m)"),

    start_z_m: float = typer.Option(0.0, "--start-z", help="Start altitude relative to home (m)"),

    total_s: float = typer.Option(30.0, "--total-s", help="Mission sim duration (s)"),

    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),

    mz_nm: float = typer.Option(0.0, "--mz-nm", help="Desired yaw moment (N*m). Keep 0 for UFO-style no-yaw motion"),

    cruise_alt_m: float = typer.Option(12.0, "--cruise-alt-m", help="Transit altitude before final approach (m)"),

    max_speed_mps: float = typer.Option(8.0, "--max-speed-mps", help="Horizontal cruise speed cap (m/s)"),

    arrival_radius_m: float = typer.Option(1.5, "--arrival-radius-m", help="Horizontal arrival radius (m)"),

    slow_radius_m: float = typer.Option(14.0, "--slow-radius-m", help="Radius where the craft starts slowing for arrival (m)"),

    descent_radius_m: float = typer.Option(18.0, "--descent-radius-m", help="Radius where the craft starts descending to the landing altitude (m)"),

    hold_s: float = typer.Option(2.0, "--hold-s", help="How long to hold at the destination before ending the trace (s)"),

    fxy_max_n: float = typer.Option(3500.0, "--fxy-max", help="Maximum horizontal force command (N)"),

    pos_k_n_per_m: float = typer.Option(120.0, "--pos-k", help="Position guidance gain (N/m)"),

    vel_k_n_per_mps: float = typer.Option(900.0, "--vel-k", help="Velocity damping gain (N per m/s)"),

    obstacle: list[str] = typer.Option([], "--obstacle", help="Obstacle as x,y,radius[,k[,swirl[,influence]]]. Repeat for multiple obstacles."),

    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),

    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),

    power_safe: bool = typer.Option(False, "--power-safe/--no-power-safe", help="Auto-tune mission aggressiveness against sustained power and flap targets"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --power-safe is enabled"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --power-safe is enabled"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --power-safe envelope search"),

    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the power-safe search is allowed to test"),

    battery_full_v: Optional[float] = typer.Option(None, "--battery-full-v", help="Override full-pack voltage for the power model"),

    battery_empty_v: Optional[float] = typer.Option(None, "--battery-empty-v", help="Override empty-pack voltage for the power model"),

    capacity_kwh: Optional[float] = typer.Option(None, "--capacity-kwh", help="Override battery capacity for the power model"),

    internal_resistance_ohm: Optional[float] = typer.Option(None, "--internal-resistance-ohm", help="Override pack internal resistance for the power model"),

    hover_power_kw: Optional[float] = typer.Option(None, "--hover-power-kw", help="Override hover power draw for the power model"),

    continuous_power_kw: Optional[float] = typer.Option(None, "--continuous-power-kw", help="Override continuous power capability for the power model"),

    peak_power_kw: Optional[float] = typer.Option(None, "--peak-power-kw", help="Override peak power capability for the power model"),

    aux_power_kw: Optional[float] = typer.Option(None, "--aux-power-kw", help="Override auxiliary power draw for the power model"),

    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or dead-fan-0+stuck-flap-0"),

    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),

):

    preset_cfg = {}

    if preset:

        if preset not in COORDINATE_PRESETS:

            raise typer.BadParameter("--preset must be one of: medium, long, very-long")

        preset_cfg = COORDINATE_PRESETS[preset]

    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)

    fault_case = normalized_fault_case



    def pick(current, key, default):

        if current is None:

            return preset_cfg.get(key, default)

        if preset_cfg and current == default:

            return preset_cfg.get(key, current)

        return current



    dest_x_m = pick(dest_x_m, "dest_x_m", None)

    dest_y_m = pick(dest_y_m, "dest_y_m", None)

    if dest_x_m is None or dest_y_m is None:

        raise typer.BadParameter("--dest-x and --dest-y are required unless --preset is used")



    dest_z_m = pick(dest_z_m, "dest_z_m", 0.0)

    total_s = pick(total_s, "total_s", 30.0)

    cruise_alt_m = pick(cruise_alt_m, "cruise_alt_m", 12.0)

    max_speed_mps = pick(max_speed_mps, "max_speed_mps", 8.0)

    arrival_radius_m = pick(arrival_radius_m, "arrival_radius_m", 1.5)

    slow_radius_m = pick(slow_radius_m, "slow_radius_m", 14.0)

    descent_radius_m = pick(descent_radius_m, "descent_radius_m", 18.0)

    hold_s = pick(hold_s, "hold_s", 2.0)

    fxy_max_n = pick(fxy_max_n, "fxy_max_n", 3500.0)

    pos_k_n_per_m = pick(pos_k_n_per_m, "pos_k_n_per_m", 120.0)

    vel_k_n_per_mps = pick(vel_k_n_per_mps, "vel_k_n_per_mps", 900.0)

    if preset_cfg and not obstacle:

        obstacle = list(preset_cfg.get("obstacles", []))



    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)

    pl = PlenumModel(tau_s=plenum_tau_s)

    obstacles = parse_mission_obstacles(obstacle)

    power = build_power_params(

        battery_full_v=battery_full_v,

        battery_empty_v=battery_empty_v,

        capacity_kwh=capacity_kwh,

        internal_resistance_ohm=internal_resistance_ohm,

        hover_power_kw=hover_power_kw,

        continuous_power_kw=continuous_power_kw,

        peak_power_kw=peak_power_kw,

        aux_power_kw=aux_power_kw,

    )

    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)

    fault = build_fault_spec(normalized_fault_case)



    mission_kwargs = build_coordinate_mission_kwargs(

        dest_x_m=dest_x_m,

        dest_y_m=dest_y_m,

        dest_z_m=dest_z_m,

        start_x_m=start_x_m,

        start_y_m=start_y_m,

        start_z_m=start_z_m,

        total_s=total_s,

        yaw_hold_deg=yaw_hold_deg,

        mz_nm=mz_nm,

        cruise_alt_m=cruise_alt_m,

        max_speed_mps=max_speed_mps,

        arrival_radius_m=arrival_radius_m,

        slow_radius_m=slow_radius_m,

        descent_radius_m=descent_radius_m,

        hold_s=hold_s,

        fxy_max_n=fxy_max_n,

        pos_k_n_per_m=pos_k_n_per_m,

        vel_k_n_per_mps=vel_k_n_per_mps,

        obstacles=obstacles,

        lim=lim,

        pl=pl,

        power=power,

        fault=fault,

    )



    if power_safe:

        out, _hist, report, tuning = tune_coordinate_mission(

            mission_kwargs,

            target_continuous_power_p95_pct=power_target_pct,

            target_flap_peak_pct=flap_target_pct,

            min_aggressiveness=tune_min_aggressiveness,

            max_iterations=tune_iterations,

        )

        tuning["report_status"] = report.get("status")

        tuning["report_warnings"] = list(report.get("warnings", []))

        out["tuning"] = tuning

    else:

        out, _hist = run_coordinate_mission_v5(**mission_kwargs)

    if preset:

        out.setdefault("mission", {})["preset"] = preset

    out["power_model"] = summarize_power_params(power)

    out["fault_case"] = summarize_fault_case(normalized_fault_case, fault)

    if trace_out:

        save_trace_json(trace_out, meta=out, hist=_hist)

    typer.echo(json.dumps(out, indent=2))





def build_power_sweep_report(

    *,

    preset: str,

    fault_case: str,

    required_status: str,

    power_target_pct: float,

    flap_target_pct: float,

    tune_iterations: int,

    tune_min_aggressiveness: float,

    battery_full_v: list[float],

    battery_empty_v: list[float],

    capacity_kwh: list[float],

    internal_resistance_ohm: list[float],

    hover_power_kw: list[float],

    continuous_power_kw: list[float],

    peak_power_kw: list[float],

    aux_power_kw: list[float],

    top: int,

) -> tuple[dict, dict | None]:

    if preset not in COORDINATE_PRESETS:

        raise typer.BadParameter('--preset must be one of: medium, long, very-long')

    preset_cfg = COORDINATE_PRESETS[preset]

    obstacles = parse_mission_obstacles(list(preset_cfg.get('obstacles', [])))

    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)

    fault = build_fault_spec(normalized_fault_case)

    fault_summary = summarize_fault_case(normalized_fault_case, fault)

    return build_power_sweep_report_for_fault(

        preset=preset,

        preset_cfg=preset_cfg,

        obstacles=obstacles,

        fault_label=normalized_fault_case,

        fault=fault,

        fault_summary=fault_summary,

        required_status=required_status,

        power_target_pct=power_target_pct,

        flap_target_pct=flap_target_pct,

        tune_iterations=tune_iterations,

        tune_min_aggressiveness=tune_min_aggressiveness,

        battery_full_v=battery_full_v,

        battery_empty_v=battery_empty_v,

        capacity_kwh=capacity_kwh,

        internal_resistance_ohm=internal_resistance_ohm,

        hover_power_kw=hover_power_kw,

        continuous_power_kw=continuous_power_kw,

        peak_power_kw=peak_power_kw,

        aux_power_kw=aux_power_kw,

        top=top,

    )





@alloc_app.command("effectiveness-report")
def alloc_effectiveness_report(
    spec: str = typer.Option("", "--spec", help="Optional geometry-seed spec JSON; defaults to the built-in Aurora ring32 seed"),
    table_in: str = typer.Option("", "--table-in", help="Optional nominal effectiveness table JSON to inspect instead of building from --spec"),
    out_dir: str = typer.Option("", "--out-dir", help="Optional directory for summary and materialized-table artifacts"),
    summary_out: str = typer.Option("", "--summary-out", help="Optional summary report path (.json, .md, .txt)"),
    summary_format: str = typer.Option("auto", "--summary-format", help="Summary output format: auto, json, markdown, text"),
    table_out: str = typer.Option("", "--table-out", help="Optional materialized table JSON path"),
    source_out: str = typer.Option("", "--source-out", help="Optional normalized source spec/table JSON path"),
):
    try:
        report, table, source_spec = build_effectiveness_report(
            spec_path=spec or None,
            table_path=table_in or None,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    report = write_effectiveness_report_outputs(
        report,
        table,
        spec=source_spec,
        out_dir=out_dir,
        summary_out=summary_out,
        summary_format=summary_format,
        table_out=table_out,
        source_out=source_out,
    )
    typer.echo(json.dumps(report, indent=2))


@alloc_app.command("effectiveness-template")
def alloc_effectiveness_template(
    out_dir: str = typer.Option("runs/effectiveness_candidate_template", "--out-dir", help="Directory for a starter candidate spec and provenance note"),
    spec_out: str = typer.Option("", "--spec-out", help="Optional output path for the candidate spec JSON"),
    note_out: str = typer.Option("", "--note-out", help="Optional output path for the provenance note markdown"),
    spec_name: str = typer.Option("", "--spec-name", help="Optional candidate spec name override"),
    provenance: str = typer.Option("", "--provenance", help="Optional provenance placeholder override"),
):
    try:
        report = write_effectiveness_candidate_template_outputs(
            out_dir=out_dir,
            spec_out=spec_out,
            note_out=note_out,
            spec_name=spec_name or None,
            provenance=provenance or None,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report, indent=2))


@alloc_app.command("effectiveness-compare")
def alloc_effectiveness_compare(
    candidate_spec: str = typer.Option("", "--candidate-spec", help="Candidate geometry-seed spec JSON to compare against the current baseline"),
    candidate_table: str = typer.Option("", "--candidate-table", help="Candidate materialized effectiveness table JSON to compare against the current baseline"),
    baseline_spec: str = typer.Option("", "--baseline-spec", help="Optional baseline geometry-seed spec JSON; defaults to the built-in Aurora ring32 seed"),
    baseline_table: str = typer.Option("", "--baseline-table", help="Optional baseline materialized table JSON instead of --baseline-spec"),
    out_dir: str = typer.Option("", "--out-dir", help="Optional directory for summary and normalized comparison artifacts"),
    summary_out: str = typer.Option("", "--summary-out", help="Optional comparison summary path (.json, .md, .txt)"),
    summary_format: str = typer.Option("auto", "--summary-format", help="Comparison summary output format: auto, json, markdown, text"),
):
    try:
        report, baseline_table_obj, baseline_spec_obj, candidate_table_obj, candidate_spec_obj = build_effectiveness_comparison_report(
            candidate_spec_path=candidate_spec or None,
            candidate_table_path=candidate_table or None,
            baseline_spec_path=baseline_spec or None,
            baseline_table_path=baseline_table or None,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    report = write_effectiveness_comparison_outputs(
        report,
        baseline_table_obj,
        candidate_table_obj,
        baseline_spec=baseline_spec_obj,
        candidate_spec=candidate_spec_obj,
        out_dir=out_dir,
        summary_out=summary_out,
        summary_format=summary_format,
    )
    typer.echo(json.dumps(report, indent=2))


@alloc_app.command("power-sweep")

def alloc_power_sweep(

    preset: str = typer.Option("medium", "--preset", help="Coordinate mission preset to evaluate"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage for the sustainable envelope search"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage for the sustainable envelope search"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for the sustainable envelope search"),

    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the sustainable envelope search is allowed to test"),

    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages"),

    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages"),

    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities"),

    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances"),

    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions"),

    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities"),

    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities"),

    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws"),

    top: int = typer.Option(10, "--top", help="How many ranked sweep results to print"),

    out: str = typer.Option("", "--out", help="Optional JSON output path for the sweep report"),

    requirements_out: str = typer.Option("", "--requirements-out", help="Optional compact requirements-table output path (.csv, .md, .txt)"),

    requirements_format: str = typer.Option("auto", "--requirements-format", help="Requirements-table output format: auto, csv, markdown, text"),

    best_trace_out: str = typer.Option("", "--best-trace-out", help="Optional JSON trace path for the best sweep candidate"),

    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or dead-fan-0+stuck-flap-0"),

):

    report, best_candidate = build_power_sweep_report(

        preset=preset,

        fault_case=fault_case,

        required_status='feasible',

        power_target_pct=power_target_pct,

        flap_target_pct=flap_target_pct,

        tune_iterations=tune_iterations,

        tune_min_aggressiveness=tune_min_aggressiveness,

        battery_full_v=battery_full_v,

        battery_empty_v=battery_empty_v,

        capacity_kwh=capacity_kwh,

        internal_resistance_ohm=internal_resistance_ohm,

        hover_power_kw=hover_power_kw,

        continuous_power_kw=continuous_power_kw,

        peak_power_kw=peak_power_kw,

        aux_power_kw=aux_power_kw,

        top=top,

    )



    if best_trace_out and best_candidate is not None:

        save_trace_json(best_trace_out, meta=best_candidate['meta'], hist=best_candidate['hist'])

        report['best_trace_out'] = best_trace_out



    title_label = preset if fault_case == 'nominal' else f"{preset} {fault_case}"

    write_power_sweep_outputs(

        report,

        preset=preset,

        out=out,

        requirements_out=requirements_out,

        requirements_format=requirements_format,

        title_label=title_label,

    )

    typer.echo(json.dumps(report, indent=2))





@alloc_app.command("sizing-pack")

def alloc_sizing_pack(

    preset: list[str] = typer.Option([], "--preset", help="Repeat to limit presets; defaults to medium, long, and very-long"),

    fault_case: list[str] = typer.Option([], "--fault-case", help="Repeat to include fault cases or composites like dead-fan-0+stuck-flap-0; overrides --fault-set when provided"),

    fault_set: str = typer.Option("nominal", "--fault-set", help="Fault set to use when --fault-case is not provided: nominal, core, stress, all"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage for the sustainable envelope search"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage for the sustainable envelope search"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for the sustainable envelope search"),

    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the sustainable envelope search is allowed to test"),

    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages"),

    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages"),

    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities"),

    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances"),

    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions"),

    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities"),

    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities"),

    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws"),

    top: int = typer.Option(10, "--top", help="How many ranked sweep results to retain per preset"),

    out_dir: str = typer.Option("runs/power_sizing_pack", "--out-dir", help="Directory for per-preset and combined sizing artifacts"),

):

    selected_presets = list(preset) if preset else list(COORDINATE_PRESETS.keys())

    invalid = [name for name in selected_presets if name not in COORDINATE_PRESETS]

    if invalid:

        raise typer.BadParameter("--preset must be one of: medium, long, very-long")

    selected_fault_cases = select_fault_cases(list(fault_case), fault_set)



    base_dir = Path(out_dir)

    base_dir.mkdir(parents=True, exist_ok=True)



    summary_rows = []

    preset_reports = []



    for preset_name in selected_presets:

        for fault_name in selected_fault_cases:

            report, best_candidate = build_power_sweep_report(

                preset=preset_name,

                fault_case=fault_name,

                required_status='feasible',

                power_target_pct=power_target_pct,

                flap_target_pct=flap_target_pct,

                tune_iterations=tune_iterations,

                tune_min_aggressiveness=tune_min_aggressiveness,

                battery_full_v=battery_full_v,

                battery_empty_v=battery_empty_v,

                capacity_kwh=capacity_kwh,

                internal_resistance_ohm=internal_resistance_ohm,

                hover_power_kw=hover_power_kw,

                continuous_power_kw=continuous_power_kw,

                peak_power_kw=peak_power_kw,

                aux_power_kw=aux_power_kw,

                top=top,

            )



            file_stub = f'{preset_name}_{fault_name}'

            title_label = preset_name if fault_name == 'nominal' else f"{preset_name} {fault_name}"

            preset_json = base_dir / f'{file_stub}_power_requirements.json'

            preset_md = base_dir / f'{file_stub}_power_requirements.md'

            preset_csv = base_dir / f'{file_stub}_power_requirements.csv'

            best_trace = base_dir / f'{file_stub}_best_trace.json'



            if best_candidate is not None:

                save_trace_json(str(best_trace), meta=best_candidate['meta'], hist=best_candidate['hist'])

                report['best_trace_out'] = str(best_trace)



            write_power_sweep_outputs(

                report,

                preset=preset_name,

                out=str(preset_json),

                requirements_out=str(preset_md),

                requirements_format='markdown',

                title_label=title_label,

            )

            render_csv = render_requirements_table(

                report['requirements_table'],

                format_name='csv',

                preset=title_label,

            )

            preset_csv.write_text(render_csv, encoding='utf-8')



            best = report.get('best') or {}

            best_model = best.get('power_model', {})

            summary_row = {

                'preset': preset_name,

                'fault_case': fault_name,

                'cases': report.get('search_space', {}).get('cases'),

                'feasible_cases': report.get('search_space', {}).get('feasible_cases'),

                'best_feasible': bool(best.get('feasible', False)),

                'battery_full_v': best_model.get('battery_full_v'),

                'capacity_kwh': best_model.get('capacity_kwh'),

                'internal_resistance_ohm': best_model.get('internal_resistance_ohm'),

                'hover_power_kw': best_model.get('hover_power_kw'),

                'continuous_power_kw': best_model.get('continuous_power_kw'),

                'peak_power_kw': best_model.get('peak_power_kw'),

                'selected_aggressiveness': best.get('selected_aggressiveness'),

                'report_status': best.get('report_status'),

                'continuous_power_p95_pct': best.get('continuous_power_p95_pct'),

                'final_goal_error_m': best.get('final_goal_error_m'),

                'arrival_time_s': best.get('arrival_time_s'),

            }

            summary_rows.append(summary_row)

            preset_reports.append({

                'preset': preset_name,

                'fault_case': report.get('fault_case', {}),

                'search_space': report.get('search_space', {}),

                'best': best,

                'artifacts': {

                    'json': str(preset_json),

                    'markdown': str(preset_md),

                    'csv': str(preset_csv),

                    'best_trace': (str(best_trace) if best_candidate is not None else ''),

                },

            })



    summary_rows.sort(key=lambda row: (

        0 if row.get('best_feasible') else 1,

        row.get('fault_case', ''),

        row.get('continuous_power_kw') if row.get('continuous_power_kw') is not None else 1e9,

        row.get('preset', ''),

    ))



    summary_report = {

        'targets': {

            'continuous_power_p95_pct': float(power_target_pct),

            'flap_limit_usage_peak_pct': float(flap_target_pct),

            'tune_iterations': int(tune_iterations),

            'tune_min_aggressiveness': float(tune_min_aggressiveness),

        },

        'presets': preset_reports,

        'summary_rows': summary_rows,

    }



    summary_json = base_dir / 'summary.json'

    summary_md = base_dir / 'summary.md'

    summary_csv = base_dir / 'summary.csv'

    summary_json.write_text(json.dumps(summary_report, indent=2), encoding='utf-8')

    summary_md.write_text(

        render_preset_summary_table(summary_rows, format_name='markdown', title='power sizing pack summary'),

        encoding='utf-8',

    )

    summary_csv.write_text(

        render_preset_summary_table(summary_rows, format_name='csv', title='power sizing pack summary'),

        encoding='utf-8',

    )

    summary_report['summary_artifacts'] = {

        'json': str(summary_json),

        'markdown': str(summary_md),

        'csv': str(summary_csv),

    }

    typer.echo(json.dumps(summary_report, indent=2))





@alloc_app.command("fault-envelope")

def alloc_fault_envelope(

    preset: str = typer.Option("very-long", "--preset", help="Coordinate mission preset to evaluate"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage for the sustainable envelope search"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage for the sustainable envelope search"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for the sustainable envelope search"),

    tune_min_aggressiveness: float = typer.Option(0.25, "--tune-min-aggr", help="Lowest aggressiveness the sustainable envelope search is allowed to test"),

    battery_full_v: Optional[float] = typer.Option(None, "--battery-full-v", help="Override full-pack voltage for the power model"),

    battery_empty_v: Optional[float] = typer.Option(None, "--battery-empty-v", help="Override empty-pack voltage for the power model"),

    capacity_kwh: Optional[float] = typer.Option(None, "--capacity-kwh", help="Override battery capacity for the power model"),

    internal_resistance_ohm: Optional[float] = typer.Option(None, "--internal-resistance-ohm", help="Override pack internal resistance for the power model"),

    hover_power_kw: Optional[float] = typer.Option(None, "--hover-power-kw", help="Override hover power draw for the power model"),

    continuous_power_kw: Optional[float] = typer.Option(None, "--continuous-power-kw", help="Override continuous power capability for the power model"),

    peak_power_kw: Optional[float] = typer.Option(None, "--peak-power-kw", help="Override peak power capability for the power model"),

    aux_power_kw: Optional[float] = typer.Option(None, "--aux-power-kw", help="Override auxiliary power draw for the power model"),

    dead_fan_scale: list[float] = typer.Option([], "--dead-fan-scale", help="Repeat to test multiple dead-fan severity values; default 0.0"),

    stuck_flap_alpha_deg: list[float] = typer.Option([], "--stuck-flap-alpha-deg", help="Repeat to test multiple stuck-flap angles; default 20 deg"),

    plenum_sector_scale: list[float] = typer.Option([], "--plenum-sector-scale", help="Repeat to test multiple plenum-sector thrust scales; default 0.7"),

    fan_group: list[int] = typer.Option([], "--fan-group", help="Repeat to limit fan groups; default all 16 groups"),

    flap_idx: list[int] = typer.Option([], "--flap-idx", help="Repeat to limit flap indices; default uses the coarse grid from --flap-step"),

    plenum_sector_idx: list[int] = typer.Option([], "--plenum-sector-idx", help="Repeat to limit plenum sectors; default uses the coarse grid from --plenum-step"),

    flap_step: int = typer.Option(4, "--flap-step", help="Stride for default flap-index sampling when --flap-idx is not provided"),

    plenum_step: int = typer.Option(4, "--plenum-step", help="Stride for default plenum-sector sampling when --plenum-sector-idx is not provided"),

    top_per_family: int = typer.Option(2, "--top-per-family", help="Worst single-fault cases per family to promote into pair and triple combinations"),

    include_pairs: bool = typer.Option(True, "--pairs/--no-pairs", help="Evaluate pair combinations built from the worst single-fault cases"),

    include_triples: bool = typer.Option(False, "--triples/--no-triples", help="Evaluate triple combinations built from the worst single-fault cases"),

    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reuse cached fault-envelope cases from the output directory when available"),

    max_new_cases: int = typer.Option(0, "--max-new-cases", help="Limit newly evaluated cases per run for resumable searches; 0 means no limit"),

    top: int = typer.Option(12, "--top", help="How many ranked worst cases to keep in the summary"),

    save_traces: int = typer.Option(5, "--save-traces", help="How many worst-case traces to write into the output directory"),

    out_dir: str = typer.Option("runs/fault_envelope", "--out-dir", help="Directory for the fault-envelope report and worst-case traces"),

):

    report = run_fault_envelope_report(
        preset=preset,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
        dead_fan_scale=dead_fan_scale,
        stuck_flap_alpha_deg=stuck_flap_alpha_deg,
        plenum_sector_scale=plenum_sector_scale,
        fan_group=fan_group,
        flap_idx=flap_idx,
        plenum_sector_idx=plenum_sector_idx,
        flap_step=flap_step,
        plenum_step=plenum_step,
        top_per_family=top_per_family,
        include_pairs=include_pairs,
        include_triples=include_triples,
        resume=resume,
        max_new_cases=max_new_cases,
        top=top,
        save_traces=save_traces,
        out_dir=out_dir,
        preset_context_resolver=resolve_coordinate_preset_context,
    )
    typer.echo(json.dumps(report, indent=2))







@alloc_app.command("fault-threshold")

def alloc_fault_threshold(

    summary: str = typer.Option(..., "--summary", help="Path to a completed fault-envelope summary.json"),

    case: list[str] = typer.Option([], "--case", help="Repeat to select specific cases from the supplied fault-envelope summary"),

    top_cases: int = typer.Option(3, "--top-cases", help="How many top fault-envelope cases to evaluate when --case is not provided"),

    include_family_worst: bool = typer.Option(True, "--include-family-worst/--no-include-family-worst", help="Also include the worst single case per family from the supplied summary"),

    required_status: str = typer.Option('feasible', "--required-status", help="Threshold qualification: feasible, pass, caution, or risk"),

    power_target_pct: Optional[float] = typer.Option(None, "--power-target-pct", help="Override the target p95 continuous-power percentage; defaults to the summary target"),

    flap_target_pct: Optional[float] = typer.Option(None, "--flap-target-pct", help="Override the target peak flap-limit usage percentage; defaults to the summary target"),

    tune_iterations: Optional[int] = typer.Option(None, "--tune-iters", help="Override the tuning iterations; defaults to the summary target"),

    tune_min_aggressiveness: Optional[float] = typer.Option(None, "--tune-min-aggr", help="Override the minimum aggressiveness; defaults to the summary target"),

    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages; defaults to the summary power model"),

    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages; defaults to the summary power model"),

    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities; defaults to the summary power model"),

    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances; defaults to the summary power model"),

    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions; defaults to the summary power model"),

    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities; defaults to a threshold sweep around the summary power model"),

    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities; defaults to the summary power model"),

    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws; defaults to the summary power model"),

    top: int = typer.Option(10, "--top", help="How many ranked sweep results to retain per fault case"),

    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reuse cached fault-threshold cases from the output directory when available"),

    max_new_cases: int = typer.Option(0, "--max-new-cases", help="Limit newly evaluated fault-threshold cases per run; 0 means no limit"),

    out_dir: str = typer.Option("runs/fault_threshold", "--out-dir", help="Directory for the worst-case threshold artifacts"),

):

    summary_report = run_fault_threshold_report(
        summary=summary,
        case_names=list(case),
        top_cases=top_cases,
        include_family_worst=include_family_worst,
        required_status=required_status,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
        top=top,
        resume=resume,
        max_new_cases=max_new_cases,
        out_dir=out_dir,
        preset_context_resolver=resolve_coordinate_preset_context,
    )

    typer.echo(json.dumps(summary_report, indent=2))





@alloc_app.command("fault-threshold-pack")
def alloc_fault_threshold_pack(
    summary: list[str] = typer.Option([], "--summary", help="Repeat to provide completed fault-envelope summary.json files; defaults to auto-discovering runs/fault_envelope*/summary.json"),
    preset: list[str] = typer.Option([], "--preset", help="Optional preset filter when auto-discovering or limiting supplied summaries"),
    case: list[str] = typer.Option([], "--case", help="Repeat to select specific cases from every supplied summary"),
    top_cases: int = typer.Option(3, "--top-cases", help="How many top fault-envelope cases to evaluate per summary when --case is not provided"),
    include_family_worst: bool = typer.Option(True, "--include-family-worst/--no-include-family-worst", help="Also include the worst single case per family from each supplied summary"),
    required_status: str = typer.Option('feasible', "--required-status", help="Threshold qualification: feasible, pass, caution, or risk"),
    power_target_pct: Optional[float] = typer.Option(None, "--power-target-pct", help="Override the target p95 continuous-power percentage; defaults to each summary target"),
    flap_target_pct: Optional[float] = typer.Option(None, "--flap-target-pct", help="Override the target peak flap-limit usage percentage; defaults to each summary target"),
    tune_iterations: Optional[int] = typer.Option(None, "--tune-iters", help="Override the tuning iterations; defaults to each summary target"),
    tune_min_aggressiveness: Optional[float] = typer.Option(None, "--tune-min-aggr", help="Override the minimum aggressiveness; defaults to each summary target"),
    battery_full_v: list[float] = typer.Option([], "--battery-full-v", help="Repeat to sweep multiple full-pack voltages; defaults to each summary power model"),
    battery_empty_v: list[float] = typer.Option([], "--battery-empty-v", help="Repeat to sweep multiple empty-pack voltages; defaults to each summary power model"),
    capacity_kwh: list[float] = typer.Option([], "--capacity-kwh", help="Repeat to sweep multiple battery capacities; defaults to each summary power model"),
    internal_resistance_ohm: list[float] = typer.Option([], "--internal-resistance-ohm", help="Repeat to sweep multiple pack internal resistances; defaults to each summary power model"),
    hover_power_kw: list[float] = typer.Option([], "--hover-power-kw", help="Repeat to sweep multiple hover power assumptions; defaults to each summary power model"),
    continuous_power_kw: list[float] = typer.Option([], "--continuous-power-kw", help="Repeat to sweep multiple continuous power capabilities; defaults to a threshold sweep around each summary power model"),
    peak_power_kw: list[float] = typer.Option([], "--peak-power-kw", help="Repeat to sweep multiple peak power capabilities; defaults to each summary power model"),
    aux_power_kw: list[float] = typer.Option([], "--aux-power-kw", help="Repeat to sweep multiple auxiliary power draws; defaults to each summary power model"),
    top: int = typer.Option(10, "--top", help="How many ranked sweep results to retain per fault case"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reuse cached fault-threshold cases from each preset output directory when available"),
    max_new_cases: int = typer.Option(0, "--max-new-cases", help="Limit newly evaluated fault-threshold cases per summary run; 0 means no limit"),
    max_new_cases_total: int = typer.Option(0, "--max-new-cases-total", help="Limit newly evaluated fault-threshold cases across the whole pack run; 0 means no limit"),
    out_dir: str = typer.Option("runs/fault_threshold_pack", "--out-dir", help="Directory for per-summary and combined threshold artifacts"),
):
    pack_report = run_fault_threshold_pack_report(
        summary_paths=list(summary),
        preset_names=list(preset),
        case_names=list(case),
        top_cases=top_cases,
        include_family_worst=include_family_worst,
        required_status=required_status,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        battery_full_v=battery_full_v,
        battery_empty_v=battery_empty_v,
        capacity_kwh=capacity_kwh,
        internal_resistance_ohm=internal_resistance_ohm,
        hover_power_kw=hover_power_kw,
        continuous_power_kw=continuous_power_kw,
        peak_power_kw=peak_power_kw,
        aux_power_kw=aux_power_kw,
        top=top,
        resume=resume,
        max_new_cases=max_new_cases,
        max_new_cases_total=max_new_cases_total,
        out_dir=out_dir,
        preset_context_resolver=resolve_coordinate_preset_context,
    )
    typer.echo(json.dumps(pack_report, indent=2))


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





@alloc_app.command("step-redirect")

def alloc_step_redirect(

    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg"),

    dir_b_deg: float = typer.Option(90.0, "--dir-b-deg"),

    fxy_n: float = typer.Option(2200.0, "--fxy"),

    step_time_s: float = typer.Option(3.0, "--step-time-s"),

    redirect_time_s: float = typer.Option(1.6, "--redirect-time-s", help="Duration of the smooth redirect blend phase"),

    total_s: float = typer.Option(9.0, "--total-s"),

    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),

    mz_nm: float = typer.Option(0.0, "--mz-nm"),

    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),

    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),

    redirect_speed_scale: float = typer.Option(0.88, "--redirect-speed-scale", help="Fraction of the pre-step speed preserved through the redirect blend"),

    redirect_cross_gain: float = typer.Option(1.0, "--redirect-cross-gain", help="Cross-velocity cleanup gain during the redirect blend"),

    eco: bool = typer.Option(False, "--eco/--no-eco", help="Use the dedicated low-power eco redirect profile"),

    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or weak-flap-0"),

    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),

    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),

    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),

    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),

    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),

    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),

):

    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)

    fault = build_fault_spec(normalized_fault_case)

    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)

    pl = PlenumModel(tau_s=plenum_tau_s)

    base_kwargs = {

        'dir_a_deg': dir_a_deg,

        'dir_b_deg': dir_b_deg,

        'fxy_n': fxy_n,

        'step_time_s': step_time_s,

        'redirect_time_s': redirect_time_s,

        'total_s': total_s,

        'yaw_hold_deg': yaw_hold_deg,

        'mz_nm': mz_nm,

        'lim': lim,

        'pl': pl,

        'fault': fault,

        'redirect_speed_scale': redirect_speed_scale,

        'redirect_cross_gain': redirect_cross_gain,

    }

    profile_name = 'step-redirect-eco' if eco else 'step-redirect'

    if maneuver_safe:

        out, _hist, assessment, tuning = tune_maneuver_profile(

            profile_name,

            base_kwargs,

            target_continuous_power_p95_pct=power_target_pct,

            target_flap_peak_pct=flap_target_pct,

            target_fan_tracking_pct=fan_target_pct,

            target_xy_tracking_rms_n=xy_target_n,

            target_yaw_coupling_deg=yaw_target_deg,

            min_aggressiveness=tune_min_aggressiveness,

            max_iterations=tune_iterations,

        )

        out['maneuver_tuning'] = tuning

    else:

        out, _hist = _run_maneuver_profile(profile_name, **base_kwargs)

        assessment = assess_maneuver_result(out, _hist, profile_name)

    out['fault'] = summarize_fault_case(normalized_fault_case, fault)

    out['maneuver_assessment'] = assessment

    if trace_out:

        save_trace_json(trace_out, meta=out, hist=_hist)

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

    redirect_hold_frac: float = typer.Option(-1.0, "--redirect-hold-frac", help="Fraction of the snap window kept as pure braking before steering toward the new direction. Negative = auto"),

    redirect_steer_scale: float = typer.Option(1.0, "--redirect-steer-scale", help="Scaling applied to the snap revector blend during sharp redirects"),

    eco: bool = typer.Option(False, "--eco/--no-eco", help="Use the dedicated low-power eco snap profile"),

    fault_case: str = typer.Option("nominal", "--fault-case", help="Fault case component or composite, for example nominal or slow-flap-0"),

    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),

    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),

    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),

    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),

    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),

    trace_out: str = typer.Option("", "--trace-out", help="Optional JSON trace path for replay"),

):

    normalized_fault_case, _fault_spec = resolve_fault_case(fault_case)

    fault = build_fault_spec(normalized_fault_case)

    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)

    pl = PlenumModel(tau_s=plenum_tau_s)

    base_kwargs = {

        'dir_a_deg': dir_a_deg,

        'dir_b_deg': dir_b_deg,

        'fxy_n': fxy_n,

        'step_time_s': step_time_s,

        'snap_stop_s': snap_stop_s,

        'total_s': total_s,

        'yaw_hold_deg': yaw_hold_deg,

        'mz_nm': mz_nm,

        'lim': lim,

        'pl': pl,

        'fault': fault,

        'brake_gain': brake_gain,

        'speed_stop_thr_mps': speed_stop_thr_mps,

        'redirect_hold_frac': redirect_hold_frac,

        'redirect_steer_scale': redirect_steer_scale,

    }



    profile_name = 'step-snap-eco' if eco else 'step-snap'

    if maneuver_safe:

        out, _hist, assessment, tuning = tune_maneuver_profile(

            profile_name,

            base_kwargs,

            target_continuous_power_p95_pct=power_target_pct,

            target_flap_peak_pct=flap_target_pct,

            target_fan_tracking_pct=fan_target_pct,

            target_xy_tracking_rms_n=xy_target_n,

            target_yaw_coupling_deg=yaw_target_deg,

            min_aggressiveness=tune_min_aggressiveness,

            max_iterations=tune_iterations,

        )

        out['maneuver_tuning'] = tuning

    else:

        out, _hist = _run_maneuver_profile(profile_name, **base_kwargs)

        assessment = assess_maneuver_result(out, _hist, profile_name)

    out['fault'] = summarize_fault_case(normalized_fault_case, fault)

    out['maneuver_assessment'] = assessment

    if trace_out:

        save_trace_json(trace_out, meta=out, hist=_hist)

    typer.echo(json.dumps(out, indent=2))





@alloc_app.command("maneuver-pack")

def alloc_maneuver_pack(

    profile: str = typer.Option("step-snap", "--profile", help="Maneuver profile: step, step-snap, step-snap-eco, step-redirect, or step-redirect-eco"),

    dir_b_deg: list[float] = typer.Option([], "--dir-b-deg", help="Redirect target angle(s). Repeat; defaults to 45, 90, 135, 180."),

    fault_case: list[str] = typer.Option([], "--fault-case", help="Fault case(s). Repeat; defaults to the maneuver-focused set."),

    fault_set: str = typer.Option("maneuver", "--fault-set", help="Fault set: maneuver, nominal, core, stress, or all"),

    dir_a_deg: float = typer.Option(0.0, "--dir-a-deg"),

    fxy_n: float = typer.Option(3000.0, "--fxy"),

    step_time_s: float = typer.Option(3.0, "--step-time-s"),

    snap_stop_s: float = typer.Option(0.8, "--snap-stop-s", help="Duration of snap-stop brake phase for step-snap"),

    total_s: float = typer.Option(9.0, "--total-s"),

    yaw_hold_deg: float = typer.Option(0.0, "--yaw-hold-deg"),

    mz_nm: float = typer.Option(0.0, "--mz-nm"),

    alpha_rate_deg_s: float = typer.Option(350.0, "--alpha-rate-deg-s"),

    plenum_tau_s: float = typer.Option(0.08, "--plenum-tau-s"),

    brake_gain: float = typer.Option(1.2, "--brake-gain"),

    speed_stop_thr_mps: float = typer.Option(0.2, "--stop-thr-mps"),

    redirect_time_s: float = typer.Option(1.6, "--redirect-time-s", help="Duration of the smooth redirect blend phase for step-redirect"),

    redirect_speed_scale: float = typer.Option(0.88, "--redirect-speed-scale", help="Fraction of the pre-step speed preserved through the redirect blend for step-redirect"),

    redirect_cross_gain: float = typer.Option(1.0, "--redirect-cross-gain", help="Cross-velocity cleanup gain during the redirect blend for step-redirect"),

    redirect_hold_frac: float = typer.Option(-1.0, "--redirect-hold-frac", help="Fraction of the snap window kept as pure braking before steering toward the new direction. Negative = auto"),

    redirect_steer_scale: float = typer.Option(1.0, "--redirect-steer-scale", help="Scaling applied to the snap revector blend during sharp redirects"),

    maneuver_safe: bool = typer.Option(False, "--maneuver-safe/--no-maneuver-safe", help="Auto-tune redirect aggressiveness against power, flap, fan, and yaw limits"),

    power_target_pct: float = typer.Option(100.0, "--power-target-pct", help="Target p95 continuous-power percentage when --maneuver-safe is enabled"),

    flap_target_pct: float = typer.Option(90.0, "--flap-target-pct", help="Target peak flap-limit usage percentage when --maneuver-safe is enabled"),

    fan_target_pct: float = typer.Option(10.0, "--fan-target-pct", help="Target fan tracking RMS percentage when --maneuver-safe is enabled"),

    xy_target_n: float = typer.Option(400.0, "--xy-target-n", help="Target XY force tracking RMS when --maneuver-safe is enabled"),

    yaw_target_deg: float = typer.Option(15.0, "--yaw-target-deg", help="Target mean absolute yaw hold error when --maneuver-safe is enabled"),

    tune_iterations: int = typer.Option(6, "--tune-iters", help="Number of refinement iterations for --maneuver-safe"),

    tune_min_aggressiveness: float = typer.Option(0.20, "--tune-min-aggr", help="Lowest aggressiveness the maneuver-safe search is allowed to test"),

    save_traces: bool = typer.Option(False, "--save-traces/--no-save-traces", help="Write per-case trace JSON files next to the summary"),

    out_dir: str = typer.Option("runs/maneuver_pack", "--out-dir", help="Output directory for summary artifacts"),

):

    profile = profile.strip().lower()

    if profile not in {"step", "step-snap", "step-snap-eco", "step-redirect", "step-redirect-eco"}:

        raise typer.BadParameter("--profile must be 'step', 'step-snap', 'step-snap-eco', 'step-redirect', or 'step-redirect-eco'")



    angles = [float(angle) for angle in (dir_b_deg or MANEUVER_DEFAULT_ANGLES)]

    selected_fault_cases = select_fault_cases(list(fault_case), fault_set)

    lim = ActuatorLimits(alpha_rate_deg_s=alpha_rate_deg_s)

    pl = PlenumModel(tau_s=plenum_tau_s)

    out_path = Path(out_dir)

    out_path.mkdir(parents=True, exist_ok=True)



    results = []

    for angle in angles:

        for case_name in selected_fault_cases:

            fault = build_fault_spec(case_name)

            base_kwargs = {

                'dir_a_deg': dir_a_deg,

                'dir_b_deg': angle,

                'fxy_n': fxy_n,

                'step_time_s': step_time_s,

                'total_s': total_s,

                'yaw_hold_deg': yaw_hold_deg,

                'mz_nm': mz_nm,

                'lim': lim,

                'pl': pl,

                'fault': fault,

                'snap_stop_s': snap_stop_s,

                'brake_gain': brake_gain,

                'speed_stop_thr_mps': speed_stop_thr_mps,

                'redirect_time_s': redirect_time_s,

                'redirect_speed_scale': redirect_speed_scale,

                'redirect_cross_gain': redirect_cross_gain,

                'redirect_hold_frac': redirect_hold_frac,

                'redirect_steer_scale': redirect_steer_scale,

            }

            if maneuver_safe:

                meta, hist, assessment, tuning = tune_maneuver_profile(

                    profile,

                    base_kwargs,

                    target_continuous_power_p95_pct=power_target_pct,

                    target_flap_peak_pct=flap_target_pct,

                    target_fan_tracking_pct=fan_target_pct,

                    target_xy_tracking_rms_n=xy_target_n,

                    target_yaw_coupling_deg=yaw_target_deg,

                    min_aggressiveness=tune_min_aggressiveness,

                    max_iterations=tune_iterations,

                )

                meta['maneuver_tuning'] = tuning

            else:

                meta, hist = _run_maneuver_profile(profile, **base_kwargs)

                assessment = assess_maneuver_result(meta, hist, profile)

                tuning = None

            meta['fault'] = summarize_fault_case(case_name, fault)

            meta['maneuver_assessment'] = assessment

            engineering = assessment['engineering']

            maneuver = assessment['maneuver']

            result = {

                'profile': profile,

                'dir_a_deg': float(dir_a_deg),

                'dir_b_deg': float(angle),

                'fault_case': case_name,

                'status': assessment['status'],

                'warnings': list(assessment['warnings']),

                'fault': meta['fault'],

                'selected_aggressiveness': (tuning['selected']['aggressiveness'] if tuning else 1.0),

                'tuning_status': (tuning['status'] if tuning else 'raw'),

                't_to_speed_below_thr_s': maneuver['t_to_speed_below_thr_s'],

                'snap_stop_distance_m': maneuver['snap_stop_distance_m'],

                't_reversal_s': maneuver['t_reversal_s'],

                't90_dir_s': maneuver['t90_dir_s'],

                'min_speed_transition_mps': maneuver['min_speed_transition_mps'],

                'peak_speed_mps': maneuver['peak_speed_mps'],

                'yaw_hold_error_mean_abs_deg': maneuver['yaw_hold_error_mean_abs_deg'],

                'yaw_track_decoupling_mean_abs_deg': maneuver['yaw_track_decoupling_mean_abs_deg'],

                'xy_tracking_rms_n': engineering['vectoring']['xy_tracking_rms_n'],

                'alignment_p95_deg': engineering['vectoring']['alignment_p95_deg'],

                'flap_limit_usage_peak_pct': engineering['flaps']['limit_usage_peak_pct'],

                'flap_tracking_rms_deg': engineering['flaps']['tracking_rms_deg'],

                'fan_tracking_rms_pct_mean_cmd': engineering['fans']['tracking_rms_pct_mean_cmd'],

                'fan_response_min_pct': engineering['fans']['response_min_pct'],

                'continuous_power_p95_pct': engineering['thermal']['continuous_power_p95_pct'],

                'continuous_power_raw_p95_pct': engineering['thermal'].get('continuous_power_raw_p95_pct'),

                'burst_reserve_min_pct': engineering['thermal'].get('burst_reserve_min_pct'),

                'burst_clip_time_s': engineering['thermal'].get('burst_clip_time_s'),

                'burst_active_time_s': engineering['thermal'].get('burst_active_time_s'),

                'thrust_scale_min_pct': engineering['power']['thrust_scale_min_pct'],

            }

            results.append(result)

            if save_traces:

                trace_name = f"{profile}_{_slugify_case(case_name)}_{int(round(angle))}.json"

                save_trace_json(str(out_path / trace_name), meta=meta, hist=hist)



    results.sort(

        key=lambda rec: (

            _status_rank(rec['status']),

            -(rec['selected_aggressiveness'] or 0.0),

            -(rec['continuous_power_p95_pct'] or 0.0),

            -(rec['fan_tracking_rms_pct_mean_cmd'] or 0.0),

            -(rec['flap_limit_usage_peak_pct'] or 0.0),

            -(rec['yaw_hold_error_mean_abs_deg'] or 0.0),

            rec['fault_case'],

            rec['dir_b_deg'],

        )

    )



    summary = {

        'profile': profile,

        'angles_deg': angles,

        'fault_set': fault_set,

        'fault_cases': selected_fault_cases,

        'maneuver_safe': bool(maneuver_safe),

        'worst_status': (results[-1]['status'] if results else None),

        'counts': {

            'pass': sum(1 for rec in results if rec['status'] == 'pass'),

            'caution': sum(1 for rec in results if rec['status'] == 'caution'),

            'risk': sum(1 for rec in results if rec['status'] == 'risk'),

        },

        'results': results,

    }



    (out_path / 'summary.json').write_text(json.dumps(summary, indent=2))

    fieldnames = [

        'profile', 'dir_a_deg', 'dir_b_deg', 'fault_case', 'status', 'selected_aggressiveness', 'tuning_status',

        't_to_speed_below_thr_s', 'snap_stop_distance_m', 't_reversal_s', 't90_dir_s',

        'min_speed_transition_mps', 'peak_speed_mps', 'yaw_hold_error_mean_abs_deg', 'yaw_track_decoupling_mean_abs_deg',

        'xy_tracking_rms_n', 'alignment_p95_deg', 'flap_limit_usage_peak_pct',

        'flap_tracking_rms_deg', 'fan_tracking_rms_pct_mean_cmd', 'fan_response_min_pct',

        'continuous_power_p95_pct', 'continuous_power_raw_p95_pct', 'burst_reserve_min_pct', 'burst_clip_time_s', 'burst_active_time_s',

        'thrust_scale_min_pct', 'warnings',

    ]

    with (out_path / 'summary.csv').open('w', newline='') as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for rec in results:

            row = dict(rec)

            row['warnings'] = ' | '.join(rec['warnings'])

            writer.writerow({key: row.get(key) for key in fieldnames})



    (out_path / 'summary.md').write_text(render_maneuver_pack_markdown(summary))

    typer.echo(json.dumps({

        'profile': profile,

        'out_dir': str(out_path),

        'maneuver_safe': bool(maneuver_safe),

        'cases': len(results),

        'worst_status': summary['worst_status'],

        'counts': summary['counts'],

    }, indent=2))

