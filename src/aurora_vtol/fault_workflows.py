from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from aurora_vtol.allocator.dynamics import ActuatorLimits, PlenumModel
from aurora_vtol.allocator.engineering import tune_coordinate_mission
from aurora_vtol.allocator.faults import FaultSpec
from aurora_vtol.allocator.model import RingGeometry
from aurora_vtol.allocator.mission_planning import MissionObstacle
from aurora_vtol.allocator.power_system import PowerSystemParams
from aurora_vtol.allocator.trace import save_trace_json
from aurora_vtol.reporting import render_fault_envelope_table, render_fault_threshold_pack_table, render_fault_threshold_table, render_requirements_table, summarize_power_params
from aurora_vtol.study_workflows import _fault_status_rank, _normalize_required_status, build_coordinate_mission_kwargs, build_power_params, build_power_sweep_report_for_fault, write_power_sweep_outputs

FAULT_CASE_PRESETS = {

    "nominal": {

        "description": "No injected actuator faults.",

        "kwargs": {},

    },

    "dead-fan-0": {

        "description": "Fan group 0 fully failed.",

        "kwargs": {"dead_fan_group": 0, "dead_fan_scale": 0.0},

    },

    "dead-fan-1": {

        "description": "Fan group 1 fully failed.",

        "kwargs": {"dead_fan_group": 1, "dead_fan_scale": 0.0},

    },

    "stuck-flap-0": {

        "description": "Flap 0 stuck at 20 degrees.",

        "kwargs": {"stuck_flap_idx": 0, "stuck_flap_alpha_deg": 20.0},

    },

    "stuck-flap-8": {

        "description": "Flap 8 stuck at 20 degrees.",

        "kwargs": {"stuck_flap_idx": 8, "stuck_flap_alpha_deg": 20.0},

    },

    "slow-flap-0": {

        "description": "Flap 0 responds at 35 percent of nominal rate.",

        "kwargs": {"slow_flap_idx": 0, "slow_flap_rate_scale": 0.35},

    },

    "slow-flap-8": {

        "description": "Flap 8 responds at 35 percent of nominal rate.",

        "kwargs": {"slow_flap_idx": 8, "slow_flap_rate_scale": 0.35},

    },

    "weak-flap-0": {

        "description": "Flap 0 has 55 percent vectoring authority.",

        "kwargs": {"weak_flap_idx": 0, "weak_flap_scale": 0.55},

    },

    "weak-flap-8": {

        "description": "Flap 8 has 55 percent vectoring authority.",

        "kwargs": {"weak_flap_idx": 8, "weak_flap_scale": 0.55},

    },

    "biased-flap-0": {

        "description": "Flap 0 carries an 8 degree bias offset.",

        "kwargs": {"biased_flap_idx": 0, "biased_flap_offset_deg": 8.0},

    },

    "biased-flap-8": {

        "description": "Flap 8 carries an 8 degree bias offset.",

        "kwargs": {"biased_flap_idx": 8, "biased_flap_offset_deg": 8.0},

    },

    "plenum-sector-0": {

        "description": "Plenum sector 0 reduced to 70 percent thrust.",

        "kwargs": {"plenum_sector_idx": 0, "plenum_sector_scale": 0.7},

    },

    "plenum-sector-1": {

        "description": "Plenum sector 1 reduced to 70 percent thrust.",

        "kwargs": {"plenum_sector_idx": 1, "plenum_sector_scale": 0.7},

    },

}



CORE_FAULT_CASES = ["nominal", "dead-fan-0", "stuck-flap-0", "plenum-sector-0"]

STRESS_FAULT_CASES = [

    "dead-fan-0+stuck-flap-8",

    "dead-fan-0+plenum-sector-1",

    "stuck-flap-8+plenum-sector-1",

    "dead-fan-0+stuck-flap-8+plenum-sector-1",

]

MANEUVER_FAULT_CASES = [

    "nominal",

    "slow-flap-0",

    "weak-flap-0",

    "biased-flap-0",

    "stuck-flap-0",

    "dead-fan-0",

    "plenum-sector-0",

    "dead-fan-0+slow-flap-8",

    "dead-fan-0+biased-flap-8",

    "weak-flap-8+plenum-sector-1",

]

FAULT_CASE_SETS = {

    "nominal": ["nominal"],

    "core": CORE_FAULT_CASES,

    "stress": STRESS_FAULT_CASES,

    "maneuver": MANEUVER_FAULT_CASES,

    "all": CORE_FAULT_CASES + STRESS_FAULT_CASES + MANEUVER_FAULT_CASES,

}





def _fault_case_sort_key(name: str) -> tuple[int, str]:

    order = {case: idx for idx, case in enumerate(CORE_FAULT_CASES + STRESS_FAULT_CASES)}

    return (order.get(name, len(order)), name)





def resolve_fault_case(case_name: str) -> tuple[str, dict]:

    parts = [part.strip() for part in case_name.split("+") if part.strip()]

    if not parts:

        raise typer.BadParameter("--fault-case must not be empty")

    if parts == ["nominal"]:

        spec = FAULT_CASE_PRESETS["nominal"]

        return "nominal", {

            "description": spec["description"],

            "kwargs": {},

            "components": ["nominal"],

        }



    merged_kwargs: dict[str, float | int] = {}

    components: list[str] = []

    seen: set[str] = set()

    for part in parts:

        if part == "nominal":

            continue

        spec = FAULT_CASE_PRESETS.get(part)

        if spec is None:

            allowed = ", ".join(sorted(FAULT_CASE_PRESETS))

            raise typer.BadParameter(

                f"--fault-case must use known components joined by '+': {allowed}"

            )

        if part in seen:

            continue

        seen.add(part)

        components.append(part)

        for key, value in spec["kwargs"].items():

            if key in merged_kwargs and merged_kwargs[key] != value:

                raise typer.BadParameter(f"conflicting composite --fault-case '{case_name}'")

            merged_kwargs[key] = value



    if not components:

        spec = FAULT_CASE_PRESETS["nominal"]

        return "nominal", {

            "description": spec["description"],

            "kwargs": {},

            "components": ["nominal"],

        }



    components.sort(key=_fault_case_sort_key)

    normalized = "+".join(components)

    description = " + ".join(FAULT_CASE_PRESETS[name]["description"] for name in components)

    return normalized, {

        "description": description,

        "kwargs": merged_kwargs,

        "components": components,

    }





def build_fault_spec(case_name: str) -> FaultSpec:

    _normalized, spec = resolve_fault_case(case_name)

    return FaultSpec(**spec["kwargs"])





def summarize_fault_case(case_name: str, fault: FaultSpec | None = None) -> dict:

    normalized_case, spec = resolve_fault_case(case_name)

    fault = fault or FaultSpec(**spec["kwargs"])

    return {

        'case': normalized_case,

        'components': list(spec['components']),

        'description': spec['description'],

        'stuck_flap_idx': fault.stuck_flap_idx,

        'stuck_flap_alpha_deg': fault.stuck_flap_alpha_deg,

        'slow_flap_idx': fault.slow_flap_idx,

        'slow_flap_rate_scale': fault.slow_flap_rate_scale,

        'weak_flap_idx': fault.weak_flap_idx,

        'weak_flap_scale': fault.weak_flap_scale,

        'biased_flap_idx': fault.biased_flap_idx,

        'biased_flap_offset_deg': fault.biased_flap_offset_deg,

        'dead_fan_group': fault.dead_fan_group,

        'dead_fan_scale': fault.dead_fan_scale,

        'plenum_sector_idx': fault.plenum_sector_idx,

        'plenum_sector_scale': fault.plenum_sector_scale,

    }





def fault_spec_from_summary(summary: dict | None) -> FaultSpec:

    data = summary or {}

    return FaultSpec(

        stuck_flap_idx=(None if data.get('stuck_flap_idx') is None else int(data['stuck_flap_idx'])),

        stuck_flap_alpha_deg=float(data.get('stuck_flap_alpha_deg', 0.0) or 0.0),

        slow_flap_idx=(None if data.get('slow_flap_idx') is None else int(data['slow_flap_idx'])),

        slow_flap_rate_scale=float(data.get('slow_flap_rate_scale', 0.5) or 0.5),

        weak_flap_idx=(None if data.get('weak_flap_idx') is None else int(data['weak_flap_idx'])),

        weak_flap_scale=float(data.get('weak_flap_scale', 0.7) or 0.7),

        biased_flap_idx=(None if data.get('biased_flap_idx') is None else int(data['biased_flap_idx'])),

        biased_flap_offset_deg=float(data.get('biased_flap_offset_deg', 0.0) or 0.0),

        dead_fan_group=(None if data.get('dead_fan_group') is None else int(data['dead_fan_group'])),

        dead_fan_scale=float(data.get('dead_fan_scale', 0.0) or 0.0),

        plenum_sector_idx=(None if data.get('plenum_sector_idx') is None else int(data['plenum_sector_idx'])),

        plenum_sector_scale=float(data.get('plenum_sector_scale', 0.7) or 0.7),

    )





def select_fault_cases(fault_cases: list[str], fault_set: str) -> list[str]:

    if fault_cases:

        requested = list(fault_cases)

    else:

        requested = FAULT_CASE_SETS.get(fault_set, [])

        if not requested:

            allowed = ", ".join(sorted(FAULT_CASE_SETS))

            raise typer.BadParameter(f"--fault-set must be one of: {allowed}")



    selected: list[str] = []

    seen: set[str] = set()

    for name in requested:

        normalized, _spec = resolve_fault_case(name)

        if normalized not in seen:

            seen.add(normalized)

            selected.append(normalized)

    return selected





def _fault_case_slug(label: str) -> str:

    chars = []

    for char in label.lower():

        if char.isalnum():

            chars.append(char)

        else:

            chars.append('-')

    slug = ''.join(chars)

    while '--' in slug:

        slug = slug.replace('--', '-')

    return slug.strip('-') or 'fault-case'





def _fault_value_token(value: float) -> str:

    text = f'{float(value):.2f}'.rstrip('0').rstrip('.')

    return text.replace('-', 'm').replace('.', 'p')





def _fault_index_values(values: list[int], *, count: int, step: int, label: str) -> list[int]:

    if values:

        indices = sorted({int(value) for value in values})

    else:

        stride = max(1, int(step))

        indices = list(range(0, count, stride))

    invalid = [value for value in indices if value < 0 or value >= count]

    if invalid:

        raise typer.BadParameter(f'{label} must be between 0 and {count - 1}')

    return indices





def _fault_severity_values(values: list[float], default: float) -> list[float]:

    return sorted({float(value) for value in (values or [default])})





def _fault_component_catalog(

    *,

    fan_groups: list[int],

    flap_indices: list[int],

    plenum_indices: list[int],

    dead_fan_scales: list[float],

    stuck_flap_angles_deg: list[float],

    plenum_scales: list[float],

) -> dict[str, list[dict]]:

    catalog = {'fan': [], 'flap': [], 'plenum': []}



    for group in fan_groups:

        for scale in dead_fan_scales:

            catalog['fan'].append({

                'family': 'fan',

                'label': f'dead-fan-{group:02d}-x{_fault_value_token(scale)}',

                'description': f'fan group {group} scale {scale:.2f}',

                'kwargs': {'dead_fan_group': int(group), 'dead_fan_scale': float(scale)},

            })



    for idx in flap_indices:

        for angle in stuck_flap_angles_deg:

            catalog['flap'].append({

                'family': 'flap',

                'label': f'stuck-flap-{idx:02d}-a{_fault_value_token(angle)}',

                'description': f'flap {idx} stuck at {angle:.1f} deg',

                'kwargs': {'stuck_flap_idx': int(idx), 'stuck_flap_alpha_deg': float(angle)},

            })



    for idx in plenum_indices:

        for scale in plenum_scales:

            catalog['plenum'].append({

                'family': 'plenum',

                'label': f'plenum-sector-{idx:02d}-x{_fault_value_token(scale)}',

                'description': f'plenum sector {idx} scale {scale:.2f}',

                'kwargs': {'plenum_sector_idx': int(idx), 'plenum_sector_scale': float(scale)},

            })



    return catalog





def _compose_fault_envelope_case(parts: list[dict]) -> dict:

    kwargs: dict[str, float | int] = {}

    for part in parts:

        for key, value in part['kwargs'].items():

            if key in kwargs and kwargs[key] != value:

                raise ValueError(f'conflicting fault parts for {key}')

            kwargs[key] = value

    labels = [part['label'] for part in parts]

    descriptions = [part['description'] for part in parts]

    families = [part['family'] for part in parts]

    fault = FaultSpec(**kwargs)

    return {

        'case': '+'.join(labels),

        'components': labels,

        'families': families,

        'description': ' + '.join(descriptions),

        'fault': fault,

        'parts': parts,

    }





def _summarize_dynamic_fault_case(case_name: str, fault: FaultSpec, *, components: list[str], description: str, families: list[str]) -> dict:

    return {

        'case': case_name,

        'components': list(components),

        'families': list(families),

        'description': description,

        'stuck_flap_idx': fault.stuck_flap_idx,

        'stuck_flap_alpha_deg': fault.stuck_flap_alpha_deg,

        'dead_fan_group': fault.dead_fan_group,

        'dead_fan_scale': fault.dead_fan_scale,

        'plenum_sector_idx': fault.plenum_sector_idx,

        'plenum_sector_scale': fault.plenum_sector_scale,

    }





def _fault_envelope_score(entry: dict) -> float:

    score = 0.0

    if not entry.get('feasible', False):

        score += 1_000_000.0

    score += float(_fault_status_rank(entry.get('report_status'))) * 100_000.0

    score += float(entry.get('continuous_power_p95_pct') or 0.0) * 100.0

    score += float(entry.get('final_goal_error_m') or 0.0) * 1000.0

    score += max(0.0, 1.0 - float(entry.get('selected_aggressiveness') or 0.0)) * 10_000.0

    score += float(entry.get('arrival_time_s') or 0.0)

    return score





def _fault_envelope_manifest(

    *,

    preset: str,

    power: PowerSystemParams,

    power_target_pct: float,

    flap_target_pct: float,

    tune_iterations: int,

    tune_min_aggressiveness: float,

    fan_indices: list[int],

    flap_indices: list[int],

    plenum_indices: list[int],

    fan_scales: list[float],

    flap_angles: list[float],

    plenum_scales: list[float],

    top_per_family: int,

    include_pairs: bool,

    include_triples: bool,

) -> dict:

    return {

        'preset': preset,

        'power_model': summarize_power_params(power),

        'targets': {

            'required_status': str(globals().get('_fault_required_status', 'feasible')),

            'continuous_power_p95_pct': float(power_target_pct),

            'flap_limit_usage_peak_pct': float(flap_target_pct),

            'tune_iterations': int(tune_iterations),

            'tune_min_aggressiveness': float(tune_min_aggressiveness),

        },

        'sampling': {

            'fan_groups': list(fan_indices),

            'flap_indices': list(flap_indices),

            'plenum_sector_indices': list(plenum_indices),

            'dead_fan_scale': list(fan_scales),

            'stuck_flap_alpha_deg': list(flap_angles),

            'plenum_sector_scale': list(plenum_scales),

            'top_per_family': int(top_per_family),

            'include_pairs': bool(include_pairs),

            'include_triples': bool(include_triples),

        },

    }





def _prepare_fault_envelope_cache(cache_dir: Path, manifest: dict, *, resume: bool) -> dict:

    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / 'manifest.json'

    if manifest_path.exists():

        existing = json.loads(manifest_path.read_text(encoding='utf-8'))

        if resume and existing != manifest:

            raise typer.BadParameter(

                'fault-envelope cache manifest mismatch; use a different --out-dir or --no-resume'

            )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    return {'path': str(manifest_path), 'manifest': manifest}





def _fault_envelope_cache_paths(cache_dir: Path, case_name: str) -> tuple[Path, Path]:

    slug = _fault_case_slug(case_name)

    return cache_dir / f'{slug}.json', cache_dir / f'{slug}.trace.json'





def _load_fault_envelope_cache(cache_path: Path) -> dict | None:

    if not cache_path.exists():

        return None

    return json.loads(cache_path.read_text(encoding='utf-8'))





def _write_fault_envelope_cache(cache_path: Path, trace_path: Path, bundle: dict) -> None:

    payload = {

        'entry': bundle['entry'],

        'parts': bundle['parts'],

        'trace_path': str(trace_path),

    }

    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')





def _evaluate_fault_envelope_case(

    *,

    preset: str,

    preset_cfg: dict,

    case: dict,

    power: PowerSystemParams,

    lim: ActuatorLimits,

    pl: PlenumModel,

    obstacles: list[MissionObstacle],

    power_target_pct: float,

    flap_target_pct: float,

    tune_iterations: int,

    tune_min_aggressiveness: float,

) -> dict:

    mission_kwargs = build_coordinate_mission_kwargs(

        dest_x_m=float(preset_cfg['dest_x_m']),

        dest_y_m=float(preset_cfg['dest_y_m']),

        dest_z_m=float(preset_cfg.get('dest_z_m', 0.0)),

        start_x_m=0.0,

        start_y_m=0.0,

        start_z_m=0.0,

        total_s=float(preset_cfg['total_s']),

        yaw_hold_deg=0.0,

        mz_nm=0.0,

        cruise_alt_m=float(preset_cfg['cruise_alt_m']),

        max_speed_mps=float(preset_cfg['max_speed_mps']),

        arrival_radius_m=float(preset_cfg['arrival_radius_m']),

        slow_radius_m=float(preset_cfg['slow_radius_m']),

        descent_radius_m=float(preset_cfg['descent_radius_m']),

        hold_s=float(preset_cfg['hold_s']),

        fxy_max_n=float(preset_cfg['fxy_max_n']),

        pos_k_n_per_m=float(preset_cfg['pos_k_n_per_m']),

        vel_k_n_per_mps=float(preset_cfg['vel_k_n_per_mps']),

        obstacles=obstacles,

        lim=lim,

        pl=pl,

        power=power,

        fault=case['fault'],

    )

    meta, hist, report, tuning = tune_coordinate_mission(

        mission_kwargs,

        target_continuous_power_p95_pct=power_target_pct,

        target_flap_peak_pct=flap_target_pct,

        min_aggressiveness=tune_min_aggressiveness,

        max_iterations=tune_iterations,

    )

    entry = {

        'preset': preset,

        'case': case['case'],

        'order': len(case['components']),

        'families': list(case['families']),

        'fault_case': _summarize_dynamic_fault_case(

            case['case'],

            case['fault'],

            components=case['components'],

            description=case['description'],

            families=case['families'],

        ),

        'feasible': bool(tuning.get('envelope', {}).get('feasible', False)),

        'selected_aggressiveness': tuning.get('selected', {}).get('aggressiveness'),

        'selected_score': tuning.get('selected', {}).get('score'),

        'report_status': report.get('status'),

        'warnings': list(report.get('warnings', [])),

        'continuous_power_p95_pct': report.get('thermal', {}).get('continuous_power_p95_pct'),

        'power_margin_min_kw': report.get('thermal', {}).get('power_margin_min_kw'),

        'flap_limit_usage_peak_pct': report.get('flaps', {}).get('limit_usage_peak_pct'),

        'xy_tracking_rms_n': report.get('vectoring', {}).get('xy_tracking_rms_n'),

        'final_goal_error_m': report.get('mission', {}).get('final_goal_error_m'),

        'arrival_time_s': report.get('mission', {}).get('arrival_time_s'),

    }

    entry['severity_score'] = _fault_envelope_score(entry)

    return {'entry': entry, 'meta': meta, 'hist': hist, 'parts': case['parts']}





def build_fault_envelope_report(

    *,

    preset: str,

    power_target_pct: float,

    flap_target_pct: float,

    tune_iterations: int,

    tune_min_aggressiveness: float,

    battery_full_v: float | None,

    battery_empty_v: float | None,

    capacity_kwh: float | None,

    internal_resistance_ohm: float | None,

    hover_power_kw: float | None,

    continuous_power_kw: float | None,

    peak_power_kw: float | None,

    aux_power_kw: float | None,

    dead_fan_scale: list[float],

    stuck_flap_alpha_deg: list[float],

    plenum_sector_scale: list[float],

    fan_group: list[int],

    flap_idx: list[int],

    plenum_sector_idx: list[int],

    flap_step: int,

    plenum_step: int,

    top_per_family: int,

    include_pairs: bool,

    include_triples: bool,

    top: int,

    cache_dir: str | None = None,

    resume: bool = True,

    max_new_cases: int = 0,

) -> tuple[dict, list[dict]]:
    geom = RingGeometry()

    lim = ActuatorLimits(alpha_rate_deg_s=350.0)

    pl = PlenumModel(tau_s=0.08)

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



    fan_indices = _fault_index_values(fan_group, count=geom.n_segments // 2, step=1, label='--fan-group')

    flap_indices = _fault_index_values(flap_idx, count=geom.n_segments, step=flap_step, label='--flap-idx')

    plenum_indices = _fault_index_values(plenum_sector_idx, count=geom.n_segments, step=plenum_step, label='--plenum-sector-idx')

    fan_scales = _fault_severity_values(dead_fan_scale, 0.0)

    flap_angles = _fault_severity_values(stuck_flap_alpha_deg, 20.0)

    plenum_scales = _fault_severity_values(plenum_sector_scale, 0.7)
    normalized_required_status = _normalize_required_status(globals().get('_fault_required_status', 'feasible'))

    manifest = _fault_envelope_manifest(
        preset=preset,
        power=power,
        required_status=normalized_required_status,

        power_target_pct=power_target_pct,

        flap_target_pct=flap_target_pct,

        tune_iterations=tune_iterations,

        tune_min_aggressiveness=tune_min_aggressiveness,

        fan_indices=fan_indices,

        flap_indices=flap_indices,

        plenum_indices=plenum_indices,

        fan_scales=fan_scales,

        flap_angles=flap_angles,

        plenum_scales=plenum_scales,

        top_per_family=top_per_family,

        include_pairs=include_pairs,

        include_triples=include_triples,

    )

    cache_state = None

    cache_base = Path(cache_dir) if cache_dir else None

    if cache_base is not None:

        cache_state = _prepare_fault_envelope_cache(cache_base, manifest, resume=resume)



    catalog = _fault_component_catalog(

        fan_groups=fan_indices,

        flap_indices=flap_indices,

        plenum_indices=plenum_indices,

        dead_fan_scales=fan_scales,

        stuck_flap_angles_deg=flap_angles,

        plenum_scales=plenum_scales,

    )



    evaluated: list[dict] = []

    seen_cases: set[str] = set()

    cached_cases = 0

    new_cases = 0

    budget_exhausted = False



    def record(parts: list[dict]) -> dict | None:

        nonlocal cached_cases, new_cases, budget_exhausted

        case = _compose_fault_envelope_case(parts)

        case_name = case['case']

        if case_name in seen_cases:

            return None



        cache_payload = None

        cache_path = None

        trace_path = None

        if cache_base is not None:

            cache_path, trace_path = _fault_envelope_cache_paths(cache_base, case_name)

            if resume:

                cache_payload = _load_fault_envelope_cache(cache_path)

        if cache_payload is not None:

            bundle = {

                'entry': cache_payload['entry'],

                'meta': None,

                'hist': None,

                'parts': case['parts'],

                'cache_trace': cache_payload.get('trace_path'),

            }

            evaluated.append(bundle)

            seen_cases.add(case_name)

            cached_cases += 1

            return bundle



        if max_new_cases > 0 and new_cases >= max_new_cases:

            budget_exhausted = True

            return None



        bundle = _evaluate_fault_envelope_case(

            preset=preset,

            preset_cfg=preset_cfg,

            case=case,

            power=power,

            lim=lim,

            pl=pl,

            obstacles=obstacles,

            power_target_pct=power_target_pct,

            flap_target_pct=flap_target_pct,

            tune_iterations=tune_iterations,

            tune_min_aggressiveness=tune_min_aggressiveness,

        )

        if cache_base is not None and cache_path is not None and trace_path is not None:

            save_trace_json(str(trace_path), meta=bundle['meta'], hist=bundle['hist'])

            bundle['cache_trace'] = str(trace_path)

            _write_fault_envelope_cache(cache_path, trace_path, bundle)

        evaluated.append(bundle)

        seen_cases.add(case_name)

        new_cases += 1

        return bundle



    family_ranked: dict[str, list[dict]] = {}

    single_counts: dict[str, int] = {}

    family_bundles = {family: [] for family in ('fan', 'flap', 'plenum')}

    max_single_candidates = max((len(catalog[family]) for family in family_bundles), default=0)

    for idx in range(max_single_candidates):

        for family in ('fan', 'flap', 'plenum'):

            if idx >= len(catalog[family]):

                continue

            bundle = record([catalog[family][idx]])

            if bundle is not None:

                family_bundles[family].append(bundle)

    for family, bundles in family_bundles.items():

        bundles.sort(key=lambda item: item['entry']['severity_score'], reverse=True)

        family_ranked[family] = bundles[:max(1, int(top_per_family))]

        single_counts[family] = len(bundles)



    pair_count = 0

    if include_pairs:

        for left_family, right_family in (('fan', 'flap'), ('fan', 'plenum'), ('flap', 'plenum')):

            for left in family_ranked[left_family]:

                for right in family_ranked[right_family]:

                    bundle = record(left['parts'] + right['parts'])

                    if bundle is not None:

                        pair_count += 1



    triple_count = 0

    if include_triples and family_ranked['fan'] and family_ranked['flap'] and family_ranked['plenum']:

        for fan_bundle in family_ranked['fan']:

            for flap_bundle in family_ranked['flap']:

                for plenum_bundle in family_ranked['plenum']:

                    bundle = record(fan_bundle['parts'] + flap_bundle['parts'] + plenum_bundle['parts'])

                    if bundle is not None:

                        triple_count += 1



    ranked = sorted(evaluated, key=lambda item: item['entry']['severity_score'], reverse=True)

    top_entries = [bundle['entry'] for bundle in ranked[:max(1, int(top))]]



    report = {

        'preset': preset,

        'power_model': summarize_power_params(power),

        'targets': {

            'continuous_power_p95_pct': float(power_target_pct),

            'flap_limit_usage_peak_pct': float(flap_target_pct),

            'tune_iterations': int(tune_iterations),

            'tune_min_aggressiveness': float(tune_min_aggressiveness),

        },

        'sampling': {

            'fan_groups': fan_indices,

            'flap_indices': flap_indices,

            'plenum_sector_indices': plenum_indices,

            'dead_fan_scale': fan_scales,

            'stuck_flap_alpha_deg': flap_angles,

            'plenum_sector_scale': plenum_scales,

            'flap_step': int(max(1, flap_step)),

            'plenum_step': int(max(1, plenum_step)),

            'top_per_family': int(top_per_family),

            'include_pairs': bool(include_pairs),

            'include_triples': bool(include_triples),

        },

        'search_space': {

            'cases': len(evaluated),

            'single_cases': int(sum(single_counts.values())),

            'pair_cases': int(pair_count),

            'triple_cases': int(triple_count),

            'cached_cases': int(cached_cases),

            'new_cases': int(new_cases),

            'complete': not budget_exhausted,

            'max_new_cases': int(max_new_cases),

        },

        'cache': ({

            'dir': str(cache_base),

            'resume': bool(resume),

            'manifest': (cache_state['path'] if cache_state is not None else ''),

        } if cache_base is not None else None),

        'worst_single_by_family': {

            family: (family_ranked[family][0]['entry'] if family_ranked[family] else None)

            for family in ('fan', 'flap', 'plenum')

        },

        'top_cases': top_entries,

    }

    return report, ranked







def _default_continuous_power_sweep(base_continuous_power_kw: float) -> list[float]:

    candidates = {

        max(5.0, float(base_continuous_power_kw) - 15.0),

        max(5.0, float(base_continuous_power_kw) - 10.0),

        max(5.0, float(base_continuous_power_kw) - 5.0),

        float(base_continuous_power_kw),

        float(base_continuous_power_kw) + 5.0,

        float(base_continuous_power_kw) + 10.0,

    }

    return sorted(candidates)





def load_fault_threshold_cases(

    summary_path: str,

    *,

    case_names: list[str],

    top_cases: int,

    include_family_worst: bool,

) -> tuple[dict, list[dict]]:

    report = json.loads(Path(summary_path).read_text(encoding='utf-8'))

    if 'preset' not in report:

        raise typer.BadParameter('--summary must point to a fault-envelope summary.json')



    available: dict[str, dict] = {}

    ordered_names: list[str] = []



    def add_entry(entry: dict | None, source: str) -> None:

        if not entry:

            return

        fault_summary = dict(entry.get('fault_case') or {})

        case_name = str(fault_summary.get('case') or entry.get('case') or '').strip()

        if not case_name:

            return

        if case_name not in available:

            families = list(entry.get('families') or fault_summary.get('families') or [])

            available[case_name] = {

                'case': case_name,

                'fault_summary': fault_summary,

                'order': int(entry.get('order') or len(fault_summary.get('components', [])) or 0),

                'families': families,

                'severity_score': entry.get('severity_score'),

                'sources': [],

            }

            ordered_names.append(case_name)

        item = available[case_name]

        if source not in item['sources']:

            item['sources'].append(source)

        if not item['families'] and fault_summary.get('families'):

            item['families'] = list(fault_summary['families'])

        if not item['order']:

            item['order'] = int(entry.get('order') or len(item['families']) or 0)

        if item.get('severity_score') is None and entry.get('severity_score') is not None:

            item['severity_score'] = entry.get('severity_score')



    if include_family_worst:

        for family, entry in (report.get('worst_single_by_family') or {}).items():

            add_entry(entry, f'family:{family}')



    for idx, entry in enumerate((report.get('top_cases') or [])[:max(0, int(top_cases))], start=1):

        add_entry(entry, f'top:{idx}')



    if case_names:

        selected = []

        missing = []

        for case_name in case_names:

            if case_name not in available:

                missing.append(case_name)

            else:

                selected.append(available[case_name])

        if missing:

            raise typer.BadParameter(

                'requested --case values are not present in the supplied summary: ' + ', '.join(missing)

            )

        return report, selected



    return report, [available[name] for name in ordered_names]





def _fault_threshold_manifest(

    *,

    summary_path: str,

    preset: str,

    selected_cases: list[dict],

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

) -> dict:

    return {

        'source_summary': str(summary_path),

        'preset': preset,

        'selected_cases': [item['case'] for item in selected_cases],

        'targets': {

            'continuous_power_p95_pct': float(power_target_pct),

            'flap_limit_usage_peak_pct': float(flap_target_pct),

            'tune_iterations': int(tune_iterations),

            'tune_min_aggressiveness': float(tune_min_aggressiveness),

        },

        'sweep_values': {

            'battery_full_v': list(battery_full_v),

            'battery_empty_v': list(battery_empty_v),

            'capacity_kwh': list(capacity_kwh),

            'internal_resistance_ohm': list(internal_resistance_ohm),

            'hover_power_kw': list(hover_power_kw),

            'continuous_power_kw': list(continuous_power_kw),

            'peak_power_kw': list(peak_power_kw),

            'aux_power_kw': list(aux_power_kw),

        },

        'top': int(top),

    }





def _fault_threshold_cache_path(cache_dir: Path, case_name: str) -> Path:

    return cache_dir / f'{_fault_case_slug(case_name)}.json'





def _write_fault_threshold_cache(cache_path: Path, payload: dict) -> None:

    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')





def run_fault_threshold_report(

    *,

    summary: str,

    case_names: list[str],

    top_cases: int,

    include_family_worst: bool,

    required_status: str,

    power_target_pct: Optional[float],

    flap_target_pct: Optional[float],

    tune_iterations: Optional[int],

    tune_min_aggressiveness: Optional[float],

    battery_full_v: list[float],

    battery_empty_v: list[float],

    capacity_kwh: list[float],

    internal_resistance_ohm: list[float],

    hover_power_kw: list[float],

    continuous_power_kw: list[float],

    peak_power_kw: list[float],

    aux_power_kw: list[float],

    top: int,

    resume: bool,

    max_new_cases: int,

    out_dir: str,

preset_context_resolver,

) -> dict:

    envelope_report, selected_cases = load_fault_threshold_cases(

        summary,

        case_names=list(case_names),

        top_cases=top_cases,

        include_family_worst=include_family_worst,

    )

    if not selected_cases:

        raise typer.BadParameter('no fault cases were selected from the supplied summary')



    default_power = PowerSystemParams()

    base_power = envelope_report.get('power_model', {})

    base_targets = envelope_report.get('targets', {})

    preset = str(envelope_report['preset'])
    preset_cfg, obstacles = preset_context_resolver(preset)



    resolved_required_status = _normalize_required_status(base_targets.get('required_status', required_status))

    resolved_power_target_pct = float(base_targets.get('continuous_power_p95_pct', 100.0) if power_target_pct is None else power_target_pct)

    resolved_flap_target_pct = float(base_targets.get('flap_limit_usage_peak_pct', 90.0) if flap_target_pct is None else flap_target_pct)

    resolved_tune_iterations = int(base_targets.get('tune_iterations', 6) if tune_iterations is None else tune_iterations)

    resolved_tune_min_aggressiveness = float(base_targets.get('tune_min_aggressiveness', 0.25) if tune_min_aggressiveness is None else tune_min_aggressiveness)



    battery_full_values = list(battery_full_v) if battery_full_v else [float(base_power.get('battery_full_v', default_power.battery_full_v))]

    battery_empty_values = list(battery_empty_v) if battery_empty_v else [float(base_power.get('battery_empty_v', default_power.battery_empty_v))]

    capacity_values = list(capacity_kwh) if capacity_kwh else [float(base_power.get('capacity_kwh', default_power.capacity_kwh))]

    resistance_values = list(internal_resistance_ohm) if internal_resistance_ohm else [float(base_power.get('internal_resistance_ohm', default_power.internal_resistance_ohm))]

    hover_values = list(hover_power_kw) if hover_power_kw else [float(base_power.get('hover_power_kw', default_power.hover_power_w / 1000.0))]

    base_continuous_kw = float(base_power.get('continuous_power_kw', default_power.continuous_power_w / 1000.0))

    continuous_values = list(continuous_power_kw) if continuous_power_kw else _default_continuous_power_sweep(base_continuous_kw)

    peak_values = list(peak_power_kw) if peak_power_kw else [float(base_power.get('peak_power_kw', default_power.peak_power_w / 1000.0))]

    aux_values = list(aux_power_kw) if aux_power_kw else [float(base_power.get('aux_power_kw', default_power.aux_power_w / 1000.0))]



    base_dir = Path(out_dir)

    base_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = base_dir / 'cache'

    manifest = _fault_threshold_manifest(

        summary_path=summary,

        preset=preset,

        selected_cases=selected_cases,

        required_status=resolved_required_status,

        power_target_pct=resolved_power_target_pct,

        flap_target_pct=resolved_flap_target_pct,

        tune_iterations=resolved_tune_iterations,

        tune_min_aggressiveness=resolved_tune_min_aggressiveness,

        battery_full_v=battery_full_values,

        battery_empty_v=battery_empty_values,

        capacity_kwh=capacity_values,

        internal_resistance_ohm=resistance_values,

        hover_power_kw=hover_values,

        continuous_power_kw=continuous_values,

        peak_power_kw=peak_values,

        aux_power_kw=aux_values,

        top=top,

    )

    cache_state = _prepare_fault_envelope_cache(cache_dir, manifest, resume=resume)



    summary_rows = []

    case_reports = []

    cached_cases = 0

    new_cases = 0

    budget_exhausted = False



    for case_info in selected_cases:

        case_name = case_info['case']

        cache_path = _fault_threshold_cache_path(cache_dir, case_name)

        cache_payload = _load_fault_envelope_cache(cache_path) if resume else None

        if cache_payload is not None:

            summary_rows.append(dict(cache_payload['summary_row']))

            case_reports.append(dict(cache_payload['case_report']))

            cached_cases += 1

            continue



        if max_new_cases < 0:

            budget_exhausted = True

            break

        if max_new_cases > 0 and new_cases >= max_new_cases:

            budget_exhausted = True

            break



        fault_summary = dict(case_info['fault_summary'])

        fault = fault_spec_from_summary(fault_summary)

        report, best_candidate = build_power_sweep_report_for_fault(

            preset=preset,

            fault_label=case_name,

            fault=fault,

            fault_summary=fault_summary,

            required_status=resolved_required_status,

            power_target_pct=resolved_power_target_pct,

            flap_target_pct=resolved_flap_target_pct,

            tune_iterations=resolved_tune_iterations,

            tune_min_aggressiveness=resolved_tune_min_aggressiveness,

            battery_full_v=battery_full_values,

            battery_empty_v=battery_empty_values,

            capacity_kwh=capacity_values,

            internal_resistance_ohm=resistance_values,

            hover_power_kw=hover_values,

            continuous_power_kw=continuous_values,

            peak_power_kw=peak_values,

            aux_power_kw=aux_values,

            top=top,

        )



        file_stub = _fault_case_slug(case_name)

        case_json = base_dir / f'{file_stub}_threshold.json'

        case_md = base_dir / f'{file_stub}_requirements.md'

        case_csv = base_dir / f'{file_stub}_requirements.csv'

        best_trace = base_dir / f'{file_stub}_best_trace.json'



        if best_candidate is not None:

            save_trace_json(str(best_trace), meta=best_candidate['meta'], hist=best_candidate['hist'])

            report['best_trace_out'] = str(best_trace)



        title_label = f'{preset} {case_name}'

        write_power_sweep_outputs(

            report,

            preset=preset,

            out=str(case_json),

            requirements_out=str(case_md),

            requirements_format='markdown',

            title_label=title_label,

        )

        case_csv.write_text(

            render_requirements_table(report['requirements_table'], format_name='csv', preset=title_label),

            encoding='utf-8',

        )



        best = report.get('best') or {}

        best_model = best.get('power_model', {})

        summary_row = {

            'case': case_name,

            'source': ','.join(case_info.get('sources', [])),

            'order': case_info.get('order'),

            'families': ','.join(case_info.get('families', [])),

            'best_feasible': bool(best.get('feasible', False)),

            'meets_threshold': bool(best.get('meets_threshold', False)),

            'required_continuous_power_kw': (best_model.get('continuous_power_kw') if best.get('meets_threshold') else None),

            'battery_full_v': best_model.get('battery_full_v'),

            'capacity_kwh': best_model.get('capacity_kwh'),

            'internal_resistance_ohm': best_model.get('internal_resistance_ohm'),

            'hover_power_kw': best_model.get('hover_power_kw'),

            'peak_power_kw': best_model.get('peak_power_kw'),

            'selected_aggressiveness': best.get('selected_aggressiveness'),

            'report_status': best.get('report_status'),

            'continuous_power_p95_pct': best.get('continuous_power_p95_pct'),

            'final_goal_error_m': best.get('final_goal_error_m'),

            'arrival_time_s': best.get('arrival_time_s'),

            'severity_score': case_info.get('severity_score'),

        }

        case_report = {

            'case': case_name,

            'fault_case': fault_summary,

            'sources': list(case_info.get('sources', [])),

            'severity_score': case_info.get('severity_score'),

            'search_space': report.get('search_space', {}),

            'best': best,

            'artifacts': {

                'json': str(case_json),

                'markdown': str(case_md),

                'csv': str(case_csv),

                'best_trace': (str(best_trace) if best_candidate is not None else ''),

            },

        }

        summary_rows.append(summary_row)

        case_reports.append(case_report)

        _write_fault_threshold_cache(cache_path, {

            'report': report,

            'summary_row': summary_row,

            'case_report': case_report,

        })

        new_cases += 1



    summary_rows.sort(key=lambda row: (

        0 if row.get('meets_threshold') else 1,

        0 if row.get('best_feasible') else 1,

        row.get('required_continuous_power_kw') if row.get('required_continuous_power_kw') is not None else 1e9,

        -(float(row.get('severity_score') or 0.0)),

        row.get('case', ''),

    ))



    completed_case_names = {item['case'] for item in case_reports}

    pending_cases = [item['case'] for item in selected_cases if item['case'] not in completed_case_names]

    summary_report = {

        'preset': preset,

        'source_summary': summary,

        'selected_cases': [item['case'] for item in selected_cases],

        'pending_cases': pending_cases,

        'targets': {

            'required_status': resolved_required_status,

            'continuous_power_p95_pct': resolved_power_target_pct,

            'flap_limit_usage_peak_pct': resolved_flap_target_pct,

            'tune_iterations': resolved_tune_iterations,

            'tune_min_aggressiveness': resolved_tune_min_aggressiveness,

        },

        'sweep_values': {

            'battery_full_v': battery_full_values,

            'battery_empty_v': battery_empty_values,

            'capacity_kwh': capacity_values,

            'internal_resistance_ohm': resistance_values,

            'hover_power_kw': hover_values,

            'continuous_power_kw': continuous_values,

            'peak_power_kw': peak_values,

            'aux_power_kw': aux_values,

        },

        'search_space': {

            'selected_cases': len(selected_cases),

            'completed_cases': len(case_reports),

            'cached_cases': int(cached_cases),

            'new_cases': int(new_cases),

            'complete': not budget_exhausted and not pending_cases,

            'max_new_cases': int(max_new_cases),

        },

        'cache': {

            'dir': str(cache_dir),

            'resume': bool(resume),

            'manifest': cache_state['path'],

        },

        'cases': case_reports,

        'summary_rows': summary_rows,

    }



    summary_json = base_dir / 'summary.json'

    summary_md = base_dir / 'summary.md'

    summary_csv = base_dir / 'summary.csv'

    summary_report['summary_artifacts'] = {

        'json': str(summary_json),

        'markdown': str(summary_md),

        'csv': str(summary_csv),

    }

    summary_json.write_text(json.dumps(summary_report, indent=2), encoding='utf-8')

    summary_md.write_text(

        render_fault_threshold_table(summary_rows, format_name='markdown', title=f'{preset} fault threshold summary'),

        encoding='utf-8',

    )

    summary_csv.write_text(

        render_fault_threshold_table(summary_rows, format_name='csv', title=f'{preset} fault threshold summary'),

        encoding='utf-8',

    )

    return summary_report







_legacy_fault_envelope_manifest = _fault_envelope_manifest

def _fault_envelope_manifest(*, preset: str, power: PowerSystemParams, required_status: str | None = None, power_target_pct: float, flap_target_pct: float, tune_iterations: int, tune_min_aggressiveness: float, fan_indices: list[int], flap_indices: list[int], plenum_indices: list[int], fan_scales: list[float], flap_angles: list[float], plenum_scales: list[float], top_per_family: int, include_pairs: bool, include_triples: bool) -> dict:
    manifest = _legacy_fault_envelope_manifest(
        preset=preset,
        power=power,
        power_target_pct=power_target_pct,
        flap_target_pct=flap_target_pct,
        tune_iterations=tune_iterations,
        tune_min_aggressiveness=tune_min_aggressiveness,
        fan_indices=fan_indices,
        flap_indices=flap_indices,
        plenum_indices=plenum_indices,
        fan_scales=fan_scales,
        flap_angles=flap_angles,
        plenum_scales=plenum_scales,
        top_per_family=top_per_family,
        include_pairs=include_pairs,
        include_triples=include_triples,
    )
    resolved = _normalize_required_status(required_status or globals().get('_fault_required_status', 'feasible'))
    manifest.setdefault('targets', {})['required_status'] = str(resolved)
    return manifest


_legacy_fault_threshold_manifest = _fault_threshold_manifest

def _fault_threshold_manifest(*, summary_path: str, preset: str, selected_cases: list[dict], required_status: str, power_target_pct: float, flap_target_pct: float, tune_iterations: int, tune_min_aggressiveness: float, battery_full_v: list[float], battery_empty_v: list[float], capacity_kwh: list[float], internal_resistance_ohm: list[float], hover_power_kw: list[float], continuous_power_kw: list[float], peak_power_kw: list[float], aux_power_kw: list[float], top: int) -> dict:
    manifest = _legacy_fault_threshold_manifest(
        summary_path=summary_path,
        preset=preset,
        selected_cases=selected_cases,
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
    manifest.setdefault('targets', {})['required_status'] = str(_normalize_required_status(required_status))
    return manifest


_legacy_build_fault_envelope_report = build_fault_envelope_report

def build_fault_envelope_report(*, preset: str, preset_cfg: dict, obstacles: list[MissionObstacle], required_status: str = 'feasible', power_target_pct: float, flap_target_pct: float, tune_iterations: int, tune_min_aggressiveness: float, battery_full_v: float | None, battery_empty_v: float | None, capacity_kwh: float | None, internal_resistance_ohm: float | None, hover_power_kw: float | None, continuous_power_kw: float | None, peak_power_kw: float | None, aux_power_kw: float | None, dead_fan_scale: list[float], stuck_flap_alpha_deg: list[float], plenum_sector_scale: list[float], fan_group: list[int], flap_idx: list[int], plenum_sector_idx: list[int], flap_step: int, plenum_step: int, top_per_family: int, include_pairs: bool, include_triples: bool, top: int, cache_dir: str | None = None, resume: bool = True, max_new_cases: int = 0) -> tuple[dict, list[dict]]:
    g = globals()
    old_preset_cfg = g.get('preset_cfg')
    old_obstacles = g.get('obstacles')
    old_required = g.get('_fault_required_status')
    normalized_required_status = _normalize_required_status(required_status)
    g['preset_cfg'] = dict(preset_cfg)
    g['obstacles'] = list(obstacles)
    g['_fault_required_status'] = normalized_required_status
    try:
        report, ranked = _legacy_build_fault_envelope_report(
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
            top=top,
            cache_dir=cache_dir,
            resume=resume,
            max_new_cases=max_new_cases,
        )
    finally:
        if old_preset_cfg is None:
            g.pop('preset_cfg', None)
        else:
            g['preset_cfg'] = old_preset_cfg
        if old_obstacles is None:
            g.pop('obstacles', None)
        else:
            g['obstacles'] = old_obstacles
        if old_required is None:
            g.pop('_fault_required_status', None)
        else:
            g['_fault_required_status'] = old_required
    report.setdefault('targets', {})['required_status'] = normalized_required_status
    return report, ranked


_legacy_run_fault_threshold_report = run_fault_threshold_report

def run_fault_threshold_report(*, summary: str, case_names: list[str], top_cases: int, include_family_worst: bool, required_status: str, power_target_pct: Optional[float], flap_target_pct: Optional[float], tune_iterations: Optional[int], tune_min_aggressiveness: Optional[float], battery_full_v: list[float], battery_empty_v: list[float], capacity_kwh: list[float], internal_resistance_ohm: list[float], hover_power_kw: list[float], continuous_power_kw: list[float], peak_power_kw: list[float], aux_power_kw: list[float], top: int, resume: bool, max_new_cases: int, out_dir: str, preset_context_resolver) -> dict:
    from aurora_vtol.reporting import render_fault_threshold_table
    g = globals()
    old_resolver = g.get('preset_context_resolver')
    old_renderer = g.get('render_fault_threshold_table')
    old_builder = g.get('build_power_sweep_report_for_fault')
    g['preset_context_resolver'] = preset_context_resolver
    g['render_fault_threshold_table'] = render_fault_threshold_table
    def _wrapped_builder(**kwargs):
        preset_name = kwargs.get('preset')
        preset_cfg, obstacles = preset_context_resolver(preset_name)
        kwargs.setdefault('preset_cfg', preset_cfg)
        kwargs.setdefault('obstacles', obstacles)
        return old_builder(**kwargs)
    g['build_power_sweep_report_for_fault'] = _wrapped_builder
    try:
        return _legacy_run_fault_threshold_report(
            summary=summary,
            case_names=case_names,
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
            preset_context_resolver=preset_context_resolver,
        )
    finally:
        if old_resolver is None:
            g.pop('preset_context_resolver', None)
        else:
            g['preset_context_resolver'] = old_resolver
        if old_renderer is None:
            g.pop('render_fault_threshold_table', None)
        else:
            g['render_fault_threshold_table'] = old_renderer
        if old_builder is None:
            g.pop('build_power_sweep_report_for_fault', None)
        else:
            g['build_power_sweep_report_for_fault'] = old_builder



def run_fault_envelope_report(
    *,
    preset: str,
    power_target_pct: float,
    flap_target_pct: float,
    tune_iterations: int,
    tune_min_aggressiveness: float,
    battery_full_v: float | None,
    battery_empty_v: float | None,
    capacity_kwh: float | None,
    internal_resistance_ohm: float | None,
    hover_power_kw: float | None,
    continuous_power_kw: float | None,
    peak_power_kw: float | None,
    aux_power_kw: float | None,
    dead_fan_scale: list[float],
    stuck_flap_alpha_deg: list[float],
    plenum_sector_scale: list[float],
    fan_group: list[int],
    flap_idx: list[int],
    plenum_sector_idx: list[int],
    flap_step: int,
    plenum_step: int,
    top_per_family: int,
    include_pairs: bool,
    include_triples: bool,
    resume: bool,
    max_new_cases: int,
    top: int,
    save_traces: int,
    out_dir: str,
    preset_context_resolver,
) -> dict:
    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / 'cache'
    preset_cfg, obstacles = preset_context_resolver(preset)

    report, ranked = build_fault_envelope_report(
        preset=preset,
        preset_cfg=preset_cfg,
        obstacles=obstacles,
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
        top=top,
        cache_dir=str(cache_dir),
        resume=resume,
        max_new_cases=max_new_cases,
    )

    trace_paths = []
    for idx, bundle in enumerate(ranked[:max(0, int(save_traces))], start=1):
        trace_path = base_dir / f'{idx:02d}_{_fault_case_slug(bundle["entry"]["case"])}.json'
        cache_trace = bundle.get('cache_trace')
        if cache_trace and Path(cache_trace).exists():
            trace_path.write_text(Path(cache_trace).read_text(encoding='utf-8'), encoding='utf-8')
        elif bundle.get('meta') is not None and bundle.get('hist') is not None:
            save_trace_json(str(trace_path), meta=bundle['meta'], hist=bundle['hist'])
        else:
            continue
        bundle['entry']['trace'] = str(trace_path)
        trace_paths.append(str(trace_path))

    summary_json = base_dir / 'summary.json'
    summary_md = base_dir / 'summary.md'
    summary_csv = base_dir / 'summary.csv'
    summary_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary_md.write_text(
        render_fault_envelope_table(report['top_cases'], format_name='markdown', title=f'{preset} fault envelope summary'),
        encoding='utf-8',
    )
    summary_csv.write_text(
        render_fault_envelope_table(report['top_cases'], format_name='csv', title=f'{preset} fault envelope summary'),
        encoding='utf-8',
    )
    report['artifacts'] = {
        'json': str(summary_json),
        'markdown': str(summary_md),
        'csv': str(summary_csv),
        'cache_dir': str(cache_dir),
        'traces': trace_paths,
    }
    return report



def run_fault_threshold_pack_report(
    *,
    summary_paths: list[str],
    preset_names: list[str],
    case_names: list[str],
    top_cases: int,
    include_family_worst: bool,
    required_status: str,
    power_target_pct: Optional[float],
    flap_target_pct: Optional[float],
    tune_iterations: Optional[int],
    tune_min_aggressiveness: Optional[float],
    battery_full_v: list[float],
    battery_empty_v: list[float],
    capacity_kwh: list[float],
    internal_resistance_ohm: list[float],
    hover_power_kw: list[float],
    continuous_power_kw: list[float],
    peak_power_kw: list[float],
    aux_power_kw: list[float],
    top: int,
    resume: bool,
    max_new_cases: int,
    max_new_cases_total: int,
    out_dir: str,
    preset_context_resolver,
) -> dict:
    requested_presets = set(preset_names)
    paths = [Path(item) for item in summary_paths] if summary_paths else sorted(Path('runs').glob('fault_envelope*/summary.json'))
    if not paths:
        raise typer.BadParameter('no fault-envelope summaries were provided or discovered')

    resolved_summaries = []
    seen = set()
    for summary_path in paths:
        if not summary_path.exists():
            raise typer.BadParameter(f'fault-envelope summary does not exist: {summary_path}')
        envelope_report = json.loads(summary_path.read_text(encoding='utf-8'))
        current_preset = str(envelope_report.get('preset') or '').strip()
        if not current_preset:
            raise typer.BadParameter(f'fault-envelope summary missing preset: {summary_path}')
        if requested_presets and current_preset not in requested_presets:
            continue
        key = (current_preset, str(summary_path))
        if key in seen:
            continue
        seen.add(key)
        resolved_summaries.append({
            'preset': current_preset,
            'summary_path': str(summary_path),
            'summary_source': summary_path.parent.name,
        })

    if not resolved_summaries:
        allowed = ', '.join(sorted(requested_presets)) if requested_presets else 'discovered summaries'
        raise typer.BadParameter(f'no summaries matched the requested presets: {allowed}')

    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    pack_rows = []
    preset_reports = []
    total_cached_cases = 0
    total_new_cases = 0
    all_complete = True
    remaining_total_budget = int(max_new_cases_total) if max_new_cases_total > 0 else None

    for item in resolved_summaries:
        preset_name = item['preset']
        summary_path = item['summary_path']
        summary_source = item['summary_source']
        preset_dir = base_dir / f'{preset_name}__{_fault_case_slug(summary_source)}'
        if remaining_total_budget is None:
            per_summary_budget = max_new_cases
        elif remaining_total_budget <= 0:
            per_summary_budget = -1
        elif max_new_cases > 0:
            per_summary_budget = min(int(max_new_cases), int(remaining_total_budget))
        else:
            per_summary_budget = int(remaining_total_budget)

        report = run_fault_threshold_report(
            summary=summary_path,
            case_names=list(case_names),
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
            max_new_cases=per_summary_budget,
            out_dir=str(preset_dir),
            preset_context_resolver=preset_context_resolver,
        )
        search_space = report.get('search_space', {})
        total_cached_cases += int(search_space.get('cached_cases', 0))
        total_new_cases += int(search_space.get('new_cases', 0))
        if remaining_total_budget is not None:
            remaining_total_budget -= int(search_space.get('new_cases', 0))
        all_complete = all_complete and bool(search_space.get('complete', False))

        for row in report.get('summary_rows', []):
            pack_rows.append({
                'preset': preset_name,
                'summary_source': summary_source,
                'case': row.get('case'),
                'source': row.get('source'),
                'order': row.get('order'),
                'families': row.get('families'),
                'best_feasible': row.get('best_feasible'),
                'meets_threshold': row.get('meets_threshold'),
                'required_continuous_power_kw': row.get('required_continuous_power_kw'),
                'battery_full_v': row.get('battery_full_v'),
                'capacity_kwh': row.get('capacity_kwh'),
                'internal_resistance_ohm': row.get('internal_resistance_ohm'),
                'hover_power_kw': row.get('hover_power_kw'),
                'peak_power_kw': row.get('peak_power_kw'),
                'selected_aggressiveness': row.get('selected_aggressiveness'),
                'report_status': row.get('report_status'),
                'continuous_power_p95_pct': row.get('continuous_power_p95_pct'),
                'final_goal_error_m': row.get('final_goal_error_m'),
                'arrival_time_s': row.get('arrival_time_s'),
                'severity_score': row.get('severity_score'),
            })

        preset_reports.append({
            'preset': preset_name,
            'summary_source': summary_source,
            'source_summary': summary_path,
            'search_space': search_space,
            'pending_cases': report.get('pending_cases', []),
            'summary_artifacts': report.get('summary_artifacts', {}),
        })

    pack_rows.sort(key=lambda row: (
        0 if row.get('meets_threshold') else 1,
        0 if row.get('best_feasible') else 1,
        row.get('required_continuous_power_kw') if row.get('required_continuous_power_kw') is not None else 1e9,
        row.get('preset', ''),
        -(float(row.get('severity_score') or 0.0)),
        row.get('case', ''),
    ))

    pack_report = {
        'summaries': preset_reports,
        'summary_rows': pack_rows,
        'search_space': {
            'summaries': len(resolved_summaries),
            'cached_cases': int(total_cached_cases),
            'new_cases': int(total_new_cases),
            'complete': bool(all_complete),
            'max_new_cases_per_summary': int(max_new_cases),
            'max_new_cases_total': int(max_new_cases_total),
        },
    }
    summary_json = base_dir / 'summary.json'
    summary_md = base_dir / 'summary.md'
    summary_csv = base_dir / 'summary.csv'
    pack_report['summary_artifacts'] = {
        'json': str(summary_json),
        'markdown': str(summary_md),
        'csv': str(summary_csv),
    }
    summary_json.write_text(json.dumps(pack_report, indent=2), encoding='utf-8')
    summary_md.write_text(
        render_fault_threshold_pack_table(pack_rows, format_name='markdown', title='fault threshold pack summary'),
        encoding='utf-8',
    )
    summary_csv.write_text(
        render_fault_threshold_pack_table(pack_rows, format_name='csv', title='fault threshold pack summary'),
        encoding='utf-8',
    )
    return pack_report
