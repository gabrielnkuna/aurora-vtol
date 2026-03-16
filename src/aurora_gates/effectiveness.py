from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

from .topology import RingActuatorTopology, default_ring_topology


@dataclass(frozen=True)
class NominalEffectivenessTable:
    schema_version: str
    table_name: str
    segment_count: int
    fan_count: int
    plenum_count: int
    fan_segment_weights: np.ndarray
    plenum_segment_weights: np.ndarray
    axial_scale_by_segment: np.ndarray
    radial_scale_by_segment: np.ndarray
    tangential_scale_by_segment: np.ndarray
    provenance: str = "provisional"

    def __post_init__(self) -> None:
        fan_weights = np.asarray(self.fan_segment_weights, dtype=float)
        plenum_weights = np.asarray(self.plenum_segment_weights, dtype=float)
        axial = np.asarray(self.axial_scale_by_segment, dtype=float)
        radial = np.asarray(self.radial_scale_by_segment, dtype=float)
        tangential = np.asarray(self.tangential_scale_by_segment, dtype=float)
        if fan_weights.shape != (self.fan_count, self.segment_count):
            raise ValueError(f"fan_segment_weights shape {fan_weights.shape} != {(self.fan_count, self.segment_count)}")
        if plenum_weights.shape != (self.plenum_count, self.segment_count):
            raise ValueError(f"plenum_segment_weights shape {plenum_weights.shape} != {(self.plenum_count, self.segment_count)}")
        for name, arr in (("axial", axial), ("radial", radial), ("tangential", tangential)):
            if arr.shape != (self.segment_count,):
                raise ValueError(f"{name}_scale_by_segment shape {arr.shape} != {(self.segment_count,)}")
        object.__setattr__(self, "fan_segment_weights", fan_weights)
        object.__setattr__(self, "plenum_segment_weights", plenum_weights)
        object.__setattr__(self, "axial_scale_by_segment", np.clip(axial, 0.0, None))
        object.__setattr__(self, "radial_scale_by_segment", np.clip(radial, 0.0, None))
        object.__setattr__(self, "tangential_scale_by_segment", np.clip(tangential, 0.0, None))

    @property
    def fan_weight_row_sums(self) -> np.ndarray:
        return np.where(self.fan_segment_weights.sum(axis=1) > 1e-12, self.fan_segment_weights.sum(axis=1), 1.0)

    def component_scale(self, component: str) -> np.ndarray:
        key = component.strip().lower()
        if key == "axial":
            return self.axial_scale_by_segment
        if key == "radial":
            return self.radial_scale_by_segment
        if key == "tangential":
            return self.tangential_scale_by_segment
        raise ValueError(f"unknown component: {component}")

    def fan_means_from_segments(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size != self.segment_count:
            raise ValueError(f"segment value count {arr.size} != {self.segment_count}")
        weighted = self.fan_segment_weights @ arr
        return weighted / self.fan_weight_row_sums

    def smooth_segment_values(self, values, component: str = "radial") -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size != self.segment_count:
            raise ValueError(f"segment value count {arr.size} != {self.segment_count}")
        fan_means = self.fan_means_from_segments(arr)
        smoothed = self.fan_segment_weights.T @ fan_means
        return smoothed * self.component_scale(component)

    def apply_tangential_effectiveness(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size != self.segment_count:
            raise ValueError(f"segment value count {arr.size} != {self.segment_count}")
        return arr * self.tangential_scale_by_segment


@dataclass(frozen=True)
class GeometrySeedSpec:
    schema_version: str
    spec_name: str
    segment_count: int
    fan_center_deg: np.ndarray
    fan_half_span_deg: np.ndarray
    fan_sigma_deg: np.ndarray
    plenum_center_deg: np.ndarray
    plenum_half_span_deg: np.ndarray
    plenum_sigma_deg: np.ndarray
    axial_scale_by_segment: np.ndarray
    radial_scale_by_segment: np.ndarray
    tangential_scale_by_segment: np.ndarray
    provenance: str = "geometry-seeded provisional"

    def __post_init__(self) -> None:
        fan_center = np.asarray(self.fan_center_deg, dtype=float)
        fan_half_span = _expand_vector(self.fan_half_span_deg, fan_center.size)
        fan_sigma = _expand_vector(self.fan_sigma_deg, fan_center.size)
        plenum_center = np.asarray(self.plenum_center_deg, dtype=float)
        plenum_half_span = _expand_vector(self.plenum_half_span_deg, plenum_center.size)
        plenum_sigma = _expand_vector(self.plenum_sigma_deg, plenum_center.size)
        axial = np.asarray(self.axial_scale_by_segment, dtype=float)
        radial = np.asarray(self.radial_scale_by_segment, dtype=float)
        tangential = np.asarray(self.tangential_scale_by_segment, dtype=float)
        for name, arr in (("axial", axial), ("radial", radial), ("tangential", tangential)):
            if arr.shape != (self.segment_count,):
                raise ValueError(f"{name}_scale_by_segment shape {arr.shape} != {(self.segment_count,)}")
        object.__setattr__(self, "fan_center_deg", fan_center)
        object.__setattr__(self, "fan_half_span_deg", fan_half_span)
        object.__setattr__(self, "fan_sigma_deg", np.clip(fan_sigma, 0.0, None))
        object.__setattr__(self, "plenum_center_deg", plenum_center)
        object.__setattr__(self, "plenum_half_span_deg", plenum_half_span)
        object.__setattr__(self, "plenum_sigma_deg", np.clip(plenum_sigma, 0.0, None))
        object.__setattr__(self, "axial_scale_by_segment", np.clip(axial, 0.0, None))
        object.__setattr__(self, "radial_scale_by_segment", np.clip(radial, 0.0, None))
        object.__setattr__(self, "tangential_scale_by_segment", np.clip(tangential, 0.0, None))

    @property
    def fan_count(self) -> int:
        return int(self.fan_center_deg.size)

    @property
    def plenum_count(self) -> int:
        return int(self.plenum_center_deg.size)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _effectiveness_asset_path(name: str) -> Path:
    return _repo_root() / "data" / "effectiveness" / name


def _effectiveness_spec_path(name: str) -> Path:
    return _repo_root() / "data" / "effectiveness_specs" / name


def _expand_vector(value, count: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(int(count), float(arr), dtype=float)
    if arr.shape != (count,):
        raise ValueError(f"expected vector of length {count}, got {arr.shape}")
    return arr.astype(float)


def _segment_angles_deg(segment_count: int) -> np.ndarray:
    return np.linspace(0.0, 360.0, int(segment_count), endpoint=False, dtype=float)


def _circular_angle_distance_deg(a_deg: float, b_deg: float) -> float:
    delta = (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0
    return abs(delta)


def _footprint_distance_deg(angle_deg: float, center_deg: float, half_span_deg: float) -> float:
    return max(0.0, _circular_angle_distance_deg(angle_deg, center_deg) - max(0.0, float(half_span_deg)))


def _weights_from_geometry(segment_angles_deg: np.ndarray, center_deg: np.ndarray, half_span_deg: np.ndarray, sigma_deg: np.ndarray) -> np.ndarray:
    count = int(segment_angles_deg.size)
    item_count = int(center_deg.size)
    raw = np.zeros((item_count, count), dtype=float)
    for item_idx in range(item_count):
        center = float(center_deg[item_idx])
        half_span = float(half_span_deg[item_idx])
        sigma = float(max(0.0, sigma_deg[item_idx]))
        for seg_idx, seg_angle in enumerate(segment_angles_deg):
            dist = _footprint_distance_deg(float(seg_angle), center, half_span)
            if sigma <= 1e-9:
                raw[item_idx, seg_idx] = 1.0 if dist <= 1e-9 else 0.0
            else:
                raw[item_idx, seg_idx] = float(np.exp(-0.5 * (dist / sigma) ** 2))
        owned = [_footprint_distance_deg(float(seg_angle), center, half_span) <= 1e-9 for seg_angle in segment_angles_deg]
        if any(owned):
            for seg_idx, is_owned in enumerate(owned):
                if is_owned:
                    raw[item_idx, seg_idx] = 1.0
    col_sums = raw.sum(axis=0)
    for seg_idx in range(count):
        if col_sums[seg_idx] <= 1e-12:
            nearest = int(np.argmin([_circular_angle_distance_deg(segment_angles_deg[seg_idx], center) for center in center_deg]))
            raw[nearest, seg_idx] = 1.0
    col_sums = np.where(raw.sum(axis=0) > 1e-12, raw.sum(axis=0), 1.0)
    return raw / col_sums[None, :]


def load_effectiveness_table(path: str | Path) -> NominalEffectivenessTable:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return NominalEffectivenessTable(
        schema_version=str(data["schema_version"]),
        table_name=str(data["table_name"]),
        segment_count=int(data["segment_count"]),
        fan_count=int(data["fan_count"]),
        plenum_count=int(data["plenum_count"]),
        fan_segment_weights=np.asarray(data["fan_segment_weights"], dtype=float),
        plenum_segment_weights=np.asarray(data["plenum_segment_weights"], dtype=float),
        axial_scale_by_segment=np.asarray(data["axial_scale_by_segment"], dtype=float),
        radial_scale_by_segment=np.asarray(data["radial_scale_by_segment"], dtype=float),
        tangential_scale_by_segment=np.asarray(data["tangential_scale_by_segment"], dtype=float),
        provenance=str(data.get("provenance", "provisional")),
    )


def load_geometry_seed_spec(path: str | Path) -> GeometrySeedSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GeometrySeedSpec(
        schema_version=str(data["schema_version"]),
        spec_name=str(data["spec_name"]),
        segment_count=int(data["segment_count"]),
        fan_center_deg=np.asarray(data["fan_center_deg"], dtype=float),
        fan_half_span_deg=data["fan_half_span_deg"],
        fan_sigma_deg=data["fan_sigma_deg"],
        plenum_center_deg=np.asarray(data["plenum_center_deg"], dtype=float),
        plenum_half_span_deg=data["plenum_half_span_deg"],
        plenum_sigma_deg=data["plenum_sigma_deg"],
        axial_scale_by_segment=np.asarray(data["axial_scale_by_segment"], dtype=float),
        radial_scale_by_segment=np.asarray(data["radial_scale_by_segment"], dtype=float),
        tangential_scale_by_segment=np.asarray(data["tangential_scale_by_segment"], dtype=float),
        provenance=str(data.get("provenance", "geometry-seeded provisional")),
    )


def build_effectiveness_table_from_geometry_seed(spec: GeometrySeedSpec) -> NominalEffectivenessTable:
    segment_angles = _segment_angles_deg(spec.segment_count)
    fan_weights = _weights_from_geometry(segment_angles, spec.fan_center_deg, spec.fan_half_span_deg, spec.fan_sigma_deg)
    plenum_weights = _weights_from_geometry(segment_angles, spec.plenum_center_deg, spec.plenum_half_span_deg, spec.plenum_sigma_deg)
    return NominalEffectivenessTable(
        schema_version="aurora.effectiveness.v1",
        table_name=f"{spec.spec_name}-table",
        segment_count=spec.segment_count,
        fan_count=spec.fan_count,
        plenum_count=spec.plenum_count,
        fan_segment_weights=fan_weights,
        plenum_segment_weights=plenum_weights,
        axial_scale_by_segment=spec.axial_scale_by_segment,
        radial_scale_by_segment=spec.radial_scale_by_segment,
        tangential_scale_by_segment=spec.tangential_scale_by_segment,
        provenance=spec.provenance,
    )


def build_seeded_effectiveness_table(topology: RingActuatorTopology) -> NominalEffectivenessTable:
    fan_weights = np.asarray(topology.fan_segment_influence, dtype=float)
    plenum_weights = np.zeros((len(topology.plenum_to_segments), topology.segment_count), dtype=float)
    for plenum_idx, segments in enumerate(topology.plenum_to_segments):
        for seg_idx in segments:
            if 0 <= int(seg_idx) < topology.segment_count:
                plenum_weights[plenum_idx, int(seg_idx)] = 1.0
    return NominalEffectivenessTable(
        schema_version="aurora.effectiveness.v1",
        table_name=f"seeded-ring-{topology.segment_count}",
        segment_count=topology.segment_count,
        fan_count=topology.fan_count,
        plenum_count=len(topology.plenum_to_segments),
        fan_segment_weights=fan_weights,
        plenum_segment_weights=plenum_weights,
        axial_scale_by_segment=np.ones(topology.segment_count, dtype=float),
        radial_scale_by_segment=np.ones(topology.segment_count, dtype=float),
        tangential_scale_by_segment=np.ones(topology.segment_count, dtype=float),
        provenance="seeded-from-topology",
    )


@lru_cache(maxsize=8)
def default_effectiveness_table(segment_count: int) -> NominalEffectivenessTable:
    count = int(segment_count)
    if count == 32:
        spec_asset = _effectiveness_spec_path("aurora_ring32_geometry_seed_v1.json")
        if spec_asset.exists():
            return build_effectiveness_table_from_geometry_seed(load_geometry_seed_spec(spec_asset))
        asset = _effectiveness_asset_path("aurora_ring32_provisional_v1.json")
        if asset.exists():
            return load_effectiveness_table(asset)
    return build_seeded_effectiveness_table(default_ring_topology(count))
