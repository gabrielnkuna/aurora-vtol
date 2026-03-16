from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .allocator.faults import FaultSpec


@dataclass(frozen=True)
class RingEffectivenessMap:
    fan_index_by_segment: tuple[int, ...]
    plenum_index_by_segment: tuple[int, ...]
    fan_scale_by_segment: tuple[float, ...]
    plenum_scale_by_segment: tuple[float, ...]

    @property
    def segment_scale(self) -> np.ndarray:
        return np.asarray(self.fan_scale_by_segment, dtype=float) * np.asarray(self.plenum_scale_by_segment, dtype=float)


@dataclass(frozen=True)
class RingActuatorTopology:
    segment_count: int
    fan_to_segments: tuple[tuple[int, ...], ...]
    plenum_to_segments: tuple[tuple[int, ...], ...]
    fan_nominal_sigma_segments: float = 0.0
    fan_fault_sigma_segments: float = 0.0
    plenum_fault_sigma_segments: float = 0.0

    @property
    def fan_count(self) -> int:
        return len(self.fan_to_segments)

    @classmethod
    def even_pairs(cls, segment_count: int) -> "RingActuatorTopology":
        fan_to_segments = tuple(
            tuple(range(start, min(start + 2, segment_count)))
            for start in range(0, segment_count, 2)
        )
        plenum_to_segments = tuple((idx,) for idx in range(segment_count))
        return cls(
            segment_count=int(segment_count),
            fan_to_segments=fan_to_segments,
            plenum_to_segments=plenum_to_segments,
            fan_nominal_sigma_segments=0.0,
            fan_fault_sigma_segments=0.0,
            plenum_fault_sigma_segments=0.0,
        )

    @classmethod
    def aurora_ring_32(cls) -> "RingActuatorTopology":
        return cls(
            segment_count=32,
            fan_to_segments=AURORA_FAN_TO_SEGMENTS_32,
            plenum_to_segments=AURORA_PLENUM_TO_SEGMENTS_32,
            fan_nominal_sigma_segments=0.85,
            fan_fault_sigma_segments=0.65,
            plenum_fault_sigma_segments=0.50,
        )

    def fan_segments(self, fan_index: int) -> tuple[int, ...]:
        if 0 <= fan_index < self.fan_count:
            return self.fan_to_segments[fan_index]
        return ()

    def plenum_segments(self, plenum_index: int) -> tuple[int, ...]:
        if 0 <= plenum_index < len(self.plenum_to_segments):
            return self.plenum_to_segments[plenum_index]
        return ()

    def _circular_segment_distance(self, a: int, b: int) -> float:
        if self.segment_count <= 0:
            return 0.0
        raw = abs(int(a) - int(b))
        return float(min(raw, self.segment_count - raw))

    def _owned_distance_profile(self, owners: tuple[int, ...]) -> np.ndarray:
        if self.segment_count <= 0:
            return np.zeros(0, dtype=float)
        if not owners:
            return np.zeros(self.segment_count, dtype=float)
        owner_list = tuple(int(idx) for idx in owners)
        out = np.zeros(self.segment_count, dtype=float)
        for seg_idx in range(self.segment_count):
            out[seg_idx] = min(self._circular_segment_distance(seg_idx, owner) for owner in owner_list)
        return out

    def _fault_influence_profile(self, owners: tuple[int, ...], sigma_segments: float) -> np.ndarray:
        dist = self._owned_distance_profile(owners)
        if dist.size == 0:
            return dist
        sigma = float(max(0.0, sigma_segments))
        if sigma <= 1e-9:
            return (dist <= 1e-9).astype(float)
        profile = np.exp(-0.5 * (dist / sigma) ** 2)
        profile = np.clip(profile, 0.0, 1.0)
        for owner in owners:
            if 0 <= int(owner) < self.segment_count:
                profile[int(owner)] = 1.0
        return profile

    @cached_property
    def fan_segment_influence(self) -> np.ndarray:
        if self.segment_count <= 0 or self.fan_count <= 0:
            return np.zeros((self.fan_count, self.segment_count), dtype=float)
        raw = np.zeros((self.fan_count, self.segment_count), dtype=float)
        for fan_idx, segments in enumerate(self.fan_to_segments):
            sigma = float(max(0.0, self.fan_nominal_sigma_segments))
            if sigma <= 1e-9:
                for seg_idx in segments:
                    if 0 <= int(seg_idx) < self.segment_count:
                        raw[fan_idx, int(seg_idx)] = 1.0
            else:
                raw[fan_idx, :] = self._fault_influence_profile(tuple(int(v) for v in segments), sigma)
        col_sums = raw.sum(axis=0)
        for seg_idx in range(self.segment_count):
            if col_sums[seg_idx] <= 1e-12:
                for fan_idx, segments in enumerate(self.fan_to_segments):
                    if seg_idx in segments:
                        raw[fan_idx, seg_idx] = 1.0
                        break
        col_sums = np.where(raw.sum(axis=0) > 1e-12, raw.sum(axis=0), 1.0)
        return raw / col_sums[None, :]

    @cached_property
    def fan_effective_segment_counts(self) -> np.ndarray:
        influence = self.fan_segment_influence
        if influence.size == 0:
            return np.zeros(self.fan_count, dtype=float)
        return np.sum(influence, axis=1)

    def smooth_segment_values(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr.copy()
        if arr.size != self.segment_count:
            return default_ring_topology(int(arr.size)).smooth_segment_values(arr)
        influence = self.fan_segment_influence
        if influence.size == 0:
            return arr.copy()
        row_sums = np.where(self.fan_effective_segment_counts > 1e-12, self.fan_effective_segment_counts, 1.0)
        fan_means = (influence @ arr) / row_sums
        return influence.T @ fan_means

    def segment_values_to_fan_means(self, values) -> list[float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return []
        if arr.size != self.segment_count:
            return default_ring_topology(int(arr.size)).segment_values_to_fan_means(arr)
        influence = self.fan_segment_influence
        row_sums = np.where(self.fan_effective_segment_counts > 1e-12, self.fan_effective_segment_counts, 1.0)
        weighted = influence @ arr
        return [float(weighted[idx] / row_sums[idx]) for idx in range(self.fan_count)]

    def distribute_fan_means_to_segments(self, fan_mean_n, segment_targets_n) -> np.ndarray:
        fan_mean = np.asarray(fan_mean_n, dtype=float)
        targets = np.asarray(segment_targets_n, dtype=float)
        if targets.size == 0:
            return targets.copy()
        if targets.size != self.segment_count:
            return default_ring_topology(int(targets.size)).distribute_fan_means_to_segments(fan_mean, targets)

        influence = self.fan_segment_influence
        row_sums = self.fan_effective_segment_counts
        targets_pos = np.clip(targets, 0.0, None)
        out = np.zeros_like(targets)
        for fan_idx in range(self.fan_count):
            if fan_idx >= fan_mean.size:
                break
            total = float(row_sums[fan_idx] * float(fan_mean[fan_idx]))
            if total <= 1e-12:
                continue
            base = influence[fan_idx]
            weights = base * targets_pos
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                weights = base.copy()
                weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                continue
            out += total * weights / weight_sum
        return out

    def fan_fault_influence_profile(self, fan_index: int) -> np.ndarray:
        return self._fault_influence_profile(
            self.fan_segments(int(fan_index)),
            self.fan_fault_sigma_segments,
        )

    def plenum_fault_influence_profile(self, plenum_index: int) -> np.ndarray:
        return self._fault_influence_profile(
            self.plenum_segments(int(plenum_index)),
            self.plenum_fault_sigma_segments,
        )

    def effectiveness_map(self, fault: FaultSpec | None = None) -> RingEffectivenessMap:
        fan_index_by_segment = [-1] * self.segment_count
        plenum_index_by_segment = [-1] * self.segment_count
        fan_scale = np.ones(self.segment_count, dtype=float)
        plenum_scale = np.ones(self.segment_count, dtype=float)

        for fan_idx, segments in enumerate(self.fan_to_segments):
            for idx in segments:
                if 0 <= idx < self.segment_count:
                    fan_index_by_segment[idx] = fan_idx
        for plenum_idx, segments in enumerate(self.plenum_to_segments):
            for idx in segments:
                if 0 <= idx < self.segment_count:
                    plenum_index_by_segment[idx] = plenum_idx

        if fault is not None:
            if fault.dead_fan_group is not None:
                influence = self.fan_fault_influence_profile(int(fault.dead_fan_group))
                fan_scale *= 1.0 - (1.0 - float(fault.dead_fan_scale)) * influence
            if fault.plenum_sector_idx is not None:
                influence = self.plenum_fault_influence_profile(int(fault.plenum_sector_idx))
                plenum_scale *= 1.0 - (1.0 - float(fault.plenum_sector_scale)) * influence

        return RingEffectivenessMap(
            fan_index_by_segment=tuple(fan_index_by_segment),
            plenum_index_by_segment=tuple(plenum_index_by_segment),
            fan_scale_by_segment=tuple(float(v) for v in np.clip(fan_scale, 0.0, None)),
            plenum_scale_by_segment=tuple(float(v) for v in np.clip(plenum_scale, 0.0, None)),
        )

    def segment_effectiveness_scales(self, fault: FaultSpec | None) -> np.ndarray:
        return self.effectiveness_map(fault).segment_scale


def default_ring_topology(segment_count: int) -> RingActuatorTopology:
    if int(segment_count) == 32:
        return RingActuatorTopology.aurora_ring_32()
    return RingActuatorTopology.even_pairs(segment_count)


AURORA_FAN_TO_SEGMENTS_32: tuple[tuple[int, ...], ...] = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (8, 9), (10, 11), (12, 13), (14, 15),
    (16, 17), (18, 19), (20, 21), (22, 23),
    (24, 25), (26, 27), (28, 29), (30, 31),
)

AURORA_PLENUM_TO_SEGMENTS_32: tuple[tuple[int, ...], ...] = (
    (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,),
    (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,),
    (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,),
    (24,), (25,), (26,), (27,), (28,), (29,), (30,), (31,),
)
