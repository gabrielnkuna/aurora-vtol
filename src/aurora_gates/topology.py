from __future__ import annotations
from dataclasses import dataclass
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
        )

    @classmethod
    def aurora_ring_32(cls) -> "RingActuatorTopology":
        return cls(
            segment_count=32,
            fan_to_segments=AURORA_FAN_TO_SEGMENTS_32,
            plenum_to_segments=AURORA_PLENUM_TO_SEGMENTS_32,
        )

    def fan_segments(self, fan_index: int) -> tuple[int, ...]:
        if 0 <= fan_index < self.fan_count:
            return self.fan_to_segments[fan_index]
        return ()

    def plenum_segments(self, plenum_index: int) -> tuple[int, ...]:
        if 0 <= plenum_index < len(self.plenum_to_segments):
            return self.plenum_to_segments[plenum_index]
        return ()

    def segment_values_to_fan_means(self, values) -> list[float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return []
        if arr.size != self.segment_count:
            return default_ring_topology(int(arr.size)).segment_values_to_fan_means(arr)
        out: list[float] = []
        for segments in self.fan_to_segments:
            if not segments:
                out.append(0.0)
                continue
            out.append(float(np.mean(arr[list(segments)])))
        return out

    def distribute_fan_means_to_segments(self, fan_mean_n, segment_targets_n) -> np.ndarray:
        fan_mean = np.asarray(fan_mean_n, dtype=float)
        targets = np.asarray(segment_targets_n, dtype=float)
        if targets.size == 0:
            return targets.copy()
        if targets.size != self.segment_count:
            return default_ring_topology(int(targets.size)).distribute_fan_means_to_segments(fan_mean, targets)

        out = np.zeros_like(targets)
        for fan_idx, segments in enumerate(self.fan_to_segments):
            if not segments:
                continue
            seg_idx = list(segments)
            pair_targets = np.clip(targets[seg_idx], 0.0, None)
            pair_sum = float(np.sum(pair_targets))
            pair_total = max(0.0, float(len(seg_idx)) * float(fan_mean[fan_idx])) if fan_idx < fan_mean.size else 0.0
            if pair_sum <= 1e-9:
                out[seg_idx] = pair_total / max(1, len(seg_idx))
            else:
                out[seg_idx] = pair_total * pair_targets / pair_sum
        return out

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
                for idx in self.fan_segments(int(fault.dead_fan_group)):
                    if 0 <= idx < self.segment_count:
                        fan_scale[idx] *= float(fault.dead_fan_scale)
            if fault.plenum_sector_idx is not None:
                for idx in self.plenum_segments(int(fault.plenum_sector_idx)):
                    if 0 <= idx < self.segment_count:
                        plenum_scale[idx] *= float(fault.plenum_sector_scale)

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
