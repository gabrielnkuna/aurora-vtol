from __future__ import annotations

from dataclasses import dataclass
import math

from .field import RepelField, repel_force_xy


def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class MissionObstacle:
    x_m: float
    y_m: float
    radius_m: float = 12.0
    k_n_per_m: float = 180.0
    fxy_max_n: float = 2500.0
    swirl_n: float = 900.0
    influence_m: float = 20.0



def mission_safety_force(x_m: float, y_m: float, obstacles: list[MissionObstacle]) -> tuple[float, float, float | None, int, float]:
    if not obstacles:
        return 0.0, 0.0, None, 0, 0.0

    fx_total = 0.0
    fy_total = 0.0
    nearest_clearance_m = None
    active_count = 0
    max_threat = 0.0

    for obs in obstacles:
        dist_center = float(math.hypot(x_m - obs.x_m, y_m - obs.y_m))
        clearance_m = dist_center - obs.radius_m
        if nearest_clearance_m is None or clearance_m < nearest_clearance_m:
            nearest_clearance_m = clearance_m

        safety_radius_m = obs.radius_m + max(4.0, 0.35 * obs.influence_m)
        field = RepelField(k_n_per_m=obs.k_n_per_m, radius_m=safety_radius_m, fxy_max_n=obs.fxy_max_n)
        fx_rep, fy_rep = repel_force_xy(field, x_m, y_m, obs.x_m, obs.y_m)
        fx_total += float(fx_rep)
        fy_total += float(fy_rep)

        if dist_center <= safety_radius_m:
            active_count += 1
        threat = 1.0 - max(0.0, dist_center - obs.radius_m) / max(obs.influence_m, 1e-6)
        max_threat = max(max_threat, _smoothstep01(threat))

    return fx_total, fy_total, nearest_clearance_m, active_count, max_threat


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    dx = bx - ax
    dy = by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-9:
        return float(math.hypot(px - ax, py - ay)), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return float(math.hypot(px - qx, py - qy)), t


def route_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += float(math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]))
    return total


def route_penalty(points: list[tuple[float, float]], obstacles: list[MissionObstacle], clearance_m: float) -> float:
    penalty = 0.0
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        for obs in obstacles:
            dist_m, frac = point_segment_distance(obs.x_m, obs.y_m, ax, ay, bx, by)
            required_m = obs.radius_m + clearance_m
            if 0.02 < frac < 0.98 and dist_m < required_m:
                penalty += (required_m - dist_m) ** 2 * 200.0
    return penalty


def choose_bypass_waypoints(ax: float, ay: float, bx: float, by: float, obs: MissionObstacle, clearance_m: float, obstacles: list[MissionObstacle]) -> list[tuple[float, float]]:
    seg_dx = bx - ax
    seg_dy = by - ay
    seg_len = max(1e-6, float(math.hypot(seg_dx, seg_dy)))
    ux = seg_dx / seg_len
    uy = seg_dy / seg_len
    px = -uy
    py = ux
    offset_m = obs.radius_m + clearance_m
    lead_m = max(6.0, clearance_m * 1.5, min(obs.influence_m, offset_m * 1.5))

    best_score = None
    best_points = None
    for side in (-1.0, 1.0):
        wp1 = (obs.x_m - ux * lead_m + side * px * offset_m, obs.y_m - uy * lead_m + side * py * offset_m)
        wp2 = (obs.x_m + ux * lead_m + side * px * offset_m, obs.y_m + uy * lead_m + side * py * offset_m)
        points = [(ax, ay), wp1, wp2, (bx, by)]
        score = route_length(points) + route_penalty(points, obstacles, clearance_m)
        if best_score is None or score < best_score:
            best_score = score
            best_points = [wp1, wp2]

    return best_points if best_points is not None else []


def plan_route_waypoints(start_x_m: float, start_y_m: float, dest_x_m: float, dest_y_m: float, obstacles: list[MissionObstacle], clearance_m: float) -> list[tuple[float, float]]:
    route = [(start_x_m, start_y_m), (dest_x_m, dest_y_m)]
    max_insertions = max(4, len(obstacles) * 4)

    for _ in range(max_insertions):
        best_block = None
        for seg_idx in range(len(route) - 1):
            ax, ay = route[seg_idx]
            bx, by = route[seg_idx + 1]
            for obs in obstacles:
                dist_m, frac = point_segment_distance(obs.x_m, obs.y_m, ax, ay, bx, by)
                required_m = obs.radius_m + clearance_m
                if 0.05 < frac < 0.95 and dist_m < required_m:
                    severity = required_m - dist_m
                    if best_block is None or severity > best_block['severity']:
                        best_block = {
                            'seg_idx': seg_idx,
                            'ax': ax,
                            'ay': ay,
                            'bx': bx,
                            'by': by,
                            'obs': obs,
                            'severity': severity,
                        }

        if best_block is None:
            break

        bypass = choose_bypass_waypoints(
            best_block['ax'],
            best_block['ay'],
            best_block['bx'],
            best_block['by'],
            best_block['obs'],
            clearance_m,
            obstacles,
        )
        if not bypass:
            break
        route = route[:best_block['seg_idx'] + 1] + bypass + route[best_block['seg_idx'] + 1:]
        if len(route) >= 18:
            break

    simplified = route[:]
    changed = True
    while changed and len(simplified) > 2:
        changed = False
        for i in range(1, len(simplified) - 1):
            direct = [simplified[i - 1], simplified[i + 1]]
            current = [simplified[i - 1], simplified[i], simplified[i + 1]]
            direct_score = route_length(direct) + route_penalty(direct, obstacles, clearance_m)
            current_score = route_length(current) + route_penalty(current, obstacles, clearance_m)
            if direct_score <= current_score + 0.5:
                del simplified[i]
                changed = True
                break

    cleaned = [simplified[0]]
    for pt in simplified[1:]:
        if math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 0.75:
            cleaned.append(pt)
    return cleaned
