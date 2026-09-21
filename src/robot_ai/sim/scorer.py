"""Authoritative deadline, settling, and hard-limit scoring."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Task


@dataclass(frozen=True)
class EpisodeScore:
    success: bool
    deadline_error_m: float | None
    deadline_speed_m_s: float | None
    max_hold_error_m: float | None
    max_hold_speed_m_s: float | None
    settling_time_s: float | None
    failure_reason: str | None
    hard_violation: bool


def joint_limit_violations(q: np.ndarray, joint_limits: np.ndarray) -> np.ndarray:
    """Return one physics-sample violation flag for the declared joint envelope.

    The rigid Warp kernel clamps a state at the limit, while native MuJoCo may
    expose a constrained state at or marginally beyond it.  Reaching either bound
    is an operating violation for this task; sampling at the physics clock keeps a
    contact between two control ticks visible to the scorer.
    """

    values = np.asarray(q, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("q must be finite with shape [sample, 2]")
    if limits.shape != (2, 2) or not np.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all():
        raise ValueError("joint_limits must be finite with shape [2, 2] and ordered bounds")
    return ((values <= limits[:, 0]) | (values >= limits[:, 1])).any(axis=1)


def joint_limit_summary(time_s: np.ndarray, q: np.ndarray, joint_limits: np.ndarray) -> dict[str, object]:
    """Summarize the exact physics-rate samples used for joint-limit scoring."""

    times = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(q, dtype=np.float64)
    if times.ndim != 1 or values.shape != (len(times), 2) or not np.isfinite(times).all():
        raise ValueError("time_s and q must have matching finite [sample] and [sample, 2] shapes")
    violations = joint_limit_violations(values, joint_limits)
    hit_indices = np.flatnonzero(violations)
    limits = np.asarray(joint_limits, dtype=np.float64)
    hit_joints = np.flatnonzero(((values <= limits[:, 0]) | (values >= limits[:, 1])).any(axis=0))
    return {
        "modeled_hard_limits": ["joint_position"],
        "joint_limit_violation_sample_count": int(violations.sum()),
        "first_joint_limit_violation_s": None if len(hit_indices) == 0 else float(times[hit_indices[0]]),
        "joint_limit_violation_joints": hit_joints.astype(int).tolist(),
        "joint_position_min_rad": values.min(axis=0).tolist(),
        "joint_position_max_rad": values.max(axis=0).tolist(),
    }


def score_episode(time_s: np.ndarray, endpoint_xz: np.ndarray, endpoint_velocity_xz: np.ndarray,
                  task: Task, *, hard_violation: np.ndarray | None = None,
                  max_sample_interval_s: float | None = None) -> EpisodeScore:
    """Score a complete physics-rate truth trace, retaining pass-through failures.

    ``max_sample_interval_s`` makes the sampling contract explicit for callers
    that have a physics clock.  The deadline and both ends of the hold interval
    must be represented; a short trace can never prove a successful hold.
    """

    times = np.asarray(time_s, dtype=np.float64)
    positions = np.asarray(endpoint_xz, dtype=np.float64)
    velocities = np.asarray(endpoint_velocity_xz, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (len(times), 2) or velocities.shape != positions.shape:
        raise ValueError("time and endpoint arrays must have matching [N, 2] shapes")
    if len(times) == 0 or not np.isfinite(times).all() or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("time, endpoint, and velocity arrays must be nonempty and finite")
    if len(times) > 1 and not (np.diff(times) > 0.0).all():
        raise ValueError("time samples must be strictly increasing")
    if max_sample_interval_s is not None and (not np.isfinite(max_sample_interval_s) or max_sample_interval_s <= 0.0):
        raise ValueError("max_sample_interval_s must be finite and positive when supplied")
    errors = np.linalg.norm(positions - task.goal_xz, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    violations = np.zeros(len(times), dtype=bool) if hard_violation is None else np.asarray(hard_violation, dtype=bool)
    if violations.shape != (len(times),):
        raise ValueError("hard_violation must have one value per physical sample")
    if violations.any():
        return EpisodeScore(False, None, None, None, None, None, "hard_operating_violation", True)

    tolerance = max(1e-12, (max_sample_interval_s or 1.0) * 1e-6)

    def complete_interval(start_s: float, end_s: float) -> tuple[bool, np.ndarray]:
        indices = np.flatnonzero((times >= start_s - tolerance) & (times <= end_s + tolerance))
        if len(indices) == 0:
            return False, indices
        selected = times[indices]
        if abs(selected[0] - start_s) > tolerance or abs(selected[-1] - end_s) > tolerance:
            return False, indices
        if (max_sample_interval_s is not None and len(selected) > 1
                and (np.diff(selected) > max_sample_interval_s + tolerance).any()):
            return False, indices
        return True, indices

    complete_hold, hold_indices = complete_interval(task.deadline_s, task.total_duration_s)
    if not complete_hold:
        return EpisodeScore(False, None, None, None, None, None, "insufficient_physical_trace", False)
    deadline_index = int(hold_indices[0])
    hold_errors = errors[hold_indices]
    hold_speeds = speeds[hold_indices]
    deadline_error = float(errors[deadline_index])
    deadline_speed = float(speeds[deadline_index])
    max_error = float(hold_errors.max())
    max_speed = float(hold_speeds.max())
    success = bool(max_error <= task.position_tolerance_m and max_speed <= task.speed_tolerance_m_s)
    settling = None
    valid = (errors <= task.position_tolerance_m) & (speeds <= task.speed_tolerance_m_s)
    for start_s in times:
        if start_s > task.deadline_s + tolerance:
            break
        complete, indices = complete_interval(float(start_s), float(start_s + task.hold_duration_s))
        if complete and valid[indices].all():
            settling = float(start_s)
            break
    reason = None if success else "deadline_or_hold_tolerance"
    return EpisodeScore(success, deadline_error, deadline_speed, max_error, max_speed, settling, reason, False)
