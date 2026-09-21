"""Authoritative primary, load, and smoothness metric implementations."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def command_slew(actions: np.ndarray, dt: float) -> float:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or dt <= 0:
        raise ValueError("actions must have shape [time, 2] and positive dt")
    if len(values) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(values, axis=0) / dt, axis=1) ** 2) * dt)


def trajectory_jerk(velocity: np.ndarray, dt: float) -> float:
    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or dt <= 0:
        raise ValueError("velocity must have shape [time, 2] and positive dt")
    if len(values) < 3:
        return 0.0
    jerk = np.diff(values, n=2, axis=0) / dt**2
    return float(np.sum(np.linalg.norm(jerk, axis=1) ** 2) * dt)


@dataclass(frozen=True)
class SmoothnessComparison:
    baseline_success: float
    candidate_success: float
    baseline_slew_median: float
    candidate_slew_median: float
    success_regression_points: float
    slew_reduction_fraction: float
    accepted: bool


def compare_smoothness(baseline_success: float, candidate_success: float, baseline_slew: np.ndarray,
                       candidate_slew: np.ndarray, *, max_success_regression_points: float = 0.01,
                       minimum_primary_success: float = 0.95) -> SmoothnessComparison:
    baseline_median = float(np.median(baseline_slew))
    candidate_median = float(np.median(candidate_slew))
    reduction = 0.0 if baseline_median <= 0 else 1.0 - candidate_median / baseline_median
    regression = baseline_success - candidate_success
    accepted = bool(baseline_success >= minimum_primary_success and candidate_success >= minimum_primary_success
                    and regression <= max_success_regression_points and reduction >= 0.20)
    return SmoothnessComparison(baseline_success, candidate_success, baseline_median, candidate_median,
                                regression, reduction, accepted)


def select_primary_then_smoothness(reports: list[dict[str, float]], *, success_floor: float) -> dict[str, float]:
    """Select a checkpoint by primary success/error before secondary smoothness."""

    feasible = [report for report in reports if report["success_fraction"] >= success_floor]
    if not feasible:
        raise ValueError("no checkpoint meets the primary success floor")
    return min(feasible, key=lambda report: (report.get("p95_deadline_error_m", float("inf")),
                                           report.get("command_slew_median", float("inf"))))
