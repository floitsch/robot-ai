"""Structural/base load utilization for the reduced P8 model."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import OperatingEnvelope


@dataclass(frozen=True)
class LoadMetrics:
    base_force_xz: np.ndarray
    support_moment: float
    base_deflection_xz: np.ndarray
    base_pitch: float
    link_flex_angles: np.ndarray

    def utilization(self, envelope: OperatingEnvelope) -> float:
        values = np.concatenate((np.abs(self.base_force_xz) / envelope.max_base_force_xz,
                                 [abs(self.support_moment) / envelope.max_support_moment],
                                 np.abs(self.base_deflection_xz) / envelope.max_base_deflection_xz,
                                 [abs(self.base_pitch) / envelope.max_base_pitch],
                                 np.abs(self.link_flex_angles) / envelope.max_link_flex_angles))
        return float(values.max())

