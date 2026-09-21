"""Reproducible scenario and fault schedule construction."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .actuators import ActuatorSettings
from .sensors import SensorFault, SensorSettings


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    actuator: ActuatorSettings
    sensors: SensorSettings
    faults: tuple[SensorFault, ...]


def _seed(global_seed: int, scenario_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{scenario_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def make_scenario(global_seed: int, scenario_id: str, *, degraded: bool = False) -> Scenario:
    rng = np.random.default_rng(_seed(global_seed, scenario_id))
    actuator = ActuatorSettings.healthy()
    sensors = SensorSettings.healthy()
    faults: tuple[SensorFault, ...] = ()
    if degraded:
        actuator = ActuatorSettings(
            effectiveness=rng.uniform(0.7, 1.0, 2), time_constant_s=rng.uniform(0.0, 0.04, 2),
            command_latency_s=rng.uniform(0.0, 0.03, 2), viscous_friction=rng.uniform(0.0, 0.1, 2),
            coulomb_friction=rng.uniform(0.0, 0.15, 2), local_resistance=rng.uniform(0.0, 0.2, 2),
            local_angle=rng.uniform(-1.0, 1.0, 2), local_width=rng.uniform(0.05, 0.2, 2),
        )
        faults = (SensorFault("dropout", (6,), 0.65, 0.95), SensorFault("freeze", (0,), 0.85, 1.15))
    return Scenario(scenario_id, _seed(global_seed, scenario_id), actuator, sensors, faults)

