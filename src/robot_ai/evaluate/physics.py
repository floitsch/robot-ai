"""Deterministic native/Warp physics comparison fixtures for P1."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RuntimeConfig, confined_path
from ..sim.backlash import BacklashSettings, BacklashTransmission
from ..sim.mechanics import gravity_torque, mechanical_energy
from ..sim.native import NativeArm, default_descriptor
from ..sim.warp_backend import WarpArmBatch


def _low_force_commands(count: int) -> np.ndarray:
    steps = np.arange(count, dtype=np.float64)
    return np.column_stack((0.15 * np.sin(steps / 20.0), -0.10 * np.cos(steps / 27.0)))


def _backend_parity(device: str) -> dict[str, Any]:
    descriptor = default_descriptor()
    initial = np.array([0.0, 0.0])
    commands = _low_force_commands(200)
    native = NativeArm(descriptor, timestep=0.001)
    native.reset(initial)
    warp = WarpArmBatch(descriptor, 1, device=device, timestep=0.001)
    warp.reset(q=initial.reshape(1, 2).astype(np.float32))
    errors: list[float] = []
    for command in commands:
        native_record = native.step(command)
        warp.step(command.reshape(1, 2).astype(np.float32))
        warp_record = warp.truth()
        errors.append(float(np.linalg.norm(native_record.endpoint_xz - warp_record.endpoint_xz)))
    return {
        "steps": len(commands), "max_endpoint_error_m": max(errors),
        "final_endpoint_error_m": errors[-1], "mean_endpoint_error_m": float(np.mean(errors)),
        "tolerance_m": 0.001, "passed": max(errors) <= 0.001,
    }


def _high_motion_parity(device: str) -> dict[str, Any]:
    """Check parity away from the low-force hanging fixture."""

    descriptor = default_descriptor()
    rng = np.random.default_rng(20260912)
    initial = np.array([0.65, 1.10])
    commands = np.clip(rng.normal(0.0, 0.28, size=(400, 2)), -0.65, 0.65)
    native = NativeArm(descriptor, timestep=0.001)
    native.reset(initial)
    warp = WarpArmBatch(descriptor, 1, device=device, timestep=0.001)
    warp.reset(q=initial.reshape(1, 2).astype(np.float32))
    errors: list[float] = []
    for command in commands:
        native_record = native.step(command)
        warp.step(command.reshape(1, 2).astype(np.float32))
        errors.append(float(np.linalg.norm(native_record.endpoint_xz - warp.truth().endpoint_xz)))
    return {"steps": len(commands), "max_endpoint_error_m": max(errors), "tolerance_m": 0.001,
            "passed": max(errors) <= 0.001}


def _period_fixture(device: str) -> dict[str, Any]:
    """Check a repeated multi-period command sequence on both backends."""

    steps = np.arange(400, dtype=np.float64)
    commands = np.column_stack((0.35 * np.sin(2.0 * np.pi * steps / 50.0),
                                0.22 * np.sin(2.0 * np.pi * steps / 50.0 + 0.7)))
    descriptor = default_descriptor()
    initial = np.array([-0.45, 0.85])
    native = NativeArm(descriptor, timestep=0.001)
    native.reset(initial)
    warp = WarpArmBatch(descriptor, 1, device=device, timestep=0.001)
    warp.reset(q=initial.reshape(1, 2).astype(np.float32))
    errors: list[float] = []
    for command in commands:
        native_record = native.step(command)
        warp.step(command.reshape(1, 2).astype(np.float32))
        errors.append(float(np.linalg.norm(native_record.endpoint_xz - warp.truth().endpoint_xz)))
    return {"period_s": 0.05, "period_count": 8, "max_endpoint_error_m": max(errors),
            "tolerance_m": 0.001, "passed": max(errors) <= 0.001}


def _timestep_convergence() -> dict[str, Any]:
    descriptor = default_descriptor()
    initial = np.array([0.1, 0.0])
    coarse = NativeArm(descriptor, timestep=0.001)
    coarse.reset(initial)
    coarse_endpoint: list[np.ndarray] = []
    for _ in range(2000):
        coarse_endpoint.append(coarse.step(np.zeros(2)).endpoint_xz.copy())
    fine = NativeArm(descriptor, timestep=0.0005)
    fine.reset(initial)
    fine_endpoint: list[np.ndarray] = []
    for index in range(4000):
        record = fine.step(np.zeros(2))
        if index % 2 == 1:
            fine_endpoint.append(record.endpoint_xz.copy())
    errors = np.linalg.norm(np.asarray(coarse_endpoint) - np.asarray(fine_endpoint), axis=1)
    return {
        "duration_s": 2.0, "coarse_dt_s": 0.001, "fine_dt_s": 0.0005,
        "max_endpoint_error_m": float(errors.max()), "final_endpoint_error_m": float(errors[-1]),
        "mean_endpoint_error_m": float(errors.mean()), "tolerance_m": 0.001,
        "passed": bool(errors.max() <= 0.001),
    }


def _energy_fixture() -> dict[str, Any]:
    descriptor = default_descriptor()
    arm = NativeArm(descriptor, timestep=0.001)
    arm.reset(np.array([0.1, 0.0]))
    energies = [float(mechanical_energy(arm.data.qpos, arm.data.qvel, descriptor))]
    for _ in range(2000):
        arm.step(np.zeros(2))
        energies.append(float(mechanical_energy(arm.data.qpos, arm.data.qvel, descriptor)))
    return {
        "duration_s": 2.0, "initial_energy_j": energies[0], "final_energy_j": energies[-1],
        "maximum_energy_j": max(energies), "minimum_energy_j": min(energies),
        "damping_nonincreasing": bool(max(energies) <= energies[0] + 1e-9),
    }


def _backlash_fixture(dt: float) -> dict[str, Any]:
    settings = BacklashSettings(half_gap_rad=0.006, stiffness_nm_per_rad=12.0,
                                damping_nms_per_rad=0.18, motor_inertia_kg_m2=0.002)
    transmission = BacklashTransmission(settings)
    transmission.reset(0.0)
    for _ in range(round(0.02 / dt)):
        transmission.step(0.01, 0.0, 0.0, dt)
    forward, forward_torque = transmission.motor_q, transmission.coupling_torque(0.0, 0.0)
    for _ in range(round(0.02 / dt)):
        transmission.step(-0.01, 0.0, 0.0, dt)
    reversal_torque = transmission.coupling_torque(0.0, 0.0)
    transmission.motor_q, transmission.motor_dq = 0.03, 0.0
    energies = [transmission.stored_energy(0.0)]
    for _ in range(round(0.4 / dt)):
        transmission.step(0.0, 0.0, 0.0, dt)
        energies.append(transmission.stored_energy(0.0))
    return {"dt_s": dt, "initial_energy_j": energies[0], "final_energy_j": energies[-1],
            "maximum_energy_j": max(energies), "reversal_lost_motion": bool(abs(forward) < settings.half_gap_rad and forward_torque == 0.0 and reversal_torque == 0.0),
            "unpowered_energy_nonincreasing": bool(max(energies) <= energies[0] + 1e-10),
            "unpowered_energy_dissipates": bool(energies[-1] < energies[0])}


def run_backlash_fixtures(output: str | Path) -> dict[str, Any]:
    """Record P7 transmission-only reversal, passivity, and dt evidence."""
    coarse, fine = _backlash_fixture(0.001), _backlash_fixture(0.0005)
    difference = abs(coarse["final_energy_j"] - fine["final_energy_j"])
    report = {"schema_version": 1, "purpose": "P7 physical backlash transmission fixtures", "coarse": coarse, "fine": fine,
              "final_energy_difference_j": difference, "final_energy_tolerance_j": 2e-4,
              "passed": bool(coarse["reversal_lost_motion"] and fine["reversal_lost_motion"] and coarse["unpowered_energy_nonincreasing"] and fine["unpowered_energy_nonincreasing"] and coarse["unpowered_energy_dissipates"] and fine["unpowered_energy_dissipates"] and difference <= 2e-4)}
    destination = confined_path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_physics_fixtures(output: str | Path, *, config: RuntimeConfig) -> dict[str, Any]:
    """Run fixed gravity, timestep, and native/Warp parity fixtures."""

    device = "cuda:0" if config.device == "cuda" else "cpu"
    descriptor = default_descriptor()
    report: dict[str, Any] = {
        "schema_version": 1,
        "device": device,
        "gravity_hanging": {
            "torque_norm": float(np.linalg.norm(gravity_torque(np.zeros(2), descriptor))),
            "passed": bool(np.allclose(gravity_torque(np.zeros(2), descriptor), 0.0, atol=1e-12)),
        },
        "backend_parity": _backend_parity(device),
        "high_motion_parity": _high_motion_parity(device),
        "period_fixture": _period_fixture(device),
        "timestep_convergence": _timestep_convergence(),
        "energy_fixture": _energy_fixture(),
    }
    report["passed"] = bool(
        report["gravity_hanging"]["passed"] and
        report["backend_parity"]["passed"] and
        report["high_motion_parity"]["passed"] and
        report["period_fixture"]["passed"] and
        report["timestep_convergence"]["passed"] and
        report["energy_fixture"]["damping_nonincreasing"]
    )
    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
