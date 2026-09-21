"""Analytic mechanics shared by the native and Warp reference backends."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import numpy as np

from ..contracts import RobotDescriptor

GRAVITY = 9.81


def forward_kinematics(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return endpoint x/z for scalar or batched two-joint angles."""

    q = np.asarray(q, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    q1 = q[..., 0]
    q12 = q1 + q[..., 1]
    return np.stack((lengths[0] * np.sin(q1) + lengths[1] * np.sin(q12),
                     -lengths[0] * np.cos(q1) - lengths[1] * np.cos(q12)), axis=-1)


def jacobian(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return the x/z endpoint Jacobian with shape ``(..., 2, 2)``."""

    q = np.asarray(q, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    q1 = q[..., 0]
    q12 = q1 + q[..., 1]
    result = np.empty(q.shape[:-1] + (2, 2), dtype=np.float64)
    result[..., 0, 0] = lengths[0] * np.cos(q1) + lengths[1] * np.cos(q12)
    result[..., 0, 1] = lengths[1] * np.cos(q12)
    result[..., 1, 0] = lengths[0] * np.sin(q1) + lengths[1] * np.sin(q12)
    result[..., 1, 1] = lengths[1] * np.sin(q12)
    return result


def endpoint_velocity(q: np.ndarray, dq: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", jacobian(q, lengths), dq)


def gravity_torque(q: np.ndarray, descriptor: RobotDescriptor) -> np.ndarray:
    """Return generalized gravity torque in the command equation ``M*qdd=tau+gravity``."""

    q = np.asarray(q, dtype=np.float64)
    q1 = q[..., 0]
    q12 = q1 + q[..., 1]
    lengths = descriptor.link_lengths
    masses = descriptor.nominal_masses
    terms = np.stack(
        (
            masses[0] * lengths[0] * 0.5 + masses[1] * lengths[0],
            masses[1] * lengths[1] * 0.5,
        )
    )
    return -GRAVITY * np.stack(
        (terms[0] * np.sin(q1) + terms[1] * np.sin(q12), terms[1] * np.sin(q12)), axis=-1
    )


def potential_energy(q: np.ndarray, descriptor: RobotDescriptor) -> np.ndarray:
    """Return gravitational potential energy with the base height as zero."""

    q = np.asarray(q, dtype=np.float64)
    q1 = q[..., 0]
    q12 = q1 + q[..., 1]
    lengths = descriptor.link_lengths
    masses = descriptor.nominal_masses
    heights = np.stack((-lengths[0] * 0.5 * np.cos(q1),
                        -lengths[0] * np.cos(q1) - lengths[1] * 0.5 * np.cos(q12)), axis=-1)
    return GRAVITY * np.sum(heights * masses, axis=-1)


def mass_matrix(q: np.ndarray, descriptor: RobotDescriptor) -> np.ndarray:
    """Return the standard planar two-link mass matrix."""

    q = np.asarray(q, dtype=np.float64)
    result = np.empty(q.shape[:-1] + (2, 2), dtype=np.float64)
    m1, m2 = descriptor.nominal_masses
    l1, l2 = descriptor.link_lengths
    i1, i2 = descriptor.nominal_inertias
    lc1, lc2 = l1 * 0.5, l2 * 0.5
    c2 = np.cos(q[..., 1])
    result[..., 0, 0] = i1 + i2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
    result[..., 0, 1] = i2 + m2 * (lc2**2 + l1 * lc2 * c2)
    result[..., 1, 0] = result[..., 0, 1]
    result[..., 1, 1] = i2 + m2 * lc2**2
    return result


def coriolis_torque(q: np.ndarray, dq: np.ndarray, descriptor: RobotDescriptor) -> np.ndarray:
    """Return the velocity-coupling term subtracted by the rigid-arm dynamics."""

    q = np.asarray(q, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64)
    if q.shape[-1:] != (2,) or dq.shape != q.shape:
        raise ValueError("q and dq must have matching final dimension 2")
    coupling = -descriptor.nominal_masses[1] * descriptor.link_lengths[0]
    coupling *= descriptor.link_lengths[1] * 0.5 * np.sin(q[..., 1])
    result = np.empty_like(q)
    result[..., 0] = coupling * (2 * dq[..., 0] * dq[..., 1] + dq[..., 1] ** 2)
    result[..., 1] = -coupling * dq[..., 0] ** 2
    return result


def kinetic_energy(q: np.ndarray, dq: np.ndarray, descriptor: RobotDescriptor) -> np.ndarray:
    """Return the kinetic energy from the analytic mass matrix."""

    velocities = np.asarray(dq, dtype=np.float64)
    matrix = mass_matrix(q, descriptor)
    return 0.5 * np.einsum("...i,...ij,...j->...", velocities, matrix, velocities)


def mechanical_energy(q: np.ndarray, dq: np.ndarray, descriptor: RobotDescriptor) -> np.ndarray:
    """Return gravitational plus kinetic energy for fixture checks."""

    return potential_energy(q, descriptor) + kinetic_energy(q, dq, descriptor)


def acceleration(q: np.ndarray, dq: np.ndarray, torque: np.ndarray, descriptor: RobotDescriptor,
                damping: float = 0.02) -> np.ndarray:
    rhs = np.asarray(torque, dtype=np.float64) + gravity_torque(q, descriptor) - damping * np.asarray(dq)
    return np.linalg.solve(mass_matrix(q, descriptor), rhs[..., None])[..., 0]
