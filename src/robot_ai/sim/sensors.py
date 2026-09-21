"""Sensor acquisition, delivery delay, corruption, and availability metadata."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Observation, PrivilegedRecord


@dataclass(frozen=True)
class SensorSettings:
    periods_s: np.ndarray
    noise_std: np.ndarray
    bias_limit: np.ndarray
    max_delay_s: np.ndarray

    def __post_init__(self) -> None:
        for name in ("periods_s", "noise_std", "bias_limit", "max_delay_s"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (8,) or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must be a finite nonnegative vector of length eight")
            object.__setattr__(self, name, value)
        if (self.periods_s <= 0).any():
            raise ValueError("sensor periods must be positive")

    @classmethod
    def healthy(cls) -> SensorSettings:
        return cls(np.array([.01, .01, .01, .01, .01, .01, .04, .04]),
                   np.array([.001, .001, .002, .002, .03, .03, .002, .002]),
                   np.array([.02, .02, .04, .04, .05, .05, .005, .005]),
                   np.array([.03, .03, .03, .03, .03, .03, .10, .10]))


@dataclass(frozen=True)
class SensorFault:
    kind: str
    channels: tuple[int, ...]
    start_s: float
    end_s: float | None = None
    offset: float = 0.0

    def active(self, time_s: float) -> bool:
        return time_s >= self.start_s and (self.end_s is None or time_s < self.end_s)


@dataclass(frozen=True)
class _Packet:
    delivery_s: float
    acquisition_s: float
    channel: int
    value: float


class SensorSuite:
    """Packetized eight-channel sensor suite with causal velocity estimates."""

    def __init__(self, settings: SensorSettings | None = None, *, seed: int = 0,
                 faults: tuple[SensorFault, ...] = ()) -> None:
        self.settings = settings or SensorSettings.healthy()
        self.seed = seed
        self.faults = faults
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._next_acquisition = np.zeros(8, dtype=np.float64)
        self._bias = self.rng.uniform(-self.settings.bias_limit, self.settings.bias_limit)
        self._queue: list[_Packet] = []
        self._values = np.zeros(8, dtype=np.float64)
        self._available = np.zeros(8, dtype=bool)
        self._fresh = np.zeros(8, dtype=bool)
        self._last_acquisition = np.zeros(8, dtype=np.float64)
        self._previous_angle = np.zeros(2, dtype=np.float64)
        self._previous_angle_time: float | None = None
        self._velocity_estimate = np.zeros(2, dtype=np.float64)

    def _fault(self, kind: str, channel: int, time_s: float) -> bool:
        return any(item.kind == kind and channel in item.channels and item.active(time_s) for item in self.faults)

    def _sample(self, truth: PrivilegedRecord, motor_torque: np.ndarray, time_s: float) -> np.ndarray:
        values = np.concatenate((truth.q, self._velocity_estimate, np.asarray(motor_torque), truth.endpoint_xz))
        noisy = values + self._bias + self.rng.normal(0.0, self.settings.noise_std)
        if self._fault("outlier", 0, time_s):
            noisy[0] += 10 * self.settings.noise_std[0]
        for fault in self.faults:
            if fault.kind == "bias" and fault.active(time_s):
                noisy[list(fault.channels)] += fault.offset
        return noisy

    def advance(self, time_s: float, truth: PrivilegedRecord, motor_torque: np.ndarray) -> None:
        """Acquire due samples and expose only packets delivered by this time."""

        while True:
            due = np.flatnonzero(time_s + 1e-12 >= self._next_acquisition)
            if len(due) == 0:
                break
            acquisition = float(self._next_acquisition[due[0]])
            channels = np.flatnonzero(np.isclose(self._next_acquisition, acquisition))
            sample = self._sample(truth, motor_torque, acquisition)
            if self._previous_angle_time is not None and acquisition > self._previous_angle_time:
                raw_velocity = (sample[:2] - self._previous_angle) / (acquisition - self._previous_angle_time)
                self._velocity_estimate = 0.5 * self._velocity_estimate + 0.5 * raw_velocity
                sample[2:4] = self._velocity_estimate
            self._previous_angle = sample[:2].copy()
            self._previous_angle_time = acquisition
            for raw_channel in channels:
                channel = int(raw_channel)
                self._next_acquisition[channel] += self.settings.periods_s[channel]
                if self._fault("dropout", channel, acquisition):
                    continue
                value = self._values[channel] if self._fault("freeze", channel, acquisition) and self._available[channel] else sample[channel]
                delay = self.rng.uniform(0.0, self.settings.max_delay_s[channel])
                self._queue.append(_Packet(acquisition + delay, acquisition, channel, float(value)))
        self._fresh[:] = False
        self._queue.sort(key=lambda item: (item.delivery_s, item.acquisition_s, item.channel))
        delivered: list[_Packet] = []
        for packet in self._queue:
            if packet.delivery_s <= time_s + 1e-12:
                delivered.append(packet)
                if packet.acquisition_s >= self._last_acquisition[packet.channel]:
                    index = packet.channel
                    self._values[index] = packet.value
                    self._available[index] = True
                    self._fresh[index] = True
                    self._last_acquisition[index] = packet.acquisition_s
        self._queue = [packet for packet in self._queue if packet not in delivered]

    def observe(self, time_s: float) -> Observation:
        age = np.where(self._available, np.maximum(0.0, time_s - self._last_acquisition), 0.0)
        return Observation(self._values.copy(), self._available.copy(), self._fresh.copy(), age, time_s)
