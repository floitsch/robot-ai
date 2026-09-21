"""Typed project configuration and project-local path validation."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when configuration violates the project contract."""


def project_root() -> Path:
    """Return the root selected by the wrapper or the source checkout."""

    configured = os.environ.get("ROBOT_AI_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2]


def confined_path(path: str | Path, *, root: Path | None = None, must_exist: bool = False) -> Path:
    """Resolve *path* and reject paths outside the project root."""

    root = (root or project_root()).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError(f"path is outside project root: {path}") from exc
    if must_exist and not resolved.exists():
        raise ConfigurationError(f"path does not exist: {resolved}")
    return resolved


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level settings shared by future simulation and training commands."""

    device: str = "cuda"
    backend: str = "warp"
    physics_dt: float = 0.001
    control_dt: float = 0.010
    seed: int = 0
    world_count: int = 1
    scenario: str = "healthy"
    output: Path = field(default_factory=lambda: project_root() / "artifacts")

    def validate(self) -> None:
        if self.device not in {"cuda", "cpu"}:
            raise ConfigurationError(f"unsupported device: {self.device}")
        if self.backend not in {"warp", "native"}:
            raise ConfigurationError(f"unsupported backend: {self.backend}")
        if self.physics_dt <= 0 or self.control_dt <= 0:
            raise ConfigurationError("timesteps must be positive")
        if self.control_dt < self.physics_dt:
            raise ConfigurationError("control_dt must be at least physics_dt")
        if self.world_count < 1:
            raise ConfigurationError("world_count must be positive")
        if self.scenario not in {"healthy", "degraded", "combined"}:
            raise ConfigurationError(f"unsupported scenario: {self.scenario}")
        confined_path(self.output)


def _reject_unknown(values: dict[str, Any], allowed: set[str], source: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {source} field(s): {', '.join(unknown)}")


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load a strict TOML runtime configuration, if supplied."""

    if path is None:
        config = RuntimeConfig()
        config.validate()
        return config
    config_path = confined_path(path, must_exist=True)
    with config_path.open("rb") as handle:
        values = tomllib.load(handle)
    _reject_unknown(values, {"runtime"}, "top-level")
    runtime = values.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ConfigurationError("[runtime] must be a TOML table")
    allowed = {"device", "backend", "physics_dt", "control_dt", "seed", "world_count", "scenario", "output"}
    _reject_unknown(runtime, allowed, "runtime")
    kwargs = dict(runtime)
    if "output" in kwargs:
        kwargs["output"] = confined_path(kwargs["output"])
    config = RuntimeConfig(**kwargs)
    config.validate()
    return config
