"""Pure setup and confinement tests for P0."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from pathlib import Path

import pytest

from robot_ai.config import ConfigurationError, RuntimeConfig, confined_path, project_root


def test_default_runtime_config_is_valid() -> None:
    config = RuntimeConfig()
    config.validate()
    assert config.physics_dt < config.control_dt


def test_paths_are_confined_to_project() -> None:
    root = project_root()
    assert confined_path("artifacts/setup/report.json", root=root).is_relative_to(root)
    with pytest.raises(ConfigurationError):
        confined_path(root.parent / "outside.json", root=root)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = project_root()
    link = tmp_path / "outside-link"
    link.symlink_to(root.parent, target_is_directory=True)
    with pytest.raises(ConfigurationError):
        confined_path(link / "escaped.json", root=root)

