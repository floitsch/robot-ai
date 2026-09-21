"""Self-contained observer/policy inference bundle."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..config import confined_path
from ..contracts import RobotDescriptor
from ..control.policy import POLICY_INPUT_DIM, PolicyNetwork
from ..control.runtime import Runtime
from .bundle import WorldBundle
from .world import WorldModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InferenceBundle:
    """Weights plus the exact schemas needed to reconstruct runtime behavior."""

    SCHEMA_VERSION = 2

    def __init__(self, world: WorldBundle, policy: PolicyNetwork, metadata: dict[str, Any]) -> None:
        self.world = world
        self.policy = policy
        self.metadata = metadata

    def save(self, output: str | Path) -> Path:
        destination = confined_path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.detach().cpu() for key, value in self.world.model.state_dict().items()}, destination / "world-weights.pt")
        torch.save({key: value.detach().cpu() for key, value in self.policy.state_dict().items()}, destination / "policy-weights.pt")
        (destination / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> InferenceBundle:
        source = confined_path(path, must_exist=True)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported inference bundle schema")
        model_metadata = metadata["world"]["model"]
        model = WorldModel(belief_dim=model_metadata["belief_dim"], observation_dim=model_metadata["observation_dim"],
                           descriptor_dim=model_metadata["descriptor_dim"],
                           endpoint_residual=model_metadata.get("endpoint_residual", False),
                           physics_prior=model_metadata.get("physics_prior", False),
                           fresh_measurement_correction=model_metadata.get(
                               "fresh_measurement_correction", False
                           ))
        world_metadata = metadata["world"]
        normalization = world_metadata.get("normalization")
        if normalization is not None:
            model.set_observation_normalization(
                torch.as_tensor(normalization["mean"], dtype=torch.float32),
                torch.as_tensor(normalization["std"], dtype=torch.float32),
            )
        # Keep serialized tensors on the host until their destination modules are
        # complete.  Mapping them directly to CUDA temporarily retains both the
        # checkpoint tensors and the destination parameters on a constrained
        # deployment GPU.
        model.load_state_dict(torch.load(source / "world-weights.pt", map_location="cpu", weights_only=True))
        world = WorldBundle(model.to(device).eval(), world_metadata)
        policy = PolicyNetwork(input_dim=metadata["policy"]["input_dim"])
        policy.load_state_dict(torch.load(source / "policy-weights.pt", map_location="cpu", weights_only=True))
        policy.to(device).eval()
        return cls(world, policy, metadata)

    def runtime(self, descriptor: RobotDescriptor, *, device: str | torch.device = "cpu") -> Runtime:
        return Runtime(self.world, descriptor, self.policy, torch.device(device))


def export_inference_bundle(run: str | Path, output: str | Path, *, world_bundle: str | Path | None = None,
                            config_identity: dict[str, str] | None = None) -> Path:
    """Package a policy run and its frozen world model into portable local files."""

    run_path = confined_path(run, must_exist=True)
    metrics_path = run_path / "metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if world_bundle is None:
        world_bundle = report["world_bundle"]
    world_path = confined_path(world_bundle, must_exist=True)
    world = WorldBundle.load(world_path, device="cpu")
    selected = report.get("selected_checkpoint")
    if not isinstance(selected, str) or not selected:
        raise ValueError("policy export requires a checkpoint selected by select-policy")
    policy_path = (run_path / "checkpoints" / "initial-policy.pt" if selected == "initial"
                   else run_path / "checkpoints" / f"{selected}.pt")
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)
    input_dim = int(report.get("policy_input_dim", POLICY_INPUT_DIM))
    if input_dim != POLICY_INPUT_DIM:
        raise ValueError(f"cannot export unsupported policy feature schema input dimension {input_dim}")
    policy = PolicyNetwork(input_dim=input_dim)
    policy.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
    metadata = {
        "schema_version": InferenceBundle.SCHEMA_VERSION,
        "world": world.metadata,
        "policy": {"input_dim": input_dim, "feature_schema": "belief-decoded-state-goal-q-v3",
                   "action_shape": [2], "action_range": [-1.0, 1.0], "deterministic_inference": "tanh(mean)"},
        "observation_schema": 1, "descriptor_schema": 1, "task_frame": "world",
        "control_dt_s": 0.01, "physics_dt_s": 0.001, "command_semantics": "dimensionless nominal torque request",
        "physical_envelope": {
            "modeled_hard_limits": ["joint_position"],
            "unmodeled_limits": ["base_force", "support_moment", "base_deflection", "base_pitch", "link_flex"],
            "note": "The accepted rigid packet bundle does not claim synthetic CombinedSettings support/flex ratings.",
        },
        "provenance": {
            "policy_run": str(run_path),
            "selected_checkpoint": selected,
            "selected_checkpoint_sha256": _sha256(policy_path),
            "training_metrics_sha256": _sha256(metrics_path),
            "world_bundle": str(world_path),
            "world_weights_sha256": _sha256(world_path / "weights.pt"),
            "world_metadata_sha256": _sha256(world_path / "metadata.json"),
            "config_identity": config_identity or report.get("config_identity"),
            "code_sha256": {
                "inference.py": _sha256(Path(__file__)),
                "policy.py": _sha256(Path(__file__).parents[1] / "control" / "policy.py"),
                "runtime.py": _sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
            },
        },
    }
    return InferenceBundle(world, policy, metadata).save(output)
