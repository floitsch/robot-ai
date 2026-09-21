"""P6 policy distribution and frozen-runtime contract tests."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import json

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from robot_ai.contracts import Command, Observation, Task
from robot_ai.control.features import CURRENT_PACKET_POLICY_INPUT_DIM, CurrentPacketFeatureAdapter
from robot_ai.control.policy import (
    LEARNED_STATE_DIM,
    POLICY_INPUT_DIM,
    CriticNetwork,
    PolicyNetwork,
)
from robot_ai.control.runtime import CurrentPacketRuntime, Runtime
from robot_ai.evaluate.policy import compare_backend_replays
from robot_ai.models.bundle import WorldBundle
from robot_ai.models.inference import InferenceBundle, export_inference_bundle
from robot_ai.models.world import WorldModel
from robot_ai.sim.actuators import ActuatorSettings
from robot_ai.sim.mechanics import forward_kinematics
from robot_ai.sim.native import ImperfectNativeArm, default_descriptor
from robot_ai.sim.sensors import SensorSettings, SensorSuite
from robot_ai.train.policy import (
    _initialize_policy,
    _new_rollout_state,
    _warp_observation_features_torch,
    audit_packet_clone_runtime_replay,
    train_packet_dagger,
)


class _ConstantStateCorrection(nn.Module):
    def __init__(self, value: list[float]) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(value))

    def forward(self, encoded_and_hidden: Tensor) -> Tensor:
        return self.value.expand(*encoded_and_hidden.shape[:-1], -1)


def _fresh_joint_packet(value: float) -> Tensor:
    observation = torch.zeros(32)
    observation[0] = value
    observation[8:12] = 1.0
    observation[16:20] = 1.0
    return observation


def test_transformed_action_log_prob_is_finite_at_bounds() -> None:
    policy = PolicyNetwork()
    features = torch.zeros((4, POLICY_INPUT_DIM))
    distribution = policy.distribution(features)
    actions = torch.tensor([[1.0, -1.0], [0.0, 0.0], [0.999, -0.999], [-0.5, 0.5]])
    log_probability = distribution.log_prob(actions).sum(-1)
    assert torch.isfinite(log_probability).all()
    sampled, sampled_log_probability = policy.sample(features)
    assert sampled.shape == (4, 2)
    assert sampled_log_probability.shape == (4,)
    assert torch.isfinite(sampled_log_probability).all()


def test_policy_and_critic_have_matching_public_feature_shape() -> None:
    features = torch.zeros((3, POLICY_INPUT_DIM))
    policy = PolicyNetwork()
    critic = CriticNetwork()
    assert policy.deterministic(features).shape == (3, 2)
    assert critic(features).shape == (3,)


def test_current_packet_runtime_uses_every_public_packet_field_without_belief(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = default_descriptor()
    metadata = {"schema_version": 4, "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                "endpoint_residual": False, "physics_prior": False, "fresh_measurement_correction": True},
                "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8}}
    bundle = WorldBundle(WorldModel().eval(), metadata)
    observation = Observation(np.arange(8.0), np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), 0.0)
    task = Task(np.array([0.1, -0.5]), deadline_s=1.5)
    adapter = CurrentPacketFeatureAdapter(bundle, descriptor, torch.device("cpu"))
    baseline = adapter.policy_features(observation, task, 0.0, Command.accepted([0.0, 0.0]))
    assert baseline.shape == (CURRENT_PACKET_POLICY_INPUT_DIM,)
    for index in range(8):
        values = observation.values.copy(); values[index] += 0.25
        values_features = adapter.policy_features(
            Observation(values, observation.available, observation.fresh, observation.age_s, 0.0),
            task, 0.0, Command.accepted([0.0, 0.0]))
        assert torch.nonzero(values_features != baseline).flatten().tolist() == [index]
        available = observation.available.copy(); available[index] = False
        available_features = adapter.policy_features(
            Observation(observation.values, available, observation.fresh, observation.age_s, 0.0),
            task, 0.0, Command.accepted([0.0, 0.0]))
        assert torch.nonzero(available_features != baseline).flatten().tolist() == [8 + index]
        fresh = observation.fresh.copy(); fresh[index] = False
        fresh_features = adapter.policy_features(
            Observation(observation.values, observation.available, fresh, observation.age_s, 0.0),
            task, 0.0, Command.accepted([0.0, 0.0]))
        assert torch.nonzero(fresh_features != baseline).flatten().tolist() == [16 + index]
        ages = observation.age_s.copy(); ages[index] = 0.01
        age_features = adapter.policy_features(
            Observation(observation.values, observation.available, observation.fresh, ages, 0.0),
            task, 0.0, Command.accepted([0.0, 0.0]))
        assert torch.nonzero(age_features != baseline).flatten().tolist() == [24 + index]
    assert torch.nonzero(adapter.policy_features(observation, task, 0.01, Command.accepted([0.0, 0.0])) != baseline).flatten().tolist() == [48]
    for index in range(2):
        command = np.zeros(2); command[index] = 0.25
        assert torch.nonzero(adapter.policy_features(observation, task, 0.0, Command.accepted(command)) != baseline).flatten().tolist() == [49 + index]

    def forbidden_observer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("current-packet runtime must not call the observer")

    monkeypatch.setattr(bundle.model, "initial", forbidden_observer)
    monkeypatch.setattr(bundle.model, "predict_prior", forbidden_observer)
    monkeypatch.setattr(bundle.model, "observe", forbidden_observer)
    torch.manual_seed(20260913)
    policy = PolicyNetwork(input_dim=CURRENT_PACKET_POLICY_INPUT_DIM, hidden_dim=164)
    first = CurrentPacketRuntime(bundle, descriptor, policy, torch.device("cpu"))
    second = CurrentPacketRuntime(bundle, descriptor, policy, torch.device("cpu"))
    first.reset(descriptor, observation, task); second.reset(descriptor, observation, task)
    first.observe(Command.accepted([0.2, -0.1]), observation)
    second.observe(Command.accepted([-0.3, 0.4]), Observation(
        observation.values + 1.0, np.zeros(8, dtype=bool), np.zeros(8, dtype=bool), np.full(8, 0.01), 0.0
    ))
    common_previous = Command.accepted([0.1, -0.2])
    common_packet = Observation(observation.values - 0.5, observation.available, observation.fresh,
                                np.full(8, 0.02), 0.01)
    first.observe(common_previous, common_packet); second.observe(common_previous, common_packet)
    np.testing.assert_allclose(first.act().values, second.act().values, atol=0.0, rtol=0.0)


def test_world_model_physical_decoder_supports_endpoint_residuals() -> None:
    model = WorldModel(endpoint_residual=True)
    belief = torch.zeros((2, model.belief_dim))
    descriptor = torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32).expand(2, -1)
    q, dq, endpoint, velocity = model.decode_physical(belief, descriptor)
    assert q.shape == (2, 2)
    assert dq.shape == (2, 2)
    assert endpoint.shape == (2, 2)
    assert velocity.shape == (2, 2)
    assert torch.isfinite(torch.cat((q, dq, endpoint, velocity), dim=-1)).all()


def test_world_model_physics_prior_preserves_state_shapes() -> None:
    model = WorldModel(physics_prior=True)
    belief = model.initial(torch.zeros(32), torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32))
    descriptor = torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32)
    prior = model.predict_prior(belief, torch.zeros(2), descriptor, 0.01)
    q, dq = model.decode(prior)
    assert q.shape == (2,)
    assert dq.shape == (2,)
    assert torch.isfinite(torch.cat((q, dq))).all()


def test_schema4_observer_can_correct_a_fresh_biased_encoder() -> None:
    model = WorldModel(physics_prior=True, fresh_measurement_correction=True)
    model.state_observer = _ConstantStateCorrection([-0.2, 0.0, 0.0, 0.0])
    belief = model.initial(_fresh_joint_packet(0.2),
                           torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32))
    q, dq = model.decode(belief)
    torch.testing.assert_close(q, torch.zeros(2))
    torch.testing.assert_close(dq, torch.zeros(2))


def test_schema3_observer_preserves_legacy_fresh_packet_bypass() -> None:
    model = WorldModel(physics_prior=True, fresh_measurement_correction=False)
    model.state_observer = _ConstantStateCorrection([-0.2, 0.0, 0.0, 0.0])
    belief = model.initial(_fresh_joint_packet(0.2),
                           torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32))
    q, _ = model.decode(belief)
    torch.testing.assert_close(q, torch.tensor([0.2, 0.0]))


def test_world_bundle_versions_fresh_measurement_behavior(tmp_path) -> None:
    metadata = {
        "schema_version": 4,
        "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                  "endpoint_residual": False, "physics_prior": True,
                  "fresh_measurement_correction": True},
        "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8},
    }
    WorldBundle(WorldModel(physics_prior=True, fresh_measurement_correction=True), metadata).save(tmp_path / "v4")
    loaded_v4 = WorldBundle.load(tmp_path / "v4")
    assert loaded_v4.model.fresh_measurement_correction
    legacy_model = {key: value for key, value in metadata["model"].items()
                    if key != "fresh_measurement_correction"}
    legacy = {**metadata, "schema_version": 3, "model": legacy_model}
    WorldBundle(WorldModel(physics_prior=True), legacy).save(tmp_path / "v3")
    loaded_v3 = WorldBundle.load(tmp_path / "v3")
    assert not loaded_v3.model.fresh_measurement_correction


def test_collector_and_runtime_match_for_the_same_packet_and_action() -> None:
    descriptor = default_descriptor()
    metadata = {"schema_version": 4,
                "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                          "endpoint_residual": False, "physics_prior": True,
                          "fresh_measurement_correction": True},
                "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8}}
    model = WorldModel(fresh_measurement_correction=True).eval()
    bundle = WorldBundle(model, metadata)
    state = _new_rollout_state(model, worlds=1, device=torch.device("cpu"),
                               mean=torch.zeros(8), std=torch.ones(8), physics_dt=0.001)
    initial = state.observation[0].detach().numpy()
    runtime = Runtime(bundle, descriptor, PolicyNetwork(), torch.device("cpu"))
    task = Task(state.goal[0].detach().numpy(), deadline_s=1.5)
    runtime.reset(descriptor, Observation(initial[:8], initial[8:16].astype(bool), initial[16:24].astype(bool),
                                          initial[24:32] * 0.1, 0.0), task)
    action = torch.zeros((1, 2))
    for _ in range(10):
        state.arm.step_torch(action)
    q_raw, dq_raw = state.arm.torch_state()
    q, dq = q_raw, dq_raw
    collector_observation = _warp_observation_features_torch(q, dq, state.descriptor_tensor[0],
                                                               torch.zeros(8), torch.ones(8))
    with torch.no_grad():
        collector_belief = model.observe(model.predict_prior(state.belief, action, state.descriptor_tensor, 0.01),
                                          collector_observation, state.descriptor_tensor)
    endpoint = forward_kinematics(q[0].numpy(), descriptor.link_lengths)
    runtime.observe(Command.accepted([0.0, 0.0]), Observation(
        np.concatenate((q[0].numpy(), dq[0].numpy(), np.zeros(2), endpoint)), np.ones(8, dtype=bool),
        np.ones(8, dtype=bool), np.zeros(8), 0.01,
    ))
    collector_q, collector_dq = model.decode(collector_belief)
    collector_features = torch.cat((collector_belief, collector_q, collector_dq, state.descriptor_tensor, state.goal,
                                    state.goal_q, torch.full((1, 1), 1.49), action), dim=-1)
    torch.testing.assert_close(runtime.features.policy_features(task, Command.accepted([0.0, 0.0])),
                               collector_features[0])


def test_memoryless_runtime_zeros_all_learned_state_features() -> None:
    descriptor = default_descriptor()
    metadata = {"schema_version": 4,
                "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                          "endpoint_residual": False, "physics_prior": False,
                          "fresh_measurement_correction": True},
                "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8}}
    bundle = WorldBundle(WorldModel().eval(), metadata)
    runtime = Runtime(bundle, descriptor, PolicyNetwork(), torch.device("cpu"), memoryless=True)
    observation = Observation(np.zeros(8), np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), 0.0)
    task = Task(np.array([0.1, -0.5]), deadline_s=1.5)
    runtime.reset(descriptor, observation, task)
    features = runtime.features.policy_features(task, Command.accepted([0.0, 0.0]))
    assert features.shape == (POLICY_INPUT_DIM,)
    masked = features.clone()
    masked[:LEARNED_STATE_DIM] = 0.0
    expected = runtime.policy.deterministic(masked).detach().numpy()
    np.testing.assert_allclose(runtime.act().values, expected)


def test_memory_reset_preserves_task_clock_and_previous_command() -> None:
    descriptor = default_descriptor()
    metadata = {"schema_version": 4,
                "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                          "endpoint_residual": False, "physics_prior": False,
                          "fresh_measurement_correction": True},
                "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8}}
    bundle = WorldBundle(WorldModel().eval(), metadata)
    runtime = Runtime(bundle, descriptor, PolicyNetwork(), torch.device("cpu"))
    initial = Observation(np.zeros(8), np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), 0.0)
    task = Task(np.array([0.1, -0.5]), deadline_s=1.5)
    runtime.reset(descriptor, initial, task)
    accepted = Command.accepted([0.2, -0.3])
    later = Observation(np.full(8, 0.1), np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), 0.01)
    runtime.observe(accepted, later)
    elapsed_before = runtime.features.elapsed_s
    previous_before = runtime.previous_command.values.copy()

    runtime.reset_memory(later)

    assert runtime.features.elapsed_s == elapsed_before
    np.testing.assert_allclose(runtime.previous_command.values, previous_before)


def test_clean_packet_features_replay_identically_through_runtime() -> None:
    descriptor = default_descriptor()
    metadata = {"schema_version": 4,
                "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                          "endpoint_residual": False, "physics_prior": False,
                          "fresh_measurement_correction": True},
                "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8}}
    bundle = WorldBundle(WorldModel().eval(), metadata)
    settings = SensorSettings(np.full(8, 0.01), np.zeros(8), np.zeros(8), np.zeros(8))
    arm = ImperfectNativeArm(descriptor, ActuatorSettings.healthy(), SensorSuite(settings, seed=7))
    arm.reset(np.array([0.1, 0.7]))
    task = Task(np.array([0.1, -0.5]), deadline_s=1.5)
    adapter = Runtime(bundle, descriptor, PolicyNetwork(), torch.device("cpu"))
    runtime = Runtime(bundle, descriptor, PolicyNetwork(), torch.device("cpu"))
    initial = arm.observation()
    adapter.reset(descriptor, initial, task)
    runtime.reset(descriptor, initial, task)
    command = Command.accepted([0.2, -0.1])
    for _ in range(2):
        torch.testing.assert_close(adapter.features.policy_features(task, adapter.previous_command),
                                   runtime.features.policy_features(task, runtime.previous_command))
        for _ in range(10):
            arm.step(command)
        observation = arm.observation()
        adapter.observe(command, observation)
        runtime.observe(command, observation)


def test_real_packet_clone_collector_replays_saved_packets_through_fresh_runtime(tmp_path) -> None:
    metadata = {"schema_version": 4,
                "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                          "endpoint_residual": False, "physics_prior": False,
                          "fresh_measurement_correction": True},
                "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8}}
    world_path = WorldBundle(WorldModel().eval(), metadata).save(tmp_path / "world")

    report = audit_packet_clone_runtime_replay(world_path, tmp_path / "replay", device="cpu", case_indices=(0,))

    assert report["replay"]["passed"]
    assert np.load(tmp_path / "replay" / "collector-traces.npz", allow_pickle=False)["packet_values"].shape == (1, 171, 8)


def test_packet_dagger_rejects_a_memoryless_source_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_AI_ROOT", str(tmp_path))
    run = tmp_path / "memoryless"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"memoryless": True}))

    with pytest.raises(ValueError, match="selected adaptive"):
        train_packet_dagger(run, tmp_path / "output", device="cpu")


def test_packet_dagger_rejects_a_nonpositive_learning_rate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_AI_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="learning_rate"):
        train_packet_dagger(tmp_path, tmp_path / "output", device="cpu", learning_rate=0.0)


def test_backend_replay_comparison_requires_matching_selected_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_AI_ROOT", str(tmp_path))
    checkpoint = tmp_path / "best-policy.pt"
    checkpoint.write_bytes(b"selected policy")
    common = {"suite": "validation", "episodes": 64, "physics_dt": 0.001,
              "control_dt": 0.01, "command_rate_hz": 100.0, "substeps_per_command": 10,
              "success_fraction": 63 / 64, "checkpoint": str(checkpoint)}
    trace = {
        "q": np.zeros((3, 2)), "dq": np.zeros((3, 2)), "endpoint": np.zeros((3, 2)),
        "endpoint_velocity": np.zeros((3, 2)), "accepted_action": np.zeros((2, 2)),
        "time_s": np.array([0.0, 0.001, 0.002]), "target": np.array([0.1, -0.5]),
        "link_length_1": 0.3, "link_length_2": 0.25,
    }
    for backend in ("warp", "native"):
        root = tmp_path / backend
        root.mkdir()
        (root / "metrics.json").write_text(json.dumps({**common, "backend": backend}))
        np.savez(root / "demo.npz", **trace)

    report = compare_backend_replays(tmp_path / "warp", tmp_path / "native", tmp_path / "report.json",
                                     driver_version="610.57.04")

    assert report["acceptance"]["passed"]
    assert report["trace_difference"]["accepted_action"]["first_over_tolerance"] is None


def test_export_packages_the_selected_policy_checkpoint_with_provenance(tmp_path) -> None:
    metadata = {"schema_version": 4,
                "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                          "endpoint_residual": False, "physics_prior": True,
                          "fresh_measurement_correction": True},
                "normalization": {"mean": [0.0] * 8, "std": [1.0] * 8}}
    world_path = WorldBundle(WorldModel(physics_prior=True, fresh_measurement_correction=True).eval(), metadata).save(tmp_path / "world")
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    initial = PolicyNetwork()
    selected = PolicyNetwork()
    for parameter in selected.parameters():
        parameter.data.fill_(0.25)
    torch.save(initial.state_dict(), checkpoints / "initial-policy.pt")
    torch.save(selected.state_dict(), checkpoints / "policy-update-000001.pt")
    (run / "metrics.json").write_text(json.dumps({
        "world_bundle": str(world_path), "policy_input_dim": POLICY_INPUT_DIM,
        "selected_checkpoint": "policy-update-000001",
        "config_identity": {"path": "configs/policy-healthy.toml", "sha256": "test"},
    }))

    bundle_path = export_inference_bundle(run, tmp_path / "export")
    loaded = InferenceBundle.load(bundle_path)

    for actual, expected in zip(loaded.policy.parameters(), selected.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert loaded.world.model.fresh_measurement_correction
    # This availability-without-freshness packet exercises the schema-4 branch;
    # dropping the flag changes the posterior, even when serialized weights match.
    reference = WorldBundle.load(world_path)
    for model in (reference.model, loaded.world.model):
        assert model.state_observer is not None
        model.state_observer[-1].bias.data.fill_(0.25)
    observation = torch.zeros((1, 32)); observation[:, 8:12] = 1.0
    descriptor = torch.zeros((1, 12)); prior = torch.zeros((1, 128))
    torch.testing.assert_close(loaded.world.model.observe(prior, observation, descriptor),
                               reference.model.observe(prior, observation, descriptor))
    assert loaded.metadata["physical_envelope"]["modeled_hard_limits"] == ["joint_position"]
    provenance = loaded.metadata["provenance"]
    assert provenance["selected_checkpoint"] == "policy-update-000001"
    assert len(provenance["selected_checkpoint_sha256"]) == 64


def test_explicit_policy_initialization_reproduces_selected_weights(tmp_path) -> None:
    selected = PolicyNetwork()
    for parameter in selected.parameters():
        parameter.data.fill_(0.125)
    source = tmp_path / "selected-policy.pt"
    torch.save(selected.state_dict(), source)
    initialized = PolicyNetwork()

    identity = _initialize_policy(initialized, source, device=torch.device("cpu"))

    assert identity is not None
    assert identity["path"] == str(source)
    assert len(identity["sha256"]) == 64
    for actual, expected in zip(initialized.parameters(), selected.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_explicit_scale_calibration_preserves_deterministic_selected_actions(tmp_path) -> None:
    selected = PolicyNetwork()
    source = tmp_path / "selected-policy.pt"
    torch.save(selected.state_dict(), source)
    initialized = PolicyNetwork()
    features = torch.randn(3, POLICY_INPUT_DIM)

    identity = _initialize_policy(initialized, source, device=torch.device("cpu"), action_scale=0.05)

    assert identity is not None
    assert identity["action_scale"] == 0.05
    torch.testing.assert_close(initialized.deterministic(features), selected.deterministic(features))
    _, scale = initialized.parameters_for_distribution(features)
    torch.testing.assert_close(scale, torch.full_like(scale, 0.05))
