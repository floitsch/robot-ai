"""Command-line entry point for the project."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .config import ConfigurationError, confined_path, load_config
from .data.collect import collect_dataset, inspect_dataset, reconstruct_episode
from .doctor import run_doctor
from .evaluate.adaptation import run_adaptation_suite
from .evaluate.baseline import run_baseline_suite, run_imperfect_reference_case
from .evaluate.metrics import compare_smoothness
from .evaluate.physics import run_backlash_fixtures, run_physics_fixtures
from .evaluate.policy import compare_backend_replays, evaluate_packet_policy, evaluate_policy
from .evaluate.world import evaluate_world
from .models.inference import export_inference_bundle
from .sim.combined import CombinedImperfectArm
from .sim.env import RigidArmEnvironment
from .sim.native import ImperfectNativeArm, default_descriptor
from .sim.scenarios import make_scenario
from .sim.sensors import SensorSuite
from .train.policy import (
    audit_current_packet_comparator,
    audit_packet_clone_runtime_replay,
    train_coverage_decoded_feedback_distillation,
    train_current_packet_comparator,
    train_decoded_feedback_correction,
    train_full_decoded_feedback_distillation,
    train_packet_behavior_clone,
    train_packet_dagger,
    train_packet_delta_clone,
    train_packet_progressive_clone,
    train_policy,
    train_support_matched_decoded_correction,
)
from .train.world import train_world
from .visualize.learning import export_learning_gif, export_learning_html
from .visualize.packet_diagnostic import (
    export_packet_diagnostic_gif,
    export_packet_diagnostic_html,
    export_packet_diagnostic_report,
)
from .visualize.replay import export_gif, export_html


def _output_file(value: str, filename: str) -> Path:
    path = confined_path(value)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="check local setup and GPU capacity")
    doctor.add_argument("--config", help="project-local TOML config")
    doctor.add_argument("--output", required=True, help="project-local JSON report path")
    doctor.add_argument("--device", choices=("cpu", "cuda"), help="requested device metadata")
    doctor.add_argument("--seed", type=int, default=0)
    simulate = subparsers.add_parser("simulate", help="run a short rigid-arm trajectory")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--steps", type=int, default=200)
    simulate.add_argument("--output", required=True)
    simulate.add_argument("--device", choices=("cpu", "cuda"))
    simulate.add_argument("--seed", type=int, default=0)
    benchmark = subparsers.add_parser("benchmark", help="measure a backend's step throughput")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--device", choices=("cpu", "cuda"))
    benchmark.add_argument("--seed", type=int, default=0)
    physics = subparsers.add_parser("physics-check", help="run deterministic native/Warp physics fixtures")
    physics.add_argument("--config", required=True)
    physics.add_argument("--output", required=True)
    backlash = subparsers.add_parser("backlash-check", help="run P7 physical backlash fixtures")
    backlash.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a controller on a fixed suite")
    evaluate.add_argument("--controller", choices=("baseline", "learned"), default="learned")
    evaluate.add_argument("--bundle", help="policy run or policy/best checkpoint")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--suite", choices=("validation", "final"), default="validation")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"))
    evaluate.add_argument("--baseline-kp", nargs=2, type=float,
                          help="optional ordinary-feedback proportional gains")
    evaluate.add_argument("--baseline-kd", nargs=2, type=float,
                          help="optional ordinary-feedback derivative gains")
    evaluate.add_argument("--baseline-computed-torque", action="store_true",
                          help="add nominal rigid-arm feedforward to the ordinary controller")
    evaluate.add_argument("--baseline-hold-goal", action="store_true",
                          help="command a static goal target, matching the PPO imitation teacher")
    evaluate.add_argument("--cases", type=int, default=64,
                          help="ordinary-reference case count; 64 is the frozen P2 suite")
    evaluate.add_argument("--baseline-case-index", type=int,
                          help="record one ordinary-reference case from the frozen suite")
    compare_backends = subparsers.add_parser("compare-backends",
                                              help="verify selected-policy native/Warp replay transfer")
    compare_backends.add_argument("--warp-evaluation", required=True)
    compare_backends.add_argument("--native-evaluation", required=True)
    compare_backends.add_argument("--driver-version", required=True)
    compare_backends.add_argument("--output", required=True)
    packet_evaluate = subparsers.add_parser("evaluate-packet-policy",
                                             help="evaluate a learned policy through R5b1 P3 packets")
    packet_evaluate.add_argument("--run", required=True)
    packet_evaluate.add_argument("--config", required=True)
    packet_evaluate.add_argument("--output", required=True)
    packet_evaluate.add_argument("--checkpoint", default="best")
    packet_evaluate.add_argument("--device", choices=("cpu", "cuda"))
    packet_evaluate.add_argument("--case-index", type=int, help="evaluate one frozen R5b packet case")
    packet_evaluate.add_argument("--case-manifest", help="versioned healthy packet-case manifest; default is frozen development-64")
    packet_evaluate.add_argument("--diagnostic-truth-teacher", action="store_true",
                                 help="store training-only same-state teacher actions in a diagnostic replay")
    packet_evaluate.add_argument("--no-demo-render", action="store_true",
                                 help="score all cases without rendering the first-case HTML/GIF replay")
    packet_diagnostic = subparsers.add_parser(
        "export-packet-diagnostic", help="export synchronized C2 packet-feedback diagnostic playback"
    )
    packet_diagnostic.add_argument("--selection", required=True, help="packet run metrics with selection reports")
    packet_diagnostic.add_argument("--trace", action="append", required=True,
                                   help="CASE:LABEL:demo.npz; repeat for reference, initial, intermediate, selected")
    packet_diagnostic.add_argument("--output", required=True)
    packet_diagnostic.add_argument("--gif", help="optional headless GIF comparison written from the same traces")
    packet_diagnostic.add_argument("--report", help="optional JSON with input hashes and exact reproduction command")
    imperfect = subparsers.add_parser("evaluate-imperfect-reference",
                                      help="record one scored sensor/motor-imperfect reference task")
    imperfect.add_argument("--config", required=True)
    imperfect.add_argument("--output", required=True)
    imperfect.add_argument("--scenario-id", default="r4-imperfect-reference")
    replay = subparsers.add_parser("replay", help="export a saved trajectory")
    replay.add_argument("--trajectory", required=True)
    replay.add_argument("--output", required=True)
    collect = subparsers.add_parser("collect", help="collect versioned recurrent-model data")
    collect.add_argument("--config", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--episodes", type=int, default=20)
    collect.add_argument("--actions-per-episode", type=int, default=50)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--profile", choices=("historical", "r3a"), default="historical")
    inspect = subparsers.add_parser("inspect-data", help="inspect a dataset manifest and arrays")
    inspect.add_argument("--dataset", required=True)
    inspect.add_argument("--output", required=True)
    reconstruct = subparsers.add_parser("reconstruct-data", help="regenerate and verify one named dataset episode")
    reconstruct.add_argument("--dataset", required=True)
    reconstruct.add_argument("--scenario-id", required=True)
    reconstruct.add_argument("--output", required=True)
    train_model = subparsers.add_parser("train-world", help="train the recurrent feeling and predictor")
    train_model.add_argument("--config", required=True)
    train_model.add_argument("--dataset", required=True)
    train_model.add_argument("--output", required=True)
    train_model.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train_model.add_argument("--updates", type=int, default=2000)
    train_model.add_argument("--seed", type=int, default=0)
    train_model.add_argument("--endpoint-residual", action="store_true",
                             help="train the P8 world-frame endpoint residual decoder")
    train_model.add_argument("--physics-prior", action="store_true",
                             help="use constant-velocity propagation plus learned dynamics residuals")
    train_model.add_argument("--control-action-weight", type=float, default=0.0,
                             help="supervise decoded state through the healthy computed-torque teacher")
    train_model.add_argument("--overfit-scenario-id",
                             help="restrict a diagnostic world-model run to one training scenario")
    eval_model = subparsers.add_parser("evaluate-world", help="evaluate a world-model bundle")
    eval_model.add_argument("--bundle", required=True)
    eval_model.add_argument("--dataset", required=True)
    eval_model.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    eval_model.add_argument("--output", required=True)
    eval_model.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    eval_model.add_argument("--trail-output", help="optional NPZ path for a held-out endpoint-trail HTML export")
    train_controller = subparsers.add_parser("train-policy", help="train a PPO policy using the shared feeling")
    train_controller.add_argument("--config", required=True)
    train_controller.add_argument("--world-bundle", required=True)
    train_controller.add_argument("--output", required=True)
    train_controller.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train_controller.add_argument("--worlds", type=int, default=4)
    train_controller.add_argument("--rollout-steps", type=int, default=128)
    train_controller.add_argument("--transitions", type=int, default=100_000)
    train_controller.add_argument("--seed", type=int, default=0)
    train_controller.add_argument("--warmup-updates", type=int, default=10,
                                  help="behavior-cloning updates before PPO; zero is useful for contract diagnostics")
    train_controller.add_argument("--init-policy", help="selected policy checkpoint used to initialize PPO")
    train_controller.add_argument("--init-action-scale", type=float,
                                  help="optional explicit stochastic action scale after checkpoint initialization")
    train_controller.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    train_controller.add_argument("--ppo-epochs", type=int, default=5)
    train_controller.add_argument("--max-approximate-kl", type=float,
                                  help="stop PPO minibatches after this post-update KL estimate")
    train_controller.add_argument("--teacher-regularization-weight", type=float, default=0.0,
                                  help="PPO-time MSE weight toward training-only computed-torque teacher actions")
    train_controller.add_argument("--anchor-regularization-weight", type=float, default=0.0,
                                  help="PPO-time MSE weight toward frozen selected-policy actions")
    train_controller.add_argument("--critic-warmup-epochs", type=int, default=0,
                                  help="critic-only regression epochs before PPO advantage recomputation")
    train_controller.add_argument("--critic-warmup-learning-rate", type=float, default=1e-3)
    train_controller.add_argument("--memoryless", action="store_true",
                                  help="zero the learned belief in collection and frozen evaluation")
    packet_clone = subparsers.add_parser("train-packet-clone",
                                         help="behavior-clone a policy from R5b1 causal P3 packets")
    packet_clone.add_argument("--config", required=True)
    packet_clone.add_argument("--world-bundle", required=True)
    packet_clone.add_argument("--output", required=True)
    packet_clone.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    packet_clone.add_argument("--episodes", type=int, default=30)
    packet_clone.add_argument("--clone-epochs", type=int, default=30)
    packet_clone.add_argument("--checkpoint-interval", type=int, default=5)
    packet_clone.add_argument("--seed", type=int, default=0)
    packet_clone.add_argument("--memoryless", action="store_true")
    packet_replay = subparsers.add_parser(
        "audit-packet-collector-replay", help="replay real clean-packet clone traces through a fresh Runtime"
    )
    packet_replay.add_argument("--config", required=True)
    packet_replay.add_argument("--world-bundle", required=True)
    packet_replay.add_argument("--output", required=True)
    packet_replay.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    packet_replay.add_argument("--case-indices", nargs="+", type=int, default=[0, 39])
    packet_dagger = subparsers.add_parser(
        "train-packet-dagger", help="run the R5b2 clean-packet DAgger correction"
    )
    packet_dagger.add_argument("--run", required=True, help="selected adaptive R5b1a packet-policy run")
    packet_dagger.add_argument("--config", required=True)
    packet_dagger.add_argument("--output", required=True)
    packet_dagger.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    packet_dagger.add_argument("--clone-epochs", type=int, default=500)
    packet_dagger.add_argument("--checkpoint-interval", type=int, default=125)
    packet_dagger.add_argument("--learning-rate", type=float, required=True,
                               help="explicit fresh-Adam learning rate for the bounded correction")
    packet_dagger.add_argument("--seed", type=int, default=20260913)
    packet_progressive = subparsers.add_parser(
        "train-packet-progressive-clone", help="run the R5b4 progressive clean-packet clone"
    )
    packet_progressive.add_argument("--config", required=True)
    packet_progressive.add_argument("--world-bundle", required=True)
    packet_progressive.add_argument("--output", required=True)
    packet_progressive.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    packet_progressive.add_argument("--updates", type=int, default=30)
    packet_progressive.add_argument("--trajectories-per-update", type=int, default=4)
    packet_progressive.add_argument("--fit-passes-per-update", type=int, default=50)
    packet_progressive.add_argument("--checkpoint-interval", type=int, default=5)
    packet_progressive.add_argument("--seed", type=int, default=20260913)
    packet_comparator = subparsers.add_parser("train-current-packet-comparator",
                                               help="train C3's fair feedforward current-packet baseline")
    packet_comparator.add_argument("--config", required=True); packet_comparator.add_argument("--world-bundle", required=True)
    packet_comparator.add_argument("--output", required=True); packet_comparator.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    packet_comparator.add_argument("--updates", type=int, default=30); packet_comparator.add_argument("--trajectories-per-update", type=int, default=4)
    packet_comparator.add_argument("--fit-passes-per-update", type=int, default=50); packet_comparator.add_argument("--checkpoint-interval", type=int, default=5); packet_comparator.add_argument("--seed", type=int, default=20260913)
    audit_comparator = subparsers.add_parser(
        "audit-current-packet-comparator", help="audit C3 current-packet field, reload, and runtime replay contracts"
    )
    audit_comparator.add_argument("--run", required=True); audit_comparator.add_argument("--trace", required=True)
    audit_comparator.add_argument("--config", required=True); audit_comparator.add_argument("--output", required=True)
    audit_comparator.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    audit_comparator.add_argument("--collector-case-indices", nargs="+", type=int, default=[0, 39])
    decoded_correction = subparsers.add_parser(
        "train-decoded-feedback-correction", help="run C3c-2's one-step decoded-feedback anchored correction"
    )
    decoded_correction.add_argument("--run", required=True); decoded_correction.add_argument("--config", required=True)
    decoded_correction.add_argument("--output", required=True); decoded_correction.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    support_correction = subparsers.add_parser("train-support-matched-decoded-correction", help="run C3c-3's checkpointed decoded-label correction")
    support_correction.add_argument("--run", required=True); support_correction.add_argument("--config", required=True)
    support_correction.add_argument("--output", required=True); support_correction.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    full_distill = subparsers.add_parser("train-full-decoded-feedback-distillation", help="run C3c-4 all-development decoded-feedback distillation")
    full_distill.add_argument("--run", required=True); full_distill.add_argument("--config", required=True); full_distill.add_argument("--output", required=True); full_distill.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    coverage_distill = subparsers.add_parser("train-coverage-decoded-feedback-distillation", help="run C3c-5 disjoint-manifest decoded-feedback coverage distillation")
    coverage_distill.add_argument("--run", required=True); coverage_distill.add_argument("--training-manifest", required=True); coverage_distill.add_argument("--config", required=True); coverage_distill.add_argument("--output", required=True); coverage_distill.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    packet_delta = subparsers.add_parser("train-packet-delta-clone", help="run the R5b5 action-delta clone")
    packet_delta.add_argument("--source", required=True); packet_delta.add_argument("--output", required=True)
    packet_delta.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    packet_delta.add_argument("--delta-weight", type=float, required=True)
    packet_delta.add_argument("--epochs", type=int, default=100); packet_delta.add_argument("--checkpoint-interval", type=int, default=25)
    select_packet = subparsers.add_parser(
        "select-packet-policy", help="select a packet-trained checkpoint on the R5b1 packet suite"
    )
    select_packet.add_argument("--run", required=True)
    select_packet.add_argument("--config", required=True)
    select_packet.add_argument("--output", required=True)
    select_packet.add_argument("--device", choices=("cpu", "cuda"))
    select_controller = subparsers.add_parser("select-policy", help="select a policy checkpoint by validation metrics")
    select_controller.add_argument("--run", required=True)
    select_controller.add_argument("--config", required=True)
    select_controller.add_argument("--output", required=True)
    select_controller.add_argument("--device", choices=("cpu", "cuda"))
    export = subparsers.add_parser("export", help="export a self-contained inference bundle")
    export.add_argument("--run", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--world-bundle")
    export.add_argument("--config", help="config identity for runs created before policy reports recorded it")
    compare = subparsers.add_parser("compare", help="compare saved policy checkpoints")
    compare.add_argument("--run", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--config", default="configs/healthy.toml")
    compare.add_argument("--device", choices=("cpu", "cuda"))
    compare.add_argument("--reference", default="artifacts/p2/eval-v7-100hz-warp-sampled-pd/metrics.json")
    compare.add_argument("--imperfect-reference", default="artifacts/p4/r4-imperfect-reference/metrics.json")
    compare.add_argument("--imperfect-config", default="configs/sensors-and-motors.toml")
    adaptation = subparsers.add_parser("evaluate-adaptation", help="run paired P7 adaptation evaluation")
    adaptation.add_argument("--run", required=True)
    adaptation.add_argument("--memoryless-run", required=True)
    adaptation.add_argument("--config", required=True)
    adaptation.add_argument("--output", required=True)
    adaptation.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    adaptation.add_argument("--seeds", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            config = load_config(args.config)
            report = run_doctor(args.output, requested_device=args.device or config.device, seed=args.seed)
            print(json.dumps(report, indent=2, sort_keys=True))
            print(f"doctor report: {args.output}")
            return 0
        if args.command == "simulate":
            config = load_config(args.config)
            if args.steps < 1:
                raise ConfigurationError("steps must be positive")
            np.random.seed(args.seed)
            device = args.device or config.device
            backend = config.backend
            if backend == "warp" and device == "cuda":
                device = "cuda:0"
            imperfect = None
            combined = None
            load_utilizations = [0.0]
            if config.scenario == "degraded":
                scenario = make_scenario(config.seed, "cli-simulate", degraded=True)
                imperfect = ImperfectNativeArm(default_descriptor(), scenario.actuator,
                                               SensorSuite(scenario.sensors, seed=scenario.seed, faults=scenario.faults),
                                               timestep=config.physics_dt)
                imperfect.reset()
                records = [imperfect.arm.truth()]
                observations = [imperfect.observation()]
                environment = None
            elif config.scenario == "combined":
                combined = CombinedImperfectArm(default_descriptor(), timestep=config.physics_dt, seed=config.seed)
                records = [combined.truth()]
                observations = [combined.observation()]
                environment = None
            else:
                environment = RigidArmEnvironment(default_descriptor(), backend=backend, device=device,
                                                   world_count=config.world_count, physics_dt=config.physics_dt)
                start = environment.reset()
                records = [start.truth[0]]
                observations = [start.observations[0]]
            actions = []
            for step in range(args.steps):
                command = np.array([0.15 * np.sin(step / 20), -0.10 * np.cos(step / 27)], dtype=np.float64)
                if imperfect is not None:
                    truth = imperfect.step(command)
                    records.append(truth)
                    observations.append(imperfect.observation())
                    load_utilizations.append(0.0)
                elif combined is not None:
                    truth = combined.step(command)
                    records.append(truth)
                    observations.append(combined.observation())
                    load_utilizations.append(combined.load_utilization())
                else:
                    assert environment is not None
                    result = environment.step(np.tile(command, (config.world_count, 1)))
                    records.append(result.truth[0])
                    observations.append(result.observations[0])
                    load_utilizations.append(0.0)
                actions.append(command)
            output = _output_file(args.output, "trajectory.npz")
            np.savez(output, q=np.stack([item.q for item in records]), dq=np.stack([item.dq for item in records]),
                     endpoint=np.stack([item.endpoint_xz for item in records]),
                     endpoint_velocity=np.stack([item.endpoint_velocity_xz for item in records]),
                     base_pose=np.stack([item.base_pose for item in records]),
                     load_summary=np.stack([item.load_summary for item in records]),
                     load_utilization=np.asarray(load_utilizations),
                     accepted_action=np.stack(actions), time_s=np.arange(args.steps + 1) * config.control_dt,
                     observation_values=np.stack([item.values for item in observations]),
                     available=np.stack([item.available for item in observations]),
                     fresh=np.stack([item.fresh for item in observations]),
                     age_s=np.stack([item.age_s for item in observations]),
                     backend=backend, device=device, scenario=config.scenario)
            print(json.dumps({"backend": backend, "device": device, "scenario": config.scenario,
                              "steps": args.steps, "output": str(output)}, indent=2))
            return 0
        if args.command == "benchmark":
            config = load_config(args.config)
            device = args.device or config.device
            if config.backend == "warp" and device == "cuda":
                device = "cuda:0"
            environment = RigidArmEnvironment(default_descriptor(), backend=config.backend, device=device,
                                               world_count=config.world_count, physics_dt=config.physics_dt)
            commands = np.zeros((config.world_count, 2), dtype=np.float64)
            for _ in range(10):
                environment.step(commands)
            started = time.perf_counter()
            steps = 100
            for _ in range(steps):
                if config.backend == "warp":
                    assert environment._warp is not None
                    environment._warp.step(commands)
                else:
                    environment.step(commands)
            elapsed = time.perf_counter() - started
            if config.backend == "warp":
                assert environment._warp is not None
                import warp as wp
                wp.synchronize_device(device)
                elapsed = time.perf_counter() - started
            report = {"backend": config.backend, "device": device, "world_count": config.world_count,
                      "steps": steps, "elapsed_seconds": elapsed, "steps_per_second": steps / elapsed}
            output = _output_file(args.output, "benchmark.json")
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({**report, "output": str(output)}, indent=2))
            return 0
        if args.command == "physics-check":
            config = load_config(args.config)
            report = run_physics_fixtures(args.output, config=config)
            print(json.dumps({**report, "output": str(args.output)}, indent=2))
            return 0
        if args.command == "backlash-check":
            report = run_backlash_fixtures(args.output)
            print(json.dumps({**report, "output": str(args.output)}, indent=2))
            return 0
        if args.command == "evaluate":
            config = load_config(args.config)
            if args.controller == "baseline":
                if args.cases < 1:
                    raise ConfigurationError("cases must be positive")
                report = run_baseline_suite(
                    args.output, config=config, count=args.cases,
                    kp=np.asarray(args.baseline_kp) if args.baseline_kp else None,
                    kd=np.asarray(args.baseline_kd) if args.baseline_kd else None,
                    computed_torque=args.baseline_computed_torque,
                    hold_goal=args.baseline_hold_goal,
                    case_index=args.baseline_case_index,
                )
                config_path = confined_path(args.config, must_exist=True)
                report["config_identity"] = {
                    "path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                }
                artifacts = report["artifacts"]
                if not isinstance(artifacts, dict) or not isinstance(artifacts.get("trajectory"), str):
                    raise RuntimeError("baseline report is missing its trajectory artifact")
                metrics = Path(artifacts["trajectory"]).parent / "metrics.json"
                metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            else:
                if args.cases != 64:
                    raise ConfigurationError("--cases applies only to the ordinary reference")
                if not args.bundle:
                    raise ConfigurationError("learned evaluation requires --bundle")
                config_path = confined_path(args.config, must_exist=True)
                report = evaluate_policy(
                    args.bundle, args.output, config=config, device=args.device, suite=args.suite,
                    config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
                )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "compare-backends":
            report = compare_backend_replays(args.warp_evaluation, args.native_evaluation, args.output,
                                             driver_version=args.driver_version)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "evaluate-packet-policy":
            config = load_config(args.config)
            config_path = confined_path(args.config, must_exist=True)
            report = evaluate_packet_policy(args.run, args.output, config=config,
                                            checkpoint=args.checkpoint, device=args.device,
                                            case_index=args.case_index,
                                            case_manifest=args.case_manifest,
                                            config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
                                            diagnostic_truth_teacher=args.diagnostic_truth_teacher,
                                            render_demo=not args.no_demo_render)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "evaluate-imperfect-reference":
            config = load_config(args.config)
            report = run_imperfect_reference_case(args.output, config=config, scenario_id=args.scenario_id)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "export-packet-diagnostic":
            traces: dict[int, dict[str, str]] = {}
            for item in args.trace:
                try:
                    case_text, label, path = item.split(":", 2)
                    case = int(case_text)
                except ValueError as error:
                    raise ConfigurationError("--trace must be CASE:LABEL:demo.npz") from error
                if not label or label in traces.setdefault(case, {}):
                    raise ConfigurationError("each packet diagnostic case/label must be unique")
                traces[case][label] = path
            output = export_packet_diagnostic_html(traces, args.selection, args.output)
            gif = export_packet_diagnostic_gif(traces, args.selection, args.gif) if args.gif else None
            command = shlex.join(["scripts/project-run", ".venv/bin/robot-ai", "export-packet-diagnostic",
                                  "--selection", args.selection,
                                  *[item for trace in args.trace for item in ("--trace", trace)],
                                  "--output", args.output,
                                  *([] if args.gif is None else ("--gif", args.gif)),
                                  *([] if args.report is None else ("--report", args.report))])
            diagnostic_report = (export_packet_diagnostic_report(
                traces, args.selection, args.report, reproduction_command=command, html=output, gif=gif
            ) if args.report else None)
            print(json.dumps({"html": str(output), "gif": str(gif) if gif else None,
                              "report": str(diagnostic_report) if diagnostic_report else None,
                              "cases": sorted(traces)}, indent=2))
            return 0
        if args.command == "replay":
            trajectory = confined_path(args.trajectory)
            if trajectory.is_dir():
                trajectory = trajectory / "demo.npz"
            if not trajectory.exists():
                raise ConfigurationError(f"trajectory does not exist: {trajectory}")
            loaded = np.load(trajectory, allow_pickle=False)
            data = {key: loaded[key] for key in loaded.files}
            output = confined_path(args.output)
            html_output = output if output.suffix == ".html" else output / "arm.html"
            gif_output = html_output.with_suffix(".gif")
            export_html(data, html_output)
            export_gif(data, gif_output, stride=5)
            print(json.dumps({"html": str(html_output), "gif": str(gif_output)}, indent=2))
            return 0
        if args.command == "collect":
            config = load_config(args.config)
            config_hash = hashlib.sha256(confined_path(args.config, must_exist=True).read_bytes()).hexdigest()
            manifest = collect_dataset(args.output, seed=args.seed, episodes=args.episodes,
                                       actions_per_episode=args.actions_per_episode,
                                       backend=config.backend, device=config.device,
                                       physics_dt=config.physics_dt, control_dt=config.control_dt,
                                       profile=args.profile, config_hash=config_hash)
            print(json.dumps({"manifest": str(manifest)}, indent=2))
            return 0
        if args.command == "inspect-data":
            report = inspect_dataset(args.dataset)
            output = confined_path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({**report, "output": str(output)}, indent=2))
            return 0
        if args.command == "reconstruct-data":
            report = reconstruct_episode(args.dataset, args.scenario_id, args.output)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train-world":
            load_config(args.config)
            if args.updates < 1:
                raise ConfigurationError("updates must be positive")
            report = train_world(args.dataset, args.output, device=args.device, updates=args.updates, seed=args.seed,
                                 endpoint_residual=args.endpoint_residual, physics_prior=args.physics_prior,
                                 overfit_scenario_id=args.overfit_scenario_id,
                                 control_action_weight=args.control_action_weight)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "evaluate-world":
            report = evaluate_world(args.bundle, args.dataset, args.output, split=args.split, device=args.device,
                                    trail_output=args.trail_output)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train-policy":
            config = load_config(args.config)
            if args.worlds < 1 or args.rollout_steps < 1 or args.transitions < 1 or args.warmup_updates < 0:
                raise ConfigurationError("worlds, rollout-steps, transitions, and warmup-updates must be valid")
            report = train_policy(args.world_bundle, args.output, device=args.device, worlds=args.worlds,
                                  rollout_steps=args.rollout_steps, transitions=args.transitions, seed=args.seed,
                                  physics_dt=config.physics_dt, control_dt=config.control_dt,
                                  warmup_updates=args.warmup_updates, memoryless=args.memoryless,
                                  init_policy=args.init_policy,
                                  init_action_scale=args.init_action_scale,
                                  ppo_learning_rate=args.ppo_learning_rate,
                                  ppo_epochs=args.ppo_epochs,
                                  max_approximate_kl=args.max_approximate_kl,
                                  teacher_regularization_weight=args.teacher_regularization_weight,
                                  anchor_regularization_weight=args.anchor_regularization_weight,
                                  critic_warmup_epochs=args.critic_warmup_epochs,
                                  critic_warmup_learning_rate=args.critic_warmup_learning_rate,
                                  config_identity={
                                      "path": str(confined_path(args.config, must_exist=True)),
                                      "sha256": hashlib.sha256(
                                          confined_path(args.config, must_exist=True).read_bytes()
                                      ).hexdigest(),
                                  })
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train-packet-clone":
            config = load_config(args.config)
            config_path = confined_path(args.config, must_exist=True)
            report = train_packet_behavior_clone(
                args.world_bundle, args.output, device=args.device, episodes=args.episodes,
                clone_epochs=args.clone_epochs, checkpoint_interval=args.checkpoint_interval, seed=args.seed,
                physics_dt=config.physics_dt, control_dt=config.control_dt, memoryless=args.memoryless,
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "audit-packet-collector-replay":
            config = load_config(args.config)
            config_path = confined_path(args.config, must_exist=True)
            report = audit_packet_clone_runtime_replay(
                args.world_bundle, args.output, device=args.device, case_indices=tuple(args.case_indices),
                physics_dt=config.physics_dt, control_dt=config.control_dt,
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train-packet-dagger":
            config = load_config(args.config)
            config_path = confined_path(args.config, must_exist=True)
            report = train_packet_dagger(
                args.run, args.output, device=args.device, clone_epochs=args.clone_epochs,
                checkpoint_interval=args.checkpoint_interval, seed=args.seed,
                physics_dt=config.physics_dt, control_dt=config.control_dt,
                learning_rate=args.learning_rate,
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train-packet-progressive-clone":
            config = load_config(args.config)
            config_path = confined_path(args.config, must_exist=True)
            report = train_packet_progressive_clone(
                args.world_bundle, args.output, device=args.device, updates=args.updates,
                trajectories_per_update=args.trajectories_per_update,
                fit_passes_per_update=args.fit_passes_per_update,
                checkpoint_interval=args.checkpoint_interval, seed=args.seed,
                physics_dt=config.physics_dt, control_dt=config.control_dt,
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "train-current-packet-comparator":
            config = load_config(args.config); config_path = confined_path(args.config, must_exist=True)
            report = train_current_packet_comparator(args.world_bundle, args.output, device=args.device, updates=args.updates,
                trajectories_per_update=args.trajectories_per_update, fit_passes_per_update=args.fit_passes_per_update,
                checkpoint_interval=args.checkpoint_interval, seed=args.seed, physics_dt=config.physics_dt, control_dt=config.control_dt,
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
            print(json.dumps(report, indent=2)); return 0
        if args.command == "audit-current-packet-comparator":
            config = load_config(args.config); config_path = confined_path(args.config, must_exist=True)
            report = audit_current_packet_comparator(
                args.run, args.trace, args.output, device=args.device,
                physics_dt=config.physics_dt, control_dt=config.control_dt,
                collector_case_indices=tuple(args.collector_case_indices),
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
            )
            print(json.dumps(report, indent=2)); return 0
        if args.command == "train-decoded-feedback-correction":
            config = load_config(args.config); config_path = confined_path(args.config, must_exist=True)
            report = train_decoded_feedback_correction(
                args.run, args.output, device=args.device, physics_dt=config.physics_dt, control_dt=config.control_dt,
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
            )
            print(json.dumps(report, indent=2)); return 0
        if args.command == "train-support-matched-decoded-correction":
            config = load_config(args.config); config_path = confined_path(args.config, must_exist=True)
            report = train_support_matched_decoded_correction(args.run, args.output, device=args.device,
                physics_dt=config.physics_dt, control_dt=config.control_dt,
                config_identity={"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
            print(json.dumps(report, indent=2)); return 0
        if args.command == "train-full-decoded-feedback-distillation":
            config=load_config(args.config); path=confined_path(args.config,must_exist=True)
            print(json.dumps(train_full_decoded_feedback_distillation(args.run,args.output,device=args.device,physics_dt=config.physics_dt,control_dt=config.control_dt,config_identity={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}),indent=2));return 0
        if args.command == "train-coverage-decoded-feedback-distillation":
            config=load_config(args.config); path=confined_path(args.config,must_exist=True)
            print(json.dumps(train_coverage_decoded_feedback_distillation(args.run,args.training_manifest,args.output,device=args.device,physics_dt=config.physics_dt,control_dt=config.control_dt,config_identity={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}),indent=2));return 0
        if args.command == "train-packet-delta-clone":
            print(json.dumps(train_packet_delta_clone(args.source, args.output, device=args.device,
                delta_weight=args.delta_weight, epochs=args.epochs, checkpoint_interval=args.checkpoint_interval), indent=2))
            return 0
        if args.command == "select-packet-policy":
            config = load_config(args.config)
            config_path = confined_path(args.config, must_exist=True)
            run = confined_path(args.run, must_exist=True)
            training = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            packet_checkpoints = run / "checkpoints"
            candidates = ["initial"] + [
                item.stem for item in sorted(packet_checkpoints.glob("policy-update-*.pt"))
            ]
            reports = [
                evaluate_packet_policy(run, confined_path(args.output) / checkpoint, config=config,
                                       checkpoint=checkpoint, device=args.device,
                                       config_identity={"path": str(config_path),
                                                        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
                for checkpoint in candidates
            ]
            selected = max(reports, key=lambda item: (item["success_fraction"],
                                                      -(item["deadline_error_median_m"] or float("inf"))))
            selected_name = candidates[reports.index(selected)]
            source = (packet_checkpoints / "initial-policy.pt" if selected_name == "initial"
                      else packet_checkpoints / f"{selected_name}.pt")
            shutil.copy2(source, run / "best-policy.pt")
            training["selected_checkpoint"] = selected_name
            training["selection"] = {
                "suite": "r5b1-packetized-healthy-64", "candidates": candidates, "reports": reports,
            }
            (run / "metrics.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"selected_checkpoint": selected_name, "reports": reports}, indent=2))
            return 0
        if args.command == "select-policy":
            config = load_config(args.config)
            run = confined_path(args.run, must_exist=True)
            training = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            updates = len(training["history"])
            candidates = ["initial", f"policy-update-{max(1, updates // 2):06d}", f"policy-update-{updates:06d}"]
            candidates = [item for item in dict.fromkeys(candidates)
                          if ((run / "checkpoints" / "initial-policy.pt").exists() if item == "initial"
                              else (run / "checkpoints" / f"{item}.pt").exists())]
            reports = []
            for checkpoint in candidates:
                reports.append(evaluate_policy(run, confined_path(args.output) / checkpoint, config=config,
                                               checkpoint=checkpoint, device=args.device, suite="validation"))
            selected = max(reports, key=lambda item: (item["success_fraction"],
                                                      -(item["deadline_error_median_m"] or float("inf"))))
            selected_name = candidates[reports.index(selected)]
            source = (run / "checkpoints" / "initial-policy.pt" if selected_name == "initial"
                      else run / "checkpoints" / f"{selected_name}.pt")
            shutil.copy2(source, run / "best-policy.pt")
            training["selected_checkpoint"] = selected_name
            training["selection"] = {"suite": "validation", "reports": reports}
            (run / "metrics.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"selected_checkpoint": selected_name, "reports": reports}, indent=2))
            return 0
        if args.command == "export":
            config_identity = None
            if args.config:
                config_path = confined_path(args.config, must_exist=True)
                config_identity = {"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
            bundle = export_inference_bundle(args.run, args.output, world_bundle=args.world_bundle,
                                             config_identity=config_identity)
            print(json.dumps({"bundle": str(bundle)}, indent=2))
            return 0
        if args.command == "compare":
            config = load_config(args.config)
            device = args.device or config.device
            output = confined_path(args.output)
            report_root = output.parent / f"{output.stem}-checkpoints" if output.suffix else output
            training = json.loads((confined_path(args.run) / "metrics.json").read_text(encoding="utf-8"))
            midpoint = f"policy-update-{max(1, len(training.get('history', [])) // 2):06d}"
            checkpoints = ["initial"]
            if (confined_path(args.run) / "checkpoints" / f"{midpoint}.pt").exists():
                checkpoints.append(midpoint)
            checkpoints.append("best")
            reports = []
            for checkpoint in checkpoints:
                checkpoint_report = evaluate_policy(args.run, report_root / checkpoint,
                                                    config=config, checkpoint=checkpoint, device=device)
                checkpoint_report["checkpoint_source"] = checkpoint_report["checkpoint"]
                checkpoint_report["checkpoint"] = checkpoint
                checkpoint_report["config_source"] = str(confined_path(args.config, must_exist=True))
                checkpoint_report["artifacts"]["metrics"] = str(report_root / checkpoint / "metrics.json")
                checkpoint_report["training_history"] = training.get("history", [])
                reports.append(checkpoint_report)
            reference_metrics = json.loads(confined_path(args.reference, must_exist=True).read_text(encoding="utf-8"))
            reports.append({
                "checkpoint": "ordinary feedback reference",
                "success_fraction": reference_metrics["success_fraction"],
                "deadline_error_median_m": reference_metrics["deadline_error_median_m"],
                "artifacts": reference_metrics["artifacts"],
                "config_source": str(confined_path(args.config, must_exist=True)),
            })
            imperfect_metrics = json.loads(
                confined_path(args.imperfect_reference, must_exist=True).read_text(encoding="utf-8")
            )
            reports.append({
                "checkpoint": "imperfect sensor/motor reference",
                "success_fraction": imperfect_metrics["success_fraction"],
                "deadline_error_median_m": imperfect_metrics["deadline_error_median_m"],
                "artifacts": imperfect_metrics["artifacts"],
                "config_source": str(confined_path(args.imperfect_config, must_exist=True)),
            })
            html_output = output if output.suffix else output / "learning.html"
            export_learning_html(reports, html_output)
            gif_output = html_output.with_suffix(".gif")
            export_learning_gif(reports, gif_output)
            baseline_scores = np.asarray([item["command_slew"] for item in reports[0]["scores"]], dtype=np.float64)
            candidate_scores = np.asarray([item["command_slew"] for item in reports[-2]["scores"]], dtype=np.float64)
            comparison = compare_smoothness(reports[0]["success_fraction"], reports[-2]["success_fraction"],
                                             baseline_scores, candidate_scores)
            comparison_result = {"output": str(html_output), "gif": str(gif_output), "checkpoints": reports, "device": device,
                                 "smoothness": comparison.__dict__}
            print(json.dumps(comparison_result, indent=2))
            return 0
        if args.command == "evaluate-adaptation":
            config = load_config(args.config)
            report = run_adaptation_suite(args.run, args.output, config=config,
                                          device=args.device, seeds=args.seeds,
                                          memoryless_run=args.memoryless_run)
            print(json.dumps({"output": str(args.output), "variants": report["variants"],
                              "seed_count": report["seed_count"]}, indent=2))
            return 0
    except (ConfigurationError, OSError) as exc:
        print(f"robot-ai: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
