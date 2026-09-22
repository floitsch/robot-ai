"""Train and compare reach controllers on populations of imperfect arms.

Three controllers face the same held-out robots, goals, and mid-episode changes:

* a PID with nominal gravity compensation, its gains grid-searched on defective robots;
* a memoryless network that sees only the current measurements;
* a recurrent network whose hidden state can accumulate a feeling for this particular robot.

Both networks are trained by PPO with a critic that may see hidden simulator truth;
the actors never do.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import argparse
import itertools
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from ..sim.chain_env import ChainEnv, ChainSummary
from ..sim.population import NOMINAL_LENGTHS, NOMINAL_MASSES, NOMINAL_TORQUES
from ..sim.reach_env import PRIVILEGED_DIM, EpisodeSummary, ReachEnv

EVAL_SEED = 987_654_321
# A 30 mrad error is a 0.03 input among inputs of order one: too faint to steer by. Fine-error
# actors also see it magnified and saturated at these scales, rad.
FINE_ERROR_SCALES = (0.05, 0.5)
ORACLE_FULL, ORACLE_CONDITION = 1, 2
# Per joint: the observation holds 6 blocks (q, dq, current, goal, error, previous command); the privileged
# part starts with true q and dq. The single arm has 2 joints; chains have more.
OBSERVATION_BLOCKS = 6
SENSOR_BLOCKS = 3  # measured q, derived dq, motor current lead the observation


class PidController:
    """Industrial-style joint controller: minimum-jerk reference, PID with velocity
    feed-forward and anti-windup, gravity compensation from the nominal model."""

    def __init__(self, gains: Sequence[float], device: torch.device) -> None:
        """gains = (kp, ki, kd, reference seconds per radian, joint-two gain ratio)."""

        kp, ki, kd, seconds_per_rad, joint2_ratio = gains
        ratio = torch.tensor([1.0, joint2_ratio], device=device)  # joint two carries far less inertia
        self.kp, self.ki, self.kd = kp * ratio, ki * ratio, kd * ratio
        self.seconds_per_rad = seconds_per_rad
        self.torque = torch.as_tensor(NOMINAL_TORQUES, dtype=torch.float32, device=device)
        l1, l2 = NOMINAL_LENGTHS
        m1, m2 = NOMINAL_MASSES
        self.g1, self.g2 = 9.81 * (m1 * l1 / 2 + m2 * l1), 9.81 * m2 * l2 / 2
        self.goal: Tensor | None = None

    def act(self, env: ReachEnv) -> Tensor:
        goal, q = env.goal(), env.measured_q
        if self.goal is None:
            self.goal, self.origin = torch.full_like(goal, torch.nan), q.clone()
            self.elapsed, self.duration = torch.zeros_like(q[:, :1]), torch.ones_like(q[:, :1])
            self.integral = torch.zeros_like(q)
        moved = (goal != self.goal).any(dim=1, keepdim=True)  # a new goal starts a new reference from here
        self.origin = torch.where(moved, q, self.origin)
        self.elapsed = torch.where(moved, torch.zeros_like(self.elapsed), self.elapsed + 0.01)
        span = (goal - self.origin).abs().amax(dim=1, keepdim=True)
        self.duration = torch.where(moved, (self.seconds_per_rad * span).clamp(min=0.25), self.duration)
        self.goal = goal.clone()
        phase = (self.elapsed / self.duration).clamp(0.0, 1.0)
        blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
        rate = 30.0 * phase**2 * (1.0 - phase) ** 2 / self.duration
        reference = self.origin + (goal - self.origin) * blend
        reference_velocity = (goal - self.origin) * rate
        error = reference - q
        self.integral = torch.where(error.abs() < 0.15, (self.integral + 0.01 * error).clamp(-0.2, 0.2), self.integral)
        hold2 = self.g2 * torch.sin(q[:, 0] + q[:, 1])
        gravity = torch.stack((self.g1 * torch.sin(q[:, 0]) + hold2, hold2), dim=1)
        torque = self.kp * error + self.ki * self.integral + self.kd * (reference_velocity - env.velocity) + gravity
        return (torque / self.torque).clamp(-1.0, 1.0)


class Actor(nn.Module):
    incremental: Tensor
    fine_scales: Tensor
    oracle: Tensor
    history: Tensor
    joints: Tensor
    privileged_dim: Tensor

    def __init__(self, *, recurrent: bool, incremental: bool = False, fine_scales: Sequence[float] = (), hidden: int = 64,
                 oracle: int = 0, insight: bool = False, history: int = 0, joints: int = 2,
                 privileged_dim: int = PRIVILEGED_DIM, initial_std: Sequence[float] | None = None,
                 extra_inputs: int = 0, message: int = 0) -> None:
        super().__init__()
        self.recurrent, self.hidden = recurrent, hidden
        # The joint count fixes every input layout; the privileged width depends on the task's hidden fields.
        self.register_buffer("joints", torch.tensor(joints))
        self.register_buffer("privileged_dim", torch.tensor(privileged_dim))
        self.n = joints
        self.observation_dim = OBSERVATION_BLOCKS * joints
        self.sensor_dim = SENSOR_BLOCKS * joints
        # The last `history` raw sensor readings ride along in the recurrent state, so delayed and noisy
        # encoders can be read as a short window rather than one sample at a time.
        self.register_buffer("history", torch.tensor(history))
        # An oracle is told the simulator's hidden truth and cannot be deployed. ORACLE_FULL also sees the true
        # joint state, which bounds what perfect sensing could buy. ORACLE_CONDITION knows only the robot's hidden
        # condition and must act through the same imperfect sensors as a real controller, so its strategy is
        # one a deployable student can actually reproduce.
        self.register_buffer("oracle", torch.tensor(int(oracle)))
        # Saved with the weights: the scales are part of what the encoder was trained to read.
        self.register_buffer("fine_scales", torch.tensor(list(fine_scales), dtype=torch.float32))
        inputs = (privileged_dim if oracle else self.observation_dim) + joints * len(fine_scales) + self.sensor_dim * history
        inputs += extra_inputs  # e.g. messages from other limbs, appended by the caller after the observation
        self.extra_inputs = extra_inputs
        # A message for other limbs, computed from the same features that drive the motors.
        self.message = nn.Sequential(nn.Linear(hidden, message), nn.Tanh()) if message else None
        # Incremental actors output command changes; saved with the weights so deployment cannot mix them up.
        self.register_buffer("incremental", torch.tensor(incremental))
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.core: nn.Module = nn.GRU(hidden, hidden) if recurrent else nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        # Training-only head: from the same features that drive the motors, name the true joint state and the
        # robot's hidden condition. It forces the feeling to filter noisy sensors and to identify the robot,
        # and it is the seed of a "this joint hurts" readout. Deployment ignores it.
        self.insight = nn.Linear(hidden, privileged_dim - self.observation_dim) if insight else None
        # Two command means and two log standard deviations: the policy explores boldly
        # while moving and can go quiet to hold still, which fixed noise never allows.
        self.head = nn.Linear(hidden, 2 * joints)
        nn.init.normal_(self.head.weight, std=0.01)
        # Exploration starts at half of rated torque per joint unless the task says otherwise: a joint rated for a
        # heavy load must not be shaken with the same fraction of its torque as a light one.
        stds = [0.5] * joints if initial_std is None else [float(v) for v in initial_std]
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.0] * joints + [float(torch.log(torch.tensor(v))) for v in stds]))

    @property
    def state_dim(self) -> int:
        return self.hidden + self.sensor_dim * int(self.history)

    def initial(self, worlds: int, device: torch.device) -> Tensor:
        return torch.zeros((worlds, self.state_dim), device=device)

    def distribution(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """observations [time, worlds, dim], feeling [worlds, hidden] -> means, stds, next feeling."""

        mean, std, feeling, _ = self.distribution_and_features(observations, feeling)
        return mean, std, feeling

    def distribution_and_features(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        extra = observations[..., observations.shape[-1] - self.extra_inputs:] if self.extra_inputs else None
        if self.extra_inputs:
            observations = observations[..., :observations.shape[-1] - self.extra_inputs]
        if self.oracle == ORACLE_CONDITION:
            observations = observations.clone()
            observations[..., self.observation_dim:self.observation_dim + 2 * self.n] = 0.0
        if len(self.fine_scales):
            error = observations[..., 4 * self.n:5 * self.n]
            observations = torch.cat((observations, *(torch.tanh(error / scale) for scale in self.fine_scales)), dim=-1)
        hidden, window = feeling[:, :self.hidden], feeling[:, self.hidden:]
        if int(self.history):
            steps, worlds = observations.shape[:2]
            k = int(self.history)
            raw = observations[..., :self.sensor_dim]
            padded = torch.cat((window.reshape(worlds, k, self.sensor_dim).transpose(0, 1), raw))  # [k + steps, worlds, sensors]
            index = torch.arange(steps, device=raw.device)[:, None] + torch.arange(k, device=raw.device)[None, :]
            stacked = padded[index].permute(0, 2, 1, 3).reshape(steps, worlds, k * self.sensor_dim)  # oldest first
            observations = torch.cat((observations, stacked), dim=-1)
            window = padded[steps:].transpose(0, 1).reshape(worlds, k * self.sensor_dim)
        if extra is not None:
            observations = torch.cat((observations, extra), dim=-1)
        encoded = self.encoder(observations)
        if self.recurrent:
            features, final = self.core(encoded, hidden[None].contiguous())
            hidden = final[0]
        else:
            features = self.core(encoded)
        feeling = torch.cat((hidden, window), dim=-1)
        output = self.head(features)
        return output[..., :self.n], output[..., self.n:].clamp(-4.0, 0.0).exp(), feeling, features

    def forward(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor]:
        mean, _, feeling = self.distribution(observations, feeling)
        return mean, feeling


def load_actor(run: Path, device: torch.device) -> Actor:
    state = torch.load(run / "actor.pt", map_location=device, weights_only=True)
    if "limbs" in state:
        from .limbs import LimbPolicy

        policy = LimbPolicy(limbs=int(state["limbs"]), message=int(state["message_dim"]),
                            hidden=state["actor.head.weight"].shape[1], fine_scales=state["actor.fine_scales"].tolist(),
                            oracle=int(state["actor.oracle"])).to(device)
        policy.load_state_dict(state)
        return policy.eval()  # type: ignore[return-value]
    state.setdefault("incremental", torch.tensor(False))
    if "fine_scales" not in state:  # checkpoints from before the scales were stored
        state["fine_scales"] = torch.tensor(FINE_ERROR_SCALES if bool(state.pop("fine_error", False)) else ())
    state.setdefault("oracle", torch.tensor(False))
    state.setdefault("history", torch.tensor(0))
    state.setdefault("joints", torch.tensor(2))
    state.setdefault("privileged_dim", torch.tensor(PRIVILEGED_DIM))
    actor = Actor(recurrent="core.weight_ih_l0" in state, incremental=bool(state["incremental"]),
                  fine_scales=state["fine_scales"].tolist(), hidden=state["head.weight"].shape[1],
                  oracle=int(state["oracle"]), insight="insight.weight" in state, history=int(state["history"]),
                  joints=int(state["joints"]), privileged_dim=int(state["privileged_dim"])).to(device)
    actor.load_state_dict(state)
    return actor.eval()


def env_privileged_dim(env: object) -> int:
    return int(getattr(env, "privileged_dim", PRIVILEGED_DIM))


def make_critic(privileged_dim: int = PRIVILEGED_DIM) -> nn.Module:
    return nn.Sequential(nn.Linear(privileged_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))


def _metrics(summary: EpisodeSummary | ChainSummary) -> dict[str, float]:
    result = {"success": summary.success.float().mean().item(), "final_error_mrad": 1000 * summary.final_error.median().item(),
              "mean_error_rad": summary.mean_error.mean().item(), "roughness": summary.roughness.mean().item(),
              "return": summary.episode_return.mean().item()}
    if isinstance(summary, EpisodeSummary):
        result["limit_time"] = summary.limit_time.mean().item()
    else:
        for limb, value in enumerate(summary.limb_success.float().mean(dim=0).tolist()):
            result[f"limb{limb}_success"] = value
    return result


@torch.no_grad()
def evaluate(controller: PidController | Actor, *, worlds: int, device: str, severity: float = 1.0,
             changes: bool = True, pushes: bool = False, seed: int = EVAL_SEED, limbs: int = 0) -> dict[str, float]:
    if limbs == 0 and isinstance(controller, nn.Module) and controller.n != 2:
        limbs = controller.n // 2
    env = make_env(worlds, device=device, seed=seed, limbs=limbs, severity=severity, changes=changes, pushes=pushes)
    observation = env.reset()
    feeling = controller.initial(worlds, env.torch_device) if isinstance(controller, nn.Module) else None
    for _ in range(env.episode_ticks):
        if isinstance(controller, nn.Module):
            seen = env.privileged(observation) if controller.oracle else observation
            mean, feeling = controller(seen[None], feeling)
            action = env.integrate(mean[0]) if controller.incremental else mean[0]
        else:
            action = controller.act(env)
        observation, _ = env.step(action)
    return _metrics(env.summary())


def tune_pid(*, worlds: int, device: str) -> tuple[tuple[float, ...], dict[str, float]]:
    """Best single set of gains for the defective population, by the task's own return."""

    best: tuple[float, tuple[float, ...], dict[str, float]] | None = None
    torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
    for gains in itertools.product((8.0, 15.0, 30.0, 45.0), (30.0, 100.0, 300.0), (0.4, 0.8, 1.5, 3.0), (0.3, 0.5, 0.8), (0.08, 0.16, 0.3)):
        result = evaluate(PidController(gains, torch_device), worlds=worlds, device=device, seed=EVAL_SEED + 1)
        if best is None or result["return"] > best[0]:
            best = (result["return"], gains, result)
    assert best is not None
    return best[1], best[2]


def make_env(worlds: int, *, device: str, seed: int, limbs: int = 0, **settings: object) -> ReachEnv:
    """The single arm, or a stacked chain of `limbs` two-joint limbs when `limbs` is given."""

    if limbs:
        allowed = {"severity", "changes", "reward_tolerance", "still_weight", "roughness_weight"}
        return ChainEnv(worlds, limbs, device=device, seed=seed, **{k: v for k, v in settings.items() if k in allowed})  # type: ignore[arg-type,return-value]
    return ReachEnv(worlds, device=device, seed=seed, **settings)  # type: ignore[arg-type]


def train(*, recurrent: bool, output: Path, device: str, worlds: int, iterations: int, seed: int, incremental: bool = False,
          fine_scales: Sequence[float] = (), hidden: int = 64, still_weight: float = 0.0,
          friction_probability: float = 0.7, oracle: int = 0, insight_weight: float = 0.0, mixed: bool = False,
          reward_tolerance: float = 0.03, pushes: bool = False, limbs: int = 0, roughness_weight: float | None = None,
          per_limb: bool = False, message: int = 0, severity: float = 1.0, initial: Path | None = None,
          chunk: int = 50, minibatch: int = 512, epochs: int = 4, gamma: float = 0.99, lam: float = 0.95,
          clip: float = 0.2, entropy: float = 0.002, learning_rate: float = 3e-4, eval_every: int = 10, eval_worlds: int = 4096) -> None:
    torch.manual_seed(seed)
    # Only explicitly set weights are passed on, so each task keeps its own defaults.
    weights = {k: v for k, v in (("still_weight", still_weight or None), ("roughness_weight", roughness_weight)) if v is not None}
    env = make_env(worlds, device=device, seed=seed, limbs=limbs, friction_probability=friction_probability, mixed=mixed,
                   reward_tolerance=reward_tolerance, pushes=pushes, severity=severity, **weights)
    dev = env.torch_device
    actor: Actor
    if initial:
        # Curriculum: continue a policy trained under easier conditions, e.g. healthy robots before defective ones.
        actor = load_actor(initial, dev).train()
        if actor.n != getattr(env, "n", 2):
            raise ValueError("the initial policy was trained for a different joint count")
    elif per_limb:
        from .limbs import LimbPolicy

        if not limbs:
            raise ValueError("--per-limb needs --limbs")
        actor = LimbPolicy(limbs=limbs, message=message, hidden=hidden, fine_scales=fine_scales, oracle=oracle,
                           initial_std=getattr(env, "initial_std", None)).to(dev)  # type: ignore[assignment]
    else:
        actor = Actor(recurrent=recurrent, incremental=incremental, fine_scales=fine_scales, hidden=hidden, oracle=oracle,
                      insight=insight_weight > 0, joints=getattr(env, "n", 2), privileged_dim=env_privileged_dim(env),
                      initial_std=getattr(env, "initial_std", None)).to(dev)
    critic = make_critic(env_privileged_dim(env)).to(dev)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=learning_rate)
    ticks = env.episode_ticks
    chunks = ticks // chunk
    observations = torch.zeros((ticks, worlds, int(actor.privileged_dim) if oracle else actor.observation_dim), device=dev)
    privileged = torch.zeros((ticks + 1, worlds, int(actor.privileged_dim)), device=dev)
    actions = torch.zeros((ticks, worlds, actor.n), device=dev)
    log_probs, rewards = torch.zeros((ticks, worlds), device=dev), torch.zeros((ticks, worlds), device=dev)
    feelings = torch.zeros((chunks, worlds, actor.state_dim), device=dev)
    offsets = torch.arange(chunk, device=dev)[:, None]
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "log.jsonl").open("a", encoding="utf-8")
    started = time.perf_counter()

    for iteration in range(1, iterations + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * (1.0 - 0.9 * (iteration - 1) / iterations)
        with torch.no_grad():
            observation = env.reset()
            feeling = actor.initial(worlds, dev)
            noise_level = 0.0
            for tick in range(ticks):
                if tick % chunk == 0:
                    feelings[tick // chunk] = feeling
                privileged[tick] = env.privileged(observation)
                observations[tick] = privileged[tick] if oracle else observation
                mean, std, feeling = actor.distribution(observations[tick][None], feeling)
                action = mean[0] + std[0] * torch.randn_like(mean[0])
                actions[tick] = action
                log_probs[tick] = torch.distributions.Normal(mean[0], std[0]).log_prob(action).sum(dim=1)
                noise_level += std.mean().item() / ticks
                applied = env.integrate(action) if incremental else action
                if isinstance(env, ChainEnv):
                    observation, rewards[tick] = env.step(applied, reference=mean[0])
                else:
                    observation, rewards[tick] = env.step(applied)
            privileged[ticks] = env.privileged(observation)
            values = torch.cat([critic(part).squeeze(-1) for part in privileged.split(chunk)])  # bounded activation memory
            advantages = torch.zeros_like(rewards)
            carry = torch.zeros(worlds, device=dev)
            for tick in reversed(range(ticks)):  # the episode end is a time limit, so bootstrap through it
                delta = rewards[tick] + gamma * values[tick + 1] - values[tick]
                carry = delta + gamma * lam * carry
                advantages[tick] = carry
            returns = advantages + values[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            rollout = _metrics(env.summary())

        for _ in range(epochs):
            order = torch.randperm(chunks * worlds, device=dev)
            for begin in range(0, len(order), minibatch):
                picked = order[begin:begin + minibatch]
                chunk_index, world_index = picked // worlds, picked % worlds
                time_index = chunk_index[None] * chunk + offsets
                world_grid = world_index[None].expand_as(time_index)
                mean, std, _, features = actor.distribution_and_features(observations[time_index, world_grid],
                                                                         feelings[chunk_index, world_index])
                policy = torch.distributions.Normal(mean, std)
                new_log_prob = policy.log_prob(actions[time_index, world_grid]).sum(dim=-1)
                ratio = (new_log_prob - log_probs[time_index, world_grid]).exp()
                advantage = advantages[time_index, world_grid]
                policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
                value_loss = (critic(privileged[time_index, world_grid]).squeeze(-1) - returns[time_index, world_grid]).square().mean()
                loss = policy_loss + 0.5 * value_loss - entropy * policy.entropy().sum(dim=-1).mean()
                if actor.insight is not None:
                    truth = privileged[time_index, world_grid][..., actor.observation_dim:]
                    insight_loss = (actor.insight(features) - truth).square().mean()
                    loss = loss + insight_weight * insight_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 1.0)
                optimizer.step()

        record: dict[str, object] = {"iteration": iteration, "elapsed_s": round(time.perf_counter() - started, 1),
                                     "robot_ticks": iteration * worlds * ticks, "std": noise_level,
                                     "value_loss": value_loss.item(), "rollout": rollout}
        if actor.insight is not None:
            record["insight_loss"] = insight_loss.item()
        if iteration % eval_every == 0 or iteration == iterations:
            record["eval"] = evaluate(actor, worlds=eval_worlds, device=device, limbs=limbs)
            torch.save(actor.state_dict(), output / "actor.pt")
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)


def distill(*, teacher: Path, output: Path, device: str, worlds: int, iterations: int, seed: int,
            initial: Path | None = None, fine_scales: Sequence[float] = FINE_ERROR_SCALES, hidden: int = 64,
            insight_weight: float = 1.0, noise: float = 0.05, mixed: bool = False, pushes: bool = False, history: int = 0,
            limbs: int = 0, per_limb: bool = False, message: int = 0, severity: float = 1.0, teacher_drive: float = 0.0, chunk: int = 50, minibatch: int = 512, epochs: int = 2,
            learning_rate: float = 1e-3, eval_every: int = 10, eval_worlds: int = 4096) -> None:
    """Teach a deployable recurrent actor to act like an oracle, using only what a real robot can sense.

    The student drives (so it is corrected on the states it actually reaches) while the oracle, fed the hidden
    truth of those same states, says what it would have done. The insight head is trained alongside.
    """

    torch.manual_seed(seed)
    env = make_env(worlds, device=device, seed=seed, limbs=limbs, mixed=mixed, pushes=pushes, severity=severity)
    dev = env.torch_device
    if str(teacher) == "computed-torque":
        # Classical teacher with perfect knowledge, reading the environment's truth directly.
        from ..control.computed_torque import ComputedTorqueTeacher

        if not isinstance(env, ChainEnv):
            raise ValueError("the computed-torque teacher drives chains; use --limbs (1 for a single arm)")
        oracle: Actor = ComputedTorqueTeacher(env).to(dev)  # type: ignore[assignment]
    else:
        oracle = load_actor(teacher, dev)
    if not oracle.oracle:
        raise ValueError("the teacher must be an oracle run")
    if oracle.n != getattr(env, "n", 2):
        raise ValueError("the teacher was trained for a different joint count")
    if initial:
        student = load_actor(initial, dev).train()
    elif per_limb:
        from .limbs import LimbPolicy

        student = LimbPolicy(limbs=limbs, message=message, hidden=hidden, fine_scales=fine_scales).to(dev)  # type: ignore[assignment]
    else:
        student = Actor(recurrent=True, fine_scales=fine_scales, hidden=hidden, insight=insight_weight > 0, history=history,
                        joints=oracle.n, privileged_dim=int(oracle.privileged_dim)).to(dev)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    ticks = env.episode_ticks
    chunks = ticks // chunk
    observations = torch.zeros((ticks, worlds, student.observation_dim), device=dev)
    truth = torch.zeros((ticks, worlds, int(student.privileged_dim) - student.observation_dim), device=dev)
    targets = torch.zeros((ticks, worlds, student.n), device=dev)
    feelings = torch.zeros((chunks, worlds, student.state_dim), device=dev)
    offsets = torch.arange(chunk, device=dev)[:, None]
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "log.jsonl").open("a", encoding="utf-8")
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * (1.0 - 0.9 * (iteration - 1) / iterations)
        with torch.no_grad():
            observation = env.reset()
            feeling, oracle_feeling = student.initial(worlds, dev), oracle.initial(worlds, dev)
            share = max(0.0, 1.0 - (iteration - 1) / (teacher_drive * iterations)) if teacher_drive > 0 else 0.0
            driven_by_teacher = torch.rand(worlds, device=dev) < share
            for tick in range(ticks):
                if tick % chunk == 0:
                    feelings[tick // chunk] = feeling
                privileged = env.privileged(observation)
                observations[tick], truth[tick] = observation, privileged[:, student.observation_dim:]
                wanted, oracle_feeling = oracle(privileged[None], oracle_feeling)
                targets[tick] = wanted[0].clamp(-1.0, 1.0)
                mean, feeling = student(observation[None], feeling)
                # DAgger mixing: early on the teacher drives some worlds, so the student first sees labels near the
                # teacher's own trajectories; the share decays to zero over the first `teacher_drive` of training.
                beta = max(0.0, 1.0 - (iteration - 1) / (teacher_drive * iterations)) if teacher_drive > 0 else 0.0
                executed = torch.where(driven_by_teacher[:, None], targets[tick], mean[0]) if beta > 0 else mean[0]
                observation, _ = env.step(executed + noise * torch.randn_like(mean[0]))
            rollout = _metrics(env.summary())
        for _ in range(epochs):
            order = torch.randperm(chunks * worlds, device=dev)
            for begin in range(0, len(order), minibatch):
                picked = order[begin:begin + minibatch]
                chunk_index, world_index = picked // worlds, picked % worlds
                time_index = chunk_index[None] * chunk + offsets
                world_grid = world_index[None].expand_as(time_index)
                mean, _, _, features = student.distribution_and_features(observations[time_index, world_grid],
                                                                         feelings[chunk_index, world_index])
                imitation = (mean - targets[time_index, world_grid]).square().mean()
                loss = imitation
                if student.insight is not None:
                    insight_loss = (student.insight(features) - truth[time_index, world_grid]).square().mean()
                    loss = loss + insight_weight * insight_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
        record: dict[str, object] = {"iteration": iteration, "elapsed_s": round(time.perf_counter() - started, 1),
                                     "robot_ticks": iteration * worlds * ticks, "std": noise,
                                     "imitation_loss": imitation.item(), "rollout": rollout}
        if iteration % eval_every == 0 or iteration == iterations:
            record["eval"] = evaluate(student.eval(), worlds=eval_worlds, device=device, limbs=limbs)
            student.train()
            torch.save(student.state_dict(), output / "actor.pt")
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)


def report(actors: dict[str, Path], *, device: str, worlds: int, output: Path) -> dict[str, object]:
    """Every controller on the same held-out robots, under increasingly hostile conditions."""

    torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
    gains, _ = tune_pid(worlds=worlds, device=device)
    controllers: dict[str, PidController | Actor] = {}
    for name, path in actors.items():
        controllers[name] = load_actor(path, torch_device)
    conditions = {"healthy": (0.0, False, False), "defective": (1.0, False, False),
                  "defective+changing": (1.0, True, False), "defective+changing+pushed (never trained on)": (1.0, True, True)}
    results: dict[str, object] = {"pid_gains": gains, "worlds": worlds}
    for condition, (severity, changes, pushes) in conditions.items():
        def score(controller: PidController | Actor, severity: float = severity, changes: bool = changes,
                  pushes: bool = pushes) -> dict[str, float]:
            return evaluate(controller, worlds=worlds, device=device, severity=severity, changes=changes, pushes=pushes)

        row = {"pid": score(PidController(gains, torch_device))}
        for name, actor in controllers.items():
            row[name] = score(actor)
        results[condition] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--memoryless", action="store_true")
    fit.add_argument("--incremental", action="store_true", help="output command changes instead of commands")
    fit.add_argument("--fine-error", type=float, nargs="*", default=None, metavar="RAD",
                     help="also feed the goal error magnified and saturated at these scales (default scales if none given)")
    fit.add_argument("--hidden", type=int, default=64)
    fit.add_argument("--limbs", type=int, default=0, help="train on a stacked chain of this many two-joint limbs")
    fit.add_argument("--severity", type=float, default=1.0, help="0 = healthy robots, 1 = the full defect population")
    fit.add_argument("--initial", type=Path, help="warm-start from this run (curriculum)")
    fit.add_argument("--reward-tolerance", type=float, default=0.03,
                     help="train against a stricter tolerance than the 0.03 rad success criterion")
    fit.add_argument("--insight-weight", type=float, default=0.0,
                     help="auxiliary loss: predict true joint state and hidden robot condition from the feeling")
    fit.add_argument("--oracle", choices=("full", "condition"),
                     help="undeployable reference: `full` sees true state and hidden condition, `condition` only the latter")
    fit.add_argument("--friction-probability", type=float, default=0.7,
                     help="train on a population where friction is rarer than in evaluation")
    fit.add_argument("--still-weight", type=float, default=0.0, help="graded reward for being on target and at rest")
    fit.add_argument("--roughness-weight", type=float, default=None, help="penalty on command changes (chain default 4)")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--iterations", type=int, default=300)
    fit.add_argument("--worlds", type=int, default=2048)
    fit.add_argument("--seed", type=int, default=1)
    teach = commands.add_parser("distill")
    teach.add_argument("--teacher", type=Path, required=True, help="oracle run directory, or `computed-torque`")
    teach.add_argument("--severity", type=float, default=1.0, help="0 = healthy robots, 1 = the full defect population")
    teach.add_argument("--teacher-drive", type=float, default=0.0,
                       help="fraction of training over which the teacher's share of driven worlds decays from 1 to 0 (DAgger)")
    teach.add_argument("--initial", type=Path, help="start the student from this run instead of from scratch")
    teach.add_argument("--output", type=Path, required=True)
    teach.add_argument("--iterations", type=int, default=200)
    teach.add_argument("--worlds", type=int, default=2048)
    teach.add_argument("--hidden", type=int, default=64)
    teach.add_argument("--seed", type=int, default=1)
    teach.add_argument("--history", type=int, default=0, help="raw sensor readings from the last N ticks as extra inputs")
    teach.add_argument("--limbs", type=int, default=0, help="distill on a stacked chain of this many two-joint limbs")
    for sub in (fit, teach):
        sub.add_argument("--minibatch", type=int, default=512, help="world-chunks per gradient step; halve it if the GPU runs out of memory")
        sub.add_argument("--per-limb", action="store_true", help="one shared two-joint policy per limb instead of one over all joints")
        sub.add_argument("--message", type=int, default=0, help="size of the message limbs exchange each tick (per-limb only)")

    for sub in (fit, teach):
        sub.add_argument("--mixed", action="store_true", help="train on robots from flawless to badly worn, some pushed around")
        sub.add_argument("--pushes", action="store_true", help="train with neighbour pushes at full severity")
    compare = commands.add_parser("report")
    compare.add_argument("--actor", action="append", default=[], metavar="NAME=RUN_DIR")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--worlds", type=int, default=4096)
    for sub in (fit, teach, compare):
        sub.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "train":
        train(recurrent=not args.memoryless, incremental=args.incremental, fine_scales=() if args.fine_error is None else (args.fine_error or FINE_ERROR_SCALES), hidden=args.hidden,
              still_weight=args.still_weight, roughness_weight=args.roughness_weight, friction_probability=args.friction_probability,
              oracle={None: 0, "full": ORACLE_FULL, "condition": ORACLE_CONDITION}[args.oracle],
              insight_weight=args.insight_weight, mixed=args.mixed, reward_tolerance=args.reward_tolerance,
              pushes=args.pushes, limbs=args.limbs, per_limb=args.per_limb, message=args.message,
              severity=args.severity, initial=args.initial, minibatch=args.minibatch,
              output=args.output, device=args.device, worlds=args.worlds,
              iterations=args.iterations, seed=args.seed)
    elif args.command == "distill":
        distill(teacher=args.teacher, initial=args.initial, output=args.output, device=args.device, worlds=args.worlds,
                iterations=args.iterations, hidden=args.hidden, seed=args.seed, mixed=args.mixed, pushes=args.pushes,
                history=args.history, limbs=args.limbs, per_limb=args.per_limb, message=args.message, severity=args.severity,
                minibatch=args.minibatch, teacher_drive=args.teacher_drive)
    else:
        actors = {name: Path(path) for name, path in (item.split("=", 1) for item in args.actor)}
        print(json.dumps(report(actors, device=args.device, worlds=args.worlds, output=args.output), indent=2))


if __name__ == "__main__":
    main()
