# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from robot_ai.control.c_export import export_c
from robot_ai.sim.reach_env import COMMAND_RATE, OBSERVATION_DIM, PRIVILEGED_DIM, ReachEnv
from robot_ai.train.reach import (
    ORACLE_CONDITION,
    Actor,
    PidController,
    distill,
    evaluate,
    load_actor,
    train,
)


def test_reach_env_is_reproducible_and_hides_the_robot() -> None:
    first, second = ReachEnv(16, device="cpu", seed=4), ReachEnv(16, device="cpu", seed=4)
    a, b = first.reset(), second.reset()
    assert a.shape == (16, OBSERVATION_DIM) and torch.equal(a, b)
    assert first.privileged(a).shape == (16, PRIVILEGED_DIM)
    for _ in range(5):
        a, reward = first.step(torch.zeros(16, 2))
        b, _ = second.step(torch.zeros(16, 2))
    assert torch.equal(a, b) and torch.isfinite(reward).all()


def test_tuned_pid_masters_healthy_arms_but_not_defective_ones() -> None:
    pid = (15.0, 30.0, 1.5, 0.3, 0.16)
    device = torch.device("cpu")
    healthy = evaluate(PidController(pid, device), worlds=256, device="cpu", severity=0.0, changes=False)
    defective = evaluate(PidController(pid, device), worlds=256, device="cpu")
    assert healthy["success"] > 0.9
    assert defective["success"] < 0.5


def test_training_runs_and_saves_a_loadable_actor(tmp_path: Path) -> None:
    for recurrent in (True, False):
        output = tmp_path / str(recurrent)
        train(recurrent=recurrent, output=output, device="cpu", worlds=32, iterations=1, seed=0,
              minibatch=32, eval_every=1, eval_worlds=32)
        record = json.loads((output / "log.jsonl").read_text().splitlines()[-1])
        assert "eval" in record and record["robot_ticks"] == 32 * 300
        assert load_actor(output, torch.device("cpu")).recurrent == recurrent


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
@pytest.mark.parametrize(("recurrent", "incremental", "fine_error", "history"),
                         [(True, False, False, 0), (False, False, False, 0), (True, True, False, 0), (True, False, True, 0),
                          (True, False, True, 3)])
def test_exported_c_controller_matches_pytorch(tmp_path: Path, recurrent: bool, incremental: bool, fine_error: bool,
                                               history: int) -> None:
    torch.manual_seed(3)
    actor = Actor(recurrent=recurrent, incremental=incremental, fine_scales=(0.01, 0.05, 0.5) if fine_error else (),
                  history=history)
    with torch.no_grad():
        actor.head.weight.normal_(std=0.5)  # the fresh head is near zero; make the comparison meaningful
    observations = torch.randn(40, OBSERVATION_DIM)
    harness = """
#include <stdio.h>
int main(void) {
  float feeling[H + K * S] = {0}, observation[O], command[2] = {0};
  for (;;) {
    for (int i = 0; i < O; i++) if (scanf("%f", &observation[i]) != 1) return 0;
    reach_policy_step(observation, feeling, command);
    printf("%.9g %.9g\\n", command[0], command[1]);
  }
}
"""
    source = tmp_path / "policy.c"
    source.write_text(export_c(actor) + harness)
    binary = tmp_path / "policy"
    subprocess.run(["cc", "-O2", "-o", str(binary), str(source), "-lm"], check=True)
    text = "\n".join(" ".join(f"{value:.9g}" for value in row) for row in observations.tolist())
    result = subprocess.run([str(binary)], input=text, capture_output=True, text=True, check=True)
    compiled = torch.tensor([[float(value) for value in line.split()] for line in result.stdout.splitlines()])
    feeling = actor.initial(1, torch.device("cpu"))
    expected = []
    command = torch.zeros(2)
    with torch.no_grad():
        for row in observations:
            mean, feeling = actor(row[None, None], feeling)
            change = mean[0, 0].clamp(-1.0, 1.0)
            command = (command + COMMAND_RATE * change).clamp(-1.0, 1.0) if incremental else change
            expected.append(command)
    assert (compiled - torch.stack(expected)).abs().max() < 1e-5


def test_condition_oracle_teaches_a_deployable_student(tmp_path: Path) -> None:
    cpu = torch.device("cpu")
    small = {"device": "cpu", "worlds": 32, "iterations": 1, "seed": 0, "minibatch": 32, "eval_every": 1, "eval_worlds": 32}
    train(recurrent=True, fine_scales=(0.05, 0.5), oracle=ORACLE_CONDITION, output=tmp_path / "teacher", **small)
    teacher = load_actor(tmp_path / "teacher", cpu)
    seen = torch.randn(1, 3, PRIVILEGED_DIM)
    moved = seen.clone()
    moved[..., OBSERVATION_DIM:OBSERVATION_DIM + 4] += 5.0  # the true joint state, which this oracle must not read
    assert torch.allclose(teacher(seen, teacher.initial(3, cpu))[0], teacher(moved, teacher.initial(3, cpu))[0])
    with pytest.raises(ValueError, match="cannot be deployed"):
        export_c(teacher)

    distill(teacher=tmp_path / "teacher", output=tmp_path / "student", **small)
    student = load_actor(tmp_path / "student", cpu)
    assert not student.oracle and student.insight is not None
    assert "reach_policy_step" in export_c(student)
    with pytest.raises(ValueError, match="must be an oracle"):
        distill(teacher=tmp_path / "student", output=tmp_path / "again", **small)


def test_history_window_is_the_same_whether_fed_one_tick_or_a_chunk_at_a_time() -> None:
    torch.manual_seed(5)
    actor = Actor(recurrent=True, history=4, hidden=16)
    with torch.no_grad():
        actor.head.weight.normal_(std=0.5)
    observations = torch.randn(30, 7, OBSERVATION_DIM)
    feeling = actor.initial(7, torch.device("cpu"))
    with torch.no_grad():
        chunked, _ = actor(observations, feeling)
        stepped = []
        for row in observations:
            mean, feeling = actor(row[None], feeling)
            stepped.append(mean[0])
    assert torch.allclose(chunked, torch.stack(stepped), atol=1e-6)
    assert torch.equal(feeling[:, 16:], observations[-4:, :, :6].transpose(0, 1).reshape(7, -1))
