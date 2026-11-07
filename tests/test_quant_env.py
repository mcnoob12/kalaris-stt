import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "envs" / "quant_env"))

from models import QuantAction  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from server.evaluator import Metrics, StubEvaluator  # noqa: E402
from server.quant_env_environment import QuantEnvironment  # noqa: E402


def finish_episode(env, bits):
    env.reset()
    observation = None
    for _ in StubEvaluator.groups:
        observation = env.step(QuantAction(bits=bits))
    return observation


def test_episode_visits_every_group_then_ends():
    env = QuantEnvironment()
    observation = env.reset()
    assert observation.current_group == "encoder_early"
    assert not observation.done
    seen = []
    while not observation.done:
        seen.append(observation.current_group)
        observation = env.step(QuantAction(bits=8))
    assert seen == StubEvaluator.groups
    assert observation.current_group is None
    assert set(observation.policy) == set(StubEvaluator.groups)


def test_metrics_and_reward_only_arrive_at_the_end():
    env = QuantEnvironment()
    env.reset()
    middle = env.step(QuantAction(bits=8))
    assert middle.reward == 0.0
    assert middle.wer is None


def test_observations_say_the_numbers_are_from_a_stub():
    assert QuantEnvironment().reset().evaluator == "stub"


def test_oversized_policy_is_punished():
    env = QuantEnvironment()
    assert finish_episode(env, 16).reward < finish_episode(env, 2).reward


def test_no_extra_reward_for_beating_the_size_target():
    class Fixed:
        name = "fixed"
        groups = ["only"]
        baseline_wer = 5.0

        def __init__(self, size_mb):
            self.size_mb = size_mb

        def evaluate(self, policy):
            return Metrics(wer=5.0, size_mb=self.size_mb, rtfx=50.0)

    def reward(size_mb):
        env = QuantEnvironment(evaluator=Fixed(size_mb))
        env.reset()
        return env.step(QuantAction(bits=8)).reward

    assert reward(50.0) == reward(200.0) == 0.0
    assert reward(400.0) < 0


def test_slow_policy_is_punished():
    class Slow:
        name = "slow"
        groups = ["only"]
        baseline_wer = 5.0

        def evaluate(self, policy):
            return Metrics(wer=5.0, size_mb=100.0, rtfx=5.0)

    env = QuantEnvironment(evaluator=Slow())
    env.reset()
    assert env.step(QuantAction(bits=8)).reward == pytest.approx(-2.5)


def test_invalid_bit_width_is_rejected():
    with pytest.raises(ValidationError):
        QuantAction(bits=3)


def test_stepping_a_finished_episode_fails_until_reset():
    env = QuantEnvironment()
    finish_episode(env, 8)
    with pytest.raises(RuntimeError):
        env.step(QuantAction(bits=8))
    assert not env.reset().done


def test_reset_clears_the_policy():
    env = QuantEnvironment()
    env.reset()
    env.step(QuantAction(bits=8))
    assert env.reset().policy == {}
