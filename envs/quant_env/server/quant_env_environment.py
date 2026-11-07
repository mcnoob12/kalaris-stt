"""Per-layer quantization search as an OpenEnv environment.

Each episode assigns a bit width to every layer group in turn. The final step
evaluates the whole policy and pays the reward:

    reward = -(WER increase in points)
             - size_weight * (fraction over the size target)
             - speed_weight * (fraction under the speed target)

Being under the size target or over the speed target earns nothing extra, so a
policy can't trade accuracy for gains the product doesn't need.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import QuantAction, QuantObservation
    from .evaluator import Evaluator, StubEvaluator
except ImportError:
    from models import QuantAction, QuantObservation
    from server.evaluator import Evaluator, StubEvaluator


class QuantEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        target_size_mb: float = 200.0,
        target_rtfx: float = 10.0,
        size_weight: float = 5.0,
        speed_weight: float = 5.0,
    ):
        self._evaluator = evaluator or StubEvaluator()
        self._target_size_mb = target_size_mb
        self._target_rtfx = target_rtfx
        self._size_weight = size_weight
        self._speed_weight = speed_weight
        self._policy: dict[str, int] = {}
        self._finished = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def _remaining(self) -> list[str]:
        return [group for group in self._evaluator.groups if group not in self._policy]

    def _observation(self, **fields) -> QuantObservation:
        remaining = self._remaining()
        return QuantObservation(
            current_group=remaining[0] if remaining else None,
            remaining_groups=remaining,
            policy=dict(self._policy),
            evaluator=self._evaluator.name,
            **fields,
        )

    def reset(self) -> QuantObservation:
        self._policy = {}
        self._finished = False
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return self._observation(done=False, reward=0.0)

    def step(self, action: QuantAction) -> QuantObservation:  # type: ignore[override]
        if self._finished:
            raise RuntimeError("Episode is finished. Call reset() first.")

        self._state.step_count += 1
        self._policy[self._remaining()[0]] = action.bits
        if self._remaining():
            return self._observation(done=False, reward=0.0)

        self._finished = True
        metrics = self._evaluator.evaluate(self._policy)
        wer_cost = metrics.wer - self._evaluator.baseline_wer
        over_size = max(0.0, metrics.size_mb / self._target_size_mb - 1.0)
        under_speed = max(0.0, 1.0 - metrics.rtfx / self._target_rtfx)
        reward = -wer_cost - self._size_weight * over_size - self._speed_weight * under_speed
        return self._observation(
            done=True,
            reward=round(reward, 4),
            wer=metrics.wer,
            size_mb=metrics.size_mb,
            rtfx=metrics.rtfx,
        )

    @property
    def state(self) -> State:
        return self._state
