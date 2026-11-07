"""Client for the quantization search environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import QuantAction, QuantObservation


class QuantEnv(EnvClient[QuantAction, QuantObservation, State]):
    def _step_payload(self, action: QuantAction) -> Dict:
        return {"bits": action.bits}

    def _parse_result(self, payload: Dict) -> StepResult[QuantObservation]:
        fields = payload.get("observation", {})
        observation = QuantObservation(
            **fields,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
