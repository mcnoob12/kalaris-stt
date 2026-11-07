"""Actions and observations for the quantization search environment.

Only bit choices and metrics cross this boundary, never audio.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class QuantAction(Action):
    """Assign a bit width to the layer group the environment is asking about."""

    bits: Literal[2, 4, 8, 16] = Field(..., description="Bits per weight for the current layer group")


class QuantObservation(Observation):
    current_group: str | None = Field(default=None, description="Layer group awaiting a choice; None when finished")
    remaining_groups: list[str] = Field(default_factory=list)
    policy: dict[str, int] = Field(default_factory=dict, description="Choices made so far")
    wer: float | None = Field(default=None, description="Dev WER in percent; set only at the end of an episode")
    size_mb: float | None = Field(default=None, description="Model size in MB; set only at the end of an episode")
    rtfx: float | None = Field(default=None, description="Speed as multiple of realtime; set only at the end")
    evaluator: str = Field(default="", description="Which evaluator produced the metrics; 'stub' means not real")
