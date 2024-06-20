from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from sherpa_ai.actions.base import BaseAction
from sherpa_ai.agents.base import BaseAgent


class FlowConnection(BaseModel):
    source: FlowNode
    target: FlowNode
    condition: Optional[str] = None


class FlowNode(BaseModel, ABC):
    name: str
    incoming_connections: list[FlowConnection] = []
    outgoing_connections: list[FlowConnection] = []

    @abstractmethod
    def execute(self, **kwargs):
        pass


class StartNode(FlowNode):
    name: str = "Start"

    def execute(self, **kwargs):
        pass


class EndNode(FlowNode):
    name: str = "End"

    def execute(self, **kwargs):
        pass


class DecisionNode(FlowNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    decision_maker: BaseAgent

    def execute(self, **kwargs):
        pass


class LoopNode(DecisionNode):
    outgoing_connections: list[FlowConnection] = Field(default=[], max_length=2)


class ActionNode(FlowNode):
    outgoing_connections: list[FlowConnection] = Field(default=[], max_length=1)
    action: BaseAction

    def execute(self, **kwargs):
        pass
