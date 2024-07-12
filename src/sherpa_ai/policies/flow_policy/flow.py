from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from sherpa_ai.actions.base import BaseAction
from sherpa_ai.actions.empty import EmptyAction
from sherpa_ai.agents.base import BaseAgent
from sherpa_ai.agents.user import UserAgent
from sherpa_ai.events import Event, EventType
from sherpa_ai.memory.belief import Belief
from sherpa_ai.policies.base import PolicyOutput

ACTION_OUTPUT_PROMPT = """
Based on the above context, find the input to the following action:
{action}

Output the input in JSON format as described below without any extra text.
Response Format:
{{"args": {{"arg name": "value"}}}}
Follow the described format strictly.
"""

feedback_prompt = """
You are an intelligent assistant helping the user to complete their task. You have the following task to complete:

{options}

Use polite and engaging language, and ask the user what they want you to help them next.  Not need to greet and keep it short and concise. 
"""


class FlowConnection(BaseModel):
    source: FlowNode
    target: FlowNode
    condition: Optional[str] = None


class FlowNode(BaseModel, ABC):
    name: str
    incoming_connections: list[FlowConnection] = []
    outgoing_connections: list[FlowConnection] = []

    @abstractmethod
    def execute(self, **kwargs) -> Tuple[Optional[PolicyOutput], Optional[FlowNode]]:
        pass


class StartNode(FlowNode):
    outgoing_connections: list[FlowConnection] = Field(default=[], max_length=1)
    name: str = "Start"

    def execute(self, **kwargs) -> Tuple[None, FlowNode]:
        logger.info("Starting the flow")
        return None, self.outgoing_connections[0].target


class EndNode(FlowNode):
    name: str = "End"
    action: EmptyAction = EmptyAction(name="End", args={}, usage="End the flow")

    def execute(self, **kwargs) -> Tuple[None, None]:
        logger.info("Ending the flow")
        return None, None


class DecisionNode(FlowNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    decision_maker: BaseAgent
    description: str
    action: EmptyAction = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action = EmptyAction(name=self.name, args={}, usage=self.description)

    def ask_for_feedback(self, options: str) -> str:
        prompt = feedback_prompt.format(options=options)

        question = self.decision_maker.llm.predict(prompt)

        feedback = input(question)

        return feedback

    def execute(
        self, context: str, state: dict, **kwargs
    ) -> Tuple[None, Optional[FlowNode]]:
        # logger.info(f"Executing decision node: {self.name}")
        # if isinstance(self.decision_maker, UserAgent):
        #     conditions = [
        #         f"{i}.{connection.target.name}"
        #         for i, connection in enumerate(self.outgoing_connections)
        #     ]
        #     condition_str = "\n".join(conditions)
        #     task = f"Based on the current context and conditions, select the best path to take. Select an integer (0-{len(conditions) - 1})\n{condition_str}"

        #     self.decision_maker.shared_memory.add(EventType.task, self.name, task)
        #     self.decision_maker.run()
        #     selection = self.decision_maker.shared_memory.events[-1].content
        #     selection = int(selection)
        #     return None, self.outgoing_connections[selection].target
        # else:

        actions = [
            connection.target.action for connection in self.outgoing_connections
        ]

        options = "\n".join([f"{i}. {action}" for i, action in enumerate(actions)])

        context = self.ask_for_feedback(options)

        belief = self.decision_maker.belief
        belief.set_actions(actions)
        belief.set_current_task(Event(EventType.task, "human", context))
        output = self.decision_maker.policy.select_action(belief)
        logger.info(f"Action selected: {output.action}, {output.args}")
        action_id = actions.index(output.action)
        state["action_args"] = output.args

        return (
            None,
            self.outgoing_connections[action_id].target,
        )


class LoopNode(DecisionNode):
    outgoing_connections: list[FlowConnection] = Field(default=[], max_length=2)


class ActionNode(FlowNode):
    outgoing_connections: list[FlowConnection] = Field(default=[], max_length=1)
    action: BaseAction
    prompt: str = ACTION_OUTPUT_PROMPT

    def execute(self, context: str, llm, **kwargs) -> PolicyOutput:
        logger.info(f"Executing action node: {self.name}")
        prompt = self.prompt.format(action=self.action)
        prompt = context + "\n\n" + prompt

        result = llm.predict(prompt)
        action_output = self.transform_output(result)

        return (
            PolicyOutput(action=self.action, args=action_output),
            self.outgoing_connections[0].target,
        )

    def transform_output(self, output_str: str) -> Tuple[str, dict]:
        try:
            return json.loads(output_str)
        except json.decoder.JSONDecodeError:
            logger.error("Output is not a proper json format {}", output_str)
            return {}
