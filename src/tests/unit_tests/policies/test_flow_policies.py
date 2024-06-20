from unittest.mock import MagicMock

from loguru import logger

from sherpa_ai.actions.base import BaseAction
from sherpa_ai.agents.base import BaseAgent
from sherpa_ai.policies.flow_policy import FlowPolicy


def test_create_flow_policy():
    policy = FlowPolicy()
    action = MagicMock(name="dummy_action", spec=BaseAction)
    action.name = "dummy_action"
    agent = MagicMock(name="dummy_agent", spec=BaseAgent)
    agent.name = "dummy_agent"

    policy.add_action_node("action", action)
    policy.add_decision_node("decision", agent)
    policy.add_action_node("action2", action)
    policy.add_action_node("action3", action)
    policy.add_loop_node("loop", agent)

    policy.add_connection("Start", "action")
    policy.add_connection("action", "decision")
    policy.add_connection("decision", "action2")
    policy.add_connection("decision", "action3")
    policy.add_connection("action2", "loop")
    policy.add_connection("action3", "loop")
    policy.add_connection("loop", "End")
    policy.add_connection("loop", "decision")

    assert len(policy.flow_nodes) == 7  # 5 nodes + Start + End
    assert len(policy.flow_connections) == 8

    logger.info(policy.visualize())
