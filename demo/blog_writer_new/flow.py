from unittest.mock import MagicMock

from actions import get_action_map
from langchain.chat_models import ChatOpenAI
from share_memory import DictSharedMemory

from sherpa_ai.agents.base import BaseAgent
from sherpa_ai.agents.user import UserAgent
from sherpa_ai.memory import Belief, SharedMemory
from sherpa_ai.policies.flow_policy import FlowPolicy


def get_flow_policy(action_map, agent) -> FlowPolicy:
    policy = FlowPolicy()

    policy.add_decision_node("start_session", agent)
    policy.add_action_node("chunk_document", action_map["chunk_document"])
    policy.add_action_node("read_outlines", action_map["read_outlines"])
    policy.add_action_node("generate_insight", action_map["generate_insight"])
    policy.add_action_node("generate_outline", action_map["generate_outline"])
    policy.add_decision_node("iterate_outlines", agent)
    policy.add_decision_node("choose_action", agent)
    policy.add_action_node("google_search", action_map["google_search"])
    policy.add_action_node("write", action_map["write"])
    policy.add_action_node("next_outline", action_map["next_outline"])
    policy.add_action_node("write_file", action_map["write_file"])

    policy.add_connection("Start", "start_session")
    policy.add_connection("start_session", "chunk_document")
    policy.add_connection("start_session", "read_outlines")
    policy.add_connection("chunk_document", "generate_insight")
    policy.add_connection("generate_insight", "generate_outline")

    policy.add_connection("generate_outline", "iterate_outlines")
    policy.add_connection("read_outlines", "iterate_outlines")

    policy.add_connection("iterate_outlines", "choose_action")
    policy.add_connection("choose_action", "google_search")
    policy.add_connection("choose_action", "write")
    policy.add_connection("choose_action", "next_outline")
    policy.add_connection("next_outline", "iterate_outlines")
    policy.add_connection("write", "iterate_outlines")
    policy.add_connection("google_search", "iterate_outlines")

    policy.add_connection("iterate_outlines", "write_file")
    policy.add_connection("write_file", "start_session")
    policy.add_connection("start_session", "End")

    return policy


shared_memory = DictSharedMemory(objective="")
agent = UserAgent(name="user", description="User agent", shared_memory=shared_memory)
policy = get_flow_policy(get_action_map(), agent)
belief = Belief()
llm = ChatOpenAI()

while True:
    policy_output = policy.select_action(belief, llm=llm)
    output = policy_output.action.execute(shared_memory.data, belief=belief)
    shared_memory.data[policy_output.action.name] = output
    if policy_output is None:
        break
