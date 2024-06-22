from typing import Optional

from loguru import logger

from sherpa_ai.actions.base import BaseAction
from sherpa_ai.agents.base import BaseAgent
from sherpa_ai.memory import Belief
from sherpa_ai.policies.base import BasePolicy, PolicyOutput
from sherpa_ai.policies.flow_policy.flow import (ActionNode, DecisionNode,
                                                 EndNode, FlowConnection,
                                                 FlowNode, LoopNode, StartNode)


class FlowPolicy(BasePolicy):
    flow_nodes: dict[str, FlowNode] = {}
    flow_connections: list[FlowConnection] = []
    _current_node: Optional[FlowNode] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        start_node = StartNode()
        end_node = EndNode()

        self.flow_nodes[start_node.name] = start_node
        self.flow_nodes[end_node.name] = end_node
        self._current_node = start_node

    def select_action(self, belief: Belief, llm, **kwargs) -> Optional[PolicyOutput]:
        policy_output, next_node = self._current_node.execute(
            context="", llm=llm, **kwargs
        )

        while policy_output is None and next_node is not None:
            policy_output, next_node = next_node.execute(
                context="", llm=llm, **kwargs
            )

        self._current_node = next_node
        return policy_output

    def add_action_node(self, name: str, action: BaseAction):
        self.check_node_does_not_exist(name)
        self.flow_nodes[name] = ActionNode(name=name, action=action)

    def add_loop_node(self, name: str, decision_maker: BaseAgent):
        self.check_node_does_not_exist(name)
        self.flow_nodes[name] = LoopNode(name=name, decision_maker=decision_maker)

    def add_decision_node(self, name: str, decision_maker: BaseAgent):
        self.check_node_does_not_exist(name)
        self.flow_nodes[name] = DecisionNode(name=name, decision_maker=decision_maker)

    def check_node_does_not_exist(self, name: str):
        if name in self.flow_nodes:
            raise ValueError(f"Node with name {name} already exists")

    def add_connection(self, source: str, target: str, condition: str = None):
        if source not in self.flow_nodes:
            raise ValueError(f"Node with name {source} does not exist")
        if target not in self.flow_nodes:
            raise ValueError(f"Node with name {target} does not exist")

        source_node = self.flow_nodes[source]
        target_node = self.flow_nodes[target]

        connection = FlowConnection(
            source=source_node, target=target_node, condition=condition
        )

        source_node.outgoing_connections.append(connection)
        target_node.incoming_connections.append(connection)

        self.flow_connections.append(connection)

    def visualize(self) -> str:
        nodes = list(self.flow_nodes.values())
        return visualize_flow(nodes, self.flow_connections)


def visualize_nodes(nodes: list[FlowNode]) -> list[str]:
    results = []
    for node in nodes:
        node_id = "_".join(node.name.split())
        name = node.name
        if isinstance(node, DecisionNode):
            name = name + "\n" + node.decision_maker.name
            results.append(f'{node_id}{{"`{name}`"}}')
        else:
            results.append(f'{node_id}["`{name}`"]')

    return results


def visualize_edges(edges: list[FlowConnection]) -> list[str]:
    results = []
    for edge in edges:
        source_id = "_".join(edge.source.name.split())
        target_id = "_".join(edge.target.name.split())
        results.append(f"{source_id} --> {target_id}")

    return results


def visualize_flow(nodes: list[FlowNode], edges: list[FlowConnection]) -> str:
    nodes_v = visualize_nodes(nodes)
    edges_v = visualize_edges(edges)

    node_v = [f"    {node}" for node in nodes_v]
    edge_v = [f"    {edge}" for edge in edges_v]

    return "flowchart LR" + "\n" + "\n".join(node_v) + "\n" + "\n".join(edge_v)
