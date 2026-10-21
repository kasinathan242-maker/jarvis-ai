"""JARVIS Lite — Task DAG (Directed Acyclic Graph) Definition.

Represents complex multi-step workflows as a DAG of dependent steps,
enabling parallel execution of independent tasks and structured data handoffs.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class NodeStatus(enum.StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class DAGNode:
    node_id: str
    name: str
    agent_type: str  # "researcher", "document_ops", "shell", "tools", "custom"
    action: str  # function / tool name to call
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


class TaskDAG:
    """Directed Acyclic Graph of workflow execution steps."""

    def __init__(self, workflow_id: str, name: str = "Unnamed Workflow") -> None:
        self.workflow_id = workflow_id
        self.name = name
        self.nodes: dict[str, DAGNode] = {}

    def add_node(
        self,
        node_id: str,
        name: str,
        agent_type: str,
        action: str,
        parameters: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
    ) -> DAGNode:
        """Add a step node to the DAG."""
        if node_id in self.nodes:
            raise ValueError(f"Duplicate node ID: {node_id}")

        node = DAGNode(
            node_id=node_id,
            name=name,
            agent_type=agent_type,
            action=action,
            parameters=parameters or {},
            dependencies=dependencies or [],
        )
        self.nodes[node_id] = node
        return node

    def validate(self) -> None:
        """Validate DAG: check that all dependencies exist and there are no cycles."""
        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{nid}' references non-existent dependency '{dep}'")

        # Cycle detection via DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(curr: str) -> None:
            visited.add(curr)
            rec_stack.add(curr)
            for dep in self.nodes[curr].dependencies:
                if dep not in visited:
                    _dfs(dep)
                elif dep in rec_stack:
                    raise ValueError(f"Cycle detected in TaskDAG involving '{curr}' and '{dep}'")
            rec_stack.remove(curr)

        for nid in self.nodes:
            if nid not in visited:
                _dfs(nid)

    def get_ready_nodes(self) -> list[DAGNode]:
        """Return nodes that are PENDING and whose dependencies have all COMPLETED."""
        ready: list[DAGNode] = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps_met = all(
                self.nodes[d].status == NodeStatus.COMPLETED
                for d in node.dependencies
            )
            if deps_met:
                ready.append(node)
        return ready

    def is_finished(self) -> bool:
        """Return True if all nodes have terminal status."""
        terminal = {NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.CANCELLED}
        return all(n.status in terminal for n in self.nodes.values())

    def has_failures(self) -> bool:
        """Return True if any node failed."""
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskDAG:
        dag = cls(workflow_id=data["workflow_id"], name=data.get("name", ""))
        for nid, ndata in data.get("nodes", {}).items():
            node = DAGNode(
                node_id=ndata["node_id"],
                name=ndata["name"],
                agent_type=ndata["agent_type"],
                action=ndata["action"],
                parameters=ndata.get("parameters", {}),
                dependencies=ndata.get("dependencies", []),
                status=NodeStatus(ndata.get("status", NodeStatus.PENDING)),
                result=ndata.get("result"),
                error=ndata.get("error"),
                started_at=ndata.get("started_at"),
                completed_at=ndata.get("completed_at"),
            )
            dag.nodes[nid] = node
        return dag
