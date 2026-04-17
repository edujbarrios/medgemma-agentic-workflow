"""Workflow engine tests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medgemma_agents.agents import BaseAgent, AgentOutput
from medgemma_agents.config import AgentConfig, WorkflowConfig
from medgemma_agents.llm import MedGemmaClient
from medgemma_agents.workflows import WorkflowEngine


class DummyAgent(BaseAgent):
    """Minimal test agent implementation."""

    def process(self, input_data):
        return AgentOutput(agent_name=self.name, output=str(input_data), success=True)


def _make_agent(name: str) -> DummyAgent:
    return DummyAgent(config=AgentConfig(name=name, tasks=[]), llm_client=MedGemmaClient())


def test_validate_config_fails_for_missing_agent_and_dependency():
    engine = WorkflowEngine()
    workflow_config = WorkflowConfig(
        name="invalid-workflow",
        tasks=[
            {"agent": "diagnosis", "depends_on": ["triage"]},
        ],
    )

    workflow = engine.create_workflow(workflow_config)

    assert workflow.validate_config() is False


def test_execute_fails_with_circular_dependency():
    engine = WorkflowEngine()
    engine.register_agent("task_a", _make_agent("task_a"))
    engine.register_agent("task_b", _make_agent("task_b"))

    workflow_config = WorkflowConfig(
        name="cyclic-workflow",
        tasks=[
            {"agent": "task_a", "depends_on": ["task_b"]},
            {"agent": "task_b", "depends_on": ["task_a"]},
        ],
    )
    workflow = engine.create_workflow(workflow_config)

    result = workflow.execute(initial_input={})

    assert result["success"] is False
    assert "Circular dependency detected" in result["error"]


def test_execute_fails_with_duplicate_task_names():
    engine = WorkflowEngine()
    engine.register_agent("task_a", _make_agent("task_a"))

    workflow_config = WorkflowConfig(
        name="duplicate-workflow",
        tasks=[
            {"agent": "task_a", "depends_on": []},
            {"agent": "task_a", "depends_on": []},
        ],
    )
    workflow = engine.create_workflow(workflow_config)

    result = workflow.execute(initial_input={})

    assert result["success"] is False
    assert "Invalid workflow configuration" in result["error"]
