"""Workflow orchestration engine."""

import time
from collections import Counter
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger
import json

from medgemma_agents.config import WorkflowConfig, AgentConfig
from medgemma_agents.agents import BaseAgent, AgentOutput


@dataclass
class TaskResult:
    """Result of task execution."""
    
    task_name: str
    agent_output: AgentOutput
    success: bool
    execution_time: float
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())


class Workflow:
    """Workflow definition and execution."""
    
    def __init__(
        self,
        config: WorkflowConfig,
        agents: Dict[str, BaseAgent]
    ):
        """Initialize workflow.
        
        Args:
            config: Workflow configuration
            agents: Dictionary of available agents
        """
        self.config = config
        self.agents = agents
        self.name = config.name
        self.results: Dict[str, TaskResult] = {}
        
        logger.info(f"Initialized workflow: {self.name}")
    
    def validate_config(self) -> bool:
        """Validate workflow configuration.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            is_valid = True
            task_names = [t.agent for t in self.config.tasks]
            duplicate_task_names = {name for name, count in Counter(task_names).items() if count > 1}

            if duplicate_task_names:
                is_valid = False
                logger.warning(f"Duplicate task names found: {sorted(duplicate_task_names)}")

            # Check all agents are available
            for task in self.config.tasks:
                if task.agent not in self.agents:
                    is_valid = False
                    logger.warning(f"Agent not found: {task.agent}")
            
            # Check task dependencies
            task_name_set = set(task_names)
            for task in self.config.tasks:
                for dep in task.depends_on:
                    if dep not in task_name_set:
                        is_valid = False
                        logger.warning(f"Dependency not found: {dep} <- {task.agent}")
            
            return is_valid
        except Exception as e:
            logger.error(f"Workflow validation failed: {e}")
            return False
    
    def execute(
        self,
        initial_input: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            initial_input: Initial input data
            verbose: Print execution logs
            
        Returns:
            Workflow results
        """
        start_time = time.time()
        
        try:
            if not self.validate_config():
                raise ValueError("Invalid workflow configuration")

            # Initialize context
            context = {
                "input": initial_input,
                "results": {}
            }
            
            logger.info(f"Starting workflow execution: {self.name}")
            
            # Execute tasks in dependency order
            task_order = self._topological_sort()
            
            for task_name in task_order:
                task = next(t for t in self.config.tasks if t.agent == task_name)
                
                # Build task input from context
                task_input = self._build_task_input(task, context)
                
                # Execute task
                result = self._execute_task(task, task_input, verbose)
                
                # Store result
                self.results[task_name] = result
                context["results"][task_name] = result.agent_output
            
            execution_time = time.time() - start_time
            
            return {
                "workflow": self.name,
                "success": all(r.success for r in self.results.values()),
                "execution_time": execution_time,
                "results": {k: v.agent_output.to_dict() for k, v in self.results.items()},
                "context": context
            }
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "workflow": self.name,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "results": {k: v.agent_output.to_dict() for k, v in self.results.items()},
            }
    
    def _topological_sort(self) -> List[str]:
        """Sort tasks by dependencies.
        
        Returns:
            List of task names in execution order
        """
        visited = set()
        in_progress = set()
        result = []
        tasks_by_name = {task.agent: task for task in self.config.tasks}
        
        def visit(task_name: str):
            if task_name in visited:
                return
            if task_name in in_progress:
                raise ValueError(f"Circular dependency detected involving task: {task_name}")
            if task_name not in tasks_by_name:
                raise ValueError(f"Dependency not found: {task_name}")
            
            in_progress.add(task_name)
            task = tasks_by_name[task_name]
            for dep in task.depends_on:
                visit(dep)
            
            in_progress.remove(task_name)
            visited.add(task_name)
            result.append(task_name)
        
        for task in self.config.tasks:
            visit(task.agent)
        
        return result
    
    def _build_task_input(
        self,
        task: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build input for task execution.
        
        Args:
            task: Task configuration
            context: Current workflow context
            
        Returns:
            Task input dictionary
        """
        task_input = {**context["input"]}
        
        # Add dependency outputs
        for dep in task.depends_on:
            if dep in context["results"]:
                task_input[f"{dep}_output"] = context["results"][dep].output
        
        # Add task parameters
        task_input.update(task.parameters)
        
        return task_input
    
    def _execute_task(
        self,
        task: Any,
        task_input: Dict[str, Any],
        verbose: bool = True
    ) -> TaskResult:
        """Execute single task.
        
        Args:
            task: Task configuration
            task_input: Task input
            verbose: Print logs
            
        Returns:
            TaskResult object
        """
        start_time = time.time()
        
        try:
            agent_name = task.agent
            
            if agent_name not in self.agents:
                raise ValueError(f"Agent not found: {agent_name}")
            
            agent = self.agents[agent_name]
            
            if verbose:
                logger.info(f"Executing task: {agent_name}")
            
            # Execute agent
            output = agent.process(task_input)
            
            execution_time = time.time() - start_time
            
            if verbose and output.success:
                logger.info(f"Task completed: {agent_name} ({execution_time:.2f}s)")
            elif verbose:
                logger.error(f"Task failed: {agent_name} - {output.error}")
            
            return TaskResult(
                task_name=agent_name,
                agent_output=output,
                success=output.success,
                execution_time=execution_time
            )
        
        except Exception as e:
            logger.error(f"Task execution failed: {task.agent} - {e}")
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_name=task.agent,
                agent_output=AgentOutput(
                    agent_name=task.agent,
                    output="",
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                ),
                success=False,
                execution_time=execution_time
            )


class WorkflowEngine:
    """Workflow creation and execution engine."""
    
    def __init__(self):
        """Initialize workflow engine."""
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        logger.info("Initialized WorkflowEngine")
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register agent.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")
    
    def create_workflow(
        self,
        config: WorkflowConfig
    ) -> Workflow:
        """Create workflow from configuration.
        
        Args:
            config: Workflow configuration
            
        Returns:
            Workflow instance
        """
        workflow = Workflow(config, self.agents)
        self.workflows[config.name] = workflow
        logger.info(f"Created workflow: {config.name}")
        return workflow
    
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """Get registered workflow.
        
        Args:
            name: Workflow name
            
        Returns:
            Workflow instance or None
        """
        return self.workflows.get(name)
    
    def execute_workflow(
        self,
        workflow_name: str,
        initial_input: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Execute registered workflow.
        
        Args:
            workflow_name: Name of workflow to execute
            initial_input: Initial input data
            verbose: Print logs
            
        Returns:
            Workflow results
        """
        workflow = self.get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_name}")
        
        return workflow.execute(initial_input, verbose=verbose)
