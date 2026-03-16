"""Reasoning agents with multi-step logic."""

import time
from typing import Dict, Any, List, Optional
from loguru import logger

from medgemma_agents.agents.base import BaseAgent, AgentOutput
from medgemma_agents.config import TaskConfig


class ReasoningAgent(BaseAgent):
    """Agent with multi-step reasoning capabilities."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process with multi-step reasoning.
        
        Args:
            input_data: Input data
            
        Returns:
            AgentOutput with reasoning steps
        """
        start_time = time.time()
        
        try:
            if not self.validate_input(input_data):
                return AgentOutput(
                    agent_name=self.name,
                    output="",
                    success=False,
                    error="Invalid input format"
                )
            
            if not self.config.tasks:
                return AgentOutput(
                    agent_name=self.name,
                    output="",
                    success=False,
                    error="No reasoning steps configured"
                )
            
            # Execute reasoning steps
            reasoning_output = self._execute_reasoning(input_data)
            
            execution_time = time.time() - start_time
            
            return AgentOutput(
                agent_name=self.name,
                output=reasoning_output,
                output_format=self.config.output_format,
                success=True,
                execution_time=execution_time,
            )
        
        except Exception as e:
            logger.error(f"Reasoning agent failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                output="",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _execute_reasoning(self, input_data: Dict[str, Any]) -> str:
        """Execute multi-step reasoning.
        
        Args:
            input_data: Input data
            
        Returns:
            Final reasoning output
        """
        context = input_data.copy()
        reasoning_steps = []
        
        for i, task in enumerate(self.config.tasks, 1):
            logger.info(f"Executing reasoning step {i}: {task.name}")
            
            # Build step prompt
            prompt = self._build_reasoning_prompt(task, context, i)
            
            # Get LLM response
            response = self.llm_client.complete(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
                top_p=0.9
            )
            
            step_output = response.content.strip()
            reasoning_steps.append({
                "step": i,
                "name": task.name,
                "output": step_output
            })
            
            # Update context with step output
            context[f"step_{i}_output"] = step_output
        
        # Format final output
        return self._format_reasoning_output(reasoning_steps)
    
    def _build_reasoning_prompt(
        self,
        task: TaskConfig,
        context: Dict[str, Any],
        step_number: int
    ) -> str:
        """Build prompt for reasoning step.
        
        Args:
            task: Task configuration
            context: Current context/state
            step_number: Step number
            
        Returns:
            Prompt string
        """
        # Try to render template
        try:
            prompt = self.render_prompt(task.prompt_template, context)
        except Exception as e:
            logger.warning(f"Failed to render template: {e}")
            prompt = self._build_default_reasoning_prompt(task, step_number)
        
        return prompt
    
    @staticmethod
    def _build_default_reasoning_prompt(task: TaskConfig, step_number: int) -> str:
        """Build default reasoning prompt.
        
        Args:
            task: Task configuration
            step_number: Step number
            
        Returns:
            Default prompt
        """
        return f"""Reasoning Step {step_number}: {task.name}

Description: {task.description}

Please think through this step carefully and provide a clear reasoning output."""
    
    @staticmethod
    def _format_reasoning_output(steps: List[Dict[str, Any]]) -> str:
        """Format reasoning output.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            Formatted output
        """
        output = "## Multi-Step Reasoning Analysis\n\n"
        
        for step in steps:
            output += f"### Step {step['step']}: {step['name']}\n"
            output += f"{step['output']}\n\n"
        
        return output


class ChainOfThoughtAgent(ReasoningAgent):
    """Agent that uses chain-of-thought reasoning."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process with chain-of-thought reasoning.
        
        Args:
            input_data: Input data
            
        Returns:
            AgentOutput with reasoning chain
        """
        logger.info("Executing chain-of-thought reasoning")
        return super().process(input_data)


class TreeOfThoughtAgent(ReasoningAgent):
    """Agent that uses tree-of-thought reasoning."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process with tree-of-thought reasoning.
        
        Args:
            input_data: Input data
            
        Returns:
            AgentOutput with reasoning tree
        """
        logger.info("Executing tree-of-thought reasoning")
        result = super().process(input_data)
        return result
