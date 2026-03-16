"""Clinical agents for healthcare workflows."""

import time
from typing import Dict, Any, Optional
from loguru import logger

from medgemma_agents.agents.base import BaseAgent, AgentOutput
from medgemma_agents.config import AgentConfig, TaskConfig
from medgemma_agents.llm import MedGemmaClient


class ClinicalAgent(BaseAgent):
    """Agent specialized for clinical tasks."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process clinical input.
        
        Args:
            input_data: Clinical data
            
        Returns:
            AgentOutput with clinical assessment
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                return AgentOutput(
                    agent_name=self.name,
                    output="",
                    success=False,
                    error="Invalid input format"
                )
            
            # Get primary task
            if not self.config.tasks:
                return AgentOutput(
                    agent_name=self.name,
                    output="",
                    success=False,
                    error="No tasks configured for agent"
                )
            
            primary_task = self.config.tasks[0]
            
            # Build prompt
            prompt = self._build_clinical_prompt(primary_task, input_data)
            
            # Get LLM response
            logger.info(f"Processing clinical task: {primary_task.name}")
            response = self.llm_client.complete(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024,
                top_p=0.9
            )
            
            # Format output
            output = self._format_output(response.content, primary_task.output_format)
            
            execution_time = time.time() - start_time
            
            return AgentOutput(
                agent_name=self.name,
                output=output,
                output_format=primary_task.output_format,
                success=True,
                tokens_used=response.total_tokens,
                execution_time=execution_time,
                metadata={
                    "task": primary_task.name,
                    "model": response.model,
                }
            )
        
        except Exception as e:
            logger.error(f"Clinical agent failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                output="",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _build_clinical_prompt(self, task: TaskConfig, input_data: Dict[str, Any]) -> str:
        """Build clinical prompt from template and data.
        
        Args:
            task: Task configuration
            input_data: Clinical input data
            
        Returns:
            Formatted prompt string
        """
        # Render template with input data
        try:
            prompt = self.render_prompt(task.prompt_template, input_data)
        except Exception as e:
            logger.warning(f"Failed to render template, using default: {e}")
            prompt = self._build_default_prompt(task, input_data)
        
        return prompt
    
    def _build_default_prompt(self, task: TaskConfig, input_data: Dict[str, Any]) -> str:
        """Build default prompt when template is unavailable.
        
        Args:
            task: Task configuration
            input_data: Clinical input data
            
        Returns:
            Default prompt string
        """
        prompt = f"""You are a clinical specialist performing the following task:

Task: {task.name}
Description: {task.description}

Patient/Clinical Data:
{self._format_dict(input_data)}

Please provide a professional clinical assessment. Return output in {task.output_format} format."""
        
        return prompt
    
    @staticmethod
    def _format_dict(data: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary for display.
        
        Args:
            data: Dictionary to format
            indent: Indentation level
            
        Returns:
            Formatted string
        """
        lines = []
        for key, value in data.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(ClinicalAgent._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)


class ClinicalNoteAgent(ClinicalAgent):
    """Agent for generating clinical notes."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Generate clinical note.
        
        Args:
            input_data: Patient information
            
        Returns:
            AgentOutput with clinical note
        """
        logger.info("Generating clinical note")
        return super().process(input_data)


class DiagnosisAgent(ClinicalAgent):
    """Agent for diagnosis assistance."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Suggest diagnoses.
        
        Args:
            input_data: Symptoms and clinical data
            
        Returns:
            AgentOutput with diagnosis suggestions
        """
        logger.info("Generating diagnosis suggestions")
        return super().process(input_data)


class TreatmentAgent(ClinicalAgent):
    """Agent for treatment recommendations."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Recommend treatments.
        
        Args:
            input_data: Diagnosis and patient data
            
        Returns:
            AgentOutput with treatment recommendations
        """
        logger.info("Generating treatment recommendations")
        return super().process(input_data)


class LabAnalysisAgent(ClinicalAgent):
    """Agent for lab result analysis."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Analyze lab results.
        
        Args:
            input_data: Lab results
            
        Returns:
            AgentOutput with lab analysis
        """
        logger.info("Analyzing lab results")
        return super().process(input_data)
