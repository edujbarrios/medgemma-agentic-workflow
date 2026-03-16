"""Base agent class."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from loguru import logger
from datetime import datetime
import json

from medgemma_agents.config import AgentConfig
from medgemma_agents.llm import MedGemmaClient, LLMResponse
from medgemma_agents.templates import TemplateEngine


@dataclass
class AgentOutput:
    """Agent execution output."""
    
    agent_name: str
    output: str
    output_format: str = "text"
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent": self.agent_name,
            "output": self.output,
            "format": self.output_format,
            "success": self.success,
            "error": self.error,
            "tokens": self.tokens_used,
            "time": self.execution_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: MedGemmaClient,
        template_engine: Optional[TemplateEngine] = None,
    ):
        """Initialize agent.
        
        Args:
            config: Agent configuration
            llm_client: LLM client instance
            template_engine: Optional template engine
        """
        self.config = config
        self.llm_client = llm_client
        self.template_engine = template_engine or TemplateEngine()
        self.name = config.name
        self.version = config.version
        
        logger.info(f"Initialized agent: {self.name} v{self.version}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(input_data, dict)
    
    def render_prompt(
        self,
        template_path: str,
        variables: Dict[str, Any]
    ) -> str:
        """Render prompt template.
        
        Args:
            template_path: Path to template file
            variables: Template variables
            
        Returns:
            Rendered prompt string
        """
        return self.template_engine.render(template_path, variables)
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process input and return output.
        
        Args:
            input_data: Input data
            
        Returns:
            AgentOutput object
        """
        pass
    
    def _format_output(self, raw_output: str, output_format: str) -> str:
        """Format output based on specified format.
        
        Args:
            raw_output: Raw output from LLM
            output_format: Output format (text, json, structured)
            
        Returns:
            Formatted output
        """
        if output_format == "json":
            try:
                # Ensure valid JSON
                if isinstance(raw_output, str):
                    return json.dumps(json.loads(raw_output), indent=2)
                return json.dumps(raw_output, indent=2)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to format output as JSON, returning as text")
                return raw_output
        
        elif output_format == "structured":
            # Parse structured format (assume Markdown-like)
            return raw_output
        
        else:
            # Return as plain text
            return raw_output
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"
