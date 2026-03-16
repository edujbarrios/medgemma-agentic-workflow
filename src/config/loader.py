"""Configuration models and validation."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import json
import yaml
from pathlib import Path


class LLMConfig(BaseModel):
    """LLM configuration model."""
    
    model: str = Field(default="medgemma-7b", description="Model name/identifier")
    base_url: Optional[str] = Field(default="http://localhost:8000", description="LLM server base URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Temperature parameter")
    max_tokens: int = Field(default=1024, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    class Config:
        """Pydantic config."""
        extra = "allow"
        str_strip_whitespace = True


class TaskConfig(BaseModel):
    """Task configuration model."""
    
    name: str = Field(description="Task name")
    description: str = Field(default="", description="Task description")
    prompt_template: str = Field(description="Path to prompt template file")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    output_format: str = Field(default="text", description="Output format (text, json, structured)")
    
    class Config:
        """Pydantic config."""
        extra = "allow"


class AgentConfig(BaseModel):
    """Agent configuration model."""
    
    name: str = Field(description="Agent name")
    description: str = Field(default="", description="Agent description")
    version: str = Field(default="1.0", description="Agent version")
    tasks: List[TaskConfig] = Field(default_factory=list, description="Agent tasks")
    output_format: str = Field(default="text", description="Agent output format")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic config."""
        extra = "allow"


class WorkflowTaskConfig(BaseModel):
    """Workflow task configuration."""
    
    agent: str = Field(description="Agent or task name to execute")
    depends_on: List[str] = Field(default_factory=list, description="Task dependencies")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    retry_on_failure: bool = Field(default=True, description="Retry on failure")
    timeout: Optional[int] = Field(default=None, description="Task timeout in seconds")
    
    class Config:
        """Pydantic config."""
        extra = "allow"


class WorkflowConfig(BaseModel):
    """Workflow configuration model."""
    
    name: str = Field(description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    version: str = Field(default="1.0", description="Workflow version")
    tasks: List[WorkflowTaskConfig] = Field(description="Workflow tasks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic config."""
        extra = "allow"
    
    @field_validator('tasks', mode='before')
    @classmethod
    def validate_tasks(cls, v):
        """Ensure tasks are valid."""
        if not v:
            raise ValueError("Workflow must have at least one task")
        return v


class Config(BaseModel):
    """Main configuration model."""
    
    agent: Optional[AgentConfig] = Field(default=None, description="Agent configuration")
    workflow: Optional[WorkflowConfig] = Field(default=None, description="Workflow configuration")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    logging: Dict[str, Any] = Field(default_factory=lambda: {"level": "INFO"}, description="Logging configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic config."""
        extra = "allow"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data is None:
                raise ValueError(f"Empty configuration file: {path}")
            return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary."""
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                json.loads(self.model_dump_json(exclude_none=True)),
                f,
                default_flow_style=False,
                sort_keys=False
            )
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(
                json.loads(self.model_dump_json(exclude_none=True)),
                f,
                indent=2
            )
