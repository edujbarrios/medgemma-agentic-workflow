"""Test suite for medgemma-agents."""

import pytest
from pathlib import Path
import tempfile
import json
import yaml

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medgemma_agents.config import Config, AgentConfig, WorkflowConfig
from medgemma_agents.llm import MedGemmaClient, LLMResponse
from medgemma_agents.agents import BaseAgent, ClinicalAgent
from medgemma_agents.templates import TemplateEngine


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_agent_config_creation(self):
        """Test creating agent configuration."""
        config = AgentConfig(
            name="Test Agent",
            description="Test description",
            version="1.0"
        )
        assert config.name == "Test Agent"
        assert config.version == "1.0"
    
    def test_config_from_dict(self):
        """Test loading config from dictionary."""
        data = {
            "agent": {
                "name": "Test Agent",
                "tasks": []
            },
            "llm": {
                "model": "test-model"
            }
        }
        config = Config.from_dict(data)
        assert config.agent.name == "Test Agent"
        assert config.llm.model == "test-model"
    
    def test_config_to_json(self):
        """Test saving config to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config.from_dict({
                "agent": {"name": "Test", "tasks": []},
                "llm": {"model": "test"}
            })
            
            output_path = Path(tmpdir) / "config.json"
            config.to_json(str(output_path))
            
            assert output_path.exists()
            
            with open(output_path) as f:
                data = json.load(f)
                assert data["agent"]["name"] == "Test"


class TestLLMClient:
    """Test LLM client."""
    
    def test_llm_client_initialization(self):
        """Test initializing LLM client."""
        client = MedGemmaClient(
            base_url="http://localhost:8000",
            model="test-model"
        )
        assert client.model == "test-model"
        assert client.base_url == "http://localhost:8000"
    
    def test_llm_response_creation(self):
        """Test creating LLM response."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            tokens_prompt=10,
            tokens_completion=20
        )
        assert response.content == "Test response"
        assert response.total_tokens == 30


class TestTemplateEngine:
    """Test template engine."""
    
    def test_template_initialization(self):
        """Test initializing template engine."""
        engine = TemplateEngine()
        assert engine.env is not None
    
    def test_render_string_template(self):
        """Test rendering string template."""
        engine = TemplateEngine()
        template = "Hello {{ name }}!"
        result = engine.render_string(template, {"name": "World"})
        assert result == "Hello World!"
    
    def test_custom_filters(self):
        """Test custom template filters."""
        engine = TemplateEngine()
        template = "{{ items | format_list }}"
        result = engine.render_string(
            template,
            {"items": ["apple", "banana", "cherry"]}
        )
        assert "apple" in result
        assert "banana" in result


class TestAgents:
    """Test agent implementations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = AgentConfig(
            name="Test Agent",
            tasks=[]
        )
        
        # Mock LLM client
        self.llm_client = MedGemmaClient()
    
    def test_agent_config_access(self):
        """Test agent configuration access."""
        from medgemma_agents.agents.base import BaseAgent
        
        # Can't instantiate abstract class, but can test subclass
        agent = ClinicalAgent(
            config=self.config,
            llm_client=self.llm_client
        )
        assert agent.name == "Test Agent"
        assert agent.version == "1.0"
    
    def test_agent_output_to_dict(self):
        """Test converting agent output to dictionary."""
        from medgemma_agents.agents.base import AgentOutput
        
        output = AgentOutput(
            agent_name="Test",
            output="Test output",
            success=True
        )
        
        as_dict = output.to_dict()
        assert as_dict["agent"] == "Test"
        assert as_dict["success"] is True


class TestIntegration:
    """Integration tests."""
    
    def test_config_workflow(self):
        """Test complete configuration workflow."""
        # Create config
        config_data = {
            "agent": {
                "name": "Integration Test Agent",
                "tasks": [
                    {
                        "name": "test_task",
                        "description": "Test task",
                        "prompt_template": "test.jinja2"
                    }
                ]
            },
            "llm": {
                "model": "test-model",
                "temperature": 0.5
            }
        }
        
        config = Config.from_dict(config_data)
        
        # Verify structure
        assert config.agent.name == "Integration Test Agent"
        assert len(config.agent.tasks) == 1
        assert config.llm.temperature == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
