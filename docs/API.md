# API Reference

## Core Classes

### Configuration

#### `Config`

```python
class Config(BaseModel):
    """Main configuration model."""
    
    agent: Optional[AgentConfig]
    workflow: Optional[WorkflowConfig]
    llm: LLMConfig
    logging: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config"
    @classmethod
    def from_json(cls, path: str) -> "Config"
    @classmethod
    def from_dict(cls, data: Dict) -> "Config"
    
    def to_yaml(self, path: str) -> None
    def to_json(self, path: str) -> None
```

### Agents

#### `BaseAgent`

```python
class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: MedGemmaClient,
        template_engine: Optional[TemplateEngine] = None
    )
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process input and return output."""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool
    def render_prompt(
        self,
        template_path: str,
        variables: Dict[str, Any]
    ) -> str
```

#### `ClinicalAgent`

```python
class ClinicalAgent(BaseAgent):
    """Agent specialized for clinical tasks."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process clinical input."""
        pass
```

**Subclasses**:
- `ClinicalNoteAgent`: Generate clinical notes
- `DiagnosisAgent`: Suggest diagnoses
- `TreatmentAgent`: Recommend treatments
- `LabAnalysisAgent`: Analyze lab results

#### `ReasoningAgent`

```python
class ReasoningAgent(BaseAgent):
    """Agent with multi-step reasoning capabilities."""
    
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process with multi-step reasoning."""
        pass
```

**Subclasses**:
- `ChainOfThoughtAgent`: Chain-of-thought reasoning
- `TreeOfThoughtAgent`: Tree-of-thought reasoning

#### `AgentOutput`

```python
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
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
```

### LLM Client

#### `MedGemmaClient`

```python
class MedGemmaClient:
    """MedGemma LLM client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "medgemma-7b",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    )
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs
    ) -> LLMResponse
    
    def complete_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse
    
    def complete_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]
    
    def health_check(self) -> bool
```

#### `LLMResponse`

```python
@dataclass
class LLMResponse:
    """LLM response data."""
    
    content: str
    model: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    timestamp: str
    metadata: Dict[str, Any]
    
    @property
    def total_tokens(self) -> int
```

### Templates

#### `TemplateEngine`

```python
class TemplateEngine:
    """Jinja2-based template engine."""
    
    def __init__(
        self,
        template_dir: Optional[str] = None,
        auto_reload: bool = True,
        trim_blocks: bool = True,
        lstrip_blocks: bool = True
    )
    
    def render(
        self,
        template_name: str,
        variables: Dict[str, Any],
        from_string: bool = False
    ) -> str
    
    def render_file(
        self,
        file_path: str,
        variables: Dict[str, Any]
    ) -> str
    
    def render_string(
        self,
        template_string: str,
        variables: Dict[str, Any]
    ) -> str
```

**Available Filters**:
- `format_list(items, separator=", ")`: Format list as string
- `format_dict(d, indent=0)`: Format dictionary nicely
- `truncate(s, length=100)`: Truncate long strings
- `upper_first(s)`: Capitalize first letter

### Workflows

#### `WorkflowEngine`

```python
class WorkflowEngine:
    """Workflow creation and execution engine."""
    
    def __init__(self)
    
    def register_agent(self, name: str, agent: BaseAgent) -> None
    
    def create_workflow(self, config: WorkflowConfig) -> Workflow
    
    def get_workflow(self, name: str) -> Optional[Workflow]
    
    def execute_workflow(
        self,
        workflow_name: str,
        initial_input: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]
```

#### `Workflow`

```python
class Workflow:
    """Workflow definition and execution."""
    
    def __init__(
        self,
        config: WorkflowConfig,
        agents: Dict[str, BaseAgent]
    )
    
    def validate_config(self) -> bool
    
    def execute(
        self,
        initial_input: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]
```

## Usage Examples

### Basic Agent Usage

```python
from medgemma_agents.config import Config
from medgemma_agents.agents import ClinicalAgent
from medgemma_agents.llm import MedGemmaClient

# Load config
config = Config.from_yaml("config.yaml")

# Create client
client = MedGemmaClient(base_url="http://localhost:8000")

# Create agent
agent = ClinicalAgent(config=config.agent, llm_client=client)

# Process input
result = agent.process({"chief_complaint": "Chest pain"})

# Check results
if result.success:
    print(result.output)
else:
    print(f"Error: {result.error}")
```

### Workflow Execution

```python
from medgemma_agents.workflows import WorkflowEngine

# Create engine
engine = WorkflowEngine()

# Register agents
engine.register_agent("note", note_agent)
engine.register_agent("diagnosis", diagnosis_agent)

# Load workflow
workflow_config = Config.from_yaml("workflow.yaml").workflow
workflow = engine.create_workflow(workflow_config)

# Execute
results = workflow.execute({"patient": patient_data})
```

### Custom Agent

```python
from medgemma_agents.agents import BaseAgent

class MyAgent(BaseAgent):
    def process(self, input_data):
        # Custom logic
        prompt = self.render_prompt("template.jinja2", input_data)
        response = self.llm_client.complete(prompt)
        return AgentOutput(
            agent_name=self.name,
            output=response.content,
            success=True
        )
```

---

For more detailed examples, see the [Examples](../examples/) directory.
