# MedGemma Agents 

**Agentic workflow powered by MedGemma from Hugging Face**

A fully parameterized, production-ready development framework for building autonomous clinical agents and workflows. Load **MedGemma directly from Hugging Face Hub** and create sophisticated multi-agent systems with specialized clinical tasks, all powered by open-source medical language models.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Alpha-yellow)

## ⚠️ Disclaimer: Proof of Concept

**This is a Proof of Concept (PoC) framework and is NOT a production-ready medical system.** 

- ❌ **NOT medical-grade**: This system is NOT endorsed, validated, or approved by healthcare professionals or regulatory bodies
- ❌ **NOT for direct clinical use**: Do NOT use this as-is for actual patient care or clinical decision-making
- ✅ **Research tool only**: This PoC demonstrates how supervised MedGemma-based agentic systems could potentially be architected for healthcare applications
- ✅ **Starting point**: Use this as a foundation to build custom, medically-supervised systems with proper validation, testing, and clinical oversight

## Features ✨

- **🤖 Multi-Agent Framework**: Build autonomous agents with specialized clinical tasks
- **📋 Clinical Workflows**: Pre-built agents for common healthcare scenarios
- **⚙️ Fully Parameterized**: Configure every aspect via YAML/JSON config files
- **🎨 Jinja2 Templates**: Dynamic prompt templates with variable substitution
- **🔧 MedGemma Integration**: Seamless integration with MedGemma LLM
- **📊 Extensible Architecture**: Easy to add new agents and workflows
- **🧪 Production Ready**: Comprehensive error handling, logging, and testing
- **📚 Well Documented**: Clear examples and API documentation

## Quick Start 🚀

### Installation

```bash
# Clone the repository
git clone https://github.com/edujbarrios/medgemma-agentic-workflow.git
cd medgemma-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with Hugging Face support
pip install -e ".[huggingface,dev]"

# (Optional) Login to Hugging Face if using gated models
huggingface-cli login
```
### Basic Usage

```python
from medgemma_agents.config import Config
from medgemma_agents.agents import ClinicalAgent
from medgemma_agents.llm import HuggingFaceMedGemmaClient

# Load configuration
config = Config.from_yaml("configs/agents/clinical_note.yaml")

# Initialize Hugging Face MedGemma client
# Auto-downloads from Hugging Face Hub on first run
llm_client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",  # Quantized version: "google/medgemma-7b-it"
    device="cuda",  # "cpu" for CPU or auto for automatic detection
    torch_dtype="float16"  # Use float16 for less VRAM
)

# Create and run agent
agent = ClinicalAgent(config=config.agent, llm_client=llm_client)
result = agent.process({
    "chief_complaint": "Chest pain",
    "duration": "2 hours",
    "patient": {"age": 45, "sex": "M"}
})

print(result.output)
```

### Configuration Example

**configs/clinical_note_agent.yaml**:
```yaml
agent:
  name: Clinical Note Generator
  description: Generates structured clinical notes from patient data
  version: 1.0
  
  tasks:
    - name: triage
      description: Initial patient assessment
      prompt_template: prompts/triage.jinja2
      parameters:
        urgency_levels: [low, medium, high, critical]
    
    - name: diagnosis_assist
      description: Suggests potential diagnoses
      prompt_template: prompts/diagnosis.jinja2
      parameters:
        max_suggestions: 5

llm:
  model: medgemma-7b
  temperature: 0.3
  max_tokens: 1024
  top_p: 0.9
```

## Core Components 🔧

### 1. **Agent System**
- `BaseAgent`: Abstract base class for all agents
- `ClinicalAgent`: Specialized for clinical tasks
- `ReasoningAgent`: Multi-step reasoning capabilities

### 2. **Configuration Management**
- YAML/JSON configuration loading
- Pydantic models for validation
- Environment variable substitution
- Template variable management

### 3. **LLM Integration**
- Hugging Face Transformers pipeline support
- Seamless model loading from Hub
- Token optimization
- Device management (CPU/GPU/Auto)

### 4. **Template Engine**
- Jinja2 integration for dynamic prompts
- Variable substitution
- Conditional logic in prompts
- Reusable template components

### 5. **Workflow Engine**
- DAG-based workflow execution
- Task sequencing and dependencies
- Error handling and recovery
- State management

## Example Agents 📋

### Clinical Note Generation
Generates comprehensive clinical notes from patient information.

```python
agent = ClinicalNoteAgent(config)
result = agent.process({
    "patient": {"age": 45, "sex": "M"},
    "chief_complaint": "Headache",
    "duration": "3 days"
})
```

### Diagnosis Assistance
Suggests potential diagnoses based on symptoms.

```python
agent = DiagnosisAgent(config)
result = agent.process({
    "symptoms": ["fever", "cough", "fatigue"],
    "vital_signs": {"temp": 38.5, "hr": 95}
})
```

### Treatment Recommendation
Recommends evidence-based treatments.

```python
agent = TreatmentAgent(config)
result = agent.process({
    "diagnosis": "Type 2 Diabetes",
    "patient_factors": {"age": 60, "comorbidities": ["hypertension"]},
    "current_meds": ["metformin"]
})
```

## Configuration Guide 📝

### Agent Configuration

Each agent is configured via YAML:

```yaml
agent:
  name: Your Agent Name
  description: Description of what the agent does
  version: 1.0
  
  tasks:
    - name: task_name
      description: Task description
      prompt_template: path/to/template.jinja2
      parameters:
        param1: value1
        param2: [list, of, values]
  
  output_format: structured | text | json
```

### LLM Configuration

```yaml
llm:
  model: medgemma-7b
  base_url: http://localhost:8000
  api_key: ${MEDGEMMA_API_KEY}  # Environment variable
  temperature: 0.3               # Lower = more deterministic
  max_tokens: 1024
  top_p: 0.9
```

### Workflow Configuration

```yaml
workflow:
  name: Clinical Assessment Workflow
  description: Multi-step clinical assessment
  
  tasks:
    - agent: triage
      depends_on: []
      params:
        urgency: ${input.urgency}
    
    - agent: diagnosis
      depends_on: [triage]
      params:
        symptoms: ${triage.output.symptoms}
    
    - agent: treatment
      depends_on: [diagnosis]
      params:
        diagnosis: ${diagnosis.output.primary}
```

## Prompt Templates 🎨

Prompts use Jinja2 templates for dynamic content


## API Reference 📚

### Agent Classes

```python
class BaseAgent:
    def process(self, input_data: dict) -> AgentOutput
    def validate_input(self, data: dict) -> bool
    def format_output(self, result: str) -> AgentOutput

class ClinicalAgent(BaseAgent):
    """Specialized for clinical tasks"""
    pass

class ReasoningAgent(BaseAgent):
    """Multi-step reasoning capabilities"""
    pass
```

### Workflow Engine

```python
from medgemma_agents.workflows import WorkflowEngine

engine = WorkflowEngine()
workflow = engine.load("configs/workflows/assessment.yaml")
result = workflow.execute(initial_input)
```

## Testing 🧪

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_agents.py::test_clinical_note_agent
```

## Development 🛠️

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# All checks
pre-commit run --all-files
```

## Environment Variables 

Create `.env` file:

```env
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_hf_token_here  # For gated models
```

## Advanced Features 🚀

### Custom Agents

Create specialized agents by extending `BaseAgent`:

```python
from medgemma_agents.agents import BaseAgent
from medgemma_agents.config import AgentConfig

class CustomAgent(BaseAgent):
    def process(self, input_data: dict) -> AgentOutput:
        # Your implementation
        pass
```

### Workflow Composition

Build complex workflows using DAG definition:

```python
workflow = {
    "triage": ["symptoms", "vitals"],
    "diagnosis": ["triage"],
    "treatment": ["diagnosis"]
}
```

### Streaming Results

Handle streaming responses:

```python
from medgemma_agents.llm import HuggingFaceMedGemmaClient

client = HuggingFaceMedGemmaClient()

for chunk in client.complete_stream(prompt):
    print(chunk, end="", flush=True)
```
## Limitations & Future Work ⚠️

Currently:
- Single-turn conversations (multi-turn in development)
- Text-only I/O (multi-modal planned)


## Citation 📖

If you find this PoC useful for your research or development, please consider citing it as:
```{txt}
Barrios, E. J. (2026). MedGemma Agents: A proof of concept framework for agentic workflows in healthcare. 
GitHub. https://github.com/edujbarrios/medgemma-agents
```



## Acknowledgments 🙏

Built with:
- [MedGemma](https://huggingface.co/spaces/google/medgemma-7b-it) - Medical language model by Google
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Model loading and inference
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Jinja2](https://jinja.palletsprojects.com/) - Template engine
- [Loguru](https://loguru.readthedocs.io/) - Logging

## License 📄

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Support & Issues 💬

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/edujbarrios/medgemma-agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/edujbarrios/medgemma-agents/discussions)

---

**Made by Eduardo J. Barrios** • [GitHub](https://github.com/edujbarrios) • [LinkedIn](https://linkedin.com/in/edujbarrios)

**Development Tool** | Open Source | Apache 2.0 License
