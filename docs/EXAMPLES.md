# Examples

Comprehensive examples demonstrating MedGemma Agents capabilities.

## Quick Start

### Example 1: Basic Clinical Note Generation

Generate a clinical note from patient data:

```bash
python examples/basic_agent.py
```

**Code Overview**:
```python
from medgemma_agents.config import Config
from medgemma_agents.agents import ClinicalAgent
from medgemma_agents.llm import MedGemmaClient

# 1. Load configuration
config = Config.from_yaml("configs/agents/clinical_note.yaml")

# 2. Create LLM client
client = MedGemmaClient(base_url="http://localhost:8000")

# 3. Create agent
agent = ClinicalAgent(config=config.agent, llm_client=client)

# 4. Process data
result = agent.process({
    "chief_complaint": "Chest pain",
    "duration": "2 hours"
})

# 5. Display output
print(result.output)
```

### Example 2: Multi-Agent Workflow

Run a complete clinical assessment with multiple agents:

```bash
python examples/clinical_workflow.py
```

**Workflow Steps**:
1. Generate clinical note (ClinicalNoteAgent)
2. Suggest diagnoses (DiagnosisAgent)
3. Recommend treatments (TreatmentAgent)

### Example 3: Chain-of-Thought Reasoning

Use advanced reasoning with step-by-step logic:

```bash
python examples/reasoning_example.py
```

**Features**:
- Multi-step reasoning
- Intermediate reasoning steps
- Detailed analysis output

## Detailed Examples

### Custom Configuration

Create a custom agent configuration:

```python
from medgemma_agents.config import AgentConfig, TaskConfig, LLMConfig, Config

# Create agent configuration
agent_config = AgentConfig(
    name="Custom Agent",
    description="My custom clinical agent",
    version="1.0",
    tasks=[
        TaskConfig(
            name="analysis",
            description="Analyze patient data",
            prompt_template="custom_template.jinja2",
            output_format="json",
            parameters={
                "max_items": 10,
                "include_confidence": True
            }
        )
    ]
)

# Create full config
config = Config(
    agent=agent_config,
    llm=LLMConfig(
        model="medgemma-7b",
        temperature=0.3,
        max_tokens=1024
    )
)

# Use the configuration
agent = CustomAgent(config=agent_config, llm_client=client)
```

### Custom Agent Implementation

Implement a specialized agent:

```python
from medgemma_agents.agents import BaseAgent
from medgemma_agents.agents.base import AgentOutput
import time

class SummaryAgent(BaseAgent):
    """Agent that summarizes clinical information."""
    
    def process(self, input_data):
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = f"""Summarize the following clinical information:
            
{input_data.get('clinical_data', '')}

Provide a brief, structured summary."""
            
            # Call LLM
            response = self.llm_client.complete(
                prompt=prompt,
                temperature=0.2,
                max_tokens=512
            )
            
            return AgentOutput(
                agent_name=self.name,
                output=response.content,
                success=True,
                tokens_used=response.total_tokens,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                output="",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
```

### Custom Prompt Template

Create a specialized prompt template:

```jinja2
{# templates/custom_prompt.jinja2 #}

You are a clinical specialist in {{ specialty }}.

Patient: {{ patient.name }} ({{ patient.age }}yo {{ patient.sex }})
Chief Complaint: {{ chief_complaint }}
Duration: {{ duration }}

Symptoms:
{% for symptom in symptoms %}
  - {{ symptom }}
{% endfor %}

Vital Signs:
  - Temperature: {{ vitals.temperature }}
  - Heart Rate: {{ vitals.heart_rate }}
  - Blood Pressure: {{ vitals.blood_pressure }}

{% if lab_results %}
Lab Results:
{{ lab_results | format_dict(indent=2) }}
{% endif %}

Please provide:
1. Clinical assessment
2. Differential diagnosis (top 3)
3. Recommended investigations
```

### Workflow Execution

Execute a complex multi-agent workflow:

```python
from medgemma_agents.config import Config
from medgemma_agents.workflows import WorkflowEngine
from medgemma_agents.agents import (
    ClinicalAgent, DiagnosisAgent, TreatmentAgent
)
from medgemma_agents.llm import MedGemmaClient

# Initialize
client = MedGemmaClient()
engine = WorkflowEngine()

# Create agents
agents = {
    "clinical_note": ClinicalAgent(...),
    "diagnosis": DiagnosisAgent(...),
    "treatment": TreatmentAgent(...)
}

# Register agents
for name, agent in agents.items():
    engine.register_agent(name, agent)

# Load workflow
workflow_config = Config.from_yaml("clinical_workflow.yaml")
workflow = engine.create_workflow(workflow_config.workflow)

# Execute
patient_data = {
    "patient": {"age": 65, "sex": "M"},
    "chief_complaint": "Shortness of breath",
    "symptoms": ["dyspnea", "orthopnea", "ankle edema"]
}

results = workflow.execute(patient_data, verbose=True)

# Process results
for task_name, result in results["results"].items():
    print(f"{task_name}: {result['success']}")
    if result['success']:
        print(f"Output: {result['output'][:100]}...")
```

### Streaming Responses

Handle streaming LLM responses:

```python
from medgemma_agents.llm import MedGemmaClient

client = MedGemmaClient()

prompt = "Generate a detailed clinical note for..."

print("Generating response: ", end="", flush=True)
for chunk in client.complete_stream(prompt):
    print(chunk, end="", flush=True)
    time.sleep(0.01)  # Simulate reading
print()
```

### Error Handling

Proper error handling in agents:

```python
from medgemma_agents.agents.base import AgentOutput

class RobustAgent(BaseAgent):
    def process(self, input_data):
        try:
            # Validate input
            if not self.validate_input(input_data):
                return AgentOutput(
                    agent_name=self.name,
                    output="",
                    success=False,
                    error="Invalid input format"
                )
            
            # Process with error handling
            try:
                prompt = self.render_prompt(
                    "template.jinja2",
                    input_data
                )
            except FileNotFoundError:
                # Template not found, use default
                prompt = self._build_default_prompt(input_data)
            
            # Call LLM with error handling
            try:
                response = self.llm_client.complete(prompt)
            except ConnectionError:
                return AgentOutput(
                    agent_name=self.name,
                    output="",
                    success=False,
                    error="LLM service unavailable"
                )
            
            return AgentOutput(
                agent_name=self.name,
                output=response.content,
                success=True
            )
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return AgentOutput(
                agent_name=self.name,
                output="",
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
```

### Configuration from Dictionary

Create configuration programmatically:

```python
from medgemma_agents.config import Config

config_dict = {
    "agent": {
        "name": "Clinical Analyzer",
        "tasks": [
            {
                "name": "analysis",
                "prompt_template": "analysis.jinja2",
                "output_format": "json"
            }
        ]
    },
    "llm": {
        "model": "medgemma-7b",
        "temperature": 0.3,
        "max_tokens": 1024
    }
}

config = Config.from_dict(config_dict)
```

## Common Use Cases

### Clinical Documentation Assistant
- Patient intake documentation
- Progress note generation
- Discharge summary creation

### Diagnostic Support
- Differential diagnosis generation
- Lab result interpretation
- Imaging report analysis

### Treatment Planning
- Medication recommendations
- Treatment protocol selection
- Monitoring plan creation

### Education and Training
- Student case review
- Teaching point identification
- Knowledge assessment

---

See the `examples/` directory for complete working code.
