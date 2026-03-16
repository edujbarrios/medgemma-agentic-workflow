# Architecture

## System Design

MedGemma Agents is built with a modular, extensible architecture for creating autonomous clinical agents and workflows.

### Core Components

```
┌─────────────────────────────────────────────────┐
│  Workflow Layer                                 │
│  - Workflow Engine                              │
│  - Task Orchestration                           │
│  - Dependency Management                        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────┐
│  Agent Layer                                    │
│  - BaseAgent (Abstract)                         │
│  - ClinicalAgent                                │
│  - ReasoningAgent                               │
│  - Custom Agents                                │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────┐
│  LLM Integration Layer                          │
│  - MedGemmaClient                               │
│  - Streaming Support                            │
│  - Error Handling & Retries                     │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────┐
│  Configuration & Templates Layer                │
│  - Config Management (YAML/JSON)                │
│  - Jinja2 Template Engine                       │
│  - Environment Variable Substitution            │
└─────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Configuration Module (`src/config/`)

**Purpose**: Load, validate, and manage all system configuration.

**Key Classes**:
- `Config`: Main configuration wrapper
- `AgentConfig`: Agent-specific configuration
- `WorkflowConfig`: Workflow definition configuration
- `LLMConfig`: LLM client configuration
- `TaskConfig`: Individual task configuration

**Features**:
- YAML/JSON file loading
- Pydantic validation
- Environment variable substitution
- Configuration persistence

### 2. LLM Module (`src/llm/`)

**Purpose**: Provide abstraction layer for LLM interactions.

**Key Classes**:
- `MedGemmaClient`: MedGemma API client
- `LLMResponse`: Response wrapper with metadata

**Features**:
- Completion endpoints (text generation)
- Chat API support
- Streaming responses
- Token counting
- Automatic retries with backoff
- Health check monitoring

### 3. Agents Module (`src/agents/`)

**Purpose**: Implement autonomous agents for clinical tasks.

**Key Classes**:
- `BaseAgent`: Abstract base class for all agents
- `ClinicalAgent`: General clinical task agent
- `ClinicalNoteAgent`: Specialized for note generation
- `DiagnosisAgent`: For diagnosis assistance
- `TreatmentAgent`: For treatment recommendations
- `LabAnalysisAgent`: For lab result analysis
- `ReasoningAgent`: Multi-step reasoning
- `ChainOfThoughtAgent`: Chain-of-thought reasoning
- `TreeOfThoughtAgent`: Tree-of-thought reasoning

**Agent Lifecycle**:
```
Input Data
    ↓
Validation
    ↓
Prompt Rendering (Jinja2)
    ↓
LLM Call
    ↓
Output Formatting
    ↓
AgentOutput
```

### 4. Templates Module (`src/templates/`)

**Purpose**: Dynamic prompt generation using Jinja2.

**Key Classes**:
- `TemplateEngine`: Jinja2-based template rendering
- Custom filters for formatting

**Features**:
- File-based templates
- String-based templates
- Custom filters (format_list, format_dict, etc.)
- Variable substitution
- Loop and conditional support

### 5. Workflows Module (`src/workflows/`)

**Purpose**: Orchestrate multi-agent workflows.

**Key Classes**:
- `WorkflowEngine`: Main workflow execution engine
- `Workflow`: Individual workflow instance
- `TaskResult`: Task execution result

**Features**:
- DAG-based task execution
- Dependency resolution
- Context passing between tasks
- Parallel task support (future)
- Error handling and recovery

## Data Flow

### Single Agent Execution

```
Agent Input Data
↓
Configuration Load
↓
Template Rendering (with variables)
↓
LLM Inference
↓
Output Formatting
↓
AgentOutput (with metrics)
```

### Workflow Execution

```
Initial Input
↓
Topological Dependency Sort
↓
For Each Task (in order):
  ├─ Build Task Input (from context + dependencies)
  ├─ Execute Agent
  ├─ Store Result in Context
  └─ Handle Errors
↓
Aggregate Results
↓
Return Workflow Summary
```

## Configuration Hierarchy

```yaml
# Top-level Config
config:
  agent:      # AgentConfig
    name: ...
    tasks:    # List of TaskConfig
      - name: ...
        prompt_template: ...
        parameters: ...
  
  workflow:   # WorkflowConfig
    name: ...
    tasks:    # List of WorkflowTaskConfig
      - agent: ...
        depends_on: ...
  
  llm:        # LLMConfig
    model: ...
    temperature: ...
  
  logging:    # Logging configuration
    level: ...
```

## Extension Points

### Custom Agents

Extend `BaseAgent`:

```python
class MyCustomAgent(BaseAgent):
    def process(self, input_data):
        # Custom logic
        pass
```

### Custom Filters

Add to `TemplateEngine`:

```python
engine.env.filters['my_filter'] = my_filter_function
```

### Custom Task Types

Create new workflows that use custom agents.

## Error Handling

### Agent-Level

- Input validation
- Template rendering errors
- LLM API errors (with retries)
- Output formatting errors

### Workflow-Level

- Missing agent errors
- Dependency resolution errors
- Task timeout handling
- Cascading error handling

## Performance Considerations

1. **Prompt Optimization**: Keep prompts focused to minimize token usage
2. **Caching**: Implement response caching for identical inputs
3. **Batching**: Process multiple inputs in batch when possible
4. **Streaming**: Use streaming responses for long outputs
5. **Async**: Support async/await operations

## Security

1. **API Keys**: Store in environment variables
2. **Input Validation**: Validate all inputs at agent level
3. **Output Sanitization**: Sanitize outputs if needed
4. **Logging**: Remove sensitive data from logs

## Scalability

- Modular design allows for horizontal scaling
- Agents can be distributed
- Workflow engine supports external task execution
- LLM client handles connection pooling

---

For more details, see the [API Reference](API.md) and [Examples](../examples/).
