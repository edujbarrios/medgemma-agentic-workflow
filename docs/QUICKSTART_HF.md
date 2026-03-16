# Quick Start: Using MedGemma from Hugging Face

This guide shows how to get started with MedGemma Agents using models from Hugging Face Hub.

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU support)
- 4GB+ RAM (8GB+ recommended)

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/edujbarrios/medgemma-agents.git
cd medgemma-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with Hugging Face support
pip install -e ".[huggingface]"
```

### 2. Authenticate with Hugging Face (Optional)

For gated models, authenticate:

```bash
huggingface-cli login
# Paste your HF token when prompted
```

Or set environment variable:
```bash
export HF_TOKEN=your_token_here
```

## Quick Example

### Basic Clinical Note Generation

```python
from medgemma_agents.config import Config
from medgemma_agents.agents import ClinicalAgent
from medgemma_agents.llm import HuggingFaceMedGemmaClient

# Initialize client (auto-downloads model on first run)
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="auto",  # "cuda" for GPU, "cpu" for CPU
    torch_dtype="float16"  # Memory efficient
)

# Load configuration
config = Config.from_yaml("configs/agents/clinical_note.yaml")

# Create agent
agent = ClinicalAgent(config=config.agent, llm_client=client)

# Process patient data
result = agent.process({
    "chief_complaint": "Chest pain",
    "duration": "2 hours",
    "patient": {"age": 45, "sex": "M"}
})

print(result.output)
```

## Available Models

### MedGemma Models

- **`google/medgemma-7b-it`** (Recommended for beginners)
  - 7B parameters
  - Instruction-tuned
  - ~6GB VRAM (with float16)
  - Fastest inference

- **`google/medgemma-7b`** (Base model)
  - 7B parameters
  - Requires prompt engineering
  - ~6GB VRAM (with float16)

### Device Configuration

```python
# Option 1: Auto (recommended)
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="auto"  # Uses GPU if available, else CPU
)

# Option 2: GPU Only
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="cuda"
)

# Option 3: CPU Only
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="cpu"
)
```

### Dtype Configuration

```python
# float16: Memory efficient, recommended for VRAM < 8GB
torch_dtype="float16"

# float32: Higher precision, uses more memory
torch_dtype="float32"

# bfloat16: Good balance (only if GPU supports it)
torch_dtype="bfloat16"
```

## Running Examples

### Example 1: Basic Agent

```bash
cd examples
python basic_agent.py
```

### Example 2: Multi-Agent Workflow

```bash
cd examples
python clinical_workflow.py
```

### Example 3: Reasoning Agent

```bash
cd examples
python reasoning_example.py
```

## Optimizing VRAM Usage

### If you have < 4GB VRAM

```python
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="cpu",  # Use CPU instead
    torch_dtype="float32"
)
```

### If you have 4-8GB VRAM

```python
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="auto",
    torch_dtype="float16"  # Memory efficient
)
```

### If you have 8GB+ VRAM

```python
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="cuda",
    torch_dtype="float16"  # Balanced performance
)
```

## Environment Variables

Create `.env` file:

```env
# Hugging Face Token (for gated models)
HUGGINGFACE_TOKEN=your_token

# Model ID
MEDGEMMA_MODEL_ID=google/medgemma-7b-it

# Device
MEDGEMMA_DEVICE=auto

# Dtype
MEDGEMMA_TORCH_DTYPE=float16

# Logging
LOG_LEVEL=INFO
```

## Troubleshooting

### Out of Memory Error

Solution: Use float16 dtype and CPU:

```python
client = HuggingFaceMedGemmaClient(
    model_id="google/medgemma-7b-it",
    device="cpu",
    torch_dtype="float16"
)
```

### Model Download Issues

- Check internet connection
- Ensure enough disk space (~15GB for model + dependencies)
- Use proxy if needed:
  ```bash
  export HF_ENDPOINT=https://huggingface.co
  ```

### Slow Inference

- Use GPU (CUDA) instead of CPU
- Use float16 instead of float32
- Use smaller batch sizes

### CUDA Not Found

Install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

1. **Read the [API Reference](API.md)** for detailed class documentation
2. **Explore [Examples](../examples/)** for more use cases
3. **Check [Configuration Guide](../README.md#configuration-guide)** for custom setup
4. **Review [Architecture](ARCHITECTURE.md)** for system design

## Performance Tips

1. **Batch Processing**: Process multiple inputs together
2. **Caching**: Cache model in memory across requests
3. **Threading**: Use multi-threading for I/O operations
4. **Quantization**: Use dynamic quantization for further speedup

```python
import torch
from transformers import pipeline

# Apply quantization
pipeline_with_quantization = pipeline(
    "text2text-generation",
    model="google/medgemma-7b-it",
    device_map="auto",
    torch_dtype=torch.float16,
    # Quantization would require additional setup
)
```

## Support

- **Issues**: [GitHub Issues](https://github.com/edujbarrios/medgemma-agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/edujbarrios/medgemma-agents/discussions)
- **Documentation**: [docs/](.)

---

**Ready to build clinical agents?** Start with [basic_agent.py](../examples/basic_agent.py)! 🚀
