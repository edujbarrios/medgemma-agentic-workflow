#!/usr/bin/env python
"""
Advanced example: Custom reasoning agent with chain-of-thought.

Demonstrates implementing a custom agent with advanced reasoning
using MedGemma from Hugging Face.
"""

from medgemma_agents.config import Config, AgentConfig, TaskConfig
from medgemma_agents.agents import ChainOfThoughtAgent
from medgemma_agents.llm import HuggingFaceMedGemmaClient
from pathlib import Path


def create_custom_config() -> AgentConfig:
    """Create custom chain-of-thought configuration."""
    return AgentConfig(
        name="Clinical Reasoning Agent",
        description="Multi-step clinical reasoning with chain-of-thought",
        version="1.0",
        tasks=[
            TaskConfig(
                name="symptom_analysis",
                description="Analyze patient symptoms and vital signs",
                prompt_template="prompts/diagnosis.jinja2",
                output_format="text",
                parameters={
                    "include_confidence": True,
                    "include_rationale": True
                }
            ),
            TaskConfig(
                name="differential_diagnosis",
                description="Develop differential diagnosis list",
                prompt_template="prompts/diagnosis.jinja2",
                output_format="json",
                parameters={
                    "max_suggestions": 3
                }
            ),
            TaskConfig(
                name="diagnostic_plan",
                description="Propose diagnostic testing strategy",
                prompt_template="prompts/treatment.jinja2",
                output_format="text",
                parameters={
                    "include_monitoring": True
                }
            )
        ],
        output_format="text"
    )


def main():
    print("="*60)
    print("CHAIN-OF-THOUGHT CLINICAL REASONING")
    print("="*60)
    
    # Initialize LLM
    print("Loading MedGemma model from Hugging Face...")
    print("(This may take a few minutes on first run)\n")
    llm_client = HuggingFaceMedGemmaClient(
        model_id="google/medgemma-7b-it",
        device="auto",
        torch_dtype="float16"
    )
    
    # Create custom agent
    print("Creating custom reasoning agent...")
    config = create_custom_config()
    agent = ChainOfThoughtAgent(
        config=config,
        llm_client=llm_client
    )
    
    # Clinical case
    case_data = {
        "age": 68,
        "sex": "F",
        "chief_complaint": "Progressive shortness of breath",
        "duration": "3 weeks",
        "symptoms": [
            "Dyspnea on exertion",
            "Orthopnea",
            "Bilateral ankle edema",
            "Weight gain (5 lbs in 1 week)"
        ],
        "vitals": {
            "heart_rate": "94",
            "blood_pressure": "138/85",
            "respiratory_rate": "22",
            "oxygen_saturation": "92% on room air"
        },
        "past_medical_history": ["Hypertension", "Hyperlipidemia", "Type 2 Diabetes"],
        "medications": [
            "Lisinopril 10mg daily",
            "Atorvastatin 20mg daily",
            "Metformin 1000mg twice daily"
        ],
        "test_results": {
            "BNP": "450 pg/mL (elevated)",
            "Creatinine": "1.1 mg/dL (normal)",
            "Hemoglobin": "13.2 g/dL (normal)"
        }
    }
    
    print(f"\nClinical Case:")
    print(f"  Patient: {case_data['age']}yo {case_data['sex']}")
    print(f"  Chief Complaint: {case_data['chief_complaint']}")
    print(f"  Duration: {case_data['duration']}")
    
    # Process with chain-of-thought
    print(f"\nExecuting chain-of-thought reasoning...")
    print(f"{'-'*60}")
    
    result = agent.process(case_data)
    
    # Display results
    print(f"\n{'-'*60}")
    print(f"REASONING OUTPUT")
    print(f"-"*60)
    print(result.output)
    
    print(f"\n{'-'*60}")
    print(f"EXECUTION SUMMARY")
    print(f"-"*60)
    print(f"Agent: {result.agent_name}")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Tokens Used: {result.tokens_used}")
    
    if not result.success:
        print(f"Error: {result.error}")
    
    # Save output
    output_file = "reasoning_output.txt"
    with open(output_file, "w") as f:
        f.write(f"Clinical Reasoning Analysis\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Case: {case_data['age']}yo {case_data['sex']} - {case_data['chief_complaint']}\n")
        f.write(f"\n" + result.output)
    
    print(f"\n✓ Output saved to {output_file}")


if __name__ == "__main__":
    main()
