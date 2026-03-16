#!/usr/bin/env python
"""
Basic example: Single agent clinical note generation.

Shows how to load configuration and run a single clinical agent
using MedGemma from Hugging Face.
"""

from medgemma_agents.config import Config
from medgemma_agents.agents import ClinicalAgent
from medgemma_agents.llm import HuggingFaceMedGemmaClient
import json


def main():
    # Load configuration
    print("Loading configuration...")
    config = Config.from_yaml("configs/agents/clinical_note.yaml")
    
    # Initialize Hugging Face LLM client
    print("Loading MedGemma model from Hugging Face...")
    print("(This may take a few minutes on first run)")
    llm_client = HuggingFaceMedGemmaClient(
        model_id="google/medgemma-7b-it",  # Instruction-tuned version
        device="auto",  # "cuda" for GPU, "cpu" for CPU
        torch_dtype="float16"  # Use float16 for memory efficiency
    )
    
    # Create agent
    print(f"Creating agent: {config.agent.name}")
    agent = ClinicalAgent(
        config=config.agent,
        llm_client=llm_client
    )
    
    # Example patient data
    patient_data = {
        "patient": {
            "age": 45,
            "sex": "M",
            "mrn": "12345"
        },
        "chief_complaint": "Chest pain for 2 hours",
        "hpi": "45-year-old male presents with acute onset retrosternal chest pain radiating to left arm, associated with shortness of breath and diaphoresis. Patient has history of hypertension.",
        "past_medical_history": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Lisinopril 10mg daily", "Metformin 1000mg twice daily"],
        "vitals": {
            "temperature": "37.2 C",
            "heart_rate": "102 bpm",
            "blood_pressure": "145/92 mmHg",
            "respiratory_rate": "18 breaths/min",
            "oxygen_saturation": "96% on room air"
        },
        "findings": "Cardiac auscultation: tachycardia, no murmurs. Lungs: clear to auscultation bilaterally. Abdomen: soft, non-tender."
    }
    
    # Process patient data
    print(f"\nProcessing clinical case...")
    result = agent.process(patient_data)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Agent: {result.agent_name}")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Tokens Used: {result.tokens_used}")
    print(f"{'='*60}\n")
    
    if result.success:
        print(f"--- Clinical Note Output ---\n")
        print(result.output)
    else:
        print(f"Error: {result.error}")
    
    # Save output to file
    with open("example_output.txt", "w") as f:
        f.write(result.output)
    print("\n✓ Output saved to example_output.txt")


if __name__ == "__main__":
    main()
