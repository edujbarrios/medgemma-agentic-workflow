#!/usr/bin/env python
"""
Multi-agent workflow example: Complete clinical assessment.

Demonstrates running a multi-step workflow with multiple agents
using MedGemma from Hugging Face.
"""

from medgemma_agents.config import Config
from medgemma_agents.agents import ClinicalAgent, DiagnosisAgent, TreatmentAgent
from medgemma_agents.llm import HuggingFaceMedGemmaClient
from medgemma_agents.workflows import WorkflowEngine
import json


def create_agents(llm_client: HuggingFaceMedGemmaClient) -> dict:
    """Create and configure agents."""
    agents = {}
    
    # Clinical Note Agent
    print("Creating Clinical Note Agent...")
    note_config = Config.from_yaml("configs/agents/clinical_note.yaml")
    agents["clinical_note"] = ClinicalAgent(
        config=note_config.agent,
        llm_client=llm_client
    )
    
    # Diagnosis Agent
    print("Creating Diagnosis Agent...")
    diagnosis_config = Config.from_yaml("configs/agents/diagnosis.yaml")
    agents["diagnosis"] = DiagnosisAgent(
        config=diagnosis_config.agent,
        llm_client=llm_client
    )
    
    # Treatment Agent
    print("Creating Treatment Agent...")
    treatment_config = Config.from_yaml("configs/agents/treatment.yaml")
    agents["treatment"] = TreatmentAgent(
        config=treatment_config.agent,
        llm_client=llm_client
    )
    
    return agents


def main():
    # Initialize Hugging Face LLM client
    print("Loading MedGemma model from Hugging Face...")
    print("(This may take a few minutes on first run)\n")
    llm_client = HuggingFaceMedGemmaClient(
        model_id="google/medgemma-7b-it",
        device="auto",
        torch_dtype="float16"
    )
    
    # Create agents
    agents = create_agents(llm_client)
    
    # Create workflow engine
    print("Initializing Workflow Engine...")
    engine = WorkflowEngine()
    
    # Register agents
    for name, agent in agents.items():
        engine.register_agent(name, agent)
    
    # Load and create workflow
    print("Loading workflow configuration...")
    workflow_config = Config.from_yaml("configs/workflows/clinical_assessment.yaml")
    workflow = engine.create_workflow(workflow_config.workflow)
    
    # Example patient data
    patient_data = {
        "patient": {
            "age": 45,
            "sex": "M",
            "mrn": "12345"
        },
        "chief_complaint": "Chest pain for 2 hours",
        "hpi": "45-year-old male with acute chest pain, shortness of breath, diaphoresis",
        "past_medical_history": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Lisinopril 10mg", "Metformin 1000mg"],
        "symptoms": ["chest pain", "shortness of breath", "diaphoresis", "fatigue"],
        "vitals": {
            "temperature": "37.2C",
            "heart_rate": "102",
            "blood_pressure": "145/92",
            "oxygen_saturation": "96%"
        }
    }
    
    # Execute workflow
    print(f"\n{'='*60}")
    print(f"Executing workflow: {workflow.name}")
    print(f"{'='*60}\n")
    
    results = workflow.execute(patient_data, verbose=True)
    
    # Display summary
    print(f"\n{'='*60}")
    print("WORKFLOW EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Workflow: {results['workflow']}")
    print(f"Success: {results['success']}")
    print(f"Total Time: {results['execution_time']:.2f}s")
    
    # Display task results
    print("\n" + "-"*60)
    print("TASK RESULTS")
    print("-"*60)
    for task_name, task_result in results['results'].items():
        print(f"\nTask: {task_name}")
        print(f"Status: {'✓ Success' if task_result['success'] else '✗ Failed'}")
        if not task_result['success']:
            print(f"Error: {task_result['error']}")
        print(f"Tokens: {task_result['tokens']}")
        print(f"Time: {task_result['time']:.2f}s")
    
    # Save detailed results
    with open("workflow_results.json", "w") as f:
        # Remove context (too verbose) for file output
        results_to_save = {k: v for k, v in results.items() if k != "context"}
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n✓ Detailed results saved to workflow_results.json")
    
    # Create agents
    agents = create_agents(llm_client)
    
    # Create workflow engine
    print("Initializing Workflow Engine...")
    engine = WorkflowEngine()
    
    # Register agents
    for name, agent in agents.items():
        engine.register_agent(name, agent)
    
    # Load and create workflow
    print("Loading workflow configuration...")
    workflow_config = Config.from_yaml("configs/workflows/clinical_assessment.yaml")
    workflow = engine.create_workflow(workflow_config.workflow)
    
    # Example patient data
    patient_data = {
        "patient": {
            "age": 45,
            "sex": "M",
            "mrn": "12345"
        },
        "chief_complaint": "Chest pain for 2 hours",
        "hpi": "45-year-old male with acute chest pain, shortness of breath, diaphoresis",
        "past_medical_history": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Lisinopril 10mg", "Metformin 1000mg"],
        "symptoms": ["chest pain", "shortness of breath", "diaphoresis", "fatigue"],
        "vitals": {
            "temperature": "37.2C",
            "heart_rate": "102",
            "blood_pressure": "145/92",
            "oxygen_saturation": "96%"
        }
    }
    
    # Execute workflow
    print(f"\n{'='*60}")
    print(f"Executing workflow: {workflow.name}")
    print(f"{'='*60}\n")
    
    results = workflow.execute(patient_data, verbose=True)
    
    # Display summary
    print(f"\n{'='*60}")
    print("WORKFLOW EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Workflow: {results['workflow']}")
    print(f"Success: {results['success']}")
    print(f"Total Time: {results['execution_time']:.2f}s")
    
    # Display task results
    print("\n" + "-"*60)
    print("TASK RESULTS")
    print("-"*60)
    for task_name, task_result in results['results'].items():
        print(f"\nTask: {task_name}")
        print(f"Status: {'✓ Success' if task_result['success'] else '✗ Failed'}")
        if not task_result['success']:
            print(f"Error: {task_result['error']}")
        print(f"Tokens: {task_result['tokens']}")
        print(f"Time: {task_result['time']:.2f}s")
    
    # Save detailed results
    with open("workflow_results.json", "w") as f:
        # Remove context (too verbose) for file output
        results_to_save = {k: v for k, v in results.items() if k != "context"}
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n✓ Detailed results saved to workflow_results.json")


if __name__ == "__main__":
    main()
