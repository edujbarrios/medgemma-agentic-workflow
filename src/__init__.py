"""
MedGemma Agents - State-of-the-art agentic workflows powered by MedGemma.

A fully parameterized, production-ready framework for building autonomous
clinical agents and workflows using MedGemma LLM.

Author: Eduardo J. Barrios
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Eduardo J. Barrios"
__license__ = "Apache-2.0"

from typing import Any

try:
    from medgemma_agents.agents import BaseAgent, ClinicalAgent, ReasoningAgent
    from medgemma_agents.config import Config, AgentConfig, WorkflowConfig
    from medgemma_agents.llm import MedGemmaClient
    from medgemma_agents.workflows import WorkflowEngine
    
    __all__ = [
        "BaseAgent",
        "ClinicalAgent", 
        "ReasoningAgent",
        "Config",
        "AgentConfig",
        "WorkflowConfig",
        "MedGemmaClient",
        "WorkflowEngine",
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import medgemma_agents components: {e}")
    __all__ = []
