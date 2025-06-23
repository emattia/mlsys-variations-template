"""
AI Agent Framework Module

Implements the AI agent roadmap components for Data Scientist → AI Agents:
- Agent frameworks (LangGraph, CrewAI)
- RAG systems with vector stores
- LLM providers (local/remote)
- Prompt engineering utilities
- Monitoring and token estimation
- Multi-agent orchestration

This module provides a minimal but extensible foundation that can be enhanced
with additional frameworks and capabilities over time.
"""

from .agents import Agent, AgentConfig, AgentRegistry, AgentType, BaseAgent
from .llm import LLMFactory, LLMProvider, TokenEstimator
from .monitoring import AgentMonitor, CostTracker, PerformanceAnalytics
from .prompts import PromptEngineering, PromptLibrary, PromptTemplate, PromptValidator
from .rag import RAGConfig, RAGPipeline, RAGSystem, VectorStore
from .tools import BaseTool, ToolRegistry

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentRegistry",
    "AgentType",
    "BaseAgent",
    "LLMFactory",
    "LLMProvider",
    "TokenEstimator",
    "AgentMonitor",
    "CostTracker",
    "PerformanceAnalytics",
    "PromptTemplate",
    "PromptLibrary",
    "PromptValidator",
    "PromptEngineering",
    "RAGConfig",
    "RAGPipeline",
    "RAGSystem",
    "VectorStore",
    "ToolRegistry",
    "BaseTool",
]
