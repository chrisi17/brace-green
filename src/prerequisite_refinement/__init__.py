"""Prerequisite Refinement Module.

This module provides tools for analyzing and refining challenge prerequisites
by identifying information gaps between what agents see and what they need to know.

Key Components:
    - PrerequisiteAnalyzer: LLM-based analyzer for identifying missing information
    - PrerequisiteRefinementWorkflow: LangGraph workflow for step-by-step analysis

Usage:
    from prerequisite_refinement import PrerequisiteAnalyzer, PrerequisiteRefinementWorkflow
    
    # Create analyzer
    analyzer = PrerequisiteAnalyzer(model="gpt-4o")
    
    # Create and run workflow
    workflow = PrerequisiteRefinementWorkflow(analyzer)
    results = workflow.run("Funbox", {"model": "gpt-4o"})
"""

from .analyzer import PrerequisiteAnalyzer
from .workflow import PrerequisiteRefinementWorkflow

__all__ = [
    "PrerequisiteAnalyzer",
    "PrerequisiteRefinementWorkflow",
]

__version__ = "0.1.0"


