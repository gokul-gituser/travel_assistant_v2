"""
Travel Assistant LangGraph Module
A reusable module for a travel chatbot with:
- Intent classification and routing
- Location-aware recommendations
- User profile memory (Redis)
- Travel history
- Nearby places context (injected by caller)
"""

from .core import run_travel_assistant

__version__ = "1.0.0"
__author__ = "gokul"
__all__ = ["run_travel_assistant"]