"""Configuration module for the bioeconomic products analyzer."""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from .settings import settings

# Remove the circular import - don't import PanAmazonPromptManager here
# Instead, provide a factory function to create it when needed

def get_prompt_manager():
    """Factory function to create prompt manager instance when needed."""
    from src.llm.prompt_manager import PanAmazonPromptManager
    return PanAmazonPromptManager()

# For backward compatibility, create a lazy-loaded prompt_manager
class LazyPromptManager:
    """Lazy-loaded prompt manager to avoid circular imports."""
    def __init__(self):
        self._instance = None
    
    def __getattr__(self, name):
        if self._instance is None:
            self._instance = get_prompt_manager()
        return getattr(self._instance, name)

prompt_manager = LazyPromptManager()

# Create alias for backward compatibility  
def PromptManager():
    """Factory function for creating PanAmazonPromptManager instances."""
    return get_prompt_manager()

__all__ = [
    "settings",
    "prompt_manager", 
    "PromptManager",
    "get_prompt_manager"
]