from abc import ABC, abstractmethod
from typing import Dict, Any

class ASRModel(ABC):
    """Abstract base class for all ASR models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the text."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata (latency, cost estimates, etc)."""
        pass
