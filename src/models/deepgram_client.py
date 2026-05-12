import time
import logging
from typing import Dict, Any
from deepgram import DeepgramClient
from src.core.interfaces import ASRModel

logger = logging.getLogger(__name__)

class DeepgramASR(ASRModel):
    def __init__(self, api_key: str, model: str = "nova-2", language: str = "hi"):
        super().__init__(model_name=f"Deepgram-{model}")
        self.client = DeepgramClient(api_key=api_key)
        self.model = model
        self.language = language
        self.last_latency = 0.0

    def transcribe(self, audio_path: str) -> str:
        start_time = time.time()
        try:
            with open(audio_path, "rb") as file:
                buffer_data = file.read()

            # In v7 SDK, we pass options as keyword arguments directly
            response = self.client.listen.v1.media.transcribe_file(
                request=buffer_data,
                model=self.model,
                language=self.language,
                smart_format=True,
                punctuate=True
            )
            
            self.last_latency = time.time() - start_time
            
            # Access the transcript from the response object
            # Note: v7 responses are often objects, but check if it's a dict
            if hasattr(response, 'results'):
                transcript = response.results.channels[0].alternatives[0].transcript
            else:
                # Fallback if it returns a dict (depending on client version)
                transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
                
            return transcript
        
        except Exception as e:
            logger.error(f"Deepgram transcription failed for {audio_path}: {e}")
            return ""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model,
            "provider": "Deepgram",
            "latency": self.last_latency
        }
