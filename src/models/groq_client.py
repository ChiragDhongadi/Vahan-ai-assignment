import os
import time
import logging
import io
import subprocess
import static_ffmpeg
from groq import Groq
from typing import Dict, Any
from src.core.interfaces import ASRModel

# Ensure ffmpeg is in the path
static_ffmpeg.add_paths()

logger = logging.getLogger(__name__)

class GroqASR(ASRModel):
    def __init__(self, api_key: str, model_id: str = "whisper-large-v3-turbo"):
        super().__init__(model_name=f"Groq-{model_id}")
        self.client = Groq(api_key=api_key)
        self.model_id = model_id
        self.last_latency = 0.0

    def transcribe(self, audio_path: str) -> str:
        start_time = time.time()
        try:
            # Groq doesn't support .aac directly. 
            # We use ffmpeg to convert it to a supported format (WAV) in memory.
            
            # Command to convert input to wav and output to stdout
            command = [
                'ffmpeg',
                '-i', audio_path,
                '-f', 'wav',
                '-ar', '16000', # Sample rate 16kHz
                '-ac', '1',     # Mono
                'pipe:1'        # Output to stdout
            ]
            
            # Run conversion
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")

            # Send converted audio to Groq
            # We name it 'audio.wav' so the API knows the format
            transcription = self.client.audio.transcriptions.create(
                file=('audio.wav', stdout),
                model=self.model_id,
                temperature=0,
                response_format="verbose_json",
            )
            
            self.last_latency = time.time() - start_time
            return transcription.text.strip()
            
        except Exception as e:
            logger.error(f"Groq transcription failed for {audio_path}: {e}")
            return ""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "provider": "Groq Cloud",
            "latency": self.last_latency
        }
