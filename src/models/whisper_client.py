import os
import sys
import time
import logging
import static_ffmpeg
from faster_whisper import WhisperModel
from typing import Dict, Any
from src.core.interfaces import ASRModel

# Ensure ffmpeg is in the path
static_ffmpeg.add_paths()

# --- WINDOWS GPU DLL HACK ---
# This part helps Python find the cublas and cudnn DLLs installed via pip
if sys.platform == "win32":
    # Get the base path of the venv
    venv_base = os.path.dirname(os.path.dirname(sys.executable))
    venv_site_packages = os.path.join(venv_base, "Lib", "site-packages")
    
    # List of specific paths where DLLs are located
    paths_to_add = [
        os.path.join(venv_site_packages, "nvidia", "cublas", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cudnn", "bin"),
    ]
    
    for path in paths_to_add:
        if os.path.exists(path):
            print(f"Adding DLL directory: {path}")
            os.add_dll_directory(path)
            # Also add to PATH for good measure
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
# ----------------------------

logger = logging.getLogger(__name__)

class WhisperASR(ASRModel):
    def __init__(self, model_size: str = "large-v3"):
        super().__init__(model_name=f"Whisper-{model_size}")
        
        # Now attempting GPU load with the new DLL paths
        try:
            print(f"Attempting to load {self.model_name} on GPU (CUDA)...")
            # We use float16 for high-speed GPU inference
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print("SUCCESS: Whisper is running on your GPU!")
        except Exception as e:
            print(f"GPU Load Failed even with hack: {e}")
            print("Falling back to CPU mode (int8)...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            
        self.last_latency = 0.0

    def transcribe(self, audio_path: str) -> str:
        start_time = time.time()
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments]).strip()
            self.last_latency = time.time() - start_time
            return transcript
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path}: {e}")
            return ""

    def get_metadata(self) -> Dict[str, Any]:
        device = "GPU (CUDA)" if self.model.model.device == "cuda" else "CPU"
        return {
            "model_id": self.model_name,
            "provider": f"Faster-Whisper ({device})",
            "latency": self.last_latency
        }
