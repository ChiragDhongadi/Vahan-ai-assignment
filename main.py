import os
import json
import pandas as pd
import requests
import editdistance
from tqdm import tqdm
from tabulate import tabulate
from src.models.deepgram_client import DeepgramASR
from src.models.whisper_client import WhisperASR
from src.models.groq_client import GroqASR
from src.evaluation.metrics import MetricEvaluator
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
AUDIO_DIR = "audio_samples"
METADATA_FILE = "metadata.json"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ---------------------

def run_benchmark():
    if not DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY not found in .env or environment variables.")
        return

    # 1. Load Metadata
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    # 2. Initialize Models
    models = [
        DeepgramASR(api_key=DEEPGRAM_API_KEY),
        WhisperASR(model_size="large-v3"),
    ]

    if GROQ_API_KEY:
        models.append(GroqASR(api_key=GROQ_API_KEY))
    else:
        print("Notice: GROQ_API_KEY not set. Skipping Groq STT.")

    results = []
    
    print(f"\nStarting Benchmark on {len(metadata)} samples...")

    # 3. Process each audio file
    for audio_file, info in tqdm(metadata.items(), desc="Processing"):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        
        if not os.path.exists(audio_path):
            print(f"Warning: File {audio_path} not found. Skipping.")
            continue

        reference_text = info["transcript"]
        target_locality = info["locality"]
        condition = info.get("condition", "Unknown")

        for model in models:
            # Inference
            transcript = model.transcribe(audio_path)
            meta = model.get_metadata()

            # Evaluation
            wer = MetricEvaluator.calculate_wer(reference_text, transcript)
            cer = MetricEvaluator.calculate_cer(reference_text, transcript)
            is_entity_correct = MetricEvaluator.calculate_entity_accuracy(target_locality, transcript)

            results.append({
                "File": audio_file,
                "Condition": condition,
                "Model": model.model_name,
                "Reference": reference_text,
                "Hypothesis": transcript,
                "Locality": target_locality,
                "Entity_Match": is_entity_correct,
                "WER": round(wer, 4),
                "CER": round(cer, 4),
                "Latency_sec": round(meta.get("latency", 0), 2)
            })

    # 4. Save and Summarize
    if not results:
        print("\nNo files were successfully processed. Check audio_samples folder and metadata.json.")
        return

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    
    # Create Aggregate Metrics Table
    summary = df.groupby("Model").agg({
        "WER": "mean",
        "CER": "mean",
        "Entity_Match": "mean",
        "Latency_sec": "mean"
    }).reset_index()

    # Rename for professional presentation as requested
    summary.columns = ["Model", "Avg WER", "Avg CER", "Entity Accuracy", "Avg Latency (s)"]
    
    print("\n" + "="*75)
    print("FINAL AGGREGATE BENCHMARK SUMMARY")
    print("="*75)
    print(tabulate(summary, headers='keys', tablefmt='grid', showindex=False))
    print("="*75)
    print(f"\nDetailed logs saved to: benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
