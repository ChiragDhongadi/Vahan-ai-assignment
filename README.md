# Multimodal ASR Benchmarking Pipeline: Hinglish Telephony Analysis

A production-grade evaluation suite designed to benchmark ASR systems on noisy Indian conversational speech (Hinglish), specifically optimized for Bangalore-based locality name extraction.

## 🚀 Key Features
- **3-Way Model Comparison**: Integrated **Deepgram Nova-2** (Cloud), **Faster-Whisper** (Local GPU), and **Groq Whisper Turbo** (LPU Cloud).
- **Cross-Script Entity Matching**: Implemented a **Transliteration Layer** + **Fuzzy Matching** to recover entities across Roman and Devanagari scripts.
- **Hardware-Aware Benchmarking**: Optimized for local CUDA acceleration (RTX 3050) and high-speed LPU inference.
- **Deep Analytics**: Professional Jupyter Notebook for data visualization and failure analysis.

## 📊 Performance Summary (20 Samples)
| Model | Avg WER | Entity Accuracy | Avg Latency | Deployment |
| :--- | :--- | :--- | :--- | :--- |
| **Groq Whisper Turbo** | 1.05 | 41.1% | **0.34s** | LPU Accelerated |
| **Deepgram Nova-2** | **0.81** | **58.8%** | 0.67s | Production API |
| **Whisper Large-v3** | 1.06 | **58.8%** | 1.84s | Local GPU |

## 📂 Project Structure
- `main.py`: Core benchmark orchestrator.
- `src/models/`: Modular ASR wrappers (Deepgram, Whisper, Groq).
- `src/evaluation/`: Custom metrics and transliteration-aware fuzzy matching.
- `ASR_Benchmark_Analysis.ipynb`: Data visualization and insights.
- `audio_samples/`: 20 real-world recordings (Traffic, Fan, Quiet Room).

## 🛠️ Setup & Usage
1. **Environment Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Configuration**:
   Add your keys to a `.env` file:
   ```text
   DEEPGRAM_API_KEY=your_key
   GROQ_API_KEY=your_key
   ```
3. **Execution**:
   ```bash
   python main.py
   ```

## 📝 Engineering Insights
This project highlights the "Script Bias" in standard ASR systems and demonstrates a software engineering solution (Transliteration) to recover 50%+ more entities from Devanagari-biased models.
