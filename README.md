# Call Audit Automation System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.6+-e92063.svg)](https://docs.pydantic.dev/)
[![Status](https://img.shields.io/badge/Status-Project%20Initialization-success.svg)](#)

An AI-powered Call Audit Automation System designed for financial services compliance (HDFC Securities). The system automates end-to-end audit workflows—from mono audio processing and speaker diarization to KPI policy validation, explanation email generation, and branch reply summarization—with final compliance decisions retained by human auditors.

---

## 1. Project Overview & Architecture

### Four Automation Stages

1. **Audio Processing (`app/audio/`, `app/transcription/`, `app/diarization/`)**
   - Ingestion and audio normalization of mono audio recordings.
   - Speech-to-text transcription powered by `faster-whisper`.
   - Speaker diarization via `pyannote.audio` and algorithmic distinction between Relationship Manager (RM) and Customer voices to eliminate false-positive misconduct flags.

2. **KPI Validation (`app/validation/`)**
   - Triangulation of verified transcripts, original AI justifications from client reports, and statutory KPI policies (`config/kpi_conditions.json`).
   - Categorization into confirmed violations, proven false positives, or cases requiring manual human review.

3. **Email Generation (`app/email_service/`)**
   - Automated generation of professional, fact-referenced explanation request emails for confirmed violations to RMs and branch supervisors.

4. **Reply Summarization & Reporting (`app/summarization/`, `app/reports/`)**
   - Synthesis of RM/branch responses and creation of consolidated compliance audit reports.

---

## 2. Directory Structure

```text
call-audit-automation/
│
├── app/                        # Main application package
│   ├── __init__.py
│   ├── main.py                 # FastAPI application factory & routes
│   ├── api/                    # API route definitions
│   │   └── __init__.py
│   ├── core/                   # Core application configurations & logging
│   │   ├── __init__.py
│   │   ├── config.py           # Pydantic Settings & dynamic pathlib path management
│   │   └── logging.py          # Rotating file & console logging setup
│   ├── audio/                  # Audio preprocessing & format normalization
│   │   └── __init__.py
│   ├── transcription/          # Faster-Whisper STT engine
│   │   └── __init__.py
│   ├── diarization/            # Pyannote speaker diarization & RM/Customer assignment
│   │   └── __init__.py
│   ├── validation/             # KPI policy validation & Gemini evaluation
│   │   └── __init__.py
│   ├── email_service/          # Discrepancy email generation service
│   │   └── __init__.py
│   ├── summarization/          # Explanation reply summarizer
│   │   └── __init__.py
│   ├── reports/                # Consolidated audit report generation
│   │   └── __init__.py
│   └── utils/                  # Shared helper functions
│       └── __init__.py
│
├── input/                      # Input artifacts (strictly excluded from Git)
│   ├── recordings/             # Mock & POC MP3/WAV audio recordings
│   └── excel/                  # Client observation & KPI condition sheets
│
├── output/                     # Generated audit artifacts (excluded from Git)
│   ├── transcripts/            # Timestamped diarized transcripts
│   ├── validation_results/     # KPI validation results & false-positive flags
│   └── reports/                # Final consolidated audit reports
│
├── config/                     # Configuration definitions
│   └── kpi_conditions.json     # 10 baseline financial compliance KPI definitions
│
├── tests/                      # Automated test suite
│   ├── __init__.py
│   └── test_health.py          # Health check & configuration tests
│
├── logs/                       # Application runtime logs (excluded from Git)
│   └── app.log
│
├── .env.example                # Environment variables template
├── .env                        # Local active configuration (ignored by Git)
├── .gitignore                  # Comprehensive Git exclusions for client data
├── requirements.txt            # Core backend dependencies
├── requirements-audio.txt      # Audio, diarization, & ML dependencies
├── README.md                   # Project documentation & setup instructions
└── run.py                      # Application runner script
```

---

## 3. Prerequisites

- **Python 3.11** (64-bit recommended)
- **Windows PowerShell** (or Bash/Zsh)
- **FFmpeg** (required for audio preprocessing and Whisper conversion):
  ```powershell
  winget install "Gyan.FFmpeg"
  ```
  Verify installation:
  ```powershell
  ffmpeg -version
  ```

---

## 4. Step-by-Step Setup (Windows PowerShell)

Follow these exact commands from the project root (`call-audit-automation/`):

### 1. Create Virtual Environment
```powershell
py -3.11 -m venv .venv
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### 3. Install Core Dependencies
Installs FastAPI, Uvicorn, Pydantic, Pandas, OpenPyXL, and Google GenAI SDK:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Audio & ML Dependencies (Separate Step)
Audio and speaker diarization dependencies (`faster-whisper`, `pyannote.audio`) require PyTorch.

**A. Install PyTorch first:**
- For CPU (Standard):
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```
- For NVIDIA GPU (CUDA 12.1+):
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

**B. Install audio processing packages:**
```powershell
pip install -r requirements-audio.txt
```

### 5. Configure Environment Variables
Copy the template configuration:
```powershell
Copy-Item .env.example .env
```
*Note: The initial health check and core backend run completely without API keys.*

### 6. Start the FastAPI Server
Using the application runner:
```powershell
python run.py
```
Or directly via Uvicorn:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 7. Test the Health Endpoint
In a separate PowerShell terminal:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```
**Expected Response:**
```json
{
  "status": "healthy",
  "service": "call-audit-automation"
}
```

---

## 5. Running Automated Tests

Run the test suite using `pytest`:
```powershell
pytest -v tests/test_health.py
```

---

## 6. API Documentation

Interactive Swagger and ReDoc documentation is available at runtime:
- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

---

## 7. Milestone 2: Audio Preprocessing & Quality Analysis

The audio processing pipeline handles discovery, quality validation, audio standardization, and audit reporting for raw telephony recordings.

### Architecture

- `app/audio/metadata.py`: Extracts container format, duration, channels, sample rate, bit depth, and bitrate via FFprobe into typed `AudioMetadata` objects.
- `app/audio/quality.py`: Evaluates average RMS volume, peak amplitude, sample saturation/clipping, and silence intervals using FFmpeg audio filters (`volumedetect`, `silencedetect`). Issues descriptive warnings rather than rejecting files.
- `app/audio/preprocess.py`: Converts input recordings to standardized **16,000 Hz, mono (1 channel), 16-bit PCM WAV (`pcm_s16le`)** without touching or modifying the original audio files.
- `app/audio/processor.py`: Orchestrates discovery, analysis, conversion, and report generation.
- `scripts/process_audio.py`: Standalone CLI tool with pre-flight binary checks and executive summary formatting.

### Executing the Audio Pipeline

Run the CLI tool from the project root:

```powershell
python scripts/process_audio.py
```

Optional flags:
```powershell
python scripts/process_audio.py --input-dir "custom/path" --output-dir "custom/output"
```

### Generated Artifacts

- **Standardized Audio Files:** Saved in `output/processed_audio/` as `<stem>.wav` (16kHz mono).
- **JSON Quality Report:** `output/audio_reports/audio_quality_report.json` containing complete metadata and metrics for all processed files.
- **CSV Summary Sheet:** `output/audio_reports/audio_quality_summary.csv` for quick inspection in Excel.

### Running Audio Tests

```powershell
pytest -v tests/test_audio.py
```

---

---

## 8. Milestone 3: Speech-to-Text Transcription (Faster-Whisper)

Milestone 3 implements local, privacy-preserving Speech-to-Text (STT) for standardized 16kHz mono WAV audio recordings using `faster-whisper` (CTranslate2-optimized Whisper engine).

### Architecture

- `app/transcription/model.py`: Singleton-style model manager (`WhisperModelManager`) ensuring model weights are loaded once in memory and reused across all audio calls. Supports both CPU (`int8`) and GPU CUDA execution.
- `app/transcription/transcript_schema.py`: Pydantic schema validation for sequential segments, word-level start/end timestamps, language probabilities, and complete call transcripts (`Transcript`, `TranscriptSegment`, `WordTimestamp`).
- `app/transcription/transcriber.py`: Core transcription service (`AudioTranscriber`). Implements automatic language detection, Silero VAD silence filtering, word-level timestamps, code-switching preservation (Hinglish/English terms), and error handling.
- `scripts/transcribe_audio.py`: Standalone CLI batch runner that scans `output/processed_audio/`, transcribes recordings sequentially, writes individual JSON transcripts to `output/transcripts/<recording_id>.json`, and produces an executive batch summary in `output/transcripts/transcription_summary.json`.

### Configuration Options (.env)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WHISPER_MODEL` | `small` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `cpu` | Device target (`cpu` or `cuda`) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute quantization precision (`int8`, `float16`, `float32`) |
| `WHISPER_BEAM_SIZE` | `5` | Beam search decoding width |
| `WHISPER_VAD_FILTER` | `true` | Silero voice activity detection filter to minimize hallucinations |
| `WHISPER_WORD_TIMESTAMPS` | `true` | Extract token-level start and end timings |

### Running Batch Transcription

```powershell
python scripts/transcribe_audio.py
```

Optional CLI overrides:
```powershell
python scripts/transcribe_audio.py --model medium --device cpu
```

### Running Transcription Tests

```powershell
pytest -v tests/test_transcription.py
```

---

## 9. Data Privacy & Security Rules

- **Confidential Recordings:** Customer calls contain financial and personally identifiable information (PII). Audio files must never be committed to Git or uploaded to unauthorized cloud storage.
- **Client Spreadsheets:** Raw observation workbooks from HDFC Securities must stay inside `input/excel/` and are strictly excluded via `.gitignore`.
- **Zero API Key Leakage:** Credentials (`GEMINI_API_KEY`, `HUGGINGFACE_TOKEN`) must reside exclusively in `.env` and must never be referenced directly in code.
