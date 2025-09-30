# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready Text-to-Speech (TTS) system called "Orpheus TTS Multilingual Server" that uses Hugging Face transformers and the SNAC audio codec for high-quality speech synthesis. The project supports multiple languages with independent servers, features multi-GPU parallel processing, intelligent caching, and advanced text preprocessing.

Based on: https://github.com/canopyai/Orpheus-TTS

## Architecture

The system uses a modular architecture with clear separation of concerns:

### Core Modules

1. **configurable_tts_server.py** - FastAPI web server
   - OpenAI-compatible `/v1/audio/speech` endpoint
   - Health checks, cache management, dictionary endpoints
   - Audio effects configuration endpoints
   - Language-specific server instances

2. **tts_engine.py** - Core TTS generation engine
   - Model initialization and GPU management
   - Text-to-audio generation via causal language model
   - SNAC token decoding to waveforms
   - Parallel GPU processing with round-robin distribution
   - Integration with cache system

3. **cache_manager.py** - Intelligent caching system
   - Thread-safe LRU cache using OrderedDict + RLock
   - Time-based expiration (configurable TTL)
   - Disk persistence with pickle serialization
   - Periodic auto-save (every 5 minutes)
   - Per-language cache directories

4. **text_preprocessor.py** - Text preprocessing pipeline
   - Quote removal
   - Language-specific conversions (numbers, SQL, etc.)
   - General text cleanup (whitespace, punctuation)
   - Custom word replacement dictionary
   - Auto-reloading dictionary every 30 seconds

5. **chunking.py** - Advanced text chunking
   - German-aware sentence boundary detection
   - Soft length limits with overshoot allowance
   - Abbreviation handling (z.B., bzw., Dr., etc.)
   - Paragraph-respecting chunking
   - Prevents mid-sentence cuts

6. **audio_postprocessor.py** - Audio effects processing
   - SoX-based effects pipeline
   - Pitch shift, speed adjustment, gain control
   - Reverb and echo effects
   - Audio normalization and limiting
   - Cross-platform SoX detection

7. **models_config.py** - Centralized configuration
   - Per-language model settings (path, port, voice, effects)
   - Shared configuration (temperature, tokens, preprocessing)
   - Dynamic word replacement loading
   - Audio effects defaults per language
   - Configuration validation

8. **multi_server_manager.py** - Multi-server orchestration
   - Start/stop/restart multiple language servers
   - Health monitoring and testing
   - Interactive management CLI
   - Process output monitoring

## Key Technologies

- **HuggingFace Transformers**: AutoModelForCausalLM for TTS generation
- **SNAC Codec**: Hierarchical audio codec (7 tokens per frame: 1+2+4 layers)
- **FastAPI**: Modern async web framework
- **PyTorch**: Deep learning framework with CUDA support
- **SoX**: Command-line audio processing

## Data Flow

```
HTTP Request → FastAPI Endpoint
    ↓
Text Preprocessing (quotes, numbers, SQL, dictionary)
    ↓
Intelligent Chunking (sentence boundaries, soft limits)
    ↓
Cache Check (SHA256 key: text + voice + language)
    ↓ (cache miss)
Parallel GPU Generation (round-robin distribution)
    ↓
SNAC Token Decoding (7 tokens → audio waveform)
    ↓
Cache Store
    ↓
Audio Post-Processing (SoX: pitch, speed, effects)
    ↓
WAV Response (24kHz)
```

## Configuration

### Models Configuration (`models_config.py`)

```python
MODEL_CONFIGS = {
    "german": {
        "enabled": True,
        "port": 5006,
        "model_path": "canopylabs/3b-de-ft-research_release",
        "default_voice": "thomas",
        "audio_effects": {
            "pitch_shift": 0.0,
            "speed_factor": 1.0,
            # ...
        }
    },
    # ... more languages
}

SHARED_CONFIG = {
    "snac_model_id": "hubertsiuzdak/snac_24khz",
    "temperature": 0.6,
    "max_new_tokens": 1200,
    "enable_intelligent_preprocessing": True,
    "enable_audio_postprocessing": True,
    # ...
}
```

### Environment Variables (`.env`)

```bash
HUGGINGFACE_HUB_TOKEN=your_token_here
LOG_LEVEL=INFO
LOG_TO_FILE=false
ENABLE_AUDIO_POSTPROCESSING=true
SOX_PATH=sox
```

### Word Replacements (`word_replacements.json`)

```json
{
  "german": {
    "SQL": "Siekwel",
    "bzw.": "beziehungsweise"
  }
}
```

## Important Constants (tts_engine.py)

```python
SAMPLE_RATE = 24000  # Audio sample rate
SNAC_TOKENS_PER_FRAME = 7  # SNAC hierarchical structure
PAUSE_DURATION = 0.5  # Silence between chunks (seconds)
PADDING_LENGTH = 4260  # Fixed model input padding
START_TOKEN = 128259  # Audio generation start marker
AUDIO_START_TOKEN = 128257  # Audio sequence start
AUDIO_END_TOKEN = 128258  # Audio sequence end
TOKEN_OFFSET = 128266  # Model token → SNAC code offset
```

## Key Algorithms

### SNAC Decoding

SNAC uses a hierarchical structure with 7 tokens per audio frame:
- Token 0: Layer 1 (1 code)
- Tokens 1, 4: Layer 2 (2 codes)
- Tokens 2, 3, 5, 6: Layer 3 (4 codes)

Each layer has offset: Layer 2 = +4096, Layer 3 = +8192 base

### Text Chunking with Soft Limits

- Target: `max_chars` (e.g., 200)
- Soft preference: Start preferring to end at `soft_max_ratio * max_chars` (e.g., 170)
- Soft maximum: Allow up to `max_chars + min(soft_allowance, max_chars * soft_allow_ratio)` (e.g., 240)
- This prevents mid-sentence cuts while maintaining reasonable chunk sizes

### LRU Cache with RLock

- OrderedDict tracks access order
- RLock allows reentrant locking (same thread can re-acquire)
- On cache hit: move_to_end() to mark as recently used
- On cache full: evict oldest (first in OrderedDict)
- Returns copy to prevent external modification

## Running the System

### Single Language Server

```bash
python configurable_tts_server.py --language german
```

### All Enabled Servers

```bash
python multi_server_manager.py --start
```

### Interactive Management

```bash
python multi_server_manager.py --interactive
```

Commands: start, stop, status, restart, test, quit

## API Usage

### Generate Speech

```bash
curl -X POST http://localhost:5006/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hallo, das ist ein Test.",
    "voice": "thomas",
    "speed": 1.0,
    "pitch_shift": 0.0
  }' \
  --output speech.wav
```

### Health Check

```bash
curl http://localhost:5006/health
```

### Cache Statistics

```bash
curl http://localhost:5006/cache/stats
```

### Clear Cache

```bash
curl -X POST http://localhost:5006/cache/clear
```

## Development Guidelines

### Adding a New Language

1. Add configuration to `MODEL_CONFIGS` in `models_config.py`
2. Set unique port number
3. Configure model path, voice, and audio effects
4. Add word replacements to `word_replacements.json`
5. Set `enabled: True`
6. Restart servers

### Modifying Text Preprocessing

Edit `text_preprocessor.py`:
- Add language-specific logic in `preprocess_text_intelligent()`
- Update SQL/number conversion functions
- Add entries to `word_replacements.json`

### Adjusting Chunk Sizes

Edit chunking parameters in `configurable_tts_server.py`:

```python
text_chunks = split_text_into_chunks(
    processed_text,
    max_chunk_length=200,      # Target max
    soft_max_ratio=0.85,       # Start preferring to end
    soft_allowance=40,         # Max overshoot in chars
    max_sentences_per_chunk=2  # Sentence limit
)
```

### Adding Audio Effects

1. Add effect to `TTSAudioPostProcessor` in `audio_postprocessor.py`
2. Add parameter to `TTSRequest` model in `configurable_tts_server.py`
3. Add to effects dict in `/v1/audio/speech` endpoint
4. Add to `audio_effects` in language config

### Debugging

Set log level to DEBUG:

```bash
LOG_LEVEL=DEBUG python configurable_tts_server.py --language german
```

Or in code:
```python
SHARED_CONFIG["log_level"] = "DEBUG"
```

## Performance Optimization

- **Multi-GPU**: Automatically uses all available GPUs with round-robin distribution
- **Caching**: Dramatically reduces repeated generation (check cache hit rate at `/cache/stats`)
- **Chunking**: Optimal chunk sizes balance quality vs. speed
- **Batch Processing**: Fixed padding length enables efficient batching

## Common Issues

### CUDA Out of Memory

- Reduce `max_new_tokens` in `SHARED_CONFIG`
- Use smaller model
- Reduce number of concurrent requests

### SoX Not Available

- Install: `sudo apt-get install sox` (Linux) or `brew install sox` (macOS)
- Or disable: `ENABLE_AUDIO_POSTPROCESSING=false`

### Cache Not Persisting

- Check write permissions on cache directory
- Verify disk space
- Check for pickle serialization errors in logs

### Poor Audio Quality

- Check preprocessing (might be over-aggressive)
- Verify model is correct for language
- Check audio effects settings
- Increase `max_new_tokens` if cutting off

## Virtual Environment

A Python virtual environment is present in `.venv/` - activate it before running:

```bash
source .venv/bin/activate
```

## Dependencies

Core dependencies in `requirements.txt`:
- `torch` - PyTorch with CUDA
- `transformers` - HuggingFace models
- `snac` - SNAC audio codec
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `soundfile` - Audio I/O
- `numpy` - Audio processing
- `python-dotenv` - Environment variables

## Testing

Test a server:

```bash
python multi_server_manager.py --test
```

Or manually:

```python
import requests
response = requests.post(
    "http://localhost:5006/v1/audio/speech",
    json={"model": "orpheus", "input": "Test", "voice": "thomas"}
)
with open("test.wav", "wb") as f:
    f.write(response.content)
```

## Important Reminders

- **Do NOT** commit `.env` file (contains sensitive token)
- **Do NOT** modify special tokens without understanding SNAC format
- **Always** test changes with multiple languages if modifying shared code
- **Cache keys** include text+voice+language - changing any invalidates cache
- **Word replacements** auto-reload every 30 seconds, no restart needed
- **Audio effects** cascade: request params → language config → processor defaults