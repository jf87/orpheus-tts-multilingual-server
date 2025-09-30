# Orpheus TTS Multilingual Server

Production-ready Text-to-Speech system using Hugging Face transformers and SNAC audio codec with multi-GPU parallel processing, intelligent caching, and advanced text preprocessing.

Based on [Orpheus-TTS by CanopyLabs](https://github.com/canopyai/Orpheus-TTS)

## Features

- **Multi-Language Support**: Independent servers for German, English, and Spanish TTS models
- **Multi-GPU Processing**: Automatic parallel processing across available GPUs
- **Intelligent Caching**: Thread-safe LRU cache with disk persistence
- **Text Preprocessing**: Language-specific text normalization and word replacements
- **Audio Post-Processing**: SoX-based effects (pitch, speed, reverb, echo)
- **Advanced Chunking**: Smart text splitting with sentence boundary preservation
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI TTS endpoints

## Architecture

### Core Modules

- **`configurable_tts_server.py`**: FastAPI server with OpenAI-compatible endpoints
- **`tts_engine.py`**: Core TTS generation using HuggingFace transformers + SNAC codec
- **`cache_manager.py`**: Thread-safe LRU cache with file persistence
- **`text_preprocessor.py`**: Intelligent text preprocessing and word replacements
- **`chunking.py`**: Advanced text chunking with German abbreviation awareness
- **`audio_postprocessor.py`**: SoX-based audio effects processing
- **`models_config.py`**: Centralized configuration for all languages
- **`multi_server_manager.py`**: Manager for running multiple language servers

### Processing Pipeline

```
Input Text
    ↓
Text Preprocessing (quotes, SQL, numbers, dictionary)
    ↓
Intelligent Chunking (sentence boundaries, soft limits)
    ↓
Parallel GPU Generation (round-robin distribution)
    ↓
Audio Post-Processing (pitch, speed, effects)
    ↓
WAV Output
```

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Token

Create `.env` file:
```bash
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

### 3. Configure Models

Edit `models_config.py` to enable/disable languages:

```python
MODEL_CONFIGS = {
    "german": {
        "enabled": True,  # Enable/disable
        "port": 5006,
        "model_path": "canopylabs/3b-de-ft-research_release",
        "default_voice": "thomas",
        # ...
    },
    # ...
}
```

### 4. Start Servers

**Option A: Start all enabled servers**
```bash
python multi_server_manager.py --start
```

**Option B: Start single language server**
```bash
python configurable_tts_server.py --language german
```

**Option C: Interactive management mode**
```bash
python multi_server_manager.py --interactive
```

## Configuration

### Model Configuration (`models_config.py`)

#### Per-Language Settings

Each language has its own configuration:
- `enabled`: Enable/disable the model
- `port`: Server port (must be unique)
- `model_path`: HuggingFace model path
- `default_voice`: Default voice identifier
- `display_name`: Human-readable name
- `sample_text`: Sample text for testing
- `audio_effects`: Language-specific audio processing defaults

#### Shared Configuration

Global settings in `SHARED_CONFIG`:
- `snac_model_id`: SNAC codec model
- `temperature`, `top_p`, `repetition_penalty`: Generation parameters
- `max_new_tokens`: Maximum tokens to generate
- `enable_intelligent_preprocessing`: Enable text preprocessing
- `enable_audio_postprocessing`: Enable audio effects
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `log_to_file`: Write logs to file

### Word Replacements (`word_replacements.json`)

Custom pronunciation dictionary per language:

```json
{
  "german": {
    "SQL": "Siekwel",
    "bzw.": "beziehungsweise",
    "1-zu-M": "Eins-zu-M"
  },
  "english": {},
  "spanish": {}
}
```

Changes are auto-reloaded every 30 seconds (no restart required).

## API Documentation

### POST `/v1/audio/speech`

Generate speech from text (OpenAI-compatible).

**Request:**
```json
{
  "model": "orpheus",
  "input": "Text to convert to speech",
  "voice": "thomas",
  "response_format": "wav",
  "speed": 1.0,
  "pitch_shift": 0.0,
  "gain_db": 0.0,
  "normalize_audio": false,
  "add_reverb": false,
  "add_echo": false
}
```

**Parameters:**
- `input` (required): Text to synthesize
- `voice` (optional): Voice identifier (default: language default)
- `speed` (optional): Speed multiplier, 0.5-2.0 (default: 1.0)
- `pitch_shift` (optional): Semitones, -12 to +12 (default: 0)
- `gain_db` (optional): Volume adjustment in dB, -20 to +20 (default: 0)
- `normalize_audio` (optional): Normalize volume (default: false)
- `add_reverb` (optional): Add reverb effect (default: false)
- `add_echo` (optional): Add echo effect (default: false)

**Response:** WAV audio file (24kHz sample rate)

**Example:**
```bash
curl -X POST http://localhost:5006/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hallo, das ist ein Test.",
    "voice": "thomas"
  }' \
  --output speech.wav
```

### GET `/health`

Health check with cache statistics.

**Response:**
```json
{
  "status": "healthy",
  "language": "german",
  "display_name": "German TTS",
  "models_loaded": true,
  "gpu_count": 1,
  "default_voice": "thomas",
  "cache_enabled": true,
  "cache_size": 42,
  "cache_hit_rate": "85.23%"
}
```

### GET `/info`

Server configuration information.

### GET `/cache/stats`

Cache statistics (hits, misses, evictions, size).

### POST `/cache/clear`

Clear the TTS cache.

### GET `/dictionary`

Get current word replacement dictionary for this language.

### GET `/audio-effects/status`

Audio post-processing system status.

### GET `/audio-effects/defaults`

Default audio effect parameters for this language.

## Advanced Features

### Intelligent Text Preprocessing

Automatic text normalization for better TTS output:

1. **Quote Removal**: Strips quote marks around words
2. **Language-Specific Conversions**:
   - German: `3-zu-1` → "Drei-zu-Eins"
   - SQL keywords: `SELECT` → "Select"
   - Dots in paths: `table.column` → "table Punkt column"
3. **General Cleanup**: Whitespace normalization, punctuation cleanup
4. **Custom Dictionary**: Word replacements from `word_replacements.json`

Configure via `enable_intelligent_preprocessing` in `models_config.py`.

### Advanced Chunking

Smart text splitting algorithm:

- **Soft Length Limits**: Target max_chars but allow overshoot to preserve sentence boundaries
- **German Abbreviation Awareness**: Don't split on abbreviations like "z.B.", "bzw.", "Dr."
- **Punctuation Preference**: End chunks at sentence boundaries when possible
- **Paragraph Respect**: Never split across paragraph breaks

Example: max_chars=200 → can extend to ~240 chars to finish a sentence cleanly.

### Intelligent Caching

- **Thread-Safe**: Uses RLock for concurrent access
- **LRU Eviction**: Automatically removes least-recently-used entries
- **Time-Based Expiration**: Entries expire after configurable duration
- **Disk Persistence**: Saves cache every 5 minutes, reloads on startup
- **Per-Language**: Separate cache directories per language

Cache key: `SHA256(text + voice + language)`

### Audio Post-Processing

SoX-based effects pipeline:
1. Normalize (optional)
2. Gain adjustment with limiter
3. Pitch shift
4. Speed/tempo adjustment
5. Reverb
6. Echo

Install SoX:
```bash
# Linux
sudo apt-get install sox

# macOS
brew install sox

# Windows
# Download from https://sourceforge.net/projects/sox/
```

## Multi-Server Management

The `multi_server_manager.py` provides centralized management:

```bash
# Start all enabled servers
python multi_server_manager.py --start

# Check server status
python multi_server_manager.py --status

# Test all servers
python multi_server_manager.py --test

# Interactive mode (start/stop/status/test)
python multi_server_manager.py --interactive
```

Interactive commands:
- `start` - Start all enabled servers
- `stop` - Stop all running servers
- `status` - Show server status
- `restart` - Restart all servers
- `test` - Test all servers with sample text
- `quit` - Exit (stops all servers)

## Supported Languages

### Currently Configured

| Language | Model | Port | Status |
|----------|-------|------|--------|
| German | `canopylabs/3b-de-ft-research_release` | 5006 | ✅ Enabled |
| English | `canopylabs/orpheus-3b-0.1-ft` | 5005 | ⏸️ Disabled |
| Spanish | `canopylabs/3b-es_it-ft-research_release` | 5007 | ⏸️ Disabled |

To enable a language, set `"enabled": True` in `models_config.py`.

## Requirements

### System Requirements
- CUDA-capable GPU(s)
- Python 3.11+
- 16GB+ RAM recommended
- SoX (optional, for audio post-processing)

### Python Dependencies
See `requirements.txt`:
- `torch` with CUDA support
- `transformers` (HuggingFace)
- `snac` (audio codec)
- `fastapi` and `uvicorn` (web server)
- `soundfile` (audio I/O)
- `numpy` (audio processing)
- `python-dotenv` (configuration)

## Troubleshooting

### SoX not found
```
SoX executable not found. Audio post-processing will be disabled.
```
**Solution**: Install SoX or disable audio post-processing in config.

### GPU out of memory
```
CUDA out of memory
```
**Solution**:
- Reduce `max_new_tokens` in `models_config.py`
- Use smaller model
- Enable CPU fallback

### Cache not loading
```
Failed to load cache from disk
```
**Solution**: Cache file may be corrupted. Clear cache via `/cache/clear` endpoint.

### Port already in use
```
Address already in use
```
**Solution**: Change port in `models_config.py` or stop conflicting service.

### Model download fails
```
Failed to download model
```
**Solution**: Verify `HUGGINGFACE_HUB_TOKEN` in `.env` and model path in config.

## Development

### Project Structure

```
orpheus-tts-multilingual-server/
├── configurable_tts_server.py   # FastAPI server
├── tts_engine.py                # Core TTS generation
├── cache_manager.py             # Caching system
├── text_preprocessor.py         # Text preprocessing
├── chunking.py                  # Text chunking
├── audio_postprocessor.py       # Audio effects
├── models_config.py             # Configuration
├── multi_server_manager.py      # Server manager
├── word_replacements.json       # Custom dictionary
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── CLAUDE.md                    # Developer guide
└── .env                         # Environment variables
```

### Logging

Configure logging via environment variables:

```bash
# Set log level
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR

# Enable file logging
LOG_TO_FILE=true
LOG_FILE_PATH=./tts_server.log
```

Or in code:
```python
SHARED_CONFIG["log_level"] = "DEBUG"
SHARED_CONFIG["log_to_file"] = True
```

## License

Based on Orpheus-TTS by CanopyLabs. See original repository for license information.

## Credits

- **Orpheus-TTS**: https://github.com/canopyai/Orpheus-TTS
- **SNAC**: https://github.com/hubertsiuzdak/snac
- **HuggingFace Transformers**: https://huggingface.co/