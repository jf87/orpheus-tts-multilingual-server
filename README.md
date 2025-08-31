# Orpheus-TTS Multilingual FastAPI Server

High-quality Text-to-Speech system using Hugging Face transformers and SNAC audio codec with multi-GPU parallel processing.
Based on https://github.com/canopyai/Orpheus-TTS

## Quick Start

1. **Setup environment:**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure token:**
   Create `.env` file:
   ```
   HUGGINGFACE_HUB_TOKEN=your_hf_token_here
   ```

3. **Configure models:**
   Edit `models_config.py` to enable/disable languages and set model paths.

4. **Start servers:**
   ```bash
   python multi_server_manager.py --start
   ```

## Configuration

### Supported Languages
The system supports multiple languages with dedicated models:
- **German** (enabled by default): `canopylabs/3b-de-ft-research_release`
- **English** (disabled): `canopylabs/orpheus-3b-0.1-ft`
- **Spanish** (disabled): `canopylabs/3b-es_it-ft-research_release`

### Models (`models_config.py`)
- **MODEL_CONFIGS**: Per-language settings (model paths, ports, voices, sample text)
- **SHARED_CONFIG**: Global settings (temperature, max tokens, SNAC model)

Enable/disable languages by setting `"enabled": True/False` in MODEL_CONFIGS. Each language has its own port and voice configuration.

### Environment Variables
- `HUGGINGFACE_HUB_TOKEN`: Required for model access

## Usage

**Start all enabled servers:**
```bash
python multi_server_manager.py --start
```

**Interactive mode:**
```bash
python multi_server_manager.py --interactive
```

**Check status:**
```bash
python multi_server_manager.py --status
```

## API Endpoints

Each enabled language runs on its configured port:
- German (port 5006): `http://localhost:5006/v1/audio/speech`
- English (port 5005): `http://localhost:5005/v1/audio/speech` 
- Spanish (port 5007): `http://localhost:5007/v1/audio/speech`

*Note: Only enabled languages in `models_config.py` will have running servers.*

**Request format (OpenAI-compatible):**
```json
{
  "model": "orpheus",
  "input": "Text to convert to speech",
  "voice": "thomas",
  "response_format": "wav"
}
```

## Architecture

- **Multi-GPU**: Automatically distributes processing across available GPUs
- **Parallel Processing**: Generates audio chunks concurrently while preserving order
- **Caching**: Built-in response caching for improved performance
- **Multi-Language**: Independent servers for each language model

## Requirements

- CUDA-capable GPU(s)
- Python 3.11+
- Dependencies in `requirements.txt`
