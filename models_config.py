#!/usr/bin/env python3
"""
Configuration for multiple TTS language models.
"""

import os
import json
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === MODEL CONFIGURATIONS ===
MODEL_CONFIGS = {
    "german": {
        "enabled": True,
        "port": 5006,
        "model_path": "canopylabs/3b-de-ft-research_release",
        "finetuned_model": "path_to_orpheus/checkpoint",
        "default_voice": "thomas",
        "language": "de",
        "display_name": "German TTS",
        "sample_text": "Hallo, das ist ein Test der deutschen Text-zu-Sprache Synthese.",
        # Audio post-processing defaults for German
        "audio_effects": {
            "pitch_shift": 0.0,          # No pitch adjustment
            "speed_factor": 1.0,         # Slightly slower for better comprehension
            "gain_db": 0.0,              # No volume adjustment
            "normalize_audio": False,    # No normalization
            "use_limiter": True,         # Keep limiter as safety
            "add_reverb": False,         # No reverb
            "reverb_amount": 50,         # Default values (unused)
            "reverb_room_scale": 50,     # Default values (unused)
            "add_echo": False,           # No echo
            "echo_delay": 0.5,           # Default values (unused)
            "echo_decay": 0.5            # Default values (unused)
        }
    },
    "english": {
        "enabled": False,
        "port": 5005,
        "model_path": "canopylabs/orpheus-3b-0.1-ft",  # Update with actual English model
        "finetuned_model": "path_to_english_orpheus/checkpoint",
        "default_voice": "sarah",
        "language": "en", 
        "display_name": "English TTS",
        "sample_text": "Hello, this is a test of the English text-to-speech synthesis.",
        # Audio post-processing defaults for English
        "audio_effects": {
            "pitch_shift": 0.0,          # No pitch adjustment for English
            "speed_factor": 1.0,        # Slightly slower for clarity
            "gain_db": 0.0,              # No gain adjustment
            "normalize_audio": False,    # Don't normalize English audio
            "use_limiter": True,
            "add_reverb": False,          # Add subtle reverb for English
            "reverb_amount": 50,
            "reverb_room_scale": 50,
            "add_echo": False,
            "echo_delay": 0.5,
            "echo_decay": 0.5
        }
    },
    "spanish": {
        "enabled": False,  # Disabled by default
        "port": 5007,
        "model_path": "canopylabs/3b-es_it-ft-research_release",  # Update with actual Spanish model
        "finetuned_model": "path_to_spanish_orpheus/checkpoint",
        "default_voice": "maria",
        "language": "es",
        "display_name": "Spanish TTS", 
        "sample_text": "Hola, esta es una prueba de la síntesis de texto a voz en español.",
        # Audio post-processing defaults for Spanish
        "audio_effects": {
            "pitch_shift": 0.5,          # Slight pitch boost for Spanish
            "speed_factor": 1.05,        # Slightly faster for natural Spanish rhythm
            "gain_db": 1.5,              # Moderate volume boost
            "normalize_audio": True,     # Normalize Spanish audio
            "use_limiter": True,
            "add_reverb": False,
            "reverb_amount": 40,
            "reverb_room_scale": 50,
            "add_echo": False,
            "echo_delay": 0.5,
            "echo_decay": 0.5
        }
    },
}

# === DYNAMIC WORD REPLACEMENT DICTIONARIES ===
_word_replacements_cache = {}
_last_reload_time = 0
_word_replacements_file_path = os.path.join(os.path.dirname(__file__), "word_replacements.json")

def get_word_replacements(language: str) -> Dict[str, str]:
    """Get word replacement dictionary for a specific language, reloading from file if needed."""
    global _word_replacements_cache, _last_reload_time
    
    current_time = time.time()
    
    # Reload every 30 seconds if file exists and was modified
    if current_time - _last_reload_time > 30:
        try:
            if os.path.exists(_word_replacements_file_path):
                with open(_word_replacements_file_path, 'r', encoding='utf-8') as f:
                    _word_replacements_cache = json.load(f)
                _last_reload_time = current_time
        except (json.JSONDecodeError, IOError):
            # Keep using cached version if file read fails
            pass
    
    return _word_replacements_cache.get(language, {})

# === SHARED CONFIGURATION ===
SHARED_CONFIG = {
    "snac_model_id": "hubertsiuzdak/snac_24khz",
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.3,
    "max_new_tokens": 1200,
    "max_chunk_length": 200,
    "hf_token": os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
    "enable_intelligent_preprocessing": True,  # Enable/disable intelligent text preprocessing
    # Logging configuration
    "log_level": os.environ.get("LOG_LEVEL", "INFO").upper(),  # DEBUG, INFO, WARNING, ERROR
    "log_to_file": os.environ.get("LOG_TO_FILE", "false").lower() == "true",
    "log_file_path": os.environ.get("LOG_FILE_PATH", "./tts_server.log"),
    # Audio post-processing configuration
    "enable_audio_postprocessing": os.environ.get("ENABLE_AUDIO_POSTPROCESSING", "true").lower() == "true",
    "sox_path": os.environ.get("SOX_PATH", "sox"),  # Path to SoX executable
}

# === HELPER FUNCTIONS ===
def get_enabled_models() -> Dict[str, Dict]:
    """Get only the enabled model configurations."""
    return {lang: config for lang, config in MODEL_CONFIGS.items() if config["enabled"]}

def get_model_config(language: str) -> Optional[Dict]:
    """Get configuration for a specific language model."""
    return MODEL_CONFIGS.get(language)

def get_ports() -> List[int]:
    """Get all ports for enabled models."""
    return [config["port"] for config in get_enabled_models().values()]

def validate_ports() -> bool:
    """Check if all enabled models have unique ports."""
    enabled_configs = get_enabled_models()
    ports = [config["port"] for config in enabled_configs.values()]
    return len(ports) == len(set(ports))

def get_audio_effects_config(language: str) -> Dict:
    """Get audio effects configuration for a specific language."""
    config = get_model_config(language)
    if not config:
        return {}
    return config.get("audio_effects", {})

def print_config_summary():
    """Print a summary of the current configuration."""
    print("🌍 Multi-Language TTS Server Configuration")
    print("=" * 50)
    
    enabled_models = get_enabled_models()
    if not enabled_models:
        print("❌ No models are enabled!")
        return
    
    for lang, config in enabled_models.items():
        print(f"✅ {config['display_name']} ({lang.upper()})")
        print(f"   Port: {config['port']}")
        print(f"   Model: {config['model_path']}")
        print(f"   Voice: {config['default_voice']}")
        
        # Show audio effects if configured
        if "audio_effects" in config:
            effects = config["audio_effects"]
            active_effects = []
            if effects.get("pitch_shift", 0.0) != 0.0:
                active_effects.append(f"pitch: {effects['pitch_shift']}")
            if effects.get("speed_factor", 1.0) != 1.0:
                active_effects.append(f"speed: {effects['speed_factor']}x")
            if effects.get("gain_db", 0.0) != 0.0:
                active_effects.append(f"gain: {effects['gain_db']}dB")
            if effects.get("normalize_audio", False):
                active_effects.append("normalize")
            if effects.get("add_reverb", False):
                active_effects.append(f"reverb: {effects.get('reverb_amount', 50)}%")
            if effects.get("add_echo", False):
                active_effects.append(f"echo: {effects.get('echo_delay', 0.5)}s")
            
            if active_effects:
                print(f"   🎚️  Effects: {', '.join(active_effects)}")
            else:
                print(f"   🎚️  Effects: none (defaults)")
        print()
    
    disabled_models = {lang: config for lang, config in MODEL_CONFIGS.items() if not config["enabled"]}
    if disabled_models:
        print("Disabled models:")
        for lang, config in disabled_models.items():
            print(f"⏸️  {config['display_name']} ({lang.upper()}) - Port {config['port']}")
    
    if not validate_ports():
        print("⚠️  WARNING: Duplicate ports detected!")

if __name__ == "__main__":
    print_config_summary()
