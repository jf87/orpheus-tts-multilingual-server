#!/usr/bin/env python3
"""
Configuration for multiple TTS language models.
"""

import os
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
    },
}

# === SHARED CONFIGURATION ===
SHARED_CONFIG = {
    "snac_model_id": "hubertsiuzdak/snac_24khz",
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.3,
    "max_new_tokens": 1200,
    "max_chunk_length": 200,
    "hf_token": os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
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
