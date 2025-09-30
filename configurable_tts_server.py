#!/usr/bin/env python3
"""
Configurable FastAPI server for Orpheus-HF TTS system with multi-language support.
"""

import soundfile as sf
import io
import sys
import argparse
from typing import List, Optional, Dict
import time
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from models_config import MODEL_CONFIGS, SHARED_CONFIG, get_model_config, get_audio_effects_config, get_word_replacements
from audio_postprocessor import TTSAudioPostProcessor
from text_preprocessor import preprocess_text_intelligent, split_text_into_chunks
from cache_manager import TTSCache
from tts_engine import initialize_models, generate_audio_parallel, get_model_info, set_cache_instance

# === LOGGING SETUP ===
def setup_logging():
    """Configure logging based on SHARED_CONFIG settings."""
    log_level = getattr(logging, SHARED_CONFIG.get("log_level", "INFO"))
    log_to_file = SHARED_CONFIG.get("log_to_file", False)
    log_file_path = SHARED_CONFIG.get("log_file_path", "./tts_server.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if enabled
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file enabled: {log_file_path}")
    
    logging.info(f"Logging configured - Level: {SHARED_CONFIG.get('log_level', 'INFO')}")
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Global cache instance (will be initialized in create_app with language-specific directory)
tts_cache = None

# Global audio post-processor instance
audio_postprocessor = TTSAudioPostProcessor(SHARED_CONFIG.get("sox_path", "sox"))

# === PYDANTIC MODELS ===
class TTSRequest(BaseModel):
    model: str = "orpheus"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    # Audio post-processing parameters
    pitch_shift: Optional[float] = 0.0
    gain_db: Optional[float] = 0.0
    normalize_audio: Optional[bool] = False
    add_reverb: Optional[bool] = False
    reverb_amount: Optional[float] = 50
    reverb_room_scale: Optional[float] = 50
    add_echo: Optional[bool] = False
    echo_delay: Optional[float] = 0.5
    echo_decay: Optional[float] = 0.5

# === FASTAPI APP ===
def create_app(language: str):
    """Create FastAPI app for a specific language."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Handle startup and shutdown events."""
        # Startup
        global tts_cache
        # Initialize language-specific cache directory
        cache_dir = f"./tts_cache_{language}"
        tts_cache = TTSCache(max_size=10000, max_age_seconds=3600*24*31, cache_dir=cache_dir)
        logger.info(f"🗂️ Initialized cache for {language} in directory: {cache_dir}")

        # Set cache instance for TTS engine
        set_cache_instance(tts_cache)

        # Initialize TTS models
        initialize_models(language)
        yield
        # Shutdown
        print("💾 Saving cache before shutdown...")
        tts_cache.save_to_disk(force=True)
    
    app = FastAPI(
        title=f"Orpheus-HF TTS API - {language.upper()}", 
        version="1.0.0",
        description=f"Text-to-Speech API for {language} language",
        lifespan=lifespan
    )

    @app.post("/v1/audio/speech")
    async def create_speech(request: TTSRequest):
        """Create speech from text using OpenAI-compatible API format."""
        try:
            # Use default voice if not specified
            config = get_model_config(language)
            voice = request.voice or config["default_voice"]
            
            # Apply intelligent text preprocessing before chunking
            original_text = request.input
            processed_text = preprocess_text_intelligent(
                original_text,
                language=language,
                enable_preprocessing=SHARED_CONFIG.get("enable_intelligent_preprocessing", True)
            )
            
            # Log preprocessing changes if significant
            if processed_text != original_text and len(original_text) > 20:
                logger.info(f"🔧 [{language.upper()}] Text preprocessing applied:")
                logger.info(f"   ORIGINAL:  {original_text}")
                logger.info(f"   PROCESSED: {processed_text}")
            
            # Split text into chunks for processing using the advanced chunking algorithm
            text_chunks = split_text_into_chunks(
                processed_text,
                max_chunk_length=200,
                prefer_end_punct=True,
                soft_max_ratio=0.85,      # start *preferring* to stop around ~170 chars
                max_sentences_per_chunk=2,
                soft_allowance=40,        # allow up to ~240 chars to finish a sentence
                soft_allow_ratio=0.2      # and cap overshoot to at most +20%
            )
            logger.info(f"📝 [{language.upper()}] Text split into {len(text_chunks)} chunks")
            for i, chunk in enumerate(text_chunks):
                logger.debug(f"   Chunk {i+1}: {chunk}")
            
            # Generate audio with caching
            audio = await generate_audio_parallel(text_chunks, voice, language)
            
            # Apply audio post-processing if enabled and SoX is available
            if SHARED_CONFIG.get("enable_audio_postprocessing", True) and audio_postprocessor.is_available():
                # Audio effects priority: request params > language config > processor defaults
                # This allows per-request customization while maintaining sensible defaults
                configured_effects = get_audio_effects_config(language)
                logger.debug(f"🎛️ [{language.upper()}] Configured audio effects: {configured_effects}")

                # Merge request parameters with language-specific defaults
                effects = {
                    'pitch_shift': request.pitch_shift if request.pitch_shift is not None else configured_effects.get('pitch_shift', 0.0),
                    'speed_factor': request.speed if request.speed is not None else configured_effects.get('speed_factor', 1.0),
                    'gain_db': request.gain_db if request.gain_db is not None else configured_effects.get('gain_db', 0.0),
                    'normalize_audio': request.normalize_audio if request.normalize_audio is not None else configured_effects.get('normalize_audio', False),
                    'use_limiter': configured_effects.get('use_limiter', True),  # Always use configured value
                    'add_reverb': request.add_reverb if request.add_reverb is not None else configured_effects.get('add_reverb', False),
                    'reverb_amount': request.reverb_amount if request.reverb_amount is not None else configured_effects.get('reverb_amount', 50),
                    'reverb_room_scale': request.reverb_room_scale if request.reverb_room_scale is not None else configured_effects.get('reverb_room_scale', 50),
                    'add_echo': request.add_echo if request.add_echo is not None else configured_effects.get('add_echo', False),
                    'echo_delay': request.echo_delay if request.echo_delay is not None else configured_effects.get('echo_delay', 0.5),
                    'echo_decay': request.echo_decay if request.echo_decay is not None else configured_effects.get('echo_decay', 0.5)
                }
                
                # Validate and apply effects
                validated_effects = audio_postprocessor.validate_effects(effects)
                logger.info(f"🎚️ [{language.upper()}] Applying audio post-processing")
                logger.debug(f"Effects: {validated_effects}")
                
                processed_audio = audio_postprocessor.process_audio(
                    audio, 
                    sample_rate=24000, 
                    **validated_effects
                )
                
                if processed_audio is not None:
                    audio = processed_audio
                    logger.info(f"🎵 [{language.upper()}] Audio post-processing completed")
                else:
                    logger.warning(f"⚠️ [{language.upper()}] Audio post-processing failed, using original")
            elif not SHARED_CONFIG.get("enable_audio_postprocessing", True):
                logger.debug("Audio post-processing disabled in configuration")
            elif not audio_postprocessor.is_available():
                logger.debug("SoX not available, skipping audio post-processing")
            
            # Convert to bytes for response
            with io.BytesIO() as buffer:
                sf.write(buffer, audio, samplerate=24000, format='WAV')
                audio_bytes = buffer.getvalue()
            
            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        cache_stats = tts_cache.get_stats()
        return {
            "status": "healthy", 
            "language": language,
            "display_name": get_model_info()["config"]["display_name"] if get_model_info()["config"] else "Unknown",
            "models_loaded": get_model_info()["models_loaded"] > 0,
            "gpu_count": get_model_info()["gpu_count"],
            "default_voice": get_model_info()["config"]["default_voice"] if get_model_info()["config"] else "Unknown",
            "cache_enabled": True,
            "cache_size": cache_stats["size"],
            "cache_hit_rate": cache_stats["hit_rate"]
        }

    @app.get("/info")
    async def server_info():
        """Get server information."""
        model_info = get_model_info()
        return {
            "language": language,
            "config": model_info["config"],
            "shared_config": SHARED_CONFIG,
            "gpu_count": model_info["gpu_count"],
            "models_loaded": model_info["models_loaded"]
        }

    @app.get("/cache/stats")
    async def cache_stats():
        """Get cache statistics."""
        return tts_cache.get_stats()

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the TTS cache."""
        old_stats = tts_cache.get_stats()
        tts_cache.clear()
        return {
            "message": "Cache cleared successfully (including disk files)",
            "previous_stats": old_stats,
            "current_stats": tts_cache.get_stats()
        }

    # === DICTIONARY MANAGEMENT ENDPOINTS ===
    @app.get("/dictionary")
    async def get_dictionary():
        """Get current dictionary entries for this language."""
        language_key = {
            'de': 'german',
            'en': 'english', 
            'es': 'spanish'
        }.get(language, 'german')
        
        dictionary_entries = get_word_replacements(language_key)
        return {
            "language": language,
            "language_key": language_key,
            "entries": dictionary_entries,
            "count": len(dictionary_entries)
        }
    
    

    # === AUDIO POST-PROCESSING API ENDPOINTS ===
    
    @app.get("/audio-effects/status")
    async def get_audio_effects_status():
        """Get audio post-processing system status."""
        return {
            "enabled": SHARED_CONFIG.get("enable_audio_postprocessing", True),
            "sox_available": audio_postprocessor.is_available(),
            "sox_path": SHARED_CONFIG.get("sox_path", "sox"),
            "language": language
        }
    
    @app.get("/audio-effects/defaults")
    async def get_audio_effects_defaults():
        """Get default audio effect parameters for this language."""
        # Get configured defaults for this language
        configured_effects = get_audio_effects_config(language)
        
        # If no configured effects, fall back to processor defaults
        if not configured_effects:
            configured_effects = audio_postprocessor.get_default_effects()
            source = "processor_defaults"
        else:
            source = "language_config"
        
        return {
            "language": language,
            "sox_available": audio_postprocessor.is_available(),
            "source": source,
            "default_effects": configured_effects,
            "note": f"These are the configured defaults for {language.upper()}, used when TTS request parameters are not specified"
        }

    return app

def main():
    """Main function to start a language-specific TTS server."""
    parser = argparse.ArgumentParser(description="Start Orpheus-HF TTS Server")
    parser.add_argument("--language", "-l", required=True, 
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Language model to serve")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_model_config(args.language)
    if not config:
        print(f"❌ No configuration found for language: {args.language}")
        sys.exit(1)
    
    if not config["enabled"]:
        print(f"❌ Language model '{args.language}' is disabled in configuration")
        print("Enable it in models_config.py to use this server")
        sys.exit(1)
    
    # Create and run app
    app = create_app(args.language)
    port = config["port"]
    
    print(f"🚀 Starting {config['display_name']} server on {args.host}:{port}")
    print(f"📝 API endpoint: http://{args.host}:{port}/v1/audio/speech")
    print(f"🏥 Health check: http://{args.host}:{port}/health")
    
    uvicorn.run(app, host=args.host, port=port)

if __name__ == "__main__":
    main()
