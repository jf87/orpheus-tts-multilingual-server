#!/usr/bin/env python3
"""
Configurable FastAPI server for Orpheus-HF TTS system with multi-language support.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC
import os
import numpy as np
import soundfile as sf
import asyncio
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login
import tempfile
import io
import sys
import argparse
from typing import List, Optional, Dict
import hashlib
import time
from collections import OrderedDict
import threading
import pickle
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from models_config import MODEL_CONFIGS, SHARED_CONFIG, get_model_config

# === GLOBAL MODELS ===
tokenizer = None
snac_model = None
models = []
DEVICE_IDS = []
GPU_COUNT = 0
SERVER_CONFIG = None

# === CACHE SYSTEM ===
class TTSCache:
    """Thread-safe LRU cache for TTS audio chunks with file persistence."""
    
    def __init__(self, max_size=1000, max_age_seconds=3600, cache_dir="./tts_cache"):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "tts_cache.pkl"
        self.stats_file = self.cache_dir / "cache_stats.json"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
            'saves': 0,
            'loads': 0
        }
        
        # Auto-save timer
        self.save_interval = 300  # Save every 5 minutes
        self.last_save = time.time()
        
        # Load existing cache if available
        self.load_from_disk()
    
    def _generate_key(self, text: str, voice: str, language: str) -> str:
        """Generate cache key from text, voice, and language."""
        content = f"{text}|{voice}|{language}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.max_age_seconds
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp > self.max_age_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.stats['evictions'] += 1
    
    def get(self, text: str, voice: str, language: str) -> Optional[np.ndarray]:
        """Get cached audio for text chunk."""
        key = self._generate_key(text, voice, language)
        
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache:
                audio_data, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.stats['hits'] += 1
                    return audio_data.copy()
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.stats['evictions'] += 1
            
            self.stats['misses'] += 1
            return None
    
    def put(self, text: str, voice: str, language: str, audio_data: np.ndarray):
        """Store audio in cache."""
        key = self._generate_key(text, voice, language)
        timestamp = time.time()
        
        with self.lock:
            # Remove oldest entries if at max size
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
            
            self.cache[key] = (audio_data.copy(), timestamp)
            self.stats['size'] = len(self.cache)
            
            # Auto-save periodically
            self.save_to_disk()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': f"{hit_rate:.2f}%",
                'evictions': self.stats['evictions'],
                'max_age_seconds': self.max_age_seconds
            }
    
    def save_to_disk(self, force=False):
        """Save cache to disk."""
        current_time = time.time()
        if not force and (current_time - self.last_save) < self.save_interval:
            return False
        
        try:
            with self.lock:
                # Clean expired entries before saving
                self._cleanup_expired()
                
                # Prepare data for serialization
                cache_data = {
                    'cache': dict(self.cache),  # Convert OrderedDict to dict for JSON serialization
                    'timestamp': current_time,
                    'version': '1.0'
                }
                
                # Save cache data using pickle (more efficient for numpy arrays)
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save stats as JSON (human readable)
                with open(self.stats_file, 'w') as f:
                    json.dump(self.stats, f, indent=2)
                
                self.last_save = current_time
                self.stats['saves'] += 1
                print(f"💾 Cache saved to disk: {len(self.cache)} entries")
                return True
                
        except Exception as e:
            print(f"❌ Failed to save cache to disk: {e}")
            return False
    
    def load_from_disk(self):
        """Load cache from disk."""
        try:
            if not self.cache_file.exists():
                print("📁 No existing cache file found, starting with empty cache")
                return False
            
            with self.lock:
                # Load cache data
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Load stats if available
                if self.stats_file.exists():
                    with open(self.stats_file, 'r') as f:
                        saved_stats = json.load(f)
                    # Preserve runtime stats but load persistent ones
                    self.stats.update(saved_stats)
                
                # Validate cache data
                if 'cache' not in cache_data:
                    print("❌ Invalid cache file format")
                    return False
                
                # Load cache entries and validate expiration
                current_time = time.time()
                loaded_count = 0
                expired_count = 0
                
                for key, (audio_data, timestamp) in cache_data['cache'].items():
                    if current_time - timestamp <= self.max_age_seconds:
                        self.cache[key] = (audio_data, timestamp)
                        loaded_count += 1
                    else:
                        expired_count += 1
                
                self.stats['loads'] += 1
                self.stats['size'] = len(self.cache)
                
                print(f"📂 Cache loaded from disk: {loaded_count} entries (expired: {expired_count})")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load cache from disk: {e}")
            # Reset to empty cache on load failure
            self.cache.clear()
            return False
    
    def clear(self):
        """Clear all cache entries and remove disk files."""
        with self.lock:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'size': 0, 'saves': 0, 'loads': 0}
            
            # Remove disk files
            try:
                if self.cache_file.exists():
                    self.cache_file.unlink()
                if self.stats_file.exists():
                    self.stats_file.unlink()
                print("🗑️ Cache files removed from disk")
            except Exception as e:
                print(f"⚠️ Warning: Failed to remove cache files: {e}")

# Global cache instance
tts_cache = TTSCache(max_size=10000, max_age_seconds=3600*24*31)

# === PYDANTIC MODELS ===
class TTSRequest(BaseModel):
    model: str = "orpheus"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0

# === HELPER FUNCTIONS ===
def split_text_into_chunks(text, max_chunk_length=200):
    """Split text into sensible chunks for TTS processing."""
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    if len(text) <= max_chunk_length:
        return [text]
    
    chunks = []
    
    import re
    sentence_endings = re.split(r'([.!?]+\s*)', text)
    
    current_chunk = ""
    
    for i in range(0, len(sentence_endings), 2):
        sentence = sentence_endings[i]
        ending = sentence_endings[i + 1] if i + 1 < len(sentence_endings) else ""
        full_sentence = sentence + ending
        
        if len(current_chunk) + len(full_sentence) > max_chunk_length:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            if len(full_sentence) > max_chunk_length:
                parts = re.split(r'(,\s*)', full_sentence)
                temp_chunk = ""
                
                for j in range(0, len(parts), 2):
                    part = parts[j]
                    comma = parts[j + 1] if j + 1 < len(parts) else ""
                    full_part = part + comma
                    
                    if len(temp_chunk) + len(full_part) > max_chunk_length:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        temp_chunk = full_part
                    else:
                        temp_chunk += full_part
                
                if temp_chunk.strip():
                    current_chunk = temp_chunk
            else:
                current_chunk = full_sentence
        else:
            current_chunk += full_sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def decode_snac(code_list):
    """Convert SNAC token sequences back to audio waveforms."""
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))

    device_ = snac_model.quantizer.quantizers[0].codebook.weight.device
    layers = [
        torch.tensor(layer_1).unsqueeze(0).to(device_),
        torch.tensor(layer_2).unsqueeze(0).to(device_),
        torch.tensor(layer_3).unsqueeze(0).to(device_),
    ]

    with torch.no_grad():
        audio = snac_model.decode(layers).squeeze().cpu().numpy()
    return audio

def tts_generate(text, voice, model, device, language=None):
    """Core TTS function that processes text through the model to generate audio tokens."""
    # Try to get from cache first
    if language:
        cached_audio = tts_cache.get(text, voice, language)
        if cached_audio is not None:
            print(f"🎯 Cache HIT for chunk: {text[:50]}...")
            return cached_audio
        print(f"🔍 Cache MISS for chunk: {text[:50]}...")
    
    prompt = f"{voice}: {text}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_token = torch.tensor([[128259]], dtype=torch.long).to(device)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.long).to(device)
    input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    pad_len = 4260 - input_ids.shape[1]
    pad = torch.full((1, pad_len), 128263, dtype=torch.long).to(device)
    input_ids = torch.cat([pad, input_ids], dim=1)

    attention_mask = torch.cat([
        torch.zeros((1, pad_len), dtype=torch.long),
        torch.ones((1, input_ids.shape[1] - pad_len), dtype=torch.long),
    ], dim=1).to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=SHARED_CONFIG["max_new_tokens"],
            do_sample=True,
            temperature=SHARED_CONFIG["temperature"],
            top_p=SHARED_CONFIG["top_p"],
            repetition_penalty=SHARED_CONFIG["repetition_penalty"],
            eos_token_id=128258,
            use_cache=True,
        )

    token_to_find = 128257
    token_to_remove = 128258
    indices = (generated == token_to_find).nonzero(as_tuple=True)
    if len(indices[1]) > 0:
        last_idx = indices[1][-1].item()
        cropped = generated[:, last_idx + 1:]
    else:
        cropped = generated
    cleaned = cropped[cropped != token_to_remove]
    trimmed = cleaned[: (len(cleaned) // 7) * 7]
    trimmed = [int(t) - 128266 for t in trimmed]

    audio = decode_snac(trimmed)
    
    # Store in cache if language is provided
    if language:
        tts_cache.put(text, voice, language, audio)
        print(f"💾 Cached chunk: {text[:50]}...")
    
    return audio

async def generate_audio_parallel(text_list, voice, language=None):
    """Generate audio chunks for each sentence in parallel on GPUs."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
        futures = []
        for idx, sentence in enumerate(text_list):
            gpu_id = idx % GPU_COUNT
            model = models[gpu_id]
            device = f"cuda:{DEVICE_IDS[gpu_id]}"
            futures.append((idx, loop.run_in_executor(executor, tts_generate, sentence, voice, model, device, language)))

        results = []
        for idx, fut in futures:
            audio_chunk = await fut
            results.append((idx, audio_chunk))

    results.sort(key=lambda x: x[0])
    audios_ordered = [chunk for _, chunk in results]
    return np.concatenate(audios_ordered)

# === INITIALIZATION ===
def initialize_models(language: str):
    """Initialize TTS models and components for a specific language."""
    global tokenizer, snac_model, models, DEVICE_IDS, GPU_COUNT, SERVER_CONFIG
    
    SERVER_CONFIG = get_model_config(language)
    if not SERVER_CONFIG:
        raise ValueError(f"No configuration found for language: {language}")
    
    if not SERVER_CONFIG["enabled"]:
        raise ValueError(f"Language model '{language}' is disabled in configuration")
    
    # Set HuggingFace token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = SHARED_CONFIG["hf_token"]
    
    GPU_COUNT = torch.cuda.device_count()
    DEVICE_IDS = list(range(GPU_COUNT))
    print(f"✅ Detected {GPU_COUNT} GPUs: {DEVICE_IDS}")

    print(f"🔠 Loading tokenizer for {SERVER_CONFIG['display_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(SERVER_CONFIG["model_path"])

    print("🎤 Loading SNAC model...")
    snac_model = SNAC.from_pretrained(SHARED_CONFIG["snac_model_id"]).to("cuda:0")

    print(f"🧠 Loading {SERVER_CONFIG['display_name']} models...")
    models = []
    for i, dev_id in enumerate(DEVICE_IDS):
        print(f"   -> Model [{SERVER_CONFIG['model_path']}] index {i} on cuda:{dev_id}")
        model = AutoModelForCausalLM.from_pretrained(
            SERVER_CONFIG["model_path"], torch_dtype=torch.float16).to(f"cuda:{dev_id}")
        model.eval()
        models.append(model)

# === FASTAPI APP ===
def create_app(language: str):
    """Create FastAPI app for a specific language."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Handle startup and shutdown events."""
        # Startup
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
            voice = request.voice or SERVER_CONFIG["default_voice"]
            
            # Split text into chunks for processing
            text_chunks = split_text_into_chunks(
                request.input, 
                max_chunk_length=SHARED_CONFIG["max_chunk_length"]
            )
            print(f"📝 [{language.upper()}] Text split into {len(text_chunks)} chunks for: {request.input[:50]}...")
            
            # Generate audio with caching
            audio = await generate_audio_parallel(text_chunks, voice, language)
            
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
            "display_name": SERVER_CONFIG["display_name"] if SERVER_CONFIG else "Unknown",
            "models_loaded": len(models) > 0,
            "gpu_count": GPU_COUNT,
            "default_voice": SERVER_CONFIG["default_voice"] if SERVER_CONFIG else "Unknown",
            "cache_enabled": True,
            "cache_size": cache_stats["size"],
            "cache_hit_rate": cache_stats["hit_rate"]
        }

    @app.get("/info")
    async def server_info():
        """Get server information."""
        return {
            "language": language,
            "config": SERVER_CONFIG,
            "shared_config": SHARED_CONFIG,
            "gpu_count": GPU_COUNT,
            "models_loaded": len(models)
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
    
    @app.post("/cache/save")
    async def save_cache():
        """Manually save cache to disk."""
        success = tts_cache.save_to_disk(force=True)
        return {
            "message": "Cache saved successfully" if success else "Failed to save cache",
            "success": success,
            "stats": tts_cache.get_stats()
        }
    
    @app.get("/cache/info")
    async def cache_info():
        """Get detailed cache information including persistence details."""
        return {
            "stats": tts_cache.get_stats(),
            "persistence": {
                "cache_dir": str(tts_cache.cache_dir),
                "cache_file": str(tts_cache.cache_file),
                "stats_file": str(tts_cache.stats_file),
                "cache_file_exists": tts_cache.cache_file.exists(),
                "stats_file_exists": tts_cache.stats_file.exists(),
                "save_interval_seconds": tts_cache.save_interval,
                "last_save": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tts_cache.last_save))
            },
            "config": {
                "max_size": tts_cache.max_size,
                "max_age_seconds": tts_cache.max_age_seconds
            }
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
