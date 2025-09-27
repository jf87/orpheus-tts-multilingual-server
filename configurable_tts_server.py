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
import logging
from datetime import datetime
import subprocess
import platform

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from models_config import MODEL_CONFIGS, SHARED_CONFIG, get_model_config, get_audio_effects_config, get_word_replacements
from audio_postprocessor import TTSAudioPostProcessor
from chunking import split_text_into_chunks_chars

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
        logger.debug(f"Cache GET - Key: {key}, Text: {text[:30]}...")
        
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache:
                audio_data, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.stats['hits'] += 1
                    logger.debug(f"Cache HIT - Key: {key}, Audio shape: {audio_data.shape}")
                    return audio_data.copy()
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.stats['evictions'] += 1
                    logger.debug(f"Cache EXPIRED - Key: {key} removed")
            
            self.stats['misses'] += 1
            logger.debug(f"Cache MISS - Key: {key}")
            return None
    
    def put(self, text: str, voice: str, language: str, audio_data: np.ndarray):
        """Store audio in cache."""
        key = self._generate_key(text, voice, language)
        timestamp = time.time()
        logger.debug(f"Cache PUT - Key: {key}, Audio shape: {audio_data.shape}, Text: {text[:30]}...")
        
        with self.lock:
            # Remove oldest entries if at max size
            evictions = 0
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
                evictions += 1
            
            if evictions > 0:
                logger.debug(f"Cache evicted {evictions} oldest entries to make space")
            
            self.cache[key] = (audio_data.copy(), timestamp)
            self.stats['size'] = len(self.cache)
            logger.debug(f"Cache stored - Key: {key}, Cache size: {len(self.cache)}/{self.max_size}")
            
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

class DictionaryEntryRequest(BaseModel):
    word: str
    replacement: str

class DictionaryUpdateRequest(BaseModel):
    entries: Dict[str, str]  # word -> replacement mapping

# === TEXT PREPROCESSING ===
import re

def preprocess_text_intelligent(text, language=None, enable_preprocessing=True):
    """
    Intelligent text preprocessing for better TTS results.
    
    Args:
        text (str): Input text to preprocess
        language (str): Language code (e.g., 'de', 'en', 'es')
        enable_preprocessing (bool): Whether to apply preprocessing
    
    Returns:
        str: Preprocessed text
    """
    logger.debug(f"Starting preprocessing - Language: {language}, Enabled: {enable_preprocessing}")
    logger.debug(f"Input text: {text}")
    
    if not enable_preprocessing or not text:
        logger.debug("Preprocessing disabled or empty text, returning original")
        return text
    
    processed_text = text
    
    # 1. Apply custom word replacements from dictionary
    logger.debug("Applying custom word replacements")
    before_dict = processed_text
    processed_text = apply_word_replacements(processed_text, language)
    if before_dict != processed_text:
        logger.debug(f"Dictionary replacements applied:")
        logger.debug(f"  BEFORE: {before_dict}")
        logger.debug(f"  AFTER:  {processed_text}")
    
    # 2. Remove quotes (straight and curly quotes)
    logger.debug("Removing quotes")
    before_quotes = processed_text
    processed_text = re.sub(r'["""\'\'`]([^"""\'\'`]*)["""\'\'`]', r'\1', processed_text)
    if before_quotes != processed_text:
        logger.debug(f"Quotes removed:")
        logger.debug(f"  BEFORE: {before_quotes}")
        logger.debug(f"  AFTER:  {processed_text}")
    
    # 3. Language-specific number-to-text conversions
    logger.debug(f"Applying language-specific conversions for: {language}")
    before_lang_specific = processed_text
    
    if language == 'de' or language is None:  # Default to German rules
        logger.debug("Applying German number conversions")
        # Convert numeric ratios to text (German)
        processed_text = re.sub(r'(\d+)-zu-(\d+)', lambda m: f"{number_to_german(int(m.group(1)))}-zu-{number_to_german(int(m.group(2)))}", processed_text)
        
        logger.debug("Applying German SQL preprocessing")
        # Convert SQL and programming syntax to German
        processed_text = preprocess_sql_commands_german(processed_text)
        
    elif language == 'en':
        logger.debug("Applying English number conversions")
        # Convert numeric ratios to text (English)
        processed_text = re.sub(r'(\d+)-to-(\d+)', lambda m: f"{number_to_english(int(m.group(1)))}-to-{number_to_english(int(m.group(2)))}", processed_text)
        
        logger.debug("Applying English SQL preprocessing")
        # Convert SQL and programming syntax to English
        processed_text = preprocess_sql_commands_english(processed_text)
        
    elif language == 'es':
        logger.debug("Applying Spanish number conversions")
        # Convert numeric ratios to text (Spanish)
        processed_text = re.sub(r'(\d+)-a-(\d+)', lambda m: f"{number_to_spanish(int(m.group(1)))}-a-{number_to_spanish(int(m.group(2)))}", processed_text)
        
        logger.debug("Applying Spanish SQL preprocessing")
        # Convert SQL and programming syntax to Spanish
        processed_text = preprocess_sql_commands_spanish(processed_text)
    
    if before_lang_specific != processed_text:
        logger.debug(f"Language-specific changes:")
        logger.debug(f"  BEFORE: {before_lang_specific}")
        logger.debug(f"  AFTER:  {processed_text}")
    
    # 4. General text cleanup
    logger.debug("Applying general text cleanup")
    before_cleanup = processed_text
    processed_text = clean_general_text(processed_text)
    if before_cleanup != processed_text:
        logger.debug(f"General cleanup applied:")
        logger.debug(f"  BEFORE: {before_cleanup}")
        logger.debug(f"  AFTER:  {processed_text}")
    
    logger.debug(f"Preprocessing completed. Final result: {processed_text}")
    
    return processed_text

def apply_word_replacements(text, language):
    """
    Apply language-specific word replacements from the configurable dictionary.
    
    Args:
        text (str): Input text
        language (str): Language code (e.g., 'de', 'en', 'es')
    
    Returns:
        str: Text with custom word replacements applied
    """
    logger.debug(f"Applying word replacements for language: {language}")
    
    if not language:
        language = 'de'  # Default to German
        logger.debug("No language specified, defaulting to German")
    
    # Get language-specific dictionary
    language_key = {
        'de': 'german',
        'en': 'english', 
        'es': 'spanish'
    }.get(language, 'german')
    
    word_dict = get_word_replacements(language_key)
    logger.debug(f"Using dictionary for {language_key}: {len(word_dict)} entries")
    
    if not word_dict:
        logger.debug("No dictionary entries found, returning original text")
        return text
    
    # Apply replacements (case-insensitive, whole word matching)
    processed_text = text
    replacements_made = []
    
    for original_word, replacement in word_dict.items():
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(original_word) + r'\b'
        matches = re.findall(pattern, processed_text, flags=re.IGNORECASE)
        if matches:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
            replacements_made.append(f"{original_word} -> {replacement} ({len(matches)} matches)")
            logger.debug(f"Replaced '{original_word}' with '{replacement}' ({len(matches)} matches)")
    
    if replacements_made:
        logger.debug(f"Dictionary replacements completed: {', '.join(replacements_made)}")
    else:
        logger.debug("No dictionary replacements applied")
    
    return processed_text

def number_to_german(num):
    """Convert numbers 1-20 to German words."""
    german_numbers = {
        1: "Eins", 2: "Zwei", 3: "Drei", 4: "Vier", 5: "Fünf",
        6: "Sechs", 7: "Sieben", 8: "Acht", 9: "Neun", 10: "Zehn",
        11: "Elf", 12: "Zwölf", 13: "Dreizehn", 14: "Vierzehn", 15: "Fünfzehn",
        16: "Sechzehn", 17: "Siebzehn", 18: "Achtzehn", 19: "Neunzehn", 20: "Zwanzig"
    }
    return german_numbers.get(num, str(num))

def number_to_english(num):
    """Convert numbers 1-20 to English words."""
    english_numbers = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen",
        16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty"
    }
    return english_numbers.get(num, str(num))

def number_to_spanish(num):
    """Convert numbers 1-20 to Spanish words."""
    spanish_numbers = {
        1: "Uno", 2: "Dos", 3: "Tres", 4: "Cuatro", 5: "Cinco",
        6: "Seis", 7: "Siete", 8: "Ocho", 9: "Nueve", 10: "Diez",
        11: "Once", 12: "Doce", 13: "Trece", 14: "Catorce", 15: "Quince",
        16: "Dieciséis", 17: "Diecisiete", 18: "Dieciocho", 19: "Diecinueve", 20: "Veinte"
    }
    return spanish_numbers.get(num, str(num))

def preprocess_sql_commands_german(text):
    """Convert SQL syntax to German pronunciation."""
    # Replace dots in table.column references
    text = re.sub(r'(\w+)\.(\w+)', r'\1 Punkt \2', text)
    
    # Replace equals signs
    text = re.sub(r'\s*=\s*', ' gleich ', text)
    
    # Replace common SQL keywords with German pronunciation (capitalized)
    sql_replacements = {
        r'\bSELECT\b': 'Select',
        r'\bFROM\b': 'From',
        r'\bWHERE\b': 'Where',
        r'\bJOIN\b': 'Join',
        r'\bON\b': 'On',
        r'\bINNER\b': 'Inner',
        r'\bLEFT\b': 'Left',
        r'\bRIGHT\b': 'Right',
        r'\bOUTER\b': 'Outer',
        r'\bORDER\b': 'Order',
        r'\bBY\b': 'By',
        r'\bGROUP\b': 'Group',
        r'\bHAVING\b': 'Having',
        r'\bINSERT\b': 'Insert',
        r'\bUPDATE\b': 'Update',
        r'\bDELETE\b': 'Delete',
    }
    
    for pattern, replacement in sql_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def preprocess_sql_commands_english(text):
    """Convert SQL syntax to English pronunciation."""
    # Replace dots in table.column references
    text = re.sub(r'(\w+)\.(\w+)', r'\1 dot \2', text)
    
    # Replace equals signs
    text = re.sub(r'\s*=\s*', ' equals ', text)
    
    # Replace common SQL keywords with English pronunciation (lowercase)
    sql_replacements = {
        r'\bSELECT\b': 'select',
        r'\bFROM\b': 'from',
        r'\bWHERE\b': 'where',
        r'\bJOIN\b': 'join',
        r'\bON\b': 'on',
        r'\bINNER\b': 'inner',
        r'\bLEFT\b': 'left',
        r'\bRIGHT\b': 'right',
        r'\bOUTER\b': 'outer',
        r'\bORDER\b': 'order',
        r'\bBY\b': 'by',
        r'\bGROUP\b': 'group',
        r'\bHAVING\b': 'having',
        r'\bINSERT\b': 'insert',
        r'\bUPDATE\b': 'update',
        r'\bDELETE\b': 'delete',
    }
    
    for pattern, replacement in sql_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def preprocess_sql_commands_spanish(text):
    """Convert SQL syntax to Spanish pronunciation."""
    # Replace dots in table.column references
    text = re.sub(r'(\w+)\.(\w+)', r'\1 punto \2', text)
    
    # Replace equals signs
    text = re.sub(r'\s*=\s*', ' igual ', text)
    
    # Replace common SQL keywords with Spanish pronunciation (lowercase)
    sql_replacements = {
        r'\bSELECT\b': 'select',
        r'\bFROM\b': 'from',
        r'\bWHERE\b': 'where',
        r'\bJOIN\b': 'join',
        r'\bON\b': 'on',
        r'\bINNER\b': 'inner',
        r'\bLEFT\b': 'left',
        r'\bRIGHT\b': 'right',
        r'\bOUTER\b': 'outer',
        r'\bORDER\b': 'order',
        r'\bBY\b': 'by',
        r'\bGROUP\b': 'group',
        r'\bHAVING\b': 'having',
        r'\bINSERT\b': 'insert',
        r'\bUPDATE\b': 'update',
        r'\bDELETE\b': 'delete',
    }
    
    for pattern, replacement in sql_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def clean_general_text(text):
    """Apply general text cleaning for better TTS."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    return text.strip()

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
    logger.debug(f"Starting TTS generation - Text: {text[:50]}..., Voice: {voice}, Device: {device}, Language: {language}")
    
    # Try to get from cache first
    if language:
        logger.debug(f"Checking cache for text chunk")
        cached_audio = tts_cache.get(text, voice, language)
        if cached_audio is not None:
            logger.info(f"🎯 Cache HIT for chunk: {text[:50]}...")
            logger.debug(f"Returning cached audio with shape: {cached_audio.shape}")
            return cached_audio
        logger.info(f"🔍 Cache MISS for chunk: {text[:50]}...")
    
    logger.debug(f"Creating prompt with voice '{voice}'")
    prompt = f"{voice}: {text}"
    logger.debug(f"Full prompt: {prompt}")
    
    logger.debug("Tokenizing prompt")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logger.debug(f"Input IDs shape: {input_ids.shape}")

    logger.debug("Adding start and end tokens")
    start_token = torch.tensor([[128259]], dtype=torch.long).to(device)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.long).to(device)
    input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    logger.debug(f"Input IDs with tokens shape: {input_ids.shape}")

    logger.debug("Applying padding")
    pad_len = 4260 - input_ids.shape[1]
    pad = torch.full((1, pad_len), 128263, dtype=torch.long).to(device)
    input_ids = torch.cat([pad, input_ids], dim=1)
    logger.debug(f"Padded input IDs shape: {input_ids.shape}, pad_len: {pad_len}")

    logger.debug("Creating attention mask")
    attention_mask = torch.cat([
        torch.zeros((1, pad_len), dtype=torch.long),
        torch.ones((1, input_ids.shape[1] - pad_len), dtype=torch.long),
    ], dim=1).to(device)
    logger.debug(f"Attention mask shape: {attention_mask.shape}")

    logger.debug("Starting model generation")
    generation_params = {
        "max_new_tokens": SHARED_CONFIG["max_new_tokens"],
        "temperature": SHARED_CONFIG["temperature"],
        "top_p": SHARED_CONFIG["top_p"],
        "repetition_penalty": SHARED_CONFIG["repetition_penalty"],
    }
    logger.debug(f"Generation parameters: {generation_params}")
    
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
    logger.debug(f"Generated tokens shape: {generated.shape}")

    logger.debug("Processing generated tokens")
    token_to_find = 128257
    token_to_remove = 128258
    indices = (generated == token_to_find).nonzero(as_tuple=True)
    logger.debug(f"Looking for token {token_to_find}, found {len(indices[1])} instances")
    
    if len(indices[1]) > 0:
        last_idx = indices[1][-1].item()
        cropped = generated[:, last_idx + 1:]
        logger.debug(f"Cropped from index {last_idx}, new shape: {cropped.shape}")
    else:
        cropped = generated
        logger.debug("No crop index found, using full generated sequence")
    
    logger.debug(f"Removing token {token_to_remove}")
    cleaned = cropped[cropped != token_to_remove]
    logger.debug(f"Cleaned tokens length: {len(cleaned)}")
    
    trimmed = cleaned[: (len(cleaned) // 7) * 7]
    logger.debug(f"Trimmed to multiple of 7: {len(trimmed)} tokens")
    
    trimmed = [int(t) - 128266 for t in trimmed]
    logger.debug(f"Converted tokens for SNAC decoding: {len(trimmed)} tokens")

    logger.debug("Decoding with SNAC")
    audio = decode_snac(trimmed)
    logger.debug(f"Generated audio shape: {audio.shape}, duration: {len(audio) / 24000:.2f}s")
    
    # Store in cache if language is provided
    if language:
        logger.debug("Storing result in cache")
        tts_cache.put(text, voice, language, audio)
        logger.info(f"💾 Cached chunk: {text[:50]}...")
    
    logger.debug("TTS generation completed successfully")
    return audio

async def generate_audio_parallel(text_list, voice, language=None):
    """Generate audio chunks for each sentence in parallel on GPUs."""
    logger.info(f"Starting parallel audio generation for {len(text_list)} chunks")
    logger.debug(f"Voice: {voice}, Language: {language}, GPU Count: {GPU_COUNT}")
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
        futures = []
        for idx, sentence in enumerate(text_list):
            gpu_id = idx % GPU_COUNT
            model = models[gpu_id]
            device = f"cuda:{DEVICE_IDS[gpu_id]}"
            logger.debug(f"Chunk {idx} -> GPU {gpu_id} ({device}): {sentence[:30]}...")
            futures.append((idx, loop.run_in_executor(executor, tts_generate, sentence, voice, model, device, language)))

        logger.debug("Waiting for all chunks to complete")
        results = []
        for idx, fut in futures:
            audio_chunk = await fut
            logger.debug(f"Chunk {idx} completed, audio shape: {audio_chunk.shape}")
            results.append((idx, audio_chunk))

    logger.debug("Sorting and concatenating audio chunks")
    results.sort(key=lambda x: x[0])
    audios_ordered = [chunk for _, chunk in results]
    final_audio = np.concatenate(audios_ordered)
    
    logger.info(f"Parallel generation completed - Final audio: {final_audio.shape}, duration: {len(final_audio) / 24000:.2f}s")
    return final_audio

# === INITIALIZATION ===
def initialize_models(language: str):
    """Initialize TTS models and components for a specific language."""
    global tokenizer, snac_model, models, DEVICE_IDS, GPU_COUNT, SERVER_CONFIG
    
    logger.info(f"Initializing models for language: {language}")
    
    SERVER_CONFIG = get_model_config(language)
    if not SERVER_CONFIG:
        logger.error(f"No configuration found for language: {language}")
        raise ValueError(f"No configuration found for language: {language}")
    
    if not SERVER_CONFIG["enabled"]:
        logger.error(f"Language model '{language}' is disabled in configuration")
        raise ValueError(f"Language model '{language}' is disabled in configuration")
    
    logger.debug(f"Using configuration: {SERVER_CONFIG}")
    
    # Set HuggingFace token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = SHARED_CONFIG["hf_token"]
    logger.debug("HuggingFace token set from configuration")
    
    GPU_COUNT = torch.cuda.device_count()
    DEVICE_IDS = list(range(GPU_COUNT))
    logger.info(f"✅ Detected {GPU_COUNT} GPUs: {DEVICE_IDS}")

    logger.info(f"🔠 Loading tokenizer for {SERVER_CONFIG['display_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(SERVER_CONFIG["model_path"])
    logger.debug(f"Tokenizer loaded: {len(tokenizer)} tokens")

    logger.info("🎤 Loading SNAC model...")
    snac_model = SNAC.from_pretrained(SHARED_CONFIG["snac_model_id"]).to("cuda:0")
    logger.debug(f"SNAC model loaded on cuda:0")

    logger.info(f"🧠 Loading {SERVER_CONFIG['display_name']} models...")
    models = []
    for i, dev_id in enumerate(DEVICE_IDS):
        logger.info(f"   -> Model [{SERVER_CONFIG['model_path']}] index {i} on cuda:{dev_id}")
        model = AutoModelForCausalLM.from_pretrained(
            SERVER_CONFIG["model_path"], torch_dtype=torch.float16).to(f"cuda:{dev_id}")
        model.eval()
        models.append(model)
        logger.debug(f"Model {i} loaded and set to eval mode")
    
    logger.info(f"Model initialization completed for {language}")

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
            
            # Split text into chunks for processing
            #text_chunks = split_text_into_chunks(
            #    processed_text, 
            #    max_chunk_length=SHARED_CONFIG["max_chunk_length"]
            #)
            text_chunks = split_text_into_chunks_chars(
                processed_text, 
                max_chars=200,
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
                # Get configured defaults for this language
                configured_effects = get_audio_effects_config(language)
                logger.debug(f"🎛️ [{language.upper()}] Configured audio effects: {configured_effects}")
                
                # Extract audio effects from request, using configured defaults as fallback
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
    
    
    @app.get("/dictionary/test/{word}")
    async def test_dictionary_entry(word: str):
        """Test how a word would be processed with current dictionary."""
        try:
            # Test the word in a simple sentence
            test_text = f"This is a test with the word {word} in it."
            
            processed_text = preprocess_text_intelligent(
                test_text,
                language=language,
                enable_preprocessing=True
            )
            
            return {
                "word": word,
                "language": language,
                "test_input": test_text,
                "processed_output": processed_text,
                "was_modified": test_text != processed_text
            }
            
        except Exception as e:
            logger.error(f"Failed to test dictionary entry: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to test dictionary entry: {str(e)}")

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
    
    class AudioEffectsRequest(BaseModel):
        pitch_shift: Optional[float] = 0.0
        speed_factor: Optional[float] = 1.0
        gain_db: Optional[float] = 0.0
        normalize_audio: Optional[bool] = False
        use_limiter: Optional[bool] = True
        add_reverb: Optional[bool] = False
        reverb_amount: Optional[float] = 50
        reverb_room_scale: Optional[float] = 50
        add_echo: Optional[bool] = False
        echo_delay: Optional[float] = 0.5
        echo_decay: Optional[float] = 0.5
    
    @app.post("/audio-effects/validate")
    async def validate_audio_effects(effects: AudioEffectsRequest):
        """Validate audio effect parameters."""
        if not audio_postprocessor.is_available():
            raise HTTPException(status_code=503, detail="SoX not available for audio processing")
        
        effects_dict = effects.dict()
        validated = audio_postprocessor.validate_effects(effects_dict)
        
        # Check for any corrections made during validation
        corrections = {}
        for key, value in effects_dict.items():
            if key in validated and validated[key] != value:
                corrections[key] = {"original": value, "corrected": validated[key]}
        
        return {
            "language": language,
            "valid": True,
            "validated_effects": validated,
            "corrections": corrections if corrections else None
        }
    
    @app.post("/audio-effects/test")
    async def test_audio_effects(effects: AudioEffectsRequest):
        """Test audio effects on a short sample."""
        if not audio_postprocessor.is_available():
            raise HTTPException(status_code=503, detail="SoX not available for audio processing")
        
        try:
            # Generate a short test tone (440Hz for 1 second)
            sample_rate = 24000
            duration = 1.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            effects_dict = effects.dict()
            validated_effects = audio_postprocessor.validate_effects(effects_dict)
            
            logger.info(f"🧪 [{language.upper()}] Testing audio effects")
            logger.debug(f"Test effects: {validated_effects}")
            
            # Apply effects
            processed_audio = audio_postprocessor.process_audio(
                test_audio,
                sample_rate=sample_rate,
                **validated_effects
            )
            
            # Check if processing was successful
            if processed_audio is not None and len(processed_audio) > 0:
                processing_successful = True
                original_length = len(test_audio)
                processed_length = len(processed_audio)
                length_change_percent = ((processed_length - original_length) / original_length) * 100 if original_length > 0 else 0
            else:
                processing_successful = False
                length_change_percent = 0
            
            return {
                "language": language,
                "test_successful": processing_successful,
                "effects_applied": validated_effects,
                "original_duration_samples": len(test_audio),
                "processed_duration_samples": len(processed_audio) if processed_audio is not None else 0,
                "length_change_percent": round(length_change_percent, 2),
                "message": "Audio effects test completed successfully" if processing_successful else "Audio effects test failed"
            }
            
        except Exception as e:
            logger.error(f"Audio effects test failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio effects test failed: {str(e)}")

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
