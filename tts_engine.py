#!/usr/bin/env python3
"""
TTS engine module for Orpheus-HF TTS system.

This module contains the core TTS functionality including model initialization,
audio generation, and SNAC decoding. It provides the pure TTS logic separate
from the web API layer.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC
import os
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Optional

from models_config import SHARED_CONFIG, get_model_config

# Get logger for this module
logger = logging.getLogger(__name__)

# === CONSTANTS ===
SAMPLE_RATE = 24000  # Audio sample rate for SNAC codec
SNAC_TOKENS_PER_FRAME = 7  # SNAC uses 7 tokens per audio frame (1 + 2 + 4 layers)
PAUSE_DURATION = 0.5  # Silence duration (seconds) between chunks
PADDING_LENGTH = 4260  # Fixed padding length for model input
START_TOKEN = 128259  # Special token marking audio generation start
END_TOKEN_1 = 128009  # First end token
END_TOKEN_2 = 128260  # Second end token
PAD_TOKEN = 128263  # Padding token
AUDIO_START_TOKEN = 128257  # Token marking start of audio sequence
AUDIO_END_TOKEN = 128258  # Token marking end of audio sequence
TOKEN_OFFSET = 128266  # Offset to convert model tokens to SNAC codes
LAYER_2_OFFSET = 4096  # Offset for SNAC layer 2 codes
LAYER_3_BASE_OFFSET = 2 * 4096  # Base offset for SNAC layer 3 codes

# === GLOBAL MODEL STATE ===
# These will be initialized by initialize_models()
tokenizer = None
snac_model = None
models = []
DEVICE_IDS = []
GPU_COUNT = 0
SERVER_CONFIG = None

# Global cache instance - will be injected from server
tts_cache = None

def set_cache_instance(cache_instance):
    """Set the cache instance for TTS operations."""
    global tts_cache
    tts_cache = cache_instance

# === CORE TTS FUNCTIONS ===

def decode_snac(code_list):
    """
    Convert SNAC token sequences back to audio waveforms.

    SNAC uses a hierarchical structure with 7 tokens per frame:
    - Token 0: Layer 1 (1 code)
    - Tokens 1, 4: Layer 2 (2 codes)
    - Tokens 2, 3, 5, 6: Layer 3 (4 codes)

    Each layer has a different offset to distinguish them in the flat token stream.
    """
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list) + 1) // SNAC_TOKENS_PER_FRAME):
        base = SNAC_TOKENS_PER_FRAME * i
        # Layer 1: 1 code per frame
        layer_1.append(code_list[base])
        # Layer 2: 2 codes per frame (at positions 1 and 4)
        layer_2.append(code_list[base + 1] - LAYER_2_OFFSET)
        # Layer 3: 4 codes per frame (at positions 2, 3, 5, 6)
        layer_3.append(code_list[base + 2] - LAYER_3_BASE_OFFSET)
        layer_3.append(code_list[base + 3] - (3 * LAYER_2_OFFSET))
        layer_2.append(code_list[base + 4] - (4 * LAYER_2_OFFSET))
        layer_3.append(code_list[base + 5] - (5 * LAYER_2_OFFSET))
        layer_3.append(code_list[base + 6] - (6 * LAYER_2_OFFSET))

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
    """
    Core TTS function that processes text through the model to generate audio tokens.

    The process:
    1. Check cache for existing audio
    2. Format prompt as "voice: text"
    3. Tokenize and add special control tokens (START, END)
    4. Apply padding to fixed length for batch processing
    5. Generate audio tokens via causal LM
    6. Extract and decode SNAC tokens to audio waveform
    7. Cache result for future requests
    """
    logger.debug(f"Starting TTS generation - Text: {text[:50]}..., Voice: {voice}, Device: {device}, Language: {language}")

    # Try to get from cache first
    if language and tts_cache:
        cached_audio = tts_cache.get(text, voice, language)
        if cached_audio is not None:
            logger.info(f"🎯 Cache HIT for chunk: {text[:50]}...")
            return cached_audio
        logger.info(f"🔍 Cache MISS for chunk: {text[:50]}...")

    # Format prompt with voice identifier
    prompt = f"{voice}: {text}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Add special control tokens
    start_token = torch.tensor([[START_TOKEN]], dtype=torch.long).to(device)
    end_tokens = torch.tensor([[END_TOKEN_1, END_TOKEN_2]], dtype=torch.long).to(device)
    input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    # Apply left-side padding to fixed length
    pad_len = PADDING_LENGTH - input_ids.shape[1]
    pad = torch.full((1, pad_len), PAD_TOKEN, dtype=torch.long).to(device)
    input_ids = torch.cat([pad, input_ids], dim=1)

    # Create attention mask (ignore padding, attend to real tokens)
    attention_mask = torch.cat([
        torch.zeros((1, pad_len), dtype=torch.long),
        torch.ones((1, input_ids.shape[1] - pad_len), dtype=torch.long),
    ], dim=1).to(device)

    # Generate audio tokens
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=SHARED_CONFIG["max_new_tokens"],
            do_sample=True,
            temperature=SHARED_CONFIG["temperature"],
            top_p=SHARED_CONFIG["top_p"],
            repetition_penalty=SHARED_CONFIG["repetition_penalty"],
            eos_token_id=AUDIO_END_TOKEN,
            use_cache=True,
        )

    # Extract audio tokens: find start marker, crop to audio sequence
    indices = (generated == AUDIO_START_TOKEN).nonzero(as_tuple=True)
    if len(indices[1]) > 0:
        last_idx = indices[1][-1].item()
        cropped = generated[:, last_idx + 1:]
    else:
        cropped = generated

    # Remove end markers and ensure length is multiple of 7 for SNAC
    cleaned = cropped[cropped != AUDIO_END_TOKEN]
    trimmed = cleaned[: (len(cleaned) // SNAC_TOKENS_PER_FRAME) * SNAC_TOKENS_PER_FRAME]

    # Convert from model token space to SNAC code space
    trimmed = [int(t) - TOKEN_OFFSET for t in trimmed]

    # Decode SNAC tokens to audio waveform
    audio = decode_snac(trimmed)
    logger.debug(f"Generated audio shape: {audio.shape}, duration: {len(audio) / SAMPLE_RATE:.2f}s")

    # Store in cache if language is provided
    if language and tts_cache:
        tts_cache.put(text, voice, language, audio)
        logger.info(f"💾 Cached chunk: {text[:50]}...")

    return audio

async def generate_audio_parallel(text_list, voice, language=None):
    """
    Generate audio chunks for each sentence in parallel on GPUs.

    Uses round-robin GPU assignment to distribute work evenly. Adds natural
    pauses between chunks that end with sentence-ending punctuation.
    """
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

        results = []
        for idx, fut in futures:
            audio_chunk = await fut
            logger.debug(f"Chunk {idx} completed, audio shape: {audio_chunk.shape}")
            results.append((idx, audio_chunk))

    # Sort by original order and concatenate with natural pauses
    results.sort(key=lambda x: x[0])
    audios_ordered = [chunk for _, chunk in results]

    final_audio_parts = []
    for i, audio_chunk in enumerate(audios_ordered):
        final_audio_parts.append(audio_chunk)

        # Add pause between chunks that end with sentence-ending punctuation
        if i < len(audios_ordered) - 1:
            current_text = text_list[i].strip()
            if current_text and current_text[-1] in '.!?:':
                pause_samples = int(PAUSE_DURATION * SAMPLE_RATE)
                silence = np.zeros(pause_samples, dtype=audios_ordered[0].dtype)
                final_audio_parts.append(silence)
                logger.debug(f"Added {PAUSE_DURATION}s pause after chunk {i} (ends with '{current_text[-1]}')")

    final_audio = np.concatenate(final_audio_parts)

    logger.info(f"Parallel generation completed - Final audio: {final_audio.shape}, duration: {len(final_audio) / SAMPLE_RATE:.2f}s")
    return final_audio

# === MODEL INITIALIZATION ===

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

# === STATE ACCESS FUNCTIONS ===

def get_model_info():
    """Get information about loaded models."""
    return {
        "config": SERVER_CONFIG,
        "gpu_count": GPU_COUNT,
        "models_loaded": len(models),
        "device_ids": DEVICE_IDS,
        "tokenizer_size": len(tokenizer) if tokenizer else 0
    }
