# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based Text-to-Speech (TTS) system called "Orpheus-HF" that uses Hugging Face transformers and the SNAC audio codec for high-quality speech synthesis. The project generates speech from text using fine-tuned language models and supports multi-GPU parallel processing.

## Architecture

The system consists of a single main script with several key components:

- **Model Loading**: Uses AutoModelForCausalLM from transformers for the core TTS model and SNAC for audio encoding/decoding
- **Multi-GPU Support**: Automatically detects available GPUs and distributes processing across them
- **Parallel Processing**: Uses asyncio and ThreadPoolExecutor to generate audio chunks in parallel while preserving sentence order
- **Audio Processing**: SNAC codec handles the conversion from model tokens to audio waveforms

## Key Functions

- `decode_snac()` - Converts SNAC token sequences back to audio waveforms
- `tts_generate()` - Core TTS function that processes text through the model to generate audio tokens
- `generate_audio_parallel()` - Orchestrates parallel processing across multiple GPUs for efficient batch generation

## Configuration

Key configuration variables are defined at the top of `generate_speech_hf.py`:
- `pretraind_model` - Base model path/ID
- `finetuned_model` - Path to fine-tuned checkpoint (if applicable)
- `snac_model_id` - SNAC codec model ID
- Temperature, top_p, repetition_penalty for generation control
- Max tokens, number of inferences, voice selection

## Dependencies

The project requires:
- PyTorch with CUDA support
- Hugging Face transformers
- SNAC audio codec library
- soundfile for audio I/O
- numpy for audio processing
- huggingface_hub for model access

## Running the System

Execute the main script directly:
```bash
python generate_speech_hf.py
```

The script will:
1. Detect available GPUs
2. Load the tokenizer, SNAC model, and TTS models on each GPU
3. Process the TEXT_SAMPLE sentences in parallel
4. Save generated audio files to the outputs directory

## Virtual Environment

A Python virtual environment is present in `.venv/` - activate it before running:
```bash
source .venv/bin/activate
```