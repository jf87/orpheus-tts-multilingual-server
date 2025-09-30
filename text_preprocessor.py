#!/usr/bin/env python3
"""
Text preprocessing module for Orpheus-HF TTS system.

This module handles all text preprocessing operations before TTS generation,
including intelligent text preprocessing, word replacements, language-specific
conversions, and text chunking.
"""

import re
import logging
from typing import List
from models_config import get_word_replacements
from chunking import split_text_into_chunks_chars

# Get logger for this module
logger = logging.getLogger(__name__)

# === MAIN PREPROCESSING FUNCTION ===

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
    if not enable_preprocessing or not text:
        return text

    logger.debug(f"Text preprocessing: language={language}, input_len={len(text)}")
    processed_text = text

    # 1. Remove quotes (straight and curly quotes)
    before_quotes = processed_text
    processed_text = re.sub(r'["""\'\'`]([^"""\'\'`]*)["""\'\'`]', r'\1', processed_text)
    if before_quotes != processed_text:
        logger.debug("Removed quotes from text")

    # 2. Language-specific number-to-text conversions
    before_lang_specific = processed_text

    if language in ['de', 'german'] or language is None:  # Default to German rules
        # Convert numeric ratios to text (German)
        def german_ratio_replacer(match):
            num1, num2 = int(match.group(1)), int(match.group(2))
            return f"{number_to_german(num1)}-zu-{number_to_german(num2)}"
        processed_text = re.sub(r'(\d+)-zu-(\d+)', german_ratio_replacer, processed_text)
        processed_text = preprocess_sql_commands_german(processed_text)

    elif language in ['en', 'english']:
        # Convert numeric ratios to text (English)
        def english_ratio_replacer(match):
            num1, num2 = int(match.group(1)), int(match.group(2))
            return f"{number_to_english(num1)}-to-{number_to_english(num2)}"
        processed_text = re.sub(r'(\d+)-to-(\d+)', english_ratio_replacer, processed_text)
        processed_text = preprocess_sql_commands_english(processed_text)

    elif language in ['es', 'spanish']:
        # Convert numeric ratios to text (Spanish)
        def spanish_ratio_replacer(match):
            num1, num2 = int(match.group(1)), int(match.group(2))
            return f"{number_to_spanish(num1)}-a-{number_to_spanish(num2)}"
        processed_text = re.sub(r'(\d+)-a-(\d+)', spanish_ratio_replacer, processed_text)
        processed_text = preprocess_sql_commands_spanish(processed_text)

    if before_lang_specific != processed_text:
        logger.debug(f"Applied language-specific conversions for {language}")

    # 3. General text cleanup
    before_cleanup = processed_text
    processed_text = clean_general_text(processed_text)
    if before_cleanup != processed_text:
        logger.debug("Applied general text cleanup")

    # 4. Apply custom word replacements from dictionary
    before_dict = processed_text
    processed_text = apply_word_replacements(processed_text, language)
    if before_dict != processed_text:
        logger.debug("Applied custom word replacements")

    # Log significant changes
    if processed_text != text:
        logger.info(f"Text preprocessing: '{text}' → '{processed_text}'")

    return processed_text

# === WORD REPLACEMENT FUNCTIONS ===

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
        'german': 'german',
        'en': 'english',
        'english': 'english',
        'es': 'spanish',
        'spanish': 'spanish'
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
        # Handle words with punctuation (like "bzw.") differently from regular words
        if original_word.endswith('.'):
            # For words ending with punctuation, use lookbehind/lookahead assertions
            # This ensures we match whole words even when they end with punctuation
            escaped_word = re.escape(original_word)
            pattern = r'(?<!\w)' + escaped_word + r'(?!\w)'
        else:
            # Use standard word boundaries for regular words
            pattern = r'\b' + re.escape(original_word) + r'\b'

        matches = re.findall(pattern, processed_text, flags=re.IGNORECASE)
        if matches:
            # Preserve capitalization: if the matched word starts with uppercase, capitalize replacement
            def replace_with_case_preservation(match):
                matched_text = match.group(0)
                if matched_text and matched_text[0].isupper():
                    return replacement.capitalize()
                else:
                    return replacement

            processed_text = re.sub(pattern, replace_with_case_preservation, processed_text, flags=re.IGNORECASE)
            replacements_made.append(f"{original_word} -> {replacement} ({len(matches)} matches)")
            logger.debug(f"Replaced '{original_word}' with '{replacement}' ({len(matches)} matches)")

    if replacements_made:
        logger.debug(f"Dictionary replacements completed: {', '.join(replacements_made)}")
    else:
        logger.debug("No dictionary replacements applied")

    return processed_text

# === NUMBER CONVERSION FUNCTIONS ===

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

# === SQL COMMAND PREPROCESSING FUNCTIONS ===

def preprocess_sql_commands_german(text):
    """Convert SQL syntax to German pronunciation."""
    # Replace dots in table.column references with 'Punkt'
    text = re.sub(r'(\w+)\.(\w+)', r'\1 Punkt \2', text)

    # Replace equals signs with 'gleich'
    text = re.sub(r'\s*=\s*', ' gleich ', text)

    # Replace common SQL keywords with German pronunciation
    sql_replacements = {
        r'\bSELECT\b': 'Select', r'\bFROM\b': 'From', r'\bWHERE\b': 'Where',
        r'\bJOIN\b': 'Join', r'\bON\b': 'On', r'\bINNER\b': 'Inner',
        r'\bLEFT\b': 'Left', r'\bRIGHT\b': 'Right', r'\bOUTER\b': 'Outer',
        r'\bORDER\b': 'Order', r'\bBY\b': 'By', r'\bGROUP\b': 'Group',
        r'\bHAVING\b': 'Having', r'\bINSERT\b': 'Insert',
        r'\bUPDATE\b': 'Update', r'\bDELETE\b': 'Delete',
    }
    for pattern, replacement in sql_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def preprocess_sql_commands_english(text):
    """Convert SQL syntax to English pronunciation."""
    # Replace dots in table.column references with 'dot'
    text = re.sub(r'(\w+)\.(\w+)', r'\1 dot \2', text)

    # Replace equals signs with 'equals'
    text = re.sub(r'\s*=\s*', ' equals ', text)

    # Replace common SQL keywords with English pronunciation
    sql_replacements = {
        r'\bSELECT\b': 'select', r'\bFROM\b': 'from', r'\bWHERE\b': 'where',
        r'\bJOIN\b': 'join', r'\bON\b': 'on', r'\bINNER\b': 'inner',
        r'\bLEFT\b': 'left', r'\bRIGHT\b': 'right', r'\bOUTER\b': 'outer',
        r'\bORDER\b': 'order', r'\bBY\b': 'by', r'\bGROUP\b': 'group',
        r'\bHAVING\b': 'having', r'\bINSERT\b': 'insert',
        r'\bUPDATE\b': 'update', r'\bDELETE\b': 'delete',
    }
    for pattern, replacement in sql_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def preprocess_sql_commands_spanish(text):
    """Convert SQL syntax to Spanish pronunciation."""
    # Replace dots in table.column references with 'punto'
    text = re.sub(r'(\w+)\.(\w+)', r'\1 punto \2', text)

    # Replace equals signs with 'igual'
    text = re.sub(r'\s*=\s*', ' igual ', text)

    # Replace common SQL keywords with Spanish pronunciation
    sql_replacements = {
        r'\bSELECT\b': 'select', r'\bFROM\b': 'from', r'\bWHERE\b': 'where',
        r'\bJOIN\b': 'join', r'\bON\b': 'on', r'\bINNER\b': 'inner',
        r'\bLEFT\b': 'left', r'\bRIGHT\b': 'right', r'\bOUTER\b': 'outer',
        r'\bORDER\b': 'order', r'\bBY\b': 'by', r'\bGROUP\b': 'group',
        r'\bHAVING\b': 'having', r'\bINSERT\b': 'insert',
        r'\bUPDATE\b': 'update', r'\bDELETE\b': 'delete',
    }
    for pattern, replacement in sql_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

# === GENERAL TEXT CLEANUP FUNCTIONS ===

def clean_general_text(text):
    """Apply general text cleaning for better TTS."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove excessive punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # Ensure single space after punctuation
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    # Final strip
    return text.strip()

# === UNIFIED CHUNKING INTERFACE ===

def split_text_into_chunks(text, max_chunk_length=200, prefer_end_punct=True,
                          soft_max_ratio=0.85, max_sentences_per_chunk=2,
                          soft_allowance=40, soft_allow_ratio=0.2):
    """
    Split text into chunks using the advanced character-based chunking algorithm.

    This is a convenience wrapper around the advanced chunking function from chunking.py
    that provides a unified interface for text chunking in the preprocessing module.

    Args:
        text (str): Text to split into chunks
        max_chunk_length (int): Maximum characters per chunk (nominal hard cap)
        prefer_end_punct (bool): Prefer to end chunks at punctuation
        soft_max_ratio (float): Start preferring to end around this ratio of max_chunk_length
        max_sentences_per_chunk (int): Maximum sentences per chunk
        soft_allowance (int): Allow up to +N chars to finish sentence
        soft_allow_ratio (float): Or up to +X% of max_chunk_length

    Returns:
        List[str]: List of text chunks
    """
    return split_text_into_chunks_chars(
        text=text,
        max_chars=max_chunk_length,
        prefer_end_punct=prefer_end_punct,
        soft_max_ratio=soft_max_ratio,
        max_sentences_per_chunk=max_sentences_per_chunk,
        soft_allowance=soft_allowance,
        soft_allow_ratio=soft_allow_ratio
    )
