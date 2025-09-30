#!/usr/bin/env python3
"""
Cache management module for Orpheus-HF TTS system.

This module provides thread-safe LRU caching functionality for TTS audio chunks
with file persistence capabilities.
"""

import time
import hashlib
import threading
import pickle
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict


class TTSCache:
    """
    Thread-safe LRU cache for TTS audio chunks with file persistence.

    Thread safety: Uses threading.RLock to allow reentrant locking (same thread
    can acquire lock multiple times). This is important because methods like
    get_stats() may be called while other operations hold the lock.

    LRU eviction: Uses OrderedDict to track access order. When cache is full,
    oldest (least recently used) entries are evicted first.

    Persistence: Periodically saves cache to disk (default: every 5 minutes)
    and reloads on startup. Uses pickle for efficient numpy array serialization.
    """

    def __init__(self, max_size=1000, max_age_seconds=3600, cache_dir="./tts_cache"):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "tts_cache.pkl"
        self.stats_file = self.cache_dir / "cache_stats.json"

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)

        # OrderedDict maintains insertion/access order for LRU behavior
        self.cache = OrderedDict()
        # RLock allows same thread to acquire lock multiple times (reentrant)
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
        """
        Get cached audio for text chunk.

        Thread-safe retrieval with LRU tracking. On cache hit, moves entry
        to end of OrderedDict to mark as recently used.
        """
        key = self._generate_key(text, voice, language)

        with self.lock:
            self._cleanup_expired()

            if key in self.cache:
                audio_data, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    # Move to end (most recently used) for LRU tracking
                    self.cache.move_to_end(key)
                    self.stats['hits'] += 1
                    return audio_data.copy()  # Return copy to prevent external modification
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
            evictions = 0
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
                evictions += 1

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