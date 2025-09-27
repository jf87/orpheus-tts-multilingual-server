#!/usr/bin/env python3
"""
Audio post-processing module for TTS output using SoX.
Adapted from orpheus_audio_effects.py for server integration.
"""

import os
import tempfile
import numpy as np
import subprocess
import soundfile as sf
import platform
import logging

logger = logging.getLogger(__name__)

class TTSAudioPostProcessor:
    """Audio post-processing using SoX for TTS output enhancement."""
    
    def __init__(self, sox_path="sox"):
        self.sox_path = sox_path
        self.sox_executable = None
        self._find_sox_executable()
    
    def _find_sox_executable(self):
        """Find the SoX executable on the system."""
        # First, check if the provided path exists
        if os.path.exists(self.sox_path) and os.access(self.sox_path, os.X_OK):
            self.sox_executable = self.sox_path
            logger.info(f"Found SoX executable at: {self.sox_path}")
            return
            
        # For Linux/macOS, check if sox is in the PATH
        if platform.system() != "Windows":
            try:
                sox_path = subprocess.check_output(["which", "sox"], text=True).strip()
                if sox_path:
                    self.sox_executable = sox_path
                    logger.info(f"Found SoX in PATH: {sox_path}")
                    return
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
            # Check common Unix locations
            unix_paths = ["/usr/bin/sox", "/usr/local/bin/sox", "/bin/sox"]
            for path in unix_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    self.sox_executable = path
                    logger.info(f"Found SoX at: {path}")
                    return
                    
            # If sox is specified as just "sox", try it anyway (might be in PATH)
            if self.sox_path == "sox":
                self.sox_executable = "sox"
                return
        else:
            # Windows paths to check
            windows_paths = [
                "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe",
                "C:\\Program Files\\sox-14-4-2\\sox.exe",
                "C:\\Program Files (x86)\\sox\\sox.exe",
                "C:\\Program Files\\sox\\sox.exe"
            ]
            
            for path in windows_paths:
                if os.path.exists(path):
                    self.sox_executable = path
                    logger.info(f"Found SoX at: {path}")
                    return
                    
        logger.warning("SoX executable not found. Audio post-processing will be disabled.")
        logger.info("Install SoX:")
        logger.info("- Linux: sudo apt-get install sox")
        logger.info("- macOS: brew install sox") 
        logger.info("- Windows: Download from https://sourceforge.net/projects/sox/")
    
    def is_available(self):
        """Check if SoX is available for audio processing."""
        return self.sox_executable is not None
    
    def get_default_effects(self):
        """Get default effect parameters."""
        return {
            'pitch_shift': 0.0,      # semitones (-12.0 to 12.0)
            'speed_factor': 1.0,     # speed multiplier (0.5 to 2.0)
            'gain_db': 0.0,         # gain in decibels (-20.0 to 20.0)
            'normalize_audio': False, # normalize audio levels
            'use_limiter': True,     # use limiter with positive gain
            'add_reverb': False,     # add reverb effect
            'reverb_amount': 50,     # reverb amount (0-100)
            'reverb_room_scale': 50, # room size (0-100)
            'add_echo': False,       # add echo effect
            'echo_delay': 0.5,       # echo delay in seconds (0.1-2.0)
            'echo_decay': 0.5        # echo decay factor (0.1-0.9)
        }
    
    def process_audio(self, audio_data, sample_rate=24000, **effects):
        """
        Apply audio effects to numpy audio data using SoX.
        
        Args:
            audio_data (np.ndarray): Input audio data
            sample_rate (int): Sample rate of the audio
            **effects: Audio effect parameters
        
        Returns:
            np.ndarray: Processed audio data
        """
        if not self.is_available():
            logger.debug("SoX not available, returning original audio")
            return audio_data
        
        # Merge with defaults
        default_effects = self.get_default_effects()
        effect_params = {**default_effects, **effects}
        
        # Extract effect parameters
        pitch_shift = effect_params['pitch_shift']
        speed_factor = effect_params['speed_factor'] 
        gain_db = effect_params['gain_db']
        normalize_audio = effect_params['normalize_audio']
        add_reverb = effect_params['add_reverb']
        reverb_amount = effect_params['reverb_amount']
        reverb_room_scale = effect_params['reverb_room_scale']
        add_echo = effect_params['add_echo']
        echo_delay = effect_params['echo_delay']
        echo_decay = effect_params['echo_decay']
        use_limiter = effect_params['use_limiter']
        
        # Check if any effects need to be applied
        if (abs(pitch_shift) < 0.01 and 
            abs(speed_factor - 1.0) < 0.01 and 
            abs(gain_db) < 0.01 and
            not normalize_audio and
            not add_reverb and 
            not add_echo):
            logger.debug("No audio effects to apply, returning original")
            return audio_data
        
        logger.debug(f"Applying audio effects: pitch={pitch_shift}, speed={speed_factor}, gain={gain_db}")
        if add_reverb:
            logger.debug(f"Reverb: amount={reverb_amount}, room_scale={reverb_room_scale}")
        if add_echo:
            logger.debug(f"Echo: delay={echo_delay}s, decay={echo_decay}")
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
                input_path = input_file.name
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
                output_path = output_file.name
            
            try:
                # Write input audio
                sf.write(input_path, audio_data, sample_rate)
                
                # Verify input file was written
                if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                    logger.error("Failed to write input audio file")
                    return audio_data
                
                # Build SoX command
                sox_cmd = [self.sox_executable, input_path, output_path]
                effects_list = []
                
                # Add effects in proper order
                
                # 1. Normalize (if requested) - must be applied first
                if normalize_audio:
                    effects_list.extend(['gain', '-n'])
                    logger.debug("Added normalize effect")
                    
                # 2. Gain adjustment with optional limiter
                if abs(gain_db) >= 0.01:
                    if gain_db > 0 and use_limiter:
                        effects_list.extend(['gain', '-l', str(gain_db)])
                        logger.debug(f"Added gain effect with limiter: {gain_db} dB")
                    else:
                        effects_list.extend(['gain', str(gain_db)])
                        logger.debug(f"Added gain effect: {gain_db} dB")
                        
                # 3. Pitch shift
                if pitch_shift != 0:
                    pitch_cents = int(pitch_shift * 100)
                    effects_list.extend(['pitch', str(pitch_cents)])
                    logger.debug(f"Added pitch effect: {pitch_cents} cents")
                    
                # 4. Speed/tempo adjustment 
                if speed_factor != 1.0:
                    effects_list.extend(['tempo', '-s', str(speed_factor)])
                    logger.debug(f"Added tempo effect: {speed_factor}x")
                    
                # 5. Reverb
                if add_reverb:
                    # SoX reverb parameters: reverberance HF-damping room-scale stereo-depth pre-delay wet-gain
                    effects_list.extend([
                        'reverb', 
                        str(int(reverb_amount)),    # reverberance (0-100)
                        '50',                       # HF-damping (default)
                        str(int(reverb_room_scale)), # room-scale (0-100)
                        '50',                       # stereo-depth (default)
                        '20',                       # pre-delay (default)
                        '0'                         # wet-gain (default)
                    ])
                    logger.debug(f"Added reverb: {reverb_amount}% reverberance, {reverb_room_scale}% room scale")
                    
                # 6. Echo
                if add_echo:
                    # SoX echo parameters: gain-in gain-out delay decay
                    delay_ms = int(echo_delay * 1000)  # Convert to milliseconds
                    effects_list.extend([
                        'echo',
                        '0.8',              # gain-in
                        '0.9',              # gain-out  
                        str(delay_ms),      # delay in ms
                        str(echo_decay)     # decay
                    ])
                    logger.debug(f"Added echo: {delay_ms}ms delay, {echo_decay} decay")
                
                # Add effects to command
                if effects_list:
                    sox_cmd.extend(effects_list)
                    
                # Execute SoX command
                logger.debug(f"Executing SoX: {' '.join(sox_cmd)}")
                result = subprocess.run(
                    sox_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                # Log SoX output for debugging
                if result.stdout and result.stdout.strip():
                    logger.debug(f"SoX stdout: {result.stdout.strip()}")
                if result.stderr and result.stderr.strip():
                    logger.debug(f"SoX stderr: {result.stderr.strip()}")
                
                if result.returncode != 0:
                    logger.error(f"SoX processing failed (code {result.returncode}): {result.stderr}")
                    return audio_data
                
                # Read processed audio
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    processed_audio, new_sample_rate = sf.read(output_path)
                    logger.debug(f"Audio post-processing completed: {processed_audio.shape} @ {new_sample_rate}Hz")
                    return processed_audio
                else:
                    logger.error("SoX output file is empty or missing")
                    return audio_data
                    
            finally:
                # Clean up temporary files
                for path in [input_path, output_path]:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except Exception as e:
                        logger.debug(f"Failed to clean up temp file {path}: {e}")
                        
        except subprocess.TimeoutExpired:
            logger.error("SoX processing timed out")
            return audio_data
        except Exception as e:
            logger.error(f"Audio post-processing failed: {e}")
            return audio_data
    
    def validate_effects(self, effects):
        """
        Validate effect parameters and return corrected values.
        
        Args:
            effects (dict): Effect parameters to validate
            
        Returns:
            dict: Validated effect parameters
        """
        validated = {}
        
        # Pitch shift validation (-12.0 to 12.0 semitones)
        if 'pitch_shift' in effects:
            pitch = float(effects['pitch_shift'])
            validated['pitch_shift'] = max(-12.0, min(12.0, pitch))
            
        # Speed factor validation (0.5 to 2.0)
        if 'speed_factor' in effects:
            speed = float(effects['speed_factor'])
            validated['speed_factor'] = max(0.5, min(2.0, speed))
            
        # Gain validation (-20.0 to 20.0 dB)
        if 'gain_db' in effects:
            gain = float(effects['gain_db'])
            validated['gain_db'] = max(-20.0, min(20.0, gain))
            
        # Boolean validations
        for bool_param in ['normalize_audio', 'use_limiter', 'add_reverb', 'add_echo']:
            if bool_param in effects:
                validated[bool_param] = bool(effects[bool_param])
                
        # Reverb parameters (0-100)
        for reverb_param in ['reverb_amount', 'reverb_room_scale']:
            if reverb_param in effects:
                value = float(effects[reverb_param])
                validated[reverb_param] = max(0, min(100, value))
                
        # Echo delay validation (0.1 to 2.0 seconds)
        if 'echo_delay' in effects:
            delay = float(effects['echo_delay'])
            validated['echo_delay'] = max(0.1, min(2.0, delay))
            
        # Echo decay validation (0.1 to 0.9)
        if 'echo_decay' in effects:
            decay = float(effects['echo_decay'])
            validated['echo_decay'] = max(0.1, min(0.9, decay))
            
        return validated