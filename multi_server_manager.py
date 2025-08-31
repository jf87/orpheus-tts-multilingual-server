#!/usr/bin/env python3
"""
Multi-server manager for running multiple language TTS servers simultaneously.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import argparse
from typing import Dict, List, Optional
import requests
import json

from models_config import get_enabled_models, print_config_summary, validate_ports

class TTSServerManager:
    """Manager for multiple TTS server processes."""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.enabled_models = get_enabled_models()
        self.running = True
        
    def start_server(self, language: str, config: Dict) -> bool:
        """Start a TTS server for a specific language."""
        try:
            print(f"🚀 Starting {config['display_name']} server on port {config['port']}...")
            
            cmd = [
                sys.executable, "configurable_tts_server.py",
                "--language", language
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes[language] = process
            
            # Start output monitoring thread
            threading.Thread(
                target=self._monitor_process_output,
                args=(language, process),
                daemon=True
            ).start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {language} server: {e}")
            return False
    
    def _monitor_process_output(self, language: str, process: subprocess.Popen):
        """Monitor and display output from a server process."""
        while self.running and process.poll() is None:
            try:
                line = process.stdout.readline()
                if line:
                    print(f"[{language.upper()}] {line.strip()}")
            except Exception as e:
                print(f"[{language.upper()}] Error reading output: {e}")
                break
    
    def stop_server(self, language: str) -> bool:
        """Stop a specific TTS server."""
        if language not in self.processes:
            print(f"❌ No server running for {language}")
            return False
        
        try:
            process = self.processes[language]
            print(f"⏹️  Stopping {language} server...")
            
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"⚠️  Force killing {language} server...")
                process.kill()
                process.wait()
            
            del self.processes[language]
            print(f"✅ {language} server stopped")
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop {language} server: {e}")
            return False
    
    def start_all_servers(self) -> bool:
        """Start all enabled TTS servers."""
        if not self.enabled_models:
            print("❌ No models are enabled!")
            return False
        
        if not validate_ports():
            print("❌ Port validation failed! Check for duplicate ports.")
            return False
        
        print(f"🚀 Starting {len(self.enabled_models)} TTS servers...")
        
        success_count = 0
        for language, config in self.enabled_models.items():
            if self.start_server(language, config):
                success_count += 1
                time.sleep(2)  # Stagger server startup
        
        print(f"✅ Started {success_count}/{len(self.enabled_models)} servers")
        return success_count > 0
    
    def stop_all_servers(self):
        """Stop all running TTS servers."""
        print("⏹️  Stopping all servers...")
        
        for language in list(self.processes.keys()):
            self.stop_server(language)
        
        self.running = False
        print("✅ All servers stopped")
    
    def check_server_health(self, language: str, config: Dict) -> bool:
        """Check if a server is healthy."""
        try:
            url = f"http://localhost:{config['port']}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_status(self) -> Dict:
        """Get status of all servers."""
        status = {}
        
        for language, config in self.enabled_models.items():
            is_running = language in self.processes and self.processes[language].poll() is None
            is_healthy = self.check_server_health(language, config) if is_running else False
            
            status[language] = {
                "running": is_running,
                "healthy": is_healthy,
                "port": config["port"],
                "display_name": config["display_name"]
            }
        
        return status
    
    def print_status(self):
        """Print current status of all servers."""
        status = self.get_status()
        
        print("\n📊 Server Status")
        print("=" * 50)
        
        for language, info in status.items():
            status_icon = "🟢" if info["healthy"] else "🔴" if info["running"] else "⚫"
            status_text = "Healthy" if info["healthy"] else "Running (Unhealthy)" if info["running"] else "Stopped"
            
            print(f"{status_icon} {info['display_name']} ({language.upper()})")
            print(f"   Port: {info['port']}")
            print(f"   Status: {status_text}")
            if info["healthy"]:
                print(f"   API: http://localhost:{info['port']}/v1/audio/speech")
            print()
    
    def run_interactive(self):
        """Run interactive management mode."""
        print("🎛️  Interactive TTS Server Manager")
        print("Commands: start, stop, status, restart, test, quit")
        
        while self.running:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit" or command == "q":
                    break
                elif command == "start":
                    self.start_all_servers()
                elif command == "stop":
                    self.stop_all_servers()
                elif command == "status":
                    self.print_status()
                elif command == "restart":
                    self.stop_all_servers()
                    time.sleep(2)
                    self.start_all_servers()
                elif command == "test":
                    self.test_all_servers()
                elif command == "help":
                    print("Commands: start, stop, status, restart, test, quit")
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                break
        
        self.stop_all_servers()
    
    def test_all_servers(self):
        """Test all running servers with sample text."""
        print("🧪 Testing all servers...")
        
        for language, config in self.enabled_models.items():
            if not self.check_server_health(language, config):
                print(f"❌ {language} server is not healthy, skipping test")
                continue
            
            try:
                url = f"http://localhost:{config['port']}/v1/audio/speech"
                payload = {
                    "model": "orpheus",
                    "input": config["sample_text"],
                    "voice": config["default_voice"],
                    "response_format": "wav"
                }
                
                print(f"🧪 Testing {language} server...")
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code == 200:
                    print(f"✅ {language} server test passed ({len(response.content)} bytes)")
                else:
                    print(f"❌ {language} server test failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {language} server test failed: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Multi-Language TTS Server Manager")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--start", action="store_true", 
                       help="Start all enabled servers")
    parser.add_argument("--status", action="store_true", 
                       help="Show server status")
    parser.add_argument("--test", action="store_true", 
                       help="Test all servers")
    
    args = parser.parse_args()
    
    # Show configuration
    print_config_summary()
    
    manager = TTSServerManager()
    
    # Set up signal handler for cleanup
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down...")
        manager.stop_all_servers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.interactive:
            if args.start:
                manager.start_all_servers()
            manager.run_interactive()
        elif args.start:
            manager.start_all_servers()
            print("\n✅ Servers started. Press Ctrl+C to stop.")
            # Keep running until interrupted
            while manager.running:
                time.sleep(1)
        elif args.status:
            manager.print_status()
        elif args.test:
            manager.test_all_servers()
        else:
            print("No action specified. Use --help for options.")
            
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all_servers()

if __name__ == "__main__":
    main()