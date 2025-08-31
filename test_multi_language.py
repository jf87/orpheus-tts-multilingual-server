#!/usr/bin/env python3
"""
Test script for multi-language TTS servers.
"""

import requests
import time
import tempfile
import os
from models_config import get_enabled_models

def test_language_server(language: str, config: dict, timeout: int = 30):
    """Test a specific language server."""
    print(f"\n🧪 Testing {config['display_name']} server...")
    print(f"Port: {config['port']}")
    print(f"Sample text: {config['sample_text'][:50]}...")
    
    base_url = f"http://localhost:{config['port']}/v1/audio/speech"
    
    payload = {
        "model": "orpheus",
        "input": config['sample_text'],
        "voice": config['default_voice'],
        "response_format": "wav",
        "speed": 1.0
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        # Test health endpoint first
        health_url = f"http://localhost:{config['port']}/health"
        health_response = requests.get(health_url, timeout=5)
        
        if health_response.status_code != 200:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False, "Health check failed"
        
        print("✅ Health check passed")
        
        # Test TTS endpoint
        print("🎤 Generating speech...")
        response = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        
        # Save audio file
        with tempfile.NamedTemporaryFile(suffix=f"_{language}.wav", delete=False) as temp_file:
            temp_file.write(response.content)
            output_path = temp_file.name
        
        print(f"✅ Success! Audio saved to: {output_path}")
        print(f"Audio size: {len(response.content)} bytes")
        
        return True, output_path
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False, str(e)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, str(e)

def test_all_servers():
    """Test all enabled language servers."""
    enabled_models = get_enabled_models()
    
    if not enabled_models:
        print("❌ No enabled models found!")
        return
    
    print(f"🚀 Testing {len(enabled_models)} language servers...")
    
    results = {}
    for language, config in enabled_models.items():
        success, result = test_language_server(language, config)
        results[language] = {
            "success": success,
            "result": result,
            "config": config
        }
        
        # Wait between tests
        if len(enabled_models) > 1:
            time.sleep(2)
    
    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    successful_tests = 0
    for language, result in results.items():
        config = result['config']
        icon = "✅" if result['success'] else "❌"
        status = "PASSED" if result['success'] else "FAILED"
        
        print(f"{icon} {config['display_name']} ({language.upper()}): {status}")
        if result['success']:
            print(f"   Audio file: {result['result']}")
            successful_tests += 1
        else:
            print(f"   Error: {result['result']}")
        print(f"   Port: {config['port']}")
        print()
    
    print(f"Overall: {successful_tests}/{len(enabled_models)} tests passed")
    
    if successful_tests == len(enabled_models):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check server status and configuration.")

def check_server_availability():
    """Check if servers are running and responsive."""
    enabled_models = get_enabled_models()
    
    print("🔍 Checking server availability...")
    
    for language, config in enabled_models.items():
        try:
            health_url = f"http://localhost:{config['port']}/health"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {config['display_name']} ({language}): {data.get('status', 'unknown')}")
                print(f"   Port: {config['port']}, GPUs: {data.get('gpu_count', 'unknown')}")
            else:
                print(f"❌ {config['display_name']} ({language}): HTTP {response.status_code}")
                
        except requests.exceptions.ConnectoinError:
            print(f"🔴 {config['display_name']} ({language}): Server not running")
        except Exception as e:
            print(f"⚠️  {config['display_name']} ({language}): {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multi-language TTS servers")
    parser.add_argument("--check", "-c", action="store_true", 
                       help="Check server availability only")
    parser.add_argument("--language", "-l", 
                       help="Test specific language only")
    parser.add_argument("--timeout", "-t", type=int, default=60,
                       help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    if args.check:
        check_server_availability()
    elif args.language:
        enabled_models = get_enabled_models()
        if args.language not in enabled_models:
            print(f"❌ Language '{args.language}' not found in enabled models")
            print(f"Available: {', '.join(enabled_models.keys())}")
            return
        
        config = enabled_models[args.language]
        test_language_server(args.language, config, args.timeout)
    else:
        test_all_servers()

if __name__ == "__main__":
    main()