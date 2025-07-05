#!/usr/bin/env python3
"""
Zonos TTS Worker Script

This script runs in a separate conda environment (tts_zonos) to avoid dependency conflicts.
It's called as a subprocess from the main application.
Supports both single-shot and persistent modes for better performance.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Zonos TTS Worker')
    parser.add_argument('--model', help='Model name (e.g., Zyphra/Zonos-v0.1-transformer)')
    parser.add_argument('--ref_audio', help='Path to reference audio file')
    parser.add_argument('--ref_text', help='Reference text')
    parser.add_argument('--gen_text', help='Text to generate')
    parser.add_argument('--output_file', help='Output audio file path')
    parser.add_argument('--config', default='{}', help='JSON configuration string')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--persistent', action='store_true', help='Run in persistent mode for faster subsequent calls')
    
    args = parser.parse_args()
    
    if args.persistent:
        # Run in persistent mode - keep model loaded
        run_persistent_mode(args)
    else:
        # Single-shot mode - current behavior
        if not all([args.model, args.ref_audio, args.ref_text, args.gen_text, args.output_file]):
            parser.error("In single-shot mode, --model, --ref_audio, --ref_text, --gen_text, and --output_file are required")
        run_single_shot(args)

def run_persistent_mode(args):
    """
    Run in persistent mode - keep model loaded and process multiple requests.
    Reads JSON requests from stdin and outputs JSON responses to stdout.
    """
    print("🚀 Starting Zonos worker in persistent mode...")
    
    # Global model cache for persistent mode
    model_cache = {}
    
    # Signal ready
    print("READY")
    sys.stdout.flush()
    
    try:
        while True:
            # Read request from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if line == "EXIT":
                break
                
            try:
                request = json.loads(line)
                response = process_request(request, model_cache)
                print(json.dumps(response))
                sys.stdout.flush()
            except Exception as e:
                error_response = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("🛑 Persistent worker shutting down...")
    except Exception as e:
        print(f"💥 Persistent worker error: {e}")

def run_single_shot(args):
    """Run in single-shot mode - current behavior."""
    try:
        # Parse config
        config = json.loads(args.config)
        
        request = {
            "model": args.model,
            "ref_audio": args.ref_audio,
            "ref_text": args.ref_text,
            "gen_text": args.gen_text,
            "output_file": args.output_file,
            "config": config,
            "device": args.device
        }
        
        # Process single request
        model_cache = {}
        result = process_request(request, model_cache)
        
        if result["success"]:
            print(f"RESULT: {json.dumps(result)}")
        else:
            print(f"ERROR: {json.dumps(result)}")
            sys.exit(1)
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(f"ERROR: {json.dumps(error_result)}")
        sys.exit(1)

def process_request(request, model_cache):
    """Process a single TTS request, using cached model if available."""
    try:
        # Extract request parameters
        model_name = request["model"]
        ref_audio = request["ref_audio"]
        ref_text = request["ref_text"]
        gen_text = request["gen_text"]
        output_file = request["output_file"]
        config = request.get("config", {})
        device_arg = request.get("device", "auto")
        
        # Import required modules
        import torch
        import torchaudio
        
        # Disable torch.compile for MPS devices early
        if device_arg == 'mps' or (device_arg == 'auto' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print(f"🍎 Disabling torch.compile for MPS compatibility")
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            os.environ['TORCHDYNAMO_DISABLE'] = '1'
            # Disable Metal API validation to prevent crashes
            os.environ['METAL_DEVICE_WRAPPER_TYPE'] = ''
            print(f"🍎 Disabled Metal API validation to prevent buffer validation crashes")
            # Disable inductor compilation
            torch._dynamo.config.disable = True
            torch._dynamo.config.suppress_errors = True
        
        # Ensure espeak-ng is available
        setup_espeak()
        
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.utils import DEFAULT_DEVICE
        
        print(f"🎤 Processing Zonos request:")
        print(f"   • Model: {model_name}")
        print(f"   • Reference Audio: {ref_audio}")
        print(f"   • Reference Text: {ref_text[:100]}{'...' if len(ref_text) > 100 else ''}")
        print(f"   • Generation Text: {gen_text[:100]}{'...' if len(gen_text) > 100 else ''}")
        print(f"   • Output File: {output_file}")
        
        # Validate input file exists
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
        
        # Determine device
        device = get_device(device_arg)
        print(f"🎯 Using device: {device}")
        
        # Get or load model from cache
        model_key = f"{model_name}_{device}"
        start_time = time.perf_counter()
        
        if model_key in model_cache:
            print(f"✓ Using cached Zonos model: {model_name}")
            model = model_cache[model_key]
            load_time = 0  # No loading time for cached model
        else:
            print(f"📥 Loading Zonos model: {model_name}")
            load_start = time.perf_counter()
            
            model = Zonos.from_pretrained(model_name, device=device)
            
            # Apply device-specific optimizations
            if str(device) == 'mps':
                print(f"🍎 Disabling torch.compile for MPS device")
                os.environ['TORCH_COMPILE_DISABLE'] = '1'
                if hasattr(model, 'autoencoder') and hasattr(model.autoencoder, 'compile'):
                    model.autoencoder.compile = lambda: model.autoencoder  # No-op
            
            # Cache the model for future requests
            model_cache[model_key] = model
            load_time = time.perf_counter() - load_start
            print(f"✓ Model loaded and cached in {load_time:.2f}s")
        
        # Process audio and generate
        result = generate_audio_with_model(
            model, device, ref_audio, ref_text, gen_text, output_file, config
        )
        
        total_time = time.perf_counter() - start_time
        result["total_time"] = total_time
        result["load_time"] = load_time
        result["cached"] = model_key in model_cache
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def setup_espeak():
    """Ensure espeak-ng is available in PATH."""
    import subprocess
    import shutil
    
    espeak_path = shutil.which('espeak-ng')
    if not espeak_path:
        common_paths = [
            '/opt/homebrew/bin/espeak-ng',
            '/usr/local/bin/espeak-ng',
            '/usr/bin/espeak-ng',
        ]
        for path in common_paths:
            if os.path.exists(path):
                bin_dir = os.path.dirname(path)
                if bin_dir not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = f"{bin_dir}:{os.environ.get('PATH', '')}"
                print(f"✓ Found espeak-ng at: {path}")
                break
        else:
            raise RuntimeError("espeak-ng not found. Please install it with: brew install espeak-ng")
    else:
        print(f"✓ espeak-ng found in PATH: {espeak_path}")

def get_device(device_arg):
    """Determine the appropriate device."""
    import torch
    
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
    
    # Check for Metal API Validation which can cause crashes
    if str(device) == 'mps' and os.environ.get('METAL_DEVICE_WRAPPER_TYPE') == 'MTLDebugDevice':
        print(f"⚠️ Metal API Validation detected, forcing CPU mode to prevent crashes")
        device = torch.device("cpu")
    
    return device

def generate_audio_with_model(model, device, ref_audio, ref_text, gen_text, output_file, config):
    """Generate audio using the provided model."""
    import torch
    import torchaudio
    from zonos.conditioning import make_cond_dict
    
    generation_start = time.perf_counter()
    
    # Load reference audio
    print("🔊 Loading reference audio...")
    wav, sampling_rate = torchaudio.load(ref_audio)
    
    # Ensure audio is on the correct device
    if str(device) != 'cpu':
        wav = wav.to(device)
        print(f"🎯 Audio moved to device: {device}")
    
    # Create speaker embedding
    print("🎭 Creating speaker embedding...")
    try:
        speaker = model.make_speaker_embedding(wav, sampling_rate)
        if str(device) != 'cpu':
            speaker = speaker.to(device)
            print(f"🎯 Speaker embedding on device: {device}")
    except RuntimeError as e:
        if "mps" in str(e).lower() or "cpu" in str(e).lower() or "device" in str(e).lower():
            print(f"⚠️ Device issue with speaker embedding, falling back to CPU: {e}")
            wav_cpu = wav.cpu()
            model_cpu = model.to('cpu')
            speaker = model_cpu.make_speaker_embedding(wav_cpu, sampling_rate)
            model = model.to(device)
            if str(device) != 'cpu':
                speaker = speaker.to(device)
                print(f"🎯 Speaker embedding moved back to device: {device}")
        else:
            raise e
    
    # Prepare conditioning
    print("🎯 Preparing conditioning...")
    
    # Language mapping and parameter extraction (same as before)
    language_mapping = {
        'en': 'en-us', 'es': 'es', 'fr': 'fr-fr', 'de': 'de', 'it': 'it',
        'pt': 'pt', 'pl': 'pl', 'tr': 'tr', 'ru': 'ru', 'nl': 'nl',
        'cs': 'cs', 'ar': 'ar', 'zh-cn': 'cmn', 'zh': 'cmn',
        'ja': 'ja', 'ko': 'ko', 'hi': 'hi', 'hu': 'hu'
    }
    
    if 'zonos_settings' in config and 'language' in config['zonos_settings']:
        raw_language = config['zonos_settings']['language']
    else:
        raw_language = config.get('language', 'en')
    
    zonos_language = language_mapping.get(raw_language.lower(), 'en-us')
    print(f"🌍 Language mapping: {raw_language} -> {zonos_language}")
    
    # Handle emotion parameters
    emotion_vector = None
    if 'zonos_settings' in config:
        zonos_config = config['zonos_settings']
        emotion_vector = [
            zonos_config.get('e1', 0.5), zonos_config.get('e2', 0.5),
            zonos_config.get('e3', 0.5), zonos_config.get('e4', 0.5),
            zonos_config.get('e5', 0.5), zonos_config.get('e6', 0.5),
            zonos_config.get('e7', 0.5), zonos_config.get('e8', 0.5),
        ]
        print(f"🎭 Emotion vector: {emotion_vector}")
    
    if emotion_vector is None:
        emotion_vector = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
        print(f"🎭 Using default emotion vector: {emotion_vector}")
    
    # Get additional parameters
    fmax, pitch_std, speaking_rate = 22050.0, 100.0, 20
    if 'zonos_settings' in config:
        zonos_config = config['zonos_settings']
        fmax = zonos_config.get('frequency_max', 22050.0)
        pitch_std = zonos_config.get('pitch_standard_deviation', 100.0)
        speaking_rate = zonos_config.get('speaking_rate', 20)
    
    print(f"🔧 Final parameters: fmax={fmax}, pitch_std={pitch_std}, speaking_rate={speaking_rate}")
    
    # Create conditioning dictionary
    print(f"🎯 Creating conditioning dictionary...")
    cond_dict = make_cond_dict(
        text=gen_text, speaker=speaker, language=zonos_language,
        emotion=emotion_vector, fmax=fmax, pitch_std=pitch_std, speaking_rate=speaking_rate
    )
    
    # Set seed if provided
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        print(f"🎲 Set seed: {config['seed']}")
    elif 'zonos_settings' in config and 'seed' in config['zonos_settings']:
        torch.manual_seed(config['zonos_settings']['seed'])
        print(f"🎲 Set seed from zonos_settings: {config['zonos_settings']['seed']}")
    
    # Move tensors to device
    print(f"🔍 Checking cond_dict tensor devices:")
    for key, value in cond_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"   • {key}: {value.device}")
            if str(value.device) != str(device):
                print(f"   • Moving {key} from {value.device} to {device}")
                cond_dict[key] = value.to(device)
    
    # Prepare conditioning
    print(f"🎯 Preparing conditioning with model...")
    try:
        conditioning = model.prepare_conditioning(cond_dict)
    except RuntimeError as e:
        if "cpu" in str(e).lower() and "mps" in str(e).lower():
            print(f"⚠️ Device mismatch in prepare_conditioning, trying with CPU tensors: {e}")
            cond_dict_cpu = {}
            for key, value in cond_dict.items():
                if isinstance(value, torch.Tensor):
                    cond_dict_cpu[key] = value.cpu()
                else:
                    cond_dict_cpu[key] = value
            
            model_cpu = model.to('cpu')
            conditioning = model_cpu.prepare_conditioning(cond_dict_cpu)
            model = model.to(device)
            
            if isinstance(conditioning, torch.Tensor):
                conditioning = conditioning.to(device)
            elif isinstance(conditioning, dict):
                for key, value in conditioning.items():
                    if isinstance(value, torch.Tensor):
                        conditioning[key] = value.to(device)
            
            print(f"✓ Successfully prepared conditioning using CPU workaround")
        else:
            raise e
    
    # Ensure conditioning is on correct device
    print(f"🔍 Checking conditioning tensor devices after preparation:")
    if isinstance(conditioning, torch.Tensor):
        print(f"   • conditioning tensor: {conditioning.device}")
        if str(conditioning.device) != str(device):
            print(f"   • Moving conditioning from {conditioning.device} to {device}")
            conditioning = conditioning.to(device)
    elif isinstance(conditioning, dict):
        for key, value in conditioning.items():
            if isinstance(value, torch.Tensor):
                print(f"   • {key}: {value.device}")
                if str(value.device) != str(device):
                    print(f"   • Moving {key} from {value.device} to {device}")
                    conditioning[key] = value.to(device)
    
    print(f"✓ All conditioning tensors confirmed on {device}")
    
    # Generate audio
    print("🎵 Generating audio...")
    try:
        with torch.inference_mode():
            codes = model.generate(conditioning)
            wavs = model.autoencoder.decode(codes).cpu()
    except RuntimeError as e:
        if "mps" in str(e).lower() and "cpu" in str(e).lower():
            print(f"⚠️ MPS device issue, falling back to CPU: {e}")
            model = model.to('cpu')
            
            if isinstance(conditioning, torch.Tensor):
                conditioning = conditioning.cpu()
            elif isinstance(conditioning, dict):
                for key, value in conditioning.items():
                    if isinstance(value, torch.Tensor):
                        conditioning[key] = value.cpu()
            
            with torch.inference_mode():
                codes = model.generate(conditioning)
                wavs = model.autoencoder.decode(codes).cpu()
            print("✓ Successfully generated audio using CPU fallback")
        else:
            raise e
    except Exception as e:
        if "Metal API Validation" in str(e) or "validateComputeFunctionArguments" in str(e):
            print(f"⚠️ Metal API validation error, falling back to CPU: {e}")
            model = model.to('cpu')
            
            if isinstance(conditioning, torch.Tensor):
                conditioning = conditioning.cpu()
            elif isinstance(conditioning, dict):
                for key, value in conditioning.items():
                    if isinstance(value, torch.Tensor):
                        conditioning[key] = value.cpu()
            
            with torch.inference_mode():
                codes = model.generate(conditioning)
                wavs = model.autoencoder.decode(codes).cpu()
            print("✓ Successfully generated audio using CPU fallback for Metal API issue")
        else:
            raise e
    
    generation_time = time.perf_counter() - generation_start
    print(f"✓ Audio generated in {generation_time:.2f}s")
    
    # Save output
    print(f"💾 Saving to: {output_file}")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(output_file, wavs[0], model.autoencoder.sampling_rate)
    
    print(f"✓ Zonos generation completed successfully")
    
    return {
        "success": True,
        "output_file": output_file,
        "generation_time": generation_time,
        "sample_rate": model.autoencoder.sampling_rate
    }

if __name__ == "__main__":
    main() 