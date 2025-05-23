from importlib.resources import files
from f5_tts.api import F5TTS
import base64
import io

# Preload the F5TTS model when the module is imported
print("Preloading F5TTS model...")
_f5tts_model = F5TTS()
print("F5TTS model preloaded successfully!")


def generate_cloned_audio(ref_file: str, ref_text: str, gen_text: str, output_file: str):
    # Use the preloaded model instead of creating a new instance
    wav, sr, spec = _f5tts_model.infer(
        ref_file=str(files("f5_tts").joinpath(ref_file)),
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=str(files("f5_tts").joinpath(output_file)),
        seed=None,
    )

    # Convert audio data to base64
    audio_buffer = io.BytesIO()
    import soundfile as sf
    sf.write(audio_buffer, wav, sr, format='WAV')
    audio_buffer.seek(0)
    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

    return audio_base64
