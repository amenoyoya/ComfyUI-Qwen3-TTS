import os
import torch
import numpy as np
import tempfile
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from huggingface_hub import snapshot_download

# Get model storage directory
try:
    import folder_paths
    BASE_MODEL_DIR = os.path.join(folder_paths.models_dir, "tts")
except ImportError:
    # Fallback if executed outside ComfyUI
    BASE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "tts"))

# Global model cache
LOADED_MODELS = {}

def get_qwen3_model(model_type):
    repo_id = f"Qwen/Qwen3-TTS-12Hz-1.7B-{model_type}"
    local_dir = os.path.join(BASE_MODEL_DIR, f"Qwen3-TTS-12Hz-1.7B-{model_type}")
    
    if repo_id not in LOADED_MODELS:
        print(f"Downloading/Verifying Qwen3-TTS model: {repo_id} to {local_dir}")
        # Use snapshot_download to explicitly download to local directory
        # Disable symlinks to avoid Windows-specific issues
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        
        print(f"Loading Qwen3-TTS model from local path: {downloaded_path}")
        model = Qwen3TTSModel.from_pretrained(
            downloaded_path,
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        LOADED_MODELS[repo_id] = model
        
    return LOADED_MODELS[repo_id]

class Qwen3TTSCustomVoice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, how are you today?"}),
                "language": (["Japanese", "English", "Chinese", "Auto"], {"default": "English"}),
                "speaker": (["ono_anna", "aiden", "dylan", "eric", "ryan", "serena", "sohee", "uncle_fu", "vivian"], {"default": "ono_anna"}),
                "instruct": ("STRING", {"multiline": False, "default": "Speak in a calm and friendly tone."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, text, language, speaker, instruct):
        model = get_qwen3_model("CustomVoice")
        lang = None if language == "Auto" else language
        
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=lang,
            speaker=speaker,
            instruct=instruct,
        )
        
        waveform = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": sr},)

class Qwen3TTSVoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
                "text": ("STRING", {"multiline": True, "default": "I am speaking with this voice."}),
                "language": (["Japanese", "English", "Chinese", "Auto"], {"default": "English"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, ref_audio, ref_text, text, language):
        model = get_qwen3_model("Base")
        
        waveform = ref_audio["waveform"]
        sr_in = ref_audio["sample_rate"]
        
        # Qwen3-TTS might take a file path or numpy array, 
        # but creating a temporary file is more robust for its internal loaders.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_data = waveform.squeeze(0).transpose(0, 1).numpy()
            sf.write(tmp_path, audio_data, sr_in)
        
        try:
            lang = None if language == "Auto" else language
            wavs, sr_out = model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=tmp_path,
                ref_text=ref_text,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        waveform_out = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
        return ({"waveform": waveform_out, "sample_rate": sr_out},)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSCustomVoice": "Qwen3-TTS Custom Voice",
    "Qwen3TTSVoiceClone": "Qwen3-TTS Voice Clone",
}
