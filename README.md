# ComfyUI-Qwen3-TTS

A ComfyUI custom node package for **Qwen3-TTS**, providing high-quality Text-to-Speech (TTS) with Custom Voice and Voice Cloning capabilities.

## Features

- **Qwen3-TTS Custom Voice**: Generate speech from text using high-quality preset speakers and natural language instructions for style/emotion.
- **Qwen3-TTS Voice Clone**: Clone any voice from a reference audio clip and transcribe text to speak in that voice.
- **Robust Model Loading**: Automatically downloads and manages models in `ComfyUI/models/tts`.
- **Zero Configuration**: No separate model loader node required. Each node manages its own model loading and caching.

## Installation

### 1. Install Dependencies

Ensure you have the required Python packages installed in your ComfyUI environment:

```bash
pip install qwen-tts torch soundfile accelerate transformers==4.57.3
```

> ⚠️ Dependency Note: The upstream qwen-tts package requires transformers==4.57.3. This may downgrade your existing transformers version. If other custom nodes require a newer version, consider using a separate Python environment.

### 2. Install Custom Node

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-Qwen3-TTS
```

Alternatively, you can manually copy the `ComfyUI-Qwen3-TTS` folder into your `custom_nodes` directory.

## Nodes

### 📢 Qwen3-TTS Custom Voice
Generates speech using optimized preset speakers.

- **text**: The text you want to synthesize.
- **language**: Output language (Japanese, English, Chinese, or Auto).
- **speaker**: Choose from supported speakers (e.g., `ono_anna`, `ryan`, `vivian`).
- **instruct**: Optional natural language instruction (e.g., "Speak in a calm and friendly tone.").

### 👤 Qwen3-TTS Voice Clone
Clones a voice from a reference audio input.

- **ref_audio**: The reference audio clip (from a `Load Audio` node).
- **ref_text**: The transcription of the reference audio clip (Required for better quality).
- **text**: The target text you want the cloned voice to speak.
- **language**: Output language.

## Model Cache
Models are automatically downloaded from Hugging Face and saved to `ComfyUI/models/tts`. 
- **Custom Voice**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **Voice Clone**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

## Acknowledgments

- **Qwen Team**: For the amazing [Qwen3-TTS](https://huggingface.co/Qwen) models.
- **kun432**: Implementation details inspired by this [Zenn scrap](https://zenn.dev/kun432/scraps/102b890a0956a1).

## License
This project is licensed under the MIT License.
