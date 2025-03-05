# Phi-4 Multimodal Model Demo

This project demonstrates the capabilities of Microsoft's Phi-4 multimodal model for processing both images and audio inputs.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for optimal performance)
  - CUDA 11.5+ (CUDA 11.6+ required for flash attention)
- Git

## Installation

Follow these steps to set up the project environment:

### 1. Clone the repository (if applicable)

```bash
git clone <repository-url>
cd microsoft-model
```

### 2. Create and activate a virtual environment

#### On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

For better installation of packages with complex dependencies:

```bash
# Install basic build dependencies first
pip install --upgrade pip setuptools wheel packaging numpy torch

# Then install the rest of the requirements
pip install -r requirements.txt
```

## CUDA Version Note

This project is configured to use the "eager" attention implementation due to compatibility with CUDA 11.5. If you have CUDA 11.6 or higher and want to use flash attention for better performance:

1. Install flash-attn: `pip install --no-build-isolation flash-attn`
2. Edit `main.py` to change `_attn_implementation="eager"` to `_attn_implementation="flash_attention_2"`

```bash
pip install --no-build-isolation flash-attn==2.5.5

pip install --no-build-isolation flash-attn==2.7.4.post1
```

## Usage

### Basic Usage

To run the demonstration script with default settings:

```bash
python main.py
```

This script will:

1. Load the Phi-4 multimodal model
2. Process an example image and generate a description
3. Process an audio file and perform transcription

### Command Line Parameters

The script supports the following command-line parameters:

```bash
python main.py [OPTIONS]
```

| Parameter | Description |
|-----------|-------------|
| `--model-path PATH` | Path to the model directory (default: "Phi-4-multimodal-instruct") |
| `--image-url URL` | URL or local path of the image to process |
| `--audio-url URL` | URL or local path of the audio to process |
| `--image-prompt PROMPT` | Custom prompt for image analysis |
| `--speech-prompt PROMPT` | Custom prompt for audio transcription |
| `--cache-dir DIR` | Directory to cache downloaded files |
| `--results-dir DIR` | Directory to save results |
| `--force-cpu` | Force CPU usage even if GPUs are available |
| `--disable-parallel` | Disable parallel processing even if multiple GPUs are available |
| `--debug` | Enable debug logging for more verbose output |
| `--config-file FILE` | Path to a custom JSON configuration file (default: config.json) |

### Examples

Process a custom image from URL:

```bash
python main.py --image-url https://example.com/my-image.jpg
```

Process a local image file:

```bash
python main.py --image-url /path/to/local/image.jpg
```

Process a local audio file with a custom prompt:

```bash
python main.py --audio-url /path/to/recording.wav --speech-prompt "Transcribe this audio with timestamps."
```

Force CPU processing:

```bash
python main.py --force-cpu
```

Use a custom configuration file:

```bash
python main.py --config-file my_custom_config.json
```

Enable debug logging:

```bash
python main.py --debug
```

### Configuration File

You can customize the application behavior through the `config.json` file. The default configuration includes:

```json
{
  "model_path": "Phi-4-multimodal-instruct",
  "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg",
  "audio_url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac",
  "cache_dir": "cached_files",
  "results_dir": "results",
  "user_prompt": "<|user|>",
  "assistant_prompt": "<|assistant|>",
  "prompt_suffix": "<|end|>",
  "force_cpu": false,
  "disable_parallel": false,
  "multi_gpu_threshold_gb": 8.0,
  "debug": false,
  "image_prompt": "What is shown in this image?",
  "speech_prompt": "Transcribe the audio to text, paying special attention to:
1. Natural pauses in speech (use commas, periods, or other appropriate punctuation)
2. Changes in tone and intonation (questions, exclamations)
3. Emphasis and rhythm of the speaker's delivery
4. Natural paragraph breaks where topics shift
5. Maintain speaker's original cadence while ensuring readability

Format as a professional transcript that preserves the speaker's natural speech patterns."
}
```

You can create a custom configuration file based on this template and specify it with the `--config-file` parameter.

## GPU Parallel Processing

If multiple GPUs are available with sufficient memory (above the threshold set in `multi_gpu_threshold_gb`), the script will automatically use parallel processing to handle image and audio tasks simultaneously on different GPUs. This can significantly improve overall processing time.

If you encounter issues with parallel processing, you can disable it using the `--disable-parallel` flag.

## Note

For systems without a compatible GPU, modify the `_attn_implementation` parameter in `main.py` from "flash_attention_2" to "eager" as noted in the original message.

## Compatible GPUs

The Phi-4 multimodal model requires an NVIDIA GPU with Ampere architecture or newer to support Flash Attention 2. Compatible GPUs include:

## GPUs by Architecture

### Ampere Architecture

- **Consumer GPUs**: RTX 3090, 3090 Ti, 3080, 3080 Ti, 3070, 3070 Ti, 3060, 3060 Ti
- **Professional GPUs**: A100, A40, A30, A10, A6000, A5000, A4000

### Newer than Ampere

#### Ada Lovelace Architecture

- **Consumer GPUs**: RTX 4090, 4080, 4080 SUPER, 4070, 4070 Ti, 4070 SUPER, 4060, 4060 Ti
- **Professional GPUs**: RTX 6000, L40

#### Hopper Architecture

- **Professional GPUs**: H100, H800

#### Blackwell Architecture

- **Professional GPUs**: B100, B200

## Memory Optimization

The Phi-4 multimodal model requires significant GPU memory. If you encounter out-of-memory errors:

1. The script now uses 4-bit quantization to reduce memory usage
2. You can further reduce memory usage by:
   - Processing smaller inputs
   - Reducing `max_new_tokens` in generation settings
   - Using CPU offloading: `device_map="auto"`
   - Running only one task (image or audio) at a time

For systems with limited GPU memory:

```bash
# Before running heavy ML tasks, free up GPU memory
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
```

## Known Issues

### Quantization and LoRA Adapters

The Phi-4 multimodal model uses LoRA adapters for vision and speech processing. These adapters require floating-point tensors with gradient support. When using 4-bit quantization (via BitsAndBytes), some tensors are converted to integer types which don't support gradients, causing the error:

```
ValueError: Cannot convert tensor of type torch.bfloat16 and with requires_grad=True to non_blocking Int2, found in parameter 'vision_tower.mm_vision_tower.visual.embeddings.patch_embedding.weight'.
```

As a workaround, the model is configured to avoid quantization for adapter components.

### Flash Attention 2 Compatibility

For PyTorch 2.0+, flash attention may require additional steps:

- Install correct version: `pip install -U flash-attn --no-build-isolation`
- Check CUDA compatibility: `python -m torch.utils.collect_env`
- If issues persist, fall back to eager mode by setting `"_attn_implementation": "eager"` in your config

## Environment Variables

You can configure the application using environment variables by prefixing any configuration option with `PHI4_`:

```bash
# Example: Override model path using environment variable
export PHI4_MODEL_PATH="/path/to/model"
python main.py
```

## Advanced Features

### Benchmark Tool

The project includes a benchmark script to compare transcription performance between Whisper and Phi-4:

```bash
python benchmarks/transcription_benchmark.py
```

This will process test audio files with both models and output detailed comparisons to the `benchmark_results` directory.

### Speaker Diarization

While Phi-4 doesn't support native speaker diarization, our documentation provides guidance on integrating third-party diarization solutions like Pyannote.Audio to identify multiple speakers in audio recordings. See `docs/speaker_diarization.md` for details.

### Model Comparison

For a comprehensive comparison of different speech recognition models including Phi-4, see `docs/model_comparison.md`.

## Project Structure

- `main.py`: Main entry point for the demo
- `utils/`: Core utility modules
  - `config_utils.py`: Configuration management
  - `file_utils.py`: File operations and caching
  - `gpu_utils.py`: GPU detection and resource management
  - `logging_utils.py`: Thread-safe logging
  - `model_utils.py`: Model loading and configuration
  - `processors.py`: Image and audio processing
  - `runner.py`: Main processing workflow
- `benchmarks/`: Performance comparison tools
- `docs/`: Documentation and guides
- `cached_files/`: Downloaded files cache
- `results/`: Processing results
- `config.json`: Default configuration

## License

This project follows the license terms of the Phi-4-multimodal-instruct model. For details, see the LICENSE file and Microsoft's usage terms for the Phi-4 model.

## Acknowledgments

This project demonstrates the capabilities of Microsoft's Phi-4 multimodal model.
