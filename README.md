# Microsoft Phi-4 Multimodal Model Demo

This project demonstrates the capabilities of Microsoft's **Phi-4 multimodal model** for processing text, images, and audio inputs in a unified architecture. The Phi-4 multimodal model represents the latest advancement in Microsoft's Phi series, offering state-of-the-art multimodal understanding in a compact form factor.

## 🌟 Features

- **Unified Multimodal Processing**: Single model handles text, image, and audio inputs
- **Layered Prompting Architecture**: Advanced multi-phase analysis system that decomposes complex visual understanding into specialized layers for systematic, comprehensive image analysis
- **128K Token Context**: Extended context length for complex scenarios  
- **14B Parameters**: Optimized for efficiency while maintaining high performance
- **Multilingual Support**: 23+ languages for text, with English support for vision and multiple languages for audio
- **Advanced Reasoning**: Enhanced mathematical and logical reasoning capabilities
- **Function Calling**: Built-in support for tool and function calling

## 🔧 System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher (required for Phi-4)
- **Memory**: 32GB+ RAM recommended
- **Storage**: 50GB+ available space for model files

### Recommended Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, H100, etc.)
- **CUDA**: 12.1+ for optimal performance
- **Memory**: 64GB+ RAM for optimal performance
- **Storage**: NVMe SSD for faster model loading

### Supported Platforms

- Linux (Ubuntu 20.04+, CentOS 8+)
- Windows 10/11 with WSL2
- macOS (CPU-only, limited performance)

## 🚀 Quick Setup

### Automated Setup (Recommended)

We provide an automated setup script that handles all dependencies:

```bash
# Make the setup script executable
chmod +x setup_environment.sh

# Run the automated setup
./setup_environment.sh

# Activate the environment  
source activate_env.sh
```

The setup script will:

- ✅ Check Python version compatibility
- ✅ Detect and configure CUDA/GPU settings
- ✅ Create an isolated virtual environment
- ✅ Install PyTorch with appropriate CUDA support
- ✅ Install all required dependencies
- ✅ Verify the installation
- ✅ Check Phi-4 model availability

### Manual Setup

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv phi4-multimodal-env
source phi4-multimodal-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Getting Started

### 1. Model Download

Ensure you have the Phi-4 multimodal model files:

```bash
# If not already present, clone the model
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
```

### 2. Basic Usage

Run the demo with default settings:

```bash
# Activate the environment first
source activate_env.sh

# Run the demo
python main.py
```

The demo will:

1. 🤖 Load the Phi-4 multimodal model
2. 🖼️ Process a sample image and generate detailed description  
3. 🎵 Process a sample audio file and perform transcription
4. 💾 Save results to the `results/` directory

### 3. Custom Inputs

#### Process Your Own Images

```bash
# From URL
python main.py --image-url "https://example.com/your-image.jpg"

# From local file
python main.py --image-url "./samples/your-image.jpg"
```

#### Process Your Own Audio

```bash
# From URL  
python main.py --audio-url "https://example.com/audio.wav"

# From local file
python main.py --audio-url "./samples/your-audio.wav"
```

#### Custom Prompts

```bash
# Custom image analysis
python main.py --image-prompt "Analyze this image for safety hazards and provide recommendations."

# Custom audio transcription
python main.py --speech-prompt "Transcribe this audio and identify the main topics discussed."
```

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path PATH` | Path to Phi-4 model directory | `Phi-4-multimodal-instruct` |
| `--image-url URL` | Image URL or local path | Default sample image |
| `--audio-url URL` | Audio URL or local path | Default sample audio |
| `--image-prompt TEXT` | Custom image analysis prompt | Detailed scene description |
| `--speech-prompt TEXT` | Custom audio transcription prompt | Professional transcription |
| `--cache-dir DIR` | Cache directory for downloads | `cached_files` |
| `--results-dir DIR` | Output directory for results | `results` |
| `--config-file FILE` | JSON configuration file | `config.json` |
| `--force-cpu` | Force CPU-only processing | Auto-detect GPU |
| `--disable-parallel` | Disable multi-GPU processing | Auto-enable if available |
| `--debug` | Enable verbose debug logging | Info level |

## 🔧 Advanced Configuration

### Configuration File

Create a custom `config.json` for your specific needs:

```json
{
  "model_path": "Phi-4-multimodal-instruct",
  "cache_dir": "my_cache",
  "results_dir": "my_results", 
  "image_prompt": "Provide a detailed technical analysis of this image.",
  "speech_prompt": "Transcribe with speaker identification and timestamps.",
  "force_cpu": false,
  "debug": true
}
```

### Performance Optimization

#### GPU Memory Management

```bash
# For systems with limited GPU memory
python main.py --force-cpu

# For multi-GPU systems
python main.py  # Automatically uses multiple GPUs if available
```

#### Model Loading Options

The application automatically optimizes based on available hardware:

- **16GB+ VRAM**: Full precision model loading
- **8-16GB VRAM**: Automatic model sharding
- **<8GB VRAM**: CPU fallback with warning

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
  "user_tag": "<|user|>",
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

## 💡 Performance Tips

### GPU Optimization

For optimal performance:

```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Run with memory monitoring
python main.py --debug
```

### Memory Management

The application includes automatic memory optimization:

- **Automatic quantization**: Uses 4-bit quantization when beneficial
- **Model sharding**: Distributes model across multiple GPUs
- **CPU offloading**: Falls back to CPU for memory-intensive operations
- **Cache management**: Efficiently manages model and data caching

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Try CPU-only mode
python main.py --force-cpu

# Or reduce precision (automatically handled)
python main.py --debug  # Shows memory usage
```

#### Flash Attention Errors

If you encounter flash attention issues:

1. The setup script handles this automatically
2. Manual fix: Set `_attn_implementation="eager"` in the model configuration

#### Model Loading Errors

```bash
# Verify model files
ls -la Phi-4-multimodal-instruct/

# Re-clone if corrupted
rm -rf Phi-4-multimodal-instruct
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
```

### System Requirements

#### GPU Compatibility

**Supported Architectures:**

- ✅ **Ampere** (RTX 30 series, A100, A40, etc.)
- ✅ **Ada Lovelace** (RTX 40 series, RTX 6000, L40)  
- ✅ **Hopper** (H100, H800)
- ✅ **Blackwell** (B100, B200)

**Memory Requirements by GPU:**

| GPU Memory | Model Configuration | Performance |
|------------|-------------------|-------------|
| 24GB+ | Full precision | Optimal |
| 16-24GB | Mixed precision | Good |
| 8-16GB | Quantized + sharding | Moderate |
| <8GB | CPU fallback | Limited |

## 📚 Additional Resources

### Official Documentation

- [Phi-4 Technical Report](https://arxiv.org/abs/2503.01743)
- [Microsoft Phi Portal](https://aka.ms/phi-4-multimodal/azure)
- [Phi Cookbook](https://github.com/microsoft/PhiCookBook)

### Online Playgrounds

- [Azure AI Studio](https://aka.ms/phi-4-multimodal/azure)
- [GitHub Models](https://github.com/marketplace/models/azureml/Phi-4-multimodal-instruct/playground)
- [NVIDIA NIM](https://aka.ms/phi-4-multimodal/nvidia)
- [Hugging Face Spaces](https://huggingface.co/spaces/microsoft/phi-4-multimodal)

### Sample Applications

- [Thoughts Organizer](https://huggingface.co/spaces/microsoft/ThoughtsOrganizer)
- [Stories Come Alive](https://huggingface.co/spaces/microsoft/StoriesComeAlive)
- [Phine Speech Translator](https://huggingface.co/spaces/microsoft/PhineSpeechTranslator)

## 🤝 Contributing

This is a demonstration project. For contributions to the Phi-4 model itself, please refer to Microsoft's official channels.

## 🆕 Changelog

### Version 2.0

- ✅ Updated for Phi-4 multimodal model
- ✅ Automated setup script with GPU detection
- ✅ Enhanced error handling and memory management
- ✅ Improved documentation and troubleshooting guides
- ✅ Support for latest CUDA versions and RTX 50 series

### Version 1.0

- Basic Phi-4 model support
- Manual setup process

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

This project is licensed under the GNU General Public License v3.0 (GPLv3); see the [LICENSE](./LICENSE) file in the project root for details.

The Phi-4 multimodal model is licensed under Microsoft's custom license – see the [LICENSE](Phi-4-multimodal-instruct/LICENSE) file in the model directory.

## Acknowledgments

This project demonstrates the capabilities of Microsoft's Phi-4 multimodal model.
