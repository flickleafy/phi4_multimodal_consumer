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

To run the demonstration script:

```bash
python main.py
```

This script will:

1. Load the Phi-4 multimodal model
2. Process an example image (a stop sign) and generate a description
3. Process an audio file and perform transcription and translation

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
