#!/bin/bash
# Setup script for Microsoft Phi-4 Multimodal Model
# Creates virtual environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up Microsoft Phi-4 Multimodal environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python 3.10+ is available
print_header "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
required_version="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    print_status "Python version $python_version is compatible (Phi-4 requires Python 3.10+)"
else
    print_error "Python 3.10+ is required for Phi-4 multimodal. Current version: $python_version"
    print_error "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Check CUDA availability
print_header "🔧 Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    print_status "CUDA $cuda_version detected"
    
    # Check GPU memory
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    print_status "GPU Information:"
    echo "$gpu_info" | while read line; do
        echo "  📊 $line MB"
    done
else
    print_warning "CUDA not detected. CPU-only mode will be used."
fi

# Create virtual environment
ENV_NAME="phi4-multimodal-env"
print_header "📦 Creating virtual environment: $ENV_NAME"

# Check if running inside a virtual environment and deactivate if needed
if [ -n "$VIRTUAL_ENV" ]; then
    print_warning "Currently running inside a virtual environment ($VIRTUAL_ENV). Deactivating before removal."
    deactivate
fi

if [ -d "$ENV_NAME" ]; then
    print_warning "Virtual environment $ENV_NAME already exists. Removing..."
    rm -rf "$ENV_NAME"
    print_status "Removed existing virtual environment $ENV_NAME. Proceeding with fresh install."
fi

python3 -m venv "$ENV_NAME"

# Check if Python executable exists in the venv
if [ ! -f "$ENV_NAME/bin/python3" ]; then
    print_error "Virtual environment was created, but $ENV_NAME/bin/python3 is missing."
    print_error "This usually means your Python installation is broken or incomplete."
    print_error "Try reinstalling Python 3.10+ and rerun this script."
    exit 1
fi

source "$ENV_NAME/bin/activate"

print_status "Virtual environment created and activated"

# Upgrade pip
print_header "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
print_header "🔥 Installing PyTorch for Phi-4 multimodal..."
if command -v nvidia-smi &> /dev/null; then
    # Check GPU memory requirements
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$gpu_memory" -lt 16000 ]; then
        print_warning "GPU has less than 16GB memory. Phi-4 multimodal may require model sharding or reduced precision."
    fi
    # Check for latest GPUs and install appropriate PyTorch
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    if echo "$gpu_info" | grep -q "RTX 50\|H100\|A100"; then
        print_warning "RTX 5090 detected - Installing PyTorch nightly for sm_120 support"
        # Pin all core ML libraries to exact nightly versions for CUDA 12.8 (RTX 5090)
        PYTORCH_VERSION="2.9.0.dev20250726+cu128"
        TORCHVISION_VERSION="0.24.0.dev20250726+cu128"
        TORCHAUDIO_VERSION="2.8.0.dev20250726+cu128"
        FSSPEC_VERSION="2025.5.1"
        NETWORKX_VERSION="3.4.2"
        JINJA2_VERSION="3.1.6"
        FILELOCK_VERSION="3.18.0"
        PILLOW_VERSION="11.3.0"
        # Install pinned versions
        pip install --pre torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} fsspec==${FSSPEC_VERSION} networkx==${NETWORKX_VERSION} jinja2==${JINJA2_VERSION} filelock==${FILELOCK_VERSION} pillow==${PILLOW_VERSION} --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        print_status "Installing PyTorch stable with CUDA 12.1 support"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi
else
    print_warning "No NVIDIA GPU detected - Installing CPU-only PyTorch"
    print_warning "Note: Phi-4 multimodal will run very slowly on CPU"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Pin fsspec to compatible version for datasets
print_status "Pinning fsspec to <=2025.3.0 for datasets compatibility..."
pip install "fsspec[http]==2025.5.1"

# Install critical dependencies first
print_header "🚀 Installing Phi-4 multimodal dependencies..."

# Install core ML libraries
print_status "Installing transformers and accelerate..."
pip install transformers==4.48.2 accelerate==1.4.0

# Install audio processing dependencies
print_status "Installing audio processing libraries..."
pip install soundfile>=0.13.0 scipy>=1.15.0

# Install image processing dependencies  
print_status "Installing image processing libraries..."
pip install pillow==${PILLOW_VERSION}

# Install additional ML libraries
print_status "Installing additional ML dependencies..."
pip install datasets>=3.3.0 pandas>=2.2.0 peft==0.14.0

# Install evaluation libraries
print_status "Installing evaluation libraries..."
pip install evaluate>=0.4.0 sacrebleu>=2.5.0

# Install optimizations (optional but recommended)
# print_status "Installing performance optimizations..."
# pip install bitsandbytes>=0.39.0 --no-warn-script-location

# Install flash-attn with explicit torch version constraint
# print_status "Attempting to install flash-attn (this may take several minutes or fail on some systems)..."
# pip install --no-build-isolation --force-reinstall flash-attn || {
#     print_warning "Flash attention installation failed. This is optional and the model will work without it."
#     print_warning "For better performance on supported GPUs, you can try installing it manually later."
# }

# Re-install core ML libraries to ensure correct versions after flash-attn
# print_status "Re-installing core ML libraries to fix any downgrades from flash-attn..."
# pip install --pre torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} fsspec==${FSSPEC_VERSION} networkx==${NETWORKX_VERSION} jinja2==${JINJA2_VERSION} filelock==${FILELOCK_VERSION} pillow==${PILLOW_VERSION} --index-url https://download.pytorch.org/whl/nightly/cu128

# Install remaining dependencies
print_header "📚 Installing remaining project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Installed dependencies from requirements.txt"
else
    print_warning "requirements.txt not found, skipping additional dependencies"
fi

# Verify installation
print_header "✅ Verifying Phi-4 multimodal installation..."
python3 -c "
import torch
import warnings
warnings.filterwarnings('ignore')

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        major, minor = torch.cuda.get_device_capability(i)
        print(f'  Device {i}: {device_name}')
        print(f'    Memory: {memory_gb:.1f} GB')
        print(f'    CUDA Capability: sm_{major}{minor}')
        
        # Check memory requirements for Phi-4
        if memory_gb >= 16:
            print(f'    ✅ Memory check: Sufficient for Phi-4 multimodal')
        elif memory_gb >= 8:
            print(f'    ⚠️  Memory check: May need model sharding or reduced precision')
        else:
            print(f'    ❌ Memory check: Insufficient for full Phi-4 model')

# Test basic CUDA operations
try:
    if torch.cuda.is_available():
        test_tensor = torch.randn(10, 10).cuda()
        result = torch.mm(test_tensor, test_tensor.t())
        print('✅ CUDA operations test: Passed')
    else:
        print('ℹ️  CUDA operations test: Skipped (no CUDA)')
except Exception as e:
    print(f'❌ CUDA operations test: Failed - {e}')

# Test core dependencies
dependencies = [
    ('transformers', 'Transformers'),
    ('accelerate', 'Accelerate'), 
    ('soundfile', 'SoundFile'),
    ('PIL', 'Pillow'),
    ('scipy', 'SciPy'),
    ('pandas', 'Pandas'),
    ('datasets', 'Datasets'),
]

print('\n📦 Checking core dependencies:')
for module_name, display_name in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {display_name}: {version}')
    except ImportError as e:
        print(f'❌ {display_name}: Not installed - {e}')

# Test optional dependencies
optional_deps = [
    ('flash_attn', 'Flash Attention'),
    ('bitsandbytes', 'BitsAndBytes'),
    ('peft', 'PEFT'),
]

print('\n🔧 Checking optional dependencies:')
for module_name, display_name in optional_deps:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {display_name}: {version}')
    except ImportError:
        print(f'⚠️  {display_name}: Not installed (optional)')

# Test if Phi-4 model files are available
import os
phi4_path = 'Phi-4-multimodal-instruct'
if os.path.exists(phi4_path):
    required_files = ['config.json', 'tokenizer.json', 'modeling_phi4mm.py']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(phi4_path, f))]
    if not missing_files:
        print(f'✅ Phi-4 model files: Available in {phi4_path}')
    else:
        print(f'⚠️  Phi-4 model files: Missing {missing_files}')
else:
    print(f'❌ Phi-4 model files: Directory {phi4_path} not found')
    print('   Run: git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct')
"

print_status "Phi-4 multimodal environment setup completed successfully!"
print_status "To activate the environment, run:"
echo -e "${BLUE}source $ENV_NAME/bin/activate${NC}"

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
source phi4-multimodal-env/bin/activate
echo "🌟 Phi-4 Multimodal environment activated!"
echo "📍 Virtual environment: $(which python)"
echo "🔧 Python version: $(python --version)"
echo "🤖 Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""
echo "💡 Quick start:"
echo "   python main.py --help                    # Show all options"
echo "   python main.py                          # Run with default settings"  
echo "   python main.py --debug                  # Run with debug logging"
EOF

chmod +x activate_env.sh
print_status "Created activation script: ./activate_env.sh"

# Additional setup instructions
echo ""
print_header "🎯 Next Steps:"
print_status "1. Activate the environment: source activate_env.sh"
print_status "2. Ensure Phi-4 model is available: ls Phi-4-multimodal-instruct/"
print_status "3. Run the demo: python main.py"
print_status "4. Check the results/ directory for outputs"

if ! command -v nvidia-smi &> /dev/null; then
    echo ""
    print_warning "⚠️  No NVIDIA GPU detected. The model will run on CPU and be very slow."
    print_warning "   For best performance, use a GPU with at least 16GB VRAM."
fi
