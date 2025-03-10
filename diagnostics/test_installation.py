#!/usr/bin/env python3
"""
Test script to verify Phi-4 multimodal installation.
This script performs basic checks without loading the full model.

Time Complexity: O(1) - Simple import and configuration checks
Space Complexity: O(1) - Minimal memory usage for basic tests
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def check_python_version() -> bool:
    """
    Check if Python version meets requirements.

    Returns:
        bool: True if Python 3.10+, False otherwise

    Time Complexity: O(1)
    """
    version_info = sys.version_info
    required_major, required_minor = 3, 10

    if version_info.major > required_major or (
        version_info.major == required_major and version_info.minor >= required_minor
    ):
        print(
            f"✅ Python {version_info.major}.{version_info.minor}.{version_info.micro} (meets requirement 3.10+)")
        return True
    else:
        print(
            f"❌ Python {version_info.major}.{version_info.minor}.{version_info.micro} (requires 3.10+)")
        return False


def check_core_dependencies() -> Dict[str, bool]:
    """
    Check availability of core dependencies.

    Returns:
        Dict[str, bool]: Mapping of dependency names to availability status

    Time Complexity: O(n) where n is number of dependencies
    """
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('PIL', 'Pillow'),
        ('soundfile', 'SoundFile'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
    ]

    results = {}
    print("\n📦 Checking core dependencies:")

    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            results[module_name] = True
        except ImportError as e:
            print(f"❌ {display_name}: Not installed")
            results[module_name] = False

    return results


def check_optional_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional performance dependencies.

    Returns:
        Dict[str, bool]: Mapping of optional dependency names to availability

    Time Complexity: O(n) where n is number of optional dependencies
    """
    optional_deps = [
        ('flash_attn', 'Flash Attention'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('peft', 'PEFT'),
        ('datasets', 'Datasets'),
        ('pandas', 'Pandas'),
    ]

    results = {}
    print("\n🔧 Checking optional dependencies:")

    for module_name, display_name in optional_deps:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            results[module_name] = True
        except ImportError:
            print(f"⚠️  {display_name}: Not installed (optional)")
            results[module_name] = False

    return results


def check_cuda_availability() -> Tuple[bool, Optional[List[Dict[str, any]]]]:
    """
    Check CUDA availability and GPU information.

    Returns:
        Tuple containing CUDA availability status and GPU information

    Time Complexity: O(n) where n is number of GPUs
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_info = []

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"\n🚀 CUDA Status: Available ({device_count} device(s))")

            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(
                    i).total_memory / 1e9
                major, minor = torch.cuda.get_device_capability(i)

                gpu_info.append({
                    'id': i,
                    'name': name,
                    'memory_gb': memory_gb,
                    'compute_capability': f"sm_{major}{minor}"
                })

                print(f"  📊 Device {i}: {name}")
                print(f"      Memory: {memory_gb:.1f} GB")
                print(f"      Compute: sm_{major}{minor}")

                # Memory recommendations for Phi-4
                if memory_gb >= 24:
                    print(f"      ✅ Excellent for Phi-4 (full precision)")
                elif memory_gb >= 16:
                    print(f"      ✅ Good for Phi-4 (mixed precision)")
                elif memory_gb >= 8:
                    print(f"      ⚠️  Limited for Phi-4 (quantization required)")
                else:
                    print(f"      ❌ Insufficient for Phi-4 (CPU recommended)")

        else:
            print(f"\n🚀 CUDA Status: Not available (CPU only)")

        return cuda_available, gpu_info

    except Exception as e:
        print(f"\n❌ Error checking CUDA: {e}")
        return False, None


def check_model_files() -> bool:
    """
    Check if Phi-4 model files are available.

    Returns:
        bool: True if model files are present, False otherwise

    Time Complexity: O(1) - File system checks are constant time
    """
    model_path = Path("Phi-4-multimodal-instruct")

    if not model_path.exists():
        print(f"\n❌ Model directory not found: {model_path}")
        print("   To download: git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct")
        return False

    required_files = [
        'config.json',
        'tokenizer.json',
        'modeling_phi4mm.py',
        'processing_phi4mm.py'
    ]

    missing_files = []
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)

    if missing_files:
        print(f"\n⚠️  Model files incomplete. Missing: {missing_files}")
        return False
    else:
        print(f"\n✅ Model files: Available in {model_path}")
        return True


def check_project_structure() -> bool:
    """
    Verify project directory structure.

    Returns:
        bool: True if structure is correct, False otherwise

    Time Complexity: O(1) - Checking fixed number of directories
    """
    required_dirs = ['utils', 'samples', 'results']
    optional_dirs = ['cached_files', 'docs']

    print(f"\n📁 Checking project structure:")

    all_present = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory present")
        else:
            print(f"❌ {dir_name}/ directory missing")
            all_present = False

    for dir_name in optional_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory present")
        else:
            print(f"ℹ️  {dir_name}/ directory will be created as needed")

    return all_present


def run_basic_torch_test() -> bool:
    """
    Run a basic PyTorch operation test.

    Returns:
        bool: True if test passes, False otherwise

    Time Complexity: O(1) - Fixed size tensor operations
    """
    try:
        import torch

        # Test CPU operations
        x = torch.randn(5, 5)
        y = torch.mm(x, x.t())

        # Test CUDA operations if available
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            y_cuda = torch.mm(x_cuda, x_cuda.t())
            print("\n✅ PyTorch: CPU and CUDA operations successful")
        else:
            print("\n✅ PyTorch: CPU operations successful")

        return True

    except Exception as e:
        print(f"\n❌ PyTorch test failed: {e}")
        return False


def main():
    """
    Main test function that runs all checks.

    Time Complexity: O(n) where n is total number of dependencies and GPUs
    """
    print("🧪 Phi-4 Multimodal Installation Test")
    print("=" * 50)

    # Run all checks
    python_ok = check_python_version()
    core_deps = check_core_dependencies()
    optional_deps = check_optional_dependencies()
    cuda_ok, gpu_info = check_cuda_availability()
    model_files_ok = check_model_files()
    structure_ok = check_project_structure()
    torch_test_ok = run_basic_torch_test()

    # Summary
    print("\n" + "=" * 50)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 50)

    total_core = len(core_deps)
    core_installed = sum(core_deps.values())
    total_optional = len(optional_deps)
    optional_installed = sum(optional_deps.values())

    print(f"Python Version: {'✅' if python_ok else '❌'}")
    print(f"Core Dependencies: {core_installed}/{total_core} installed")
    print(
        f"Optional Dependencies: {optional_installed}/{total_optional} installed")
    print(f"CUDA Support: {'✅' if cuda_ok else '❌'}")
    print(f"Model Files: {'✅' if model_files_ok else '❌'}")
    print(f"Project Structure: {'✅' if structure_ok else '❌'}")
    print(f"PyTorch Test: {'✅' if torch_test_ok else '❌'}")

    # Overall status
    critical_checks = [python_ok, core_installed == total_core, torch_test_ok]

    if all(critical_checks):
        print(f"\n🎉 Installation Status: READY")
        print("You can run: python main.py")
        if not model_files_ok:
            print("\n💡 Note: Download the model files to get started")
    else:
        print(f"\n⚠️  Installation Status: INCOMPLETE")
        print("Please run the setup script: ./setup_environment.sh")

    return 0 if all(critical_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
