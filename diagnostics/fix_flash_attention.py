#!/usr/bin/env python3
"""
Flash Attention Diagnostic and Fix Script

This script diagnoses and attempts to fix flash attention import issues
that can occur with the Phi-4 multimodal model.

"""

import os
import sys
import subprocess
import importlib.util
from typing import List, Dict, Any, Optional, Tuple


def set_environment_variables() -> None:
    """Set all possible environment variables to disable flash attention."""
    env_vars = {
        "DISABLE_FLASH_ATTN": "1",
        "FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE",
        "USE_FLASH_ATTENTION": "false",
        "ATTN_IMPLEMENTATION": "eager",
        "TORCH_DISABLE_FLASH_ATTENTION": "1",
        "TRANSFORMERS_FORCE_EAGER_ATTENTION": "1",
        "DISABLE_FLASHATTENTION": "1",
        "NO_FLASH_ATTN": "1",
        "PYTORCH_DISABLE_FLASH_ATTENTION": "1",
        "TORCH_USE_EAGER_ATTENTION": "1"
    }

    print("🔧 Setting environment variables to disable flash attention...")
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   {var}={value}")


def check_flash_attn_installation() -> Tuple[bool, Optional[str]]:
    """Check if flash_attn is installed and get version info."""
    try:
        import flash_attn
        return True, getattr(flash_attn, "__version__", "unknown")
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error importing flash_attn: {e}"


def check_torch_cuda_compatibility() -> Dict[str, Any]:
    """Check PyTorch and CUDA compatibility."""
    info = {}

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["device_count"] = torch.cuda.device_count(
        ) if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
    except Exception as e:
        info["error"] = str(e)

    return info


def attempt_transformers_import() -> Tuple[bool, Optional[str]]:
    """Attempt to import transformers and check for flash attention issues."""
    try:
        # Clear any existing flash_attn modules
        modules_to_remove = [
            mod for mod in sys.modules.keys() if 'flash_attn' in mod.lower()]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        from transformers import AutoModelForCausalLM, AutoProcessor
        return True, "Successfully imported transformers"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def run_subprocess_command(command: List[str]) -> Tuple[bool, str]:
    """Run a subprocess command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 60 seconds"
    except Exception as e:
        return False, str(e)


def get_package_info(package_name: str) -> Optional[Dict[str, str]]:
    """Get package installation information."""
    success, output = run_subprocess_command(
        [sys.executable, "-m", "pip", "show", package_name])
    if not success:
        return None

    info = {}
    for line in output.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            info[key.strip().lower()] = value.strip()
    return info


def diagnose_flash_attention() -> Dict[str, Any]:
    """Run comprehensive flash attention diagnostics."""
    print("🔍 Running Flash Attention Diagnostics...\n")

    diagnosis = {}

    # Check flash_attn installation
    flash_installed, flash_info = check_flash_attn_installation()
    diagnosis["flash_attn_installed"] = flash_installed
    diagnosis["flash_attn_info"] = flash_info

    print(
        f"Flash Attention Installed: {'✅ Yes' if flash_installed else '❌ No'}")
    if flash_installed:
        print(f"Flash Attention Version: {flash_info}")
    else:
        print(f"Flash Attention Error: {flash_info}")
    print()

    # Check PyTorch/CUDA
    torch_info = check_torch_cuda_compatibility()
    diagnosis["torch_info"] = torch_info

    print("PyTorch & CUDA Information:")
    for key, value in torch_info.items():
        print(f"  {key}: {value}")
    print()

    # Check package versions
    packages = ["torch", "transformers", "flash-attn"]
    package_info = {}
    for package in packages:
        info = get_package_info(package)
        package_info[package] = info
        if info:
            print(f"{package}: {info.get('version', 'unknown')}")
        else:
            print(f"{package}: Not installed or error getting info")

    diagnosis["package_info"] = package_info
    print()

    # Test transformers import
    set_environment_variables()
    transformers_success, transformers_msg = attempt_transformers_import()
    diagnosis["transformers_import"] = {
        "success": transformers_success,
        "message": transformers_msg
    }

    print(
        f"Transformers Import: {'✅ Success' if transformers_success else '❌ Failed'}")
    print(f"Message: {transformers_msg}")
    print()

    return diagnosis


def provide_recommendations(diagnosis: Dict[str, Any]) -> List[str]:
    """Provide recommendations based on diagnosis."""
    recommendations = []

    # Check if transformers import failed
    if not diagnosis["transformers_import"]["success"]:
        error_msg = diagnosis["transformers_import"]["message"].lower()

        if "flash_attn" in error_msg or "undefined symbol" in error_msg:
            recommendations.append(
                "🔧 CRITICAL: Flash attention compatibility issue detected.\n"
                "   The installed flash-attn package is incompatible with your PyTorch/CUDA versions."
            )

            # Check if flash_attn is installed
            if diagnosis["flash_attn_installed"]:
                recommendations.extend([
                    "💡 SOLUTION 1 (Recommended): Uninstall flash-attn completely:\n"
                    "   pip uninstall flash-attn\n"
                    "   The Phi-4 model will work without flash attention using eager attention.",

                    "💡 SOLUTION 2: Rebuild flash-attn for your environment:\n"
                    "   pip uninstall flash-attn\n"
                    "   pip install flash-attn --no-build-isolation --force-reinstall\n"
                    "   (This may take 10-30 minutes to compile)",
                ])

            # Check CUDA availability
            torch_info = diagnosis["torch_info"]
            if not torch_info.get("cuda_available", False):
                recommendations.append(
                    "💡 SOLUTION 3: Use CPU-only mode:\n"
                    "   export CUDA_VISIBLE_DEVICES=\n"
                    "   python main.py --device cpu"
                )

            # Version compatibility
            if torch_info.get("torch_version"):
                recommendations.append(
                    "💡 SOLUTION 4: Install compatible versions:\n"
                    f"   Current PyTorch: {torch_info['torch_version']}\n"
                    "   For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                    "   For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
                )

    # Check if transformers import succeeded
    elif diagnosis["transformers_import"]["success"]:
        recommendations.append(
            "✅ All imports successful! Your environment should work correctly.")

        if diagnosis["flash_attn_installed"]:
            recommendations.append(
                "ℹ️  Flash attention is installed but disabled via environment variables.\n"
                "   This is the safest configuration for compatibility."
            )

    # General recommendations
    recommendations.extend([
        "\n📋 General Recommendations:",
        "• Always run the environment setup script: ./setup_environment.sh",
        "• Use the test scripts: python test_imports.py && python test_installation.py",
        "• For CPU-only usage, set: export CUDA_VISIBLE_DEVICES=",
        "• Check the README.md for detailed troubleshooting steps"
    ])

    return recommendations


def main() -> None:
    """Main function to run flash attention diagnostics and provide fixes."""
    print("🚀 Flash Attention Diagnostic and Fix Tool")
    print("=" * 50)

    # Run diagnostics
    diagnosis = diagnose_flash_attention()

    # Get recommendations
    recommendations = provide_recommendations(diagnosis)

    # Display recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 50)
    for rec in recommendations:
        print(rec)
        print()

    # Offer to implement solution 1 (uninstall flash-attn)
    if not diagnosis["transformers_import"]["success"] and diagnosis["flash_attn_installed"]:
        print("\n🔧 AUTOMATIC FIX AVAILABLE")
        print("=" * 50)
        response = input(
            "Would you like to automatically uninstall flash-attn? (y/N): ").strip().lower()

        if response in ['y', 'yes']:
            print("🗑️  Uninstalling flash-attn...")
            success, output = run_subprocess_command(
                [sys.executable, "-m", "pip", "uninstall", "flash-attn", "-y"])

            if success:
                print("✅ Successfully uninstalled flash-attn")
                print("🔄 Testing imports again...")

                # Clear modules and test again
                modules_to_remove = [
                    mod for mod in sys.modules.keys() if 'flash_attn' in mod.lower()]
                for mod in modules_to_remove:
                    sys.modules.pop(mod, None)

                set_environment_variables()
                success, msg = attempt_transformers_import()

                if success:
                    print("🎉 SUCCESS! Transformers import now works correctly.")
                    print("You can now run: python main.py")
                else:
                    print(f"⚠️  Import still failing: {msg}")
                    print("Please try the other solutions listed above.")
            else:
                print(f"❌ Failed to uninstall flash-attn: {output}")
        else:
            print(
                "ℹ️  No automatic fix applied. Please follow the recommendations above.")

    print("\n✅ Diagnostic complete. Check the recommendations above for next steps.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Flash Attention Diagnostic and Fix Script

This script diagnoses and attempts to fix flash attention import issues
that can occur with the Phi-4 multimodal model.

"""


def set_environment_variables() -> None:
    """Set all possible environment variables to disable flash attention."""
    env_vars = {
        "DISABLE_FLASH_ATTN": "1",
        "FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE",
        "USE_FLASH_ATTENTION": "false",
        "ATTN_IMPLEMENTATION": "eager",
        "TORCH_DISABLE_FLASH_ATTENTION": "1",
        "TRANSFORMERS_FORCE_EAGER_ATTENTION": "1",
        "DISABLE_FLASHATTENTION": "1",
        "NO_FLASH_ATTN": "1",
        "PYTORCH_DISABLE_FLASH_ATTENTION": "1",
        "TORCH_USE_EAGER_ATTENTION": "1"
    }

    print("🔧 Setting environment variables to disable flash attention...")
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   {var}={value}")


def check_flash_attn_installation() -> Tuple[bool, Optional[str]]:
    """Check if flash_attn is installed and get version info."""
    try:
        import flash_attn
        return True, getattr(flash_attn, "__version__", "unknown")
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error importing flash_attn: {e}"


def check_torch_cuda_compatibility() -> Dict[str, Any]:
    """Check PyTorch and CUDA compatibility."""
    info = {}

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["device_count"] = torch.cuda.device_count(
        ) if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
    except Exception as e:
        info["error"] = str(e)

    return info


def attempt_transformers_import() -> Tuple[bool, Optional[str]]:
    """Attempt to import transformers and check for flash attention issues."""
    try:
        # Clear any existing flash_attn modules
        modules_to_remove = [
            mod for mod in sys.modules.keys() if 'flash_attn' in mod.lower()]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        from transformers import AutoModelForCausalLM, AutoProcessor
        return True, "Successfully imported transformers"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def run_subprocess_command(command: List[str]) -> Tuple[bool, str]:
    """Run a subprocess command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 60 seconds"
    except Exception as e:
        return False, str(e)


def get_package_info(package_name: str) -> Optional[Dict[str, str]]:
    """Get package installation information."""
    success, output = run_subprocess_command(
        [sys.executable, "-m", "pip", "show", package_name])
    if not success:
        return None

    info = {}
    for line in output.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            info[key.strip().lower()] = value.strip()
    return info


def diagnose_flash_attention() -> Dict[str, Any]:
    """Run comprehensive flash attention diagnostics."""
    print("🔍 Running Flash Attention Diagnostics...\n")

    diagnosis = {}

    # Check flash_attn installation
    flash_installed, flash_info = check_flash_attn_installation()
    diagnosis["flash_attn_installed"] = flash_installed
    diagnosis["flash_attn_info"] = flash_info

    print(
        f"Flash Attention Installed: {'✅ Yes' if flash_installed else '❌ No'}")
    if flash_installed:
        print(f"Flash Attention Version: {flash_info}")
    else:
        print(f"Flash Attention Error: {flash_info}")
    print()

    # Check PyTorch/CUDA
    torch_info = check_torch_cuda_compatibility()
    diagnosis["torch_info"] = torch_info

    print("PyTorch & CUDA Information:")
    for key, value in torch_info.items():
        print(f"  {key}: {value}")
    print()

    # Check package versions
    packages = ["torch", "transformers", "flash-attn"]
    package_info = {}
    for package in packages:
        info = get_package_info(package)
        package_info[package] = info
        if info:
            print(f"{package}: {info.get('version', 'unknown')}")
        else:
            print(f"{package}: Not installed or error getting info")

    diagnosis["package_info"] = package_info
    print()

    # Test transformers import
    set_environment_variables()
    transformers_success, transformers_msg = attempt_transformers_import()
    diagnosis["transformers_import"] = {
        "success": transformers_success,
        "message": transformers_msg
    }

    print(
        f"Transformers Import: {'✅ Success' if transformers_success else '❌ Failed'}")
    print(f"Message: {transformers_msg}")
    print()

    return diagnosis


def provide_recommendations(diagnosis: Dict[str, Any]) -> List[str]:
    """Provide recommendations based on diagnosis."""
    recommendations = []

    # Check if transformers import failed
    if not diagnosis["transformers_import"]["success"]:
        error_msg = diagnosis["transformers_import"]["message"].lower()

        if "flash_attn" in error_msg or "undefined symbol" in error_msg:
            recommendations.append(
                "🔧 CRITICAL: Flash attention compatibility issue detected.\n"
                "   The installed flash-attn package is incompatible with your PyTorch/CUDA versions."
            )

            # Check if flash_attn is installed
            if diagnosis["flash_attn_installed"]:
                recommendations.extend([
                    "💡 SOLUTION 1 (Recommended): Uninstall flash-attn completely:\n"
                    "   pip uninstall flash-attn\n"
                    "   The Phi-4 model will work without flash attention using eager attention.",

                    "💡 SOLUTION 2: Rebuild flash-attn for your environment:\n"
                    "   pip uninstall flash-attn\n"
                    "   pip install flash-attn --no-build-isolation --force-reinstall\n"
                    "   (This may take 10-30 minutes to compile)",
                ])

            # Check CUDA availability
            torch_info = diagnosis["torch_info"]
            if not torch_info.get("cuda_available", False):
                recommendations.append(
                    "💡 SOLUTION 3: Use CPU-only mode:\n"
                    "   export CUDA_VISIBLE_DEVICES=\n"
                    "   python main.py --device cpu"
                )

            # Version compatibility
            if torch_info.get("torch_version"):
                recommendations.append(
                    "💡 SOLUTION 4: Install compatible versions:\n"
                    f"   Current PyTorch: {torch_info['torch_version']}\n"
                    "   For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                    "   For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
                )

    # Check if transformers import succeeded
    elif diagnosis["transformers_import"]["success"]:
        recommendations.append(
            "✅ All imports successful! Your environment should work correctly.")

        if diagnosis["flash_attn_installed"]:
            recommendations.append(
                "ℹ️  Flash attention is installed but disabled via environment variables.\n"
                "   This is the safest configuration for compatibility."
            )

    # General recommendations
    recommendations.extend([
        "\n📋 General Recommendations:",
        "• Always run the environment setup script: ./setup_environment.sh",
        "• Use the test scripts: python test_imports.py && python test_installation.py",
        "• For CPU-only usage, set: export CUDA_VISIBLE_DEVICES=",
        "• Check the README.md for detailed troubleshooting steps"
    ])

    return recommendations


def main() -> None:
    """Main function to run flash attention diagnostics and provide fixes."""
    print("🚀 Flash Attention Diagnostic and Fix Tool")
    print("=" * 50)

    # Run diagnostics
    diagnosis = diagnose_flash_attention()

    # Get recommendations
    recommendations = provide_recommendations(diagnosis)

    # Display recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 50)
    for rec in recommendations:
        print(rec)
        print()

    # Offer to implement solution 1 (uninstall flash-attn)
    if not diagnosis["transformers_import"]["success"] and diagnosis["flash_attn_installed"]:
        print("\n🔧 AUTOMATIC FIX AVAILABLE")
        print("=" * 50)
        response = input(
            "Would you like to automatically uninstall flash-attn? (y/N): ").strip().lower()

        if response in ['y', 'yes']:
            print("🗑️  Uninstalling flash-attn...")
            success, output = run_subprocess_command(
                [sys.executable, "-m", "pip", "uninstall", "flash-attn", "-y"])

            if success:
                print("✅ Successfully uninstalled flash-attn")
                print("🔄 Testing imports again...")

                # Clear modules and test again
                modules_to_remove = [
                    mod for mod in sys.modules.keys() if 'flash_attn' in mod.lower()]
                for mod in modules_to_remove:
                    sys.modules.pop(mod, None)

                set_environment_variables()
                success, msg = attempt_transformers_import()

                if success:
                    print("🎉 SUCCESS! Transformers import now works correctly.")
                    print("You can now run: python main.py")
                else:
                    print(f"⚠️  Import still failing: {msg}")
                    print("Please try the other solutions listed above.")
            else:
                print(f"❌ Failed to uninstall flash-attn: {output}")
        else:
            print(
                "ℹ️  No automatic fix applied. Please follow the recommendations above.")

    print("\n✅ Diagnostic complete. Check the recommendations above for next steps.")


if __name__ == "__main__":
    main()
