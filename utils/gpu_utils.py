"""GPU detection and optimization utilities."""

from typing import Dict, List, Optional, TypedDict, Union, Tuple
import torch


class GPUInfo(TypedDict):
    """Type definition for GPU information dictionary."""

    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: str


class OptimalSettings(TypedDict):
    """Type definition for model configuration settings."""

    device_map: Union[str, int]
    quantization: bool
    precision: torch.dtype
    max_new_tokens: int
    parallel_processing: bool
    multi_gpu: bool
    available_gpus: List[int]


def get_gpu_info() -> List[GPUInfo]:
    """
    Get information about available GPUs and their memory.

    Returns:
        List[GPUInfo]: Information about each available GPU
    """
    gpu_info: List[GPUInfo] = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s)")

        for i in range(gpu_count):
            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            free_memory = total_memory - allocated

            gpu_info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": total_memory,
                    "free_memory_gb": free_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

            print(
                f"GPU {i}: {props.name}, "
                f"Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total"
            )
    else:
        print("No GPU available, using CPU")

    return gpu_info


def get_optimal_settings(gpu_info: List[GPUInfo]) -> OptimalSettings:
    """
    Determine optimal model settings based on available GPU resources.

    Args:
        gpu_info: List of GPU information dictionaries

    Returns:
        OptimalSettings: Optimized configuration for model loading
    """
    settings: OptimalSettings = {
        "device_map": "cpu",  # Default to CPU if no suitable GPU
        "quantization": False,  # Avoid LoRA issues
        "precision": torch.float16,
        "max_new_tokens": 100,
        "parallel_processing": False,
        "multi_gpu": False,
        "available_gpus": [],
    }

    if not gpu_info:
        return settings

    # Find GPU with most free memory
    best_gpu = max(gpu_info, key=lambda x: x["free_memory_gb"])
    device_id = best_gpu["index"]
    best_gpu_memory = best_gpu["free_memory_gb"]
    settings["device_map"] = f"cuda:{device_id}"
    print(f"Using GPU {device_id} with {best_gpu_memory:.2f}GB free memory")

    # Check for multiple usable GPUs (with at least 8GB free memory)
    usable_gpus = [gpu["index"] for gpu in gpu_info if gpu["free_memory_gb"] >= 8.0]
    settings["available_gpus"] = usable_gpus

    if len(usable_gpus) > 1:
        settings["multi_gpu"] = True
        print(f"Multi-GPU mode enabled with GPUs: {usable_gpus}")

    # Adjust maximum tokens based on available memory
    if best_gpu_memory > 24:
        settings["max_new_tokens"] = 500
        settings["quantization"] = False
    elif best_gpu_memory > 16:
        settings["max_new_tokens"] = 300
    elif best_gpu_memory > 8:
        settings["max_new_tokens"] = 200
    else:
        settings["max_new_tokens"] = 100

    # Enable parallel processing if we have multiple usable GPUs
    if len(usable_gpus) > 1:
        settings["parallel_processing"] = True

    print(
        f"Optimized settings: max_new_tokens={settings['max_new_tokens']}, "
        f"quantization={settings['quantization']}, "
        f"device_map={settings['device_map']}, "
        f"parallel_processing={settings['parallel_processing']}"
    )

    return settings


def clear_memory(device_id: Optional[int] = None) -> None:
    """
    Free GPU memory and run garbage collection.

    Args:
        device_id: Specific GPU device ID to clear memory for, or None for all devices
    """
    if device_id is not None:
        # Clear memory for specific device
        with torch.cuda.device(f"cuda:{device_id}"):
            torch.cuda.empty_cache()
    else:
        # Clear memory for all devices
        torch.cuda.empty_cache()

    import gc

    gc.collect()
