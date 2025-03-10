"""GPU detection and optimization utilities."""

from typing import Dict, List, Optional, TypedDict, Union, Tuple
import torch
import logging
import gc
import os
import time


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
    logger = logging.getLogger("phi4_demo")
    gpu_info: List[GPUInfo] = []
    if torch.cuda.is_available():
        # Force garbage collection to get more accurate memory readings
        gc.collect()
        torch.cuda.empty_cache()

        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")

        for i in range(gpu_count):
            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB

            # Get current memory usage - retry a few times to get stable readings
            # Sometimes immediate readings after cache clearing aren't accurate
            retries = 3
            for _ in range(retries):
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                time.sleep(0.1)  # Short delay for memory stats to stabilize

            # Compute actual free memory considering both allocated and reserved memory
            # This is more accurate than just looking at allocated memory
            free_memory = total_memory - allocated

            # Add information to the list
            gpu_info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": total_memory,
                    "free_memory_gb": free_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

            logger.info(
                f"GPU {i}: {props.name}, "
                f"Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total, "
                f"({allocated:.2f}GB allocated, {reserved:.2f}GB reserved)"
            )
    else:
        logger.info("No GPU available, using CPU")

    return gpu_info


def get_optimal_settings(
    gpu_info: List[GPUInfo], multi_gpu_threshold_gb: float = 8.0
) -> OptimalSettings:
    """
    Determine optimal model settings based on available GPU resources.

    Args:
        gpu_info: List of GPU information dictionaries
        multi_gpu_threshold_gb: Minimum free memory in GB required for a GPU to be usable

    Returns:
        OptimalSettings: Optimized configuration for model loading
    """
    logger = logging.getLogger("phi4_demo")
    settings: OptimalSettings = {
        "device_map": "cpu",  # Default to CPU if no suitable GPU
        "quantization": False,
        "precision": torch.float16,
        "max_new_tokens": 100,
        "parallel_processing": False,
        "multi_gpu": False,
        "available_gpus": [],
    }

    # Check if CUDA is available at all
    if not torch.cuda.is_available() or not gpu_info:
        logger.info("No suitable GPU available, falling back to CPU")
        settings["precision"] = torch.float32  # Use float32 on CPU
        return settings

    # Clear memory before making decisions
    clear_memory()

    # Find GPU with most free memory
    best_gpu = max(gpu_info, key=lambda x: x["free_memory_gb"])
    device_id = best_gpu["index"]
    best_gpu_memory = best_gpu["free_memory_gb"]

    # Set device_map based on available memory
    settings["device_map"] = f"cuda:{device_id}"
    logger.info(
        f"Using GPU {device_id} with {best_gpu_memory:.2f}GB free memory")

    # Check for multiple usable GPUs (with enough free memory)
    usable_gpus = [
        gpu["index"]
        for gpu in gpu_info
        if gpu["free_memory_gb"] >= multi_gpu_threshold_gb
    ]
    settings["available_gpus"] = usable_gpus

    # Enable multi-GPU mode if we have at least 2 usable GPUs
    if len(usable_gpus) > 1:
        settings["multi_gpu"] = True
        settings["parallel_processing"] = True
        logger.info(f"Multi-GPU mode enabled with GPUs: {usable_gpus}")

    # Set precision based on capability - use bfloat16 if available, otherwise float16
    bf16_capable = False
    for gpu in gpu_info:
        cc_major = int(gpu["compute_capability"].split(".")[0])
        cc_minor = int(gpu["compute_capability"].split(".")[1])
        if cc_major >= 8:  # Ampere or newer has BF16 support
            bf16_capable = True

    # BF16 has better numerical stability than FP16, use it if available
    if bf16_capable and torch.cuda.is_bf16_supported():
        settings["precision"] = torch.bfloat16
        logger.info("Using bfloat16 precision (better numerical stability)")
    else:
        settings["precision"] = torch.float16
        logger.info("Using float16 precision")

    # Enable quantization if we have limited memory (<12GB free)
    # But disable if we have very limited memory (<6GB) to avoid overhead
    if 6.0 <= best_gpu_memory < 12.0:
        settings["quantization"] = True
        logger.info("Enabling 8-bit quantization to save memory")
    else:
        settings["quantization"] = False

    # Adjust maximum tokens based on available memory using dynamic estimation
    settings["max_new_tokens"] = estimate_max_tokens(
        total_gpu_memory_gb=best_gpu["total_memory_gb"],
        model_weight_gb=14.0,  # Update if model size changes
        memory_per_token_kb=4.0,  # Adjust for precision/implementation
        batch_size=1,
        reserved_memory_gb=2.0
    )
    if settings["max_new_tokens"] <= 50:
        logger.warning("Very limited GPU memory, reducing max tokens to 50")

    logger.info(
        f"Optimized settings: max_new_tokens={settings['max_new_tokens']}, "
        f"precision={settings['precision']}, "
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
    logger = logging.getLogger("phi4_demo")

    # First, run Python garbage collection to free objects
    gc.collect()

    try:
        if torch.cuda.is_available():
            # Determine which devices to clean
            devices = []
            if device_id is not None:
                devices = [device_id]
            else:
                devices = list(range(torch.cuda.device_count()))

            for dev in devices:
                # Get memory stats before clearing
                before_alloc = torch.cuda.memory_allocated(dev) / (1024**3)
                before_reserved = torch.cuda.memory_reserved(dev) / (1024**3)

                # Clear memory
                with torch.cuda.device(f"cuda:{dev}"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(dev)

                # Get memory stats after clearing
                after_alloc = torch.cuda.memory_allocated(dev) / (1024**3)
                after_reserved = torch.cuda.memory_reserved(dev) / (1024**3)

                freed_memory = before_reserved - after_reserved
                logger.debug(
                    f"Cleared memory on GPU {dev}: "
                    f"freed {freed_memory:.2f}GB reserved memory, "
                    f"now using {after_alloc:.2f}GB allocated, {after_reserved:.2f}GB reserved"
                )

    except Exception as e:
        logger.warning(f"Error clearing GPU memory: {e}")


def force_release_memory() -> None:
    """
    Aggressively attempt to release memory when facing OOM issues.
    This should only be called when dealing with critical memory situations.
    """
    logger = logging.getLogger("phi4_demo")

    try:
        # Run multiple GC cycles
        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            # Try to report about fragmentation
            if hasattr(torch.cuda, "memory_stats"):
                for device in range(torch.cuda.device_count()):
                    try:
                        stats = torch.cuda.memory_stats(device)
                        if "allocated_bytes.all.peak" in stats:
                            peak_bytes = stats["allocated_bytes.all.peak"]
                            current_bytes = stats["allocated_bytes.all.current"]
                            logger.info(
                                f"GPU {device} - Peak allocation: {peak_bytes/(1024**3):.2f}GB, "
                                f"Current allocation: {current_bytes/(1024**3):.2f}GB"
                            )
                    except:
                        pass

            # Suggest to optimize memory allocation
            logger.info(
                "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce memory fragmentation"
            )

            # Try to optimize CUDA memory allocator behavior
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    except Exception as e:
        logger.warning(f"Error during aggressive memory cleanup: {e}")


def check_memory_requirements(
    expected_memory_gb: float, device: Union[str, int] = None
) -> bool:
    """
    Check if there's enough free memory for an operation requiring a specific amount.

    Args:
        expected_memory_gb: Expected memory requirement in GB
        device: Device to check, can be a device ID or full device string

    Returns:
        bool: True if enough memory is available, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # Handle different device parameter formats
    device_id = 0  # Default to first GPU
    if device is None:
        pass  # Use default
    elif isinstance(device, int):
        device_id = device
    elif isinstance(device, str) and device.startswith("cuda:"):
        device_id = int(device.split(":")[1])

    # Check if device ID is valid
    if device_id >= torch.cuda.device_count():
        return False

    # Get current free memory
    clear_memory(device_id)  # Clear memory first for more accurate reading

    total_memory = torch.cuda.get_device_properties(
        device_id).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
    free_memory = total_memory - allocated

    # Add some buffer because memory stats aren't perfectly accurate
    buffer_factor = (
        0.90  # Consider only 90% of reported free memory as actually available
    )

    return (free_memory * buffer_factor) >= expected_memory_gb


def estimate_max_tokens(
    total_gpu_memory_gb: float,
    model_weight_gb: float = 14.0,
    memory_per_token_kb: float = 4.0,
    batch_size: int = 1,
    reserved_memory_gb: float = 2.0
) -> int:
    """
    Estimate the maximum number of tokens that can fit in GPU memory.

    Args:
        total_gpu_memory_gb: Total GPU memory in GB
        model_weight_gb: Model weights size in GB
        memory_per_token_kb: Estimated memory per token in KB
        batch_size: Batch size for inference
        reserved_memory_gb: Memory reserved for system/other processes

    Returns:
        int: Estimated maximum number of tokens
    """
    # O(1) complexity: direct calculation
    available_memory_gb = total_gpu_memory_gb - model_weight_gb - reserved_memory_gb
    if available_memory_gb <= 0:
        return 50  # Minimum fallback
    available_memory_kb = available_memory_gb * 1024 * 1024
    max_tokens = int(available_memory_kb / (memory_per_token_kb * batch_size))
    # Clamp to reasonable limits (e.g., model's max context window)
    return min(max_tokens, 131072)
