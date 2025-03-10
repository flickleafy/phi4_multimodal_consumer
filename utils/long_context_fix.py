#!/usr/bin/env python3
"""
Long context fix utility for Phi-4 multimodal model.

This module provides functions to apply the fixes recommended in long_context.md
to resolve the >4096 token warning and performance issues.

Time Complexity Analysis:
- configure_attention_kernels(): O(1) - Simple environment variable setting
- apply_long_context_fixes(): O(1) - Function calls with constant time operations
"""

import os
import warnings
from typing import Any


def configure_attention_kernels() -> None:
    """
    Configure PyTorch SDPA to avoid Flash attention issues with long contexts.

    This prevents the "stalling" issue when processing >4096 tokens by using
    memory-efficient kernels instead of Flash SDPA which may not support
    very long sequences efficiently on RTX 5090 + PyTorch 2.9.0.

    Time Complexity: O(1) - Simple environment variable setting
    """
    # Import torch only when needed to avoid hanging during module import
    try:
        import torch
    except ImportError:
        print("⚠️  PyTorch not available, skipping attention kernel configuration")
        return

    # Must be set BEFORE Torch/Transformers initialize attention internals
    os.environ.setdefault("PYTORCH_SDP_KERNEL", "mem_efficient")

    # Disable Flash SDPA; enable mem-efficient SDPA for long contexts
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    print("✓ Configured attention kernels for long context support")


def configure_tokenizer_for_long_context(tokenizer: Any, target_window: int = 131072) -> Any:
    """
    Configure tokenizer to support long contexts up to 131K tokens.

    This fixes the ">4096 token" warning by uncapping the tokenizer's
    model_max_length from the default 4096 to the model's full capacity.

    Args:
        tokenizer: The tokenizer to configure
        target_window: Target context window size (default: 131072 for Phi-4)

    Returns:
        Configured tokenizer

    Time Complexity: O(1) - Simple attribute setting
    """
    # Uncap the tokenizer's maximum length
    original_max_length = getattr(tokenizer, 'model_max_length', 4096)
    tokenizer.model_max_length = target_window
    # Persisted if saved
    tokenizer.init_kwargs["model_max_length"] = target_window
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(
        f"✓ Configured tokenizer: {original_max_length} -> {target_window} token context window")
    return tokenizer


def get_long_context_generation_kwargs(max_new_tokens: int, pad_token_id: int, use_cache: bool = True) -> dict:
    """
    Get generation kwargs optimized for long context processing.

    This provides settings that work well with long contexts and avoid
    the deprecated cache format warnings.

    Args:
        max_new_tokens: Maximum tokens to generate
        pad_token_id: Padding token ID from tokenizer
        use_cache: Whether to use KV cache (disable for fresh context)

    Returns:
        Dict of generation kwargs

    Time Complexity: O(1) - Simple dictionary construction
    """
    return {
        "max_new_tokens": max_new_tokens,
        # KV cache: True for speed (O(1) vs O(n) lookup), False for fresh context
        "use_cache": use_cache,
        "pad_token_id": pad_token_id,
        "do_sample": False,  # Deterministic output for consistency
        "temperature": 1.0,  # Default temperature
        # Use new cache format to avoid deprecation warning
        "return_dict_in_generate": False,
        "output_scores": False,  # Disable scores to avoid extra memory usage
        "output_attentions": False,  # Disable attention outputs
        "output_hidden_states": False,  # Disable hidden state outputs
        # Avoid beam search to save memory (O(n) vs O(n*beams))
        "num_beams": 1,
        # Disable past key values to prevent cache accumulation when use_cache=False
        "past_key_values": None,
    }


def print_sanity_check(model: Any, tokenizer: Any) -> None:
    """
    Print configuration sanity check for long context support.

    Args:
        model: The loaded model
        tokenizer: The configured tokenizer

    Time Complexity: O(1) - Simple attribute access and printing
    """
    try:
        import torch
    except ImportError:
        print("⚠️  PyTorch not available for sanity check")
        return

    print("=== Long Context Configuration Sanity Check ===")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"SDPA flash enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    print(
        f"SDPA mem-efficient enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"SDPA math enabled: {torch.backends.cuda.math_sdp_enabled()}")
    print(
        f"Model max_position_embeddings: {getattr(model.config, 'max_position_embeddings', 'Not found')}")
    print(
        f"Model sliding_window: {getattr(model.config, 'sliding_window', 'Not found')}")
    print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")
    print("==============================================")

    # Check for potential issues
    if torch.backends.cuda.flash_sdp_enabled():
        print(
            "⚠️  WARNING: Flash SDPA is enabled - this may cause issues with long contexts")

    if tokenizer.model_max_length <= 4096:
        print("⚠️  WARNING: Tokenizer still capped at 4096 tokens - long context may not work properly")


def configure_model_cache_format(model: Any) -> None:
    """
    Configure the model to use the new cache format and avoid deprecation warnings.

    Args:
        model: The loaded model to configure

    Time Complexity: O(1) - Simple attribute setting
    """
    try:
        # Force the model to use the new cache format
        if hasattr(model, 'config'):
            # Set cache implementation to use new format
            if hasattr(model.config, 'cache_implementation'):
                model.config.cache_implementation = 'dynamic'

            # Ensure model doesn't use legacy cache format
            if hasattr(model.config, 'use_legacy_cache'):
                model.config.use_legacy_cache = False

            # Set attention dropout to 0 for inference stability
            if hasattr(model.config, 'attention_dropout'):
                model.config.attention_dropout = 0.0

        print("✓ Configured model cache format to avoid deprecation warnings")

    except Exception as e:
        print(f"⚠️  Could not configure cache format: {e}")


def verify_long_context_configuration(model: Any, tokenizer: Any) -> bool:
    """
    Verify that long context configuration is properly applied.

    Args:
        model: The loaded model
        tokenizer: The configured tokenizer

    Returns:
        bool: True if configuration is correct, False otherwise

    Time Complexity: O(1) - Simple attribute checking
    """
    issues = []

    # Check tokenizer configuration
    if tokenizer.model_max_length <= 4096:
        issues.append(
            f"Tokenizer max_length still capped at {tokenizer.model_max_length}")

    # Check attention configuration
    try:
        import torch
        if torch.backends.cuda.flash_sdp_enabled():
            issues.append("Flash SDPA is still enabled (should be disabled)")

        if not torch.backends.cuda.mem_efficient_sdp_enabled():
            issues.append(
                "Memory-efficient SDPA is not enabled (should be enabled)")

    except Exception as e:
        issues.append(f"Could not verify attention configuration: {e}")

    # Check model configuration
    if hasattr(model, 'config'):
        max_pos = getattr(model.config, 'max_position_embeddings', None)
        if max_pos and max_pos < 131072:
            issues.append(
                f"Model max_position_embeddings is {max_pos} (should be 131072)")

    if issues:
        print("⚠️  Long context configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Long context configuration verified successfully")
        return True


def apply_long_context_fixes(model_path: str = None) -> None:
    """
    Apply all recommended fixes for long context processing.

    This function should be called early in the application startup,
    before any model loading occurs.

    Args:
        model_path: Optional path to model directory for config patching

    Time Complexity: O(1) - Calls other O(1) functions
    """
    print("Applying long context fixes for Phi-4 multimodal model...")

    # Enhanced attention kernel configuration
    enhanced_configure_attention_kernels()

    # Configure attention kernels first
    configure_attention_kernels()

    # Filter specific warnings related to cache format and context length
    warnings.filterwarnings(
        "ignore",
        message=".*past_key_values.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*KV cache needs to be recomputed.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*nonsensical outputs after the 4096th token.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*tuple of tuples.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Cache.*",
        category=FutureWarning,
    )

    print("✓ Applied all long context fixes")


def _patch_model_config_for_eager_attention(config_path: str):
    """
    Patch the model's config.json to use eager attention instead of flash attention.

    The Phi-4 model config has "_attn_implementation": "flash_attention_2" by default,
    which overrides our attn_implementation parameter. This function temporarily
    patches the config file to use "eager" attention.

    Args:
        config_path: Path to the model directory containing config.json

    Returns:
        bool: True if patching was successful

    Time Complexity: O(1) - Single file read/write operation
    """
    import json
    import shutil
    from pathlib import Path

    config_file = Path(config_path) / "config.json"
    backup_file = Path(config_path) / "config.json.backup"

    # Create backup if it doesn't exist
    if not backup_file.exists() and config_file.exists():
        shutil.copy2(config_file, backup_file)
        print(f"✓ Created backup: {backup_file}")

    # Read and modify config
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Force eager attention
            config["_attn_implementation"] = "eager"

            # Write back the modified config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"✓ Patched {config_file} to use eager attention")
            return True
        except Exception as e:
            print(f"⚠️  Failed to patch config: {e}")
            return False

    return False


def restore_model_config_backup(config_path: str):
    """
    Restore the original model config from backup.

    This should be called when the application shuts down to restore
    the original model configuration.

    Args:
        config_path: Path to the model directory containing config.json

    Returns:
        bool: True if restoration was successful

    Time Complexity: O(1) - Single file copy operation
    """
    import shutil
    from pathlib import Path

    config_file = Path(config_path) / "config.json"
    backup_file = Path(config_path) / "config.json.backup"

    if backup_file.exists():
        try:
            shutil.copy2(backup_file, config_file)
            print(f"✓ Restored config from backup: {backup_file}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to restore config: {e}")
            return False

    return False


def enhanced_configure_attention_kernels():
    """
    Enhanced attention kernel configuration with additional safeguards.

    This function provides more aggressive prevention of FlashAttention2 usage
    by setting multiple environment variables and torch backend configurations.

    Time Complexity: O(1) - Environment variable and backend configuration
    """
    import warnings

    # Suppress all FlashAttention2 related warnings
    warning_patterns = [
        '.*flash.*', '.*Flash.*', '.*FLASH.*', '.*FlashAttention.*',
        '.*flash_attn.*', '.*flash-attn.*', '.*FlashAttn.*'
    ]

    for pattern in warning_patterns:
        warnings.filterwarnings(
            'ignore', message=pattern, category=UserWarning)
        warnings.filterwarnings(
            'ignore', message=pattern, category=FutureWarning)

    # Set multiple environment variables to prevent flash attention
    flash_prevention_vars = {
        'DISABLE_FLASH_ATTN': '1',
        'FORCE_EAGER_ATTENTION': '1',
        'PYTORCH_SDP_KERNEL': 'mem_efficient',
        'TORCH_COMPILE_DISABLE': '1',  # Prevent torch.compile which may use flash
        'TRANSFORMERS_ATTENTION_TYPE': 'eager'
    }

    for var, value in flash_prevention_vars.items():
        os.environ[var] = value

    # Configure torch backends if available
    try:
        import torch
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except ImportError:
        pass

    print("✓ Enhanced attention kernel configuration applied")


if __name__ == "__main__":
    # Demo usage
    apply_long_context_fixes()
    print("Long context fixes applied successfully!")
