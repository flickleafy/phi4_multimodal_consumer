"""Model loading and configuration utilities."""

import os
import warnings
from typing import Optional, Union, Dict, Any, Tuple, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from contextlib import contextmanager
from .gpu_utils import OptimalSettings
from .long_context_fix import (
    configure_attention_kernels,
    configure_tokenizer_for_long_context,
    configure_model_cache_format,
    verify_long_context_configuration,
    print_sanity_check,
)

PHI4_CONTEXT_WINDOW = 131072


def configure_environment() -> None:
    """Configure environment variables and warning filters for optimal operation."""
    # Add environment variables for better performance
    os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable deterministic algorithms
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CHECKPOINT_REENTRANT"] = "0"

    # Filter specific warnings
    warnings.filterwarnings(
        "ignore",
        message="The image_processor_class argument is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Please specify CheckpointImpl.NO_REENTRANT as CheckpointImpl.*",
        module=".*",
    )


@contextmanager
def suppress_checkpoint_warnings():
    """Context manager to specifically suppress checkpoint gradient warnings."""
    default_filters = warnings.filters.copy()
    warnings.filterwarnings(
        "ignore",
        message="None of the inputs have requires_grad=True. Gradients will be None",
        module="torch.utils.checkpoint",
    )
    try:
        yield
    finally:
        warnings.filters = default_filters


class ModelLoader:
    """Class to handle loading and configuring the Phi-4 multimodal model."""

    def __init__(
        self,
        model_path: str,
        settings: OptimalSettings,
        specific_gpu: Optional[int] = None,
        disable_audio: bool = False,  # New parameter to disable audio components
        target_context_window: int = PHI4_CONTEXT_WINDOW,  # Support full 128K context
    ):
        """
        Initialize the model loader.

        Args:
            model_path: Path to the model
            settings: Optimization settings for model loading
            specific_gpu: Optional specific GPU to load the model on
            disable_audio: If True, disable audio components in the processor
            target_context_window: Target context window size (default: PHI4_CONTEXT_WINDOW)
        """
        self.model_path = model_path
        self.settings = settings
        self.specific_gpu = specific_gpu
        self.disable_audio = disable_audio
        self.target_context_window = target_context_window
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.generation_config = None

    def load(self) -> Tuple[Any, Any, Any]:
        """
        Load the model, processor and generation config with long context support.

        Returns:
            Tuple containing (model, processor, generation_config)
        """
        # Configure attention kernels BEFORE loading model
        configure_attention_kernels()

        # Configure device map based on specific GPU if provided
        device_map = self.settings["device_map"]
        if self.specific_gpu is not None:
            device_map = f"cuda:{self.specific_gpu}"
            print(f"Loading model on specific GPU: {device_map}")

        # Configure quantization if needed
        quantization_config = None
        if self.settings["quantization"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.settings["precision"],
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Load the tokenizer separately and configure for long context
        print(
            f"Loading tokenizer with {self.target_context_window} token context window...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,  # Model doesn't have a fast tokenizer
        )
        self.tokenizer = configure_tokenizer_for_long_context(
            self.tokenizer, self.target_context_window
        )

        # Load the processor
        print(f"Loading processor with device_map={device_map}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,  # Model doesn't have a fast processor
        )

        # Use the configured tokenizer in the processor
        self.processor.tokenizer = self.tokenizer

        # Explicitly disable audio components if requested
        # TODO: check implementation, this never happens
        if self.disable_audio:
            # Override specific audio attributes in the processor
            if hasattr(self.processor, "feature_extractor_audio"):
                # Keep reference but disable functionality
                original_fe = self.processor.feature_extractor_audio
                self.processor.feature_extractor_audio = None
                print("✓ Disabled audio feature extractor")

            # Override the processing method to skip audio processing
            if hasattr(self.processor, "_process_audio"):
                original_process_audio = self.processor._process_audio

                # Create a stub function that does nothing
                def disabled_process_audio(*args, **kwargs):
                    return None

                # Replace with our stub
                self.processor._process_audio = disabled_process_audio
                print("✓ Disabled audio processing method")

            # Set a flag to indicate audio is disabled
            self.processor._audio_disabled = True

        # Try loading the model with fallbacks
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                torch_dtype=self.settings["precision"],
                trust_remote_code=True,
                quantization_config=None,  # Force no quantization for LoRA compatibility
                _attn_implementation="eager",  # Use eager attention for long context compatibility
                low_cpu_mem_usage=True,
            )
        except ValueError as e:
            print(f"Error loading model with {device_map}: {e}")
            print("Falling back to other options...")
            self._try_fallback_options()

        # Set model to evaluation mode
        self.model.eval()

        # Configure model cache format to avoid deprecation warnings
        configure_model_cache_format(self.model)

        # Load generation config and set max tokens
        self.generation_config = GenerationConfig.from_pretrained(
            self.model_path)
        self.generation_config.max_length = self.settings["max_new_tokens"] + 50

        # Print device info and configuration sanity check
        model_device = next(self.model.parameters()).device
        print(f"Model loaded on: {model_device}")

        # Verify configuration is correct
        verify_long_context_configuration(self.model, self.tokenizer)

        # Print detailed sanity check
        print_sanity_check(self.model, self.tokenizer)

        return self.model, self.processor, self.generation_config

    def _load_model_with_fallbacks(
        self, quantization_config: Optional[BitsAndBytesConfig]
    ) -> None:
        """
        Load model with fallback options if primary loading fails.

        Args:
            quantization_config: Configuration for quantization, if any
        """
        try:
            # First attempt with eager attention for long context support
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.settings["device_map"],
                torch_dtype=self.settings["precision"],
                trust_remote_code=True,
                quantization_config=None,  # Force no quantization for LoRA compatibility
                _attn_implementation="eager",  # Use eager attention for long context compatibility
                low_cpu_mem_usage=True,
            )
        except ValueError as e:
            print(
                f"Error loading model with {self.settings['device_map']}: {e}")
            print("Falling back to single GPU mode...")

            self._try_fallback_options()

    def _try_fallback_options(self) -> None:
        """Try fallback options for loading the model."""
        if torch.cuda.is_available():
            try:
                # Second attempt: Force to first GPU with eager attention
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="cuda:0",
                    torch_dtype=self.settings["precision"],
                    trust_remote_code=True,
                    quantization_config=None,
                    _attn_implementation="eager",  # Use eager attention for long context compatibility
                    low_cpu_mem_usage=True,
                )
            except Exception as e2:
                print(f"Error with fallback approach: {e2}")
                self._load_cpu_model()
        else:
            self._load_cpu_model()

    def _load_cpu_model(self) -> None:
        """Load model on CPU as last resort."""
        print("Loading to CPU (will be slow)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="cpu",
            torch_dtype=torch.float32,  # CPU works better with float32
            trust_remote_code=True,
            _attn_implementation="eager",  # Use eager attention for long context compatibility
            low_cpu_mem_usage=True,
        )
