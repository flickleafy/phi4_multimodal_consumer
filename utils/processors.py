"""Image and audio processing utilities."""

from typing import Any, Dict, List, Tuple, Optional, Union, cast
import torch
import numpy as np
from PIL import Image
import traceback
from .model_utils import suppress_checkpoint_warnings


def process_image(
    processor: Any,
    model: Any,
    text_prompt: str,
    image: Image.Image,
    max_new_tokens: int,
    generation_config: Any,
) -> str:
    """
    Process an image with the model and return the generated text.

    Args:
        processor: The model processor
        model: The loaded model
        text_prompt: The text prompt to use
        image: The input image
        max_new_tokens: Maximum number of tokens to generate
        generation_config: Generation configuration

    Returns:
        str: Generated text response
    """
    try:
        print("Processing image with model...")

        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
            print("Converted image to RGB mode")

        # Process inputs
        inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        print(f"Processor returned keys: {list(inputs.keys())}")

        # Move inputs to the model's device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

        # Generate with no grad
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
                use_cache=True,
            )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response

    except Exception as e:
        print(f"Error details: {traceback.format_exc()}")
        return f"Error in image processing: {str(e)}"


def process_audio(
    processor: Any,
    model: Any,
    text_prompt: str,
    audio_data: np.ndarray,
    sample_rate: int,
    max_new_tokens: int,
    generation_config: Any,
) -> str:
    """
    Process audio data with the model and return the generated text.

    Args:
        processor: The model processor
        model: The loaded model
        text_prompt: The text prompt to use
        audio_data: The input audio data
        sample_rate: The audio sample rate
        max_new_tokens: Maximum number of tokens to generate
        generation_config: Generation configuration

    Returns:
        str: Generated text response
    """
    try:
        print("Processing audio with model...")

        # Ensure audio data is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Use JIT settings to improve performance
        prev_profiling_executor = torch._C._jit_set_profiling_executor(False)
        prev_profiling_mode = torch._C._jit_set_profiling_mode(False)

        try:
            # Suppress specific checkpoint warnings
            with suppress_checkpoint_warnings():
                # Process inputs
                inputs = processor(
                    text=text_prompt,
                    audios=[(audio_data, sample_rate)],
                    return_tensors="pt",
                )

                print(f"Processor returned keys: {list(inputs.keys())}")

                # Move inputs to the model's device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

                # Generate with no grad
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        generation_config=generation_config,
                        use_cache=True,
                    )

                generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
                response = processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                return response

        finally:
            # Restore JIT settings
            torch._C._jit_set_profiling_executor(prev_profiling_executor)
            torch._C._jit_set_profiling_mode(prev_profiling_mode)

    except Exception as e:
        print(f"Error details: {traceback.format_exc()}")
        return f"Error in audio processing: {str(e)}"


def refine_transcription(
    processor: Any,
    model: Any,
    text: str,
    max_new_tokens: int,
    generation_config: Any,
    user_prompt: str,
    assistant_prompt: str,
    prompt_suffix: str,
) -> str:
    """
    Refine a transcription using a second pass with the model.

    Args:
        processor: The model processor
        model: The loaded model
        text: The text to refine
        max_new_tokens: Maximum number of tokens to generate
        generation_config: Generation configuration
        user_prompt: User prompt prefix
        assistant_prompt: Assistant prompt prefix
        prompt_suffix: Prompt suffix

    Returns:
        str: Refined text
    """
    refinement_prompt = (
        f"{user_prompt}Please add proper punctuation to this transcript "
        f"while preserving the original meaning. The transcript is from a "
        f'spoken lecture: "{text}"{prompt_suffix}{assistant_prompt}'
    )

    with torch.no_grad():
        inputs = processor(text=refinement_prompt, return_tensors="pt").to(model.device)
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
            use_cache=True,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        refined_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    return refined_text
