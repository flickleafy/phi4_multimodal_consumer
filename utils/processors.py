"""Image and audio processing utilities."""

from typing import Any, Dict, List, Tuple, Optional, Union, cast
import gc
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

        # Pre-processing memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Resize image if it's too large (helps with memory consumption)
        # max_dim = 300
        # if max(image.width, image.height) > max_dim:
        #     print(
        #         f"Resizing image from {image.width}x{image.height} to max dimension {max_dim}"
        #     )
        #     if image.width > image.height:
        #         new_width = max_dim
        #         new_height = int(image.height * (max_dim / image.width))
        #     else:
        #         new_height = max_dim
        #         new_width = int(image.width * (max_dim / image.height))
        #     image = image.resize((new_width, new_height), Image.LANCZOS)
        #     print(f"Image resized to {new_width}x{new_height}")

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

        # Free up memory from any unused objects
        del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use memory-efficient generation settings
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "attention_mask": inputs.get("attention_mask", None),
        }

        # Only use few beams if we have enough memory
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory > 14 * 1024**3
        ):
            generation_kwargs["num_beams"] = 1  # No beam search to save memory

        # Generate with no grad and careful memory management
        with torch.no_grad():
            try:
                # Try with custom generation config
                generate_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                    generation_config=generation_config,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(
                        "First attempt failed due to OOM, trying with reduced settings..."
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Reduce settings further for second attempt
                    generation_kwargs["max_new_tokens"] = min(max_new_tokens, 50)
                    generation_kwargs["num_beams"] = 1
                    generation_kwargs["do_sample"] = False

                    # Try again with reduced settings
                    generate_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                else:
                    raise

        # Process output
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Final cleanup
        del inputs
        del generate_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    except Exception as e:
        print(f"Error details: {traceback.format_exc()}")
        # Try to reclaim memory on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

        # Pre-processing memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

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

                # Free memory
                del audio_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Generate with no grad and careful memory management
                with torch.no_grad():
                    try:
                        # Use memory-efficient generation settings
                        generation_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "use_cache": True,
                            "pad_token_id": processor.tokenizer.pad_token_id,
                            "attention_mask": inputs.get("attention_mask", None),
                            "num_beams": 1,  # No beam search to save memory
                        }

                        # Generate response
                        generate_ids = model.generate(
                            **inputs,
                            **generation_kwargs,
                            generation_config=generation_config,
                        )
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(
                                "First attempt failed due to OOM, trying with reduced settings..."
                            )
                            torch.cuda.empty_cache()
                            gc.collect()

                            # Reduce settings further for second attempt
                            generation_kwargs["max_new_tokens"] = min(
                                max_new_tokens, 50
                            )

                            # Try again with reduced settings
                            generate_ids = model.generate(
                                **inputs,
                                **generation_kwargs,
                            )
                        else:
                            raise

                # Process output
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
                response = processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Final cleanup
                del inputs
                del generate_ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return response

        finally:
            # Restore JIT settings
            torch._C._jit_set_profiling_executor(prev_profiling_executor)
            torch._C._jit_set_profiling_mode(prev_profiling_mode)

            # Additional cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error details: {traceback.format_exc()}")
        # Try to reclaim memory on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    refinement_instruction: str,
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
        refinement_instruction: The instruction for refining the text

    Returns:
        str: Refined text
    """
    try:
        # Pre-processing memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        refinement_prompt = f'{user_prompt}{refinement_instruction}"{text}"{prompt_suffix}{assistant_prompt}'

        with torch.no_grad():
            inputs = processor(text=refinement_prompt, return_tensors="pt")

            # Move inputs to the model's device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

            # Memory-efficient generation settings
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "num_beams": 1,  # No beam search to save memory
                "pad_token_id": processor.tokenizer.pad_token_id,
                "attention_mask": inputs.get("attention_mask", None),
            }

            try:
                generate_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                    generation_config=generation_config,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(
                        "Refinement attempt failed due to OOM, trying with reduced settings..."
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Reduce settings for second attempt
                    generation_kwargs["max_new_tokens"] = min(max_new_tokens, 50)

                    generate_ids = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                else:
                    raise

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            refined_text = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        # Final cleanup
        del inputs
        del generate_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return refined_text

    except Exception as e:
        print(f"Error in refining transcription: {traceback.format_exc()}")
        # Try to reclaim memory on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text  # Return original text on error
