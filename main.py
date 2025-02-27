"""
Main script for demonstrating the Phi-4 multimodal model on images and audio.

This script shows how to use the Phi-4 multimodal model to:
1. Process images and generate descriptions
2. Transcribe audio with proper punctuation
"""

import io
import os
import concurrent.futures
from threading import Lock
from PIL import Image
import numpy as np
import time

# Import our utility modules
from utils.gpu_utils import get_gpu_info, get_optimal_settings, clear_memory
from utils.model_utils import configure_environment, ModelLoader
from utils.processors import process_image, process_audio, refine_transcription
from utils.file_utils import get_image, get_audio

# Configure environment (warnings, variables)
configure_environment()

# Clear GPU memory before starting
clear_memory()

# Define prompt structure
USER_PROMPT = "<|user|>"
ASSISTANT_PROMPT = "<|assistant|>"
PROMPT_SUFFIX = "<|end|>"

# Model information
MODEL_PATH = "Phi-4-multimodal-instruct"

# File paths
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_files")
IMAGE_URL = "https://www.ilankelman.org/stopsigns/australia.jpg"
AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"

# Global lock for print statements to avoid interleaved output in multithreaded execution
print_lock = Lock()


def synchronized_print(*args, **kwargs):
    """Thread-safe print function to avoid interleaved output."""
    with print_lock:
        print(*args, **kwargs)


def run_image_demo(model, processor, generation_config, settings, task_id=""):
    """Run the image processing demonstration."""
    synchronized_print(f"\n--- IMAGE PROCESSING {task_id}---")
    prompt = (
        f"{USER_PROMPT}<|image_1|>What is shown in this image?"
        f"{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    )
    synchronized_print(f">>> Prompt\n{prompt}")

    try:
        # Get the image file (downloads only if not cached)
        image = get_image(IMAGE_URL, CACHE_DIR)

        # Process with the model
        response = process_image(
            processor,
            model,
            prompt,
            image,
            settings["max_new_tokens"],
            generation_config,
        )
        synchronized_print(f">>> Response {task_id}\n{response}")
        return response
    except Exception as e:
        synchronized_print(f"Error in image processing {task_id}: {e}")
        return None


def run_audio_demo(model, processor, generation_config, settings, task_id=""):
    """Run the audio processing demonstration."""
    synchronized_print(f"\n--- AUDIO PROCESSING {task_id}---")
    try:
        # Create detailed prompt for natural speech patterns
        speech_prompt = """Transcribe the audio to text, paying special attention to:
1. Natural pauses in speech (use commas, periods, or other appropriate punctuation)
2. Changes in tone and intonation (questions, exclamations)
3. Emphasis and rhythm of the speaker's delivery
4. Natural paragraph breaks where topics shift
5. Maintain speaker's original cadence while ensuring readability

Format as a professional transcript that preserves the speaker's natural speech patterns."""

        prompt = (
            f"{USER_PROMPT}<|audio_1|>{speech_prompt}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
        )
        synchronized_print(f">>> Prompt\n{prompt}")

        # Get the audio file (downloads only if not cached)
        audio, samplerate = get_audio(AUDIO_URL, CACHE_DIR)

        # Process with the model
        response = process_audio(
            processor,
            model,
            prompt,
            audio,
            samplerate,
            settings["max_new_tokens"],
            generation_config,
        )
        synchronized_print(f">>> Response {task_id}\n{response}")

        # If results lack proper punctuation, do a second pass
        if len(response.split(".")) < 3:  # Heuristic for poor punctuation
            synchronized_print(
                f"\nPerforming second pass to improve transcription quality {task_id}..."
            )

            refined_response = refine_transcription(
                processor,
                model,
                response,
                settings["max_new_tokens"],
                generation_config,
                USER_PROMPT,
                ASSISTANT_PROMPT,
                PROMPT_SUFFIX,
            )

            synchronized_print(f">>> Refined Response {task_id}\n{refined_response}")
            return refined_response

        return response

    except Exception as e:
        synchronized_print(f"Error in audio processing {task_id}: {e}")
        return None


def run_sequential_processing(model, processor, generation_config, settings):
    """Run image and audio processing sequentially on a single GPU."""
    # Run image processing demo
    run_image_demo(model, processor, generation_config, settings)

    # Clear memory between tasks
    clear_memory()

    # Run audio processing demo
    run_audio_demo(model, processor, generation_config, settings)


def run_parallel_processing(settings):
    """Run image and audio processing in parallel on multiple GPUs."""
    # Use the first two GPUs from the available list
    image_gpu = settings["available_gpus"][0]
    audio_gpu = settings["available_gpus"][1]

    synchronized_print(
        f"Running parallel processing: Image on GPU {image_gpu}, Audio on GPU {audio_gpu}"
    )

    # Create two separate model loaders for different GPUs
    image_loader = ModelLoader(MODEL_PATH, settings, specific_gpu=image_gpu)
    audio_loader = ModelLoader(MODEL_PATH, settings, specific_gpu=audio_gpu)

    # Load models on separate GPUs
    image_model, image_processor, image_gen_config = image_loader.load()
    audio_model, audio_processor, audio_gen_config = audio_loader.load()

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit tasks
        image_future = executor.submit(
            run_image_demo,
            image_model,
            image_processor,
            image_gen_config,
            settings,
            f"(GPU {image_gpu})",
        )

        audio_future = executor.submit(
            run_audio_demo,
            audio_model,
            audio_processor,
            audio_gen_config,
            settings,
            f"(GPU {audio_gpu})",
        )

        # Wait for completion and get results
        image_result = image_future.result()
        audio_result = audio_future.result()

    # Clear memory on both GPUs
    clear_memory(image_gpu)
    clear_memory(audio_gpu)

    return image_result, audio_result


def ensure_directory_exists(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_result_to_file(result: str, filename: str, description: str) -> None:
    """
    Save a processing result to a file in the results directory.

    Args:
        result: The text content to save
        filename: The name of the file to save to
        description: Description of the content for the file header
    """
    if result:
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        ensure_directory_exists(result_dir)
        filepath = os.path.join(result_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"{description}:\n{result}")

        synchronized_print(f"Result saved to results/{filename}")


def main():
    """Main function to run the demonstration."""
    # Get GPU information
    gpu_info = get_gpu_info()

    # Get optimal settings based on GPU availability
    settings = get_optimal_settings(gpu_info)

    try:
        if settings["multi_gpu"] and len(settings["available_gpus"]) > 1:
            # Run parallel processing if multiple GPUs with enough memory are available
            synchronized_print("Using multiple GPUs for parallel processing")
            start_time = time.time()
            image_result, audio_result = run_parallel_processing(settings)
            end_time = time.time()

            # Report on the results and performance gain
            synchronized_print(
                f"Parallel processing completed in {end_time - start_time:.2f} seconds"
            )

            # Save results using the utility function
            save_result_to_file(
                image_result, "image_analysis.txt", "Image Analysis Result"
            )
            save_result_to_file(
                audio_result, "audio_transcript.txt", "Audio Transcript Result"
            )

        else:
            # Fall back to sequential processing on a single GPU
            synchronized_print("Using single GPU for sequential processing")
            start_time = time.time()
            loader = ModelLoader(MODEL_PATH, settings)
            model, processor, generation_config = loader.load()
            run_sequential_processing(model, processor, generation_config, settings)
            end_time = time.time()
            synchronized_print(
                f"Sequential processing completed in {end_time - start_time:.2f} seconds"
            )

    finally:
        # Final memory cleanup
        clear_memory()
        print("\nDemo completed.")


if __name__ == "__main__":
    main()
