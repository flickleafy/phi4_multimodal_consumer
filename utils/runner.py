"""Model runner for the Phi-4 multimodal demo."""

import os
import time
import concurrent.futures
from threading import Lock
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
from PIL import Image
import numpy as np

from utils.config_utils import AppConfig
from utils.gpu_utils import get_gpu_info, get_optimal_settings, clear_memory
from utils.model_utils import configure_environment, ModelLoader
from utils.processors import process_image, process_audio, refine_transcription
from utils.file_utils import get_image, get_audio, save_result_to_file
from utils.logging_utils import set_task_context, setup_logging

# Global lock for thread-safe operations
print_lock = Lock()


class ModelRunner:
    """Main class for running the Phi-4 multimodal demonstrations."""

    def __init__(self, config: AppConfig):
        """Initialize with configuration settings."""
        self.config = config

        # Set up logging
        self.logger = setup_logging(debug=config.debug)

        self.gpu_info = []
        self.settings = {}

        # Configure environment early
        configure_environment()
        clear_memory()

        # Get GPU information
        if not self.config.force_cpu:
            self.gpu_info = get_gpu_info()
            self.settings = get_optimal_settings(
                self.gpu_info, multi_gpu_threshold_gb=self.config.multi_gpu_threshold_gb
            )

            # Override parallel processing if requested
            if self.config.disable_parallel:
                self.settings["parallel_processing"] = False
                self.settings["multi_gpu"] = False
        else:
            self.logger.info("Forced CPU mode, no GPU will be used")
            self.settings = {
                "device_map": "cpu",
                "quantization": False,
                "precision": torch.float32,
                "max_new_tokens": 100,
                "parallel_processing": False,
                "multi_gpu": False,
                "available_gpus": [],
            }

    def synchronized_print(self, *args, **kwargs):
        """Thread-safe print function with task context."""
        with print_lock:
            print(*args, **kwargs)

    def run_image_demo(self, model, processor, generation_config, settings, task_id=""):
        """Run the image processing demonstration."""
        # Set task context for this thread
        set_task_context(task_id or "image")

        self.logger.info(f"Starting image processing")

        prompt = (
            f"{self.config.user_prompt}<|image_1|>{self.config.image_prompt}"
            f"{self.config.prompt_suffix}{self.config.assistant_prompt}"
        )
        self.logger.info(f"Using prompt: {prompt}")

        try:
            # Get the image
            image = get_image(self.config.image_url, self.config.cache_dir)

            # Process with the model
            response = process_image(
                processor,
                model,
                prompt,
                image,
                settings["max_new_tokens"],
                generation_config,
            )

            self.logger.info(f"Image processing response: {response}")
            return response

        except Exception as e:
            self.logger.exception(f"Error in image processing")
            return None

    def run_audio_demo(self, model, processor, generation_config, settings, task_id=""):
        """Run the audio processing demonstration."""
        # Set task context for this thread
        set_task_context(task_id or "audio")

        self.logger.info(f"Starting audio processing")

        try:
            # Use the speech prompt from config instead of a hardcoded one
            prompt = (
                f"{self.config.user_prompt}<|audio_1|>{self.config.speech_prompt}"
                f"{self.config.prompt_suffix}{self.config.assistant_prompt}"
            )
            self.logger.debug(f"Using prompt: {prompt}")

            # Get the audio file
            audio, samplerate = get_audio(self.config.audio_url, self.config.cache_dir)

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

            self.logger.info(f"Audio processing initial response: {response}")

            # Evaluate need for refinement
            if len(response.split(".")) < 3:  # Heuristic for poor punctuation
                self.logger.info(
                    "Performing second pass to improve transcription quality"
                )

                refined_response = refine_transcription(
                    processor,
                    model,
                    response,
                    settings["max_new_tokens"],
                    generation_config,
                    self.config.user_prompt,
                    self.config.assistant_prompt,
                    self.config.prompt_suffix,
                    self.config.refinement_instruction,
                )

                self.logger.info(f"Refined audio response: {refined_response}")
                return refined_response

            return response

        except Exception as e:
            self.logger.exception(f"Error in audio processing")
            return None

    def run_sequential_processing(self, model, processor, generation_config, settings):
        """Run image and audio processing sequentially on a single device."""
        self.logger.info("Starting sequential processing")

        # Process image with task context
        set_task_context("image")
        image_result = self.run_image_demo(
            model, processor, generation_config, settings
        )

        # Clear memory between tasks
        clear_memory()

        # Process audio with task context
        set_task_context("audio")
        audio_result = self.run_audio_demo(
            model, processor, generation_config, settings
        )

        # Save results using the function from file_utils.py with source file information
        save_result_to_file(
            image_result,
            "image_analysis.txt",
            "Image Analysis Result",
            self.config.results_dir,
            source_file=self.config.image_url,
        )

        save_result_to_file(
            audio_result,
            "audio_transcript.txt",
            "Audio Transcript Result",
            self.config.results_dir,
            source_file=self.config.audio_url,
        )

        return image_result, audio_result

    def run_parallel_processing(self, settings):
        """Run image and audio processing in parallel on multiple GPUs."""
        if len(settings["available_gpus"]) < 2:
            self.logger.warning("Not enough GPUs available for parallel processing")
            return None, None

        image_gpu = settings["available_gpus"][0]
        audio_gpu = settings["available_gpus"][1]

        self.logger.info(
            f"Running parallel processing: Image on GPU {image_gpu}, Audio on GPU {audio_gpu}"
        )

        # Create separate model loaders for different GPUs
        image_loader = ModelLoader(
            self.config.model_path, settings, specific_gpu=image_gpu
        )
        audio_loader = ModelLoader(
            self.config.model_path, settings, specific_gpu=audio_gpu
        )

        # Define worker functions that set up task context
        def worker_image():
            set_task_context(f"image_gpu{image_gpu}")
            return self.run_image_demo(
                image_model,
                image_processor,
                image_gen_config,
                settings,
                f"GPU{image_gpu}",
            )

        def worker_audio():
            set_task_context(f"audio_gpu{audio_gpu}")
            return self.run_audio_demo(
                audio_model,
                audio_processor,
                audio_gen_config,
                settings,
                f"GPU{audio_gpu}",
            )

        try:
            # Load models on separate GPUs
            image_model, image_processor, image_gen_config = image_loader.load()
            audio_model, audio_processor, audio_gen_config = audio_loader.load()

            # Use ThreadPoolExecutor for parallel execution with custom workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit tasks
                image_future = executor.submit(worker_image)
                audio_future = executor.submit(worker_audio)

                # Wait for completion and get results
                image_result = image_future.result()
                audio_result = audio_future.result()

            # Clear task context
            set_task_context("")

            # Save results using the function from file_utils.py
            img_path = save_result_to_file(
                image_result,
                "image_analysis.txt",
                f"Image Analysis Result (GPU {image_gpu})",
                self.config.results_dir,
                f"gpu{image_gpu}",
            )

            audio_path = save_result_to_file(
                audio_result,
                "audio_transcript.txt",
                f"Audio Transcript Result (GPU {audio_gpu})",
                self.config.results_dir,
                f"gpu{audio_gpu}",
            )

            self.logger.info(f"Results saved to {img_path} and {audio_path}")
            return image_result, audio_result

        finally:
            # Clear memory on both GPUs
            clear_memory(image_gpu)
            clear_memory(audio_gpu)

            # Clear task context
            set_task_context("")

    def run(self):
        """Main entry point to run the demo."""
        run_id = f"run_{int(time.time())}"
        set_task_context(run_id)
        self.logger.info(f"Starting Phi-4 multimodal demo ({run_id})")

        try:
            # Determine whether to use parallel or sequential processing
            if (
                self.settings.get("multi_gpu", False)
                and len(self.settings.get("available_gpus", [])) > 1
                and not self.config.disable_parallel
            ):
                # Run parallel processing across multiple GPUs
                self.logger.info("Using multiple GPUs for parallel processing")
                start_time = time.time()
                results = self.run_parallel_processing(self.settings)
                end_time = time.time()

                if results[0] is not None:  # At least one result was successful
                    self.logger.info(
                        f"Parallel processing completed in {end_time - start_time:.2f} seconds"
                    )
                else:
                    self.logger.warning(
                        "Parallel processing failed, falling back to sequential"
                    )
                    # Fall back to sequential processing
                    start_time = time.time()

                    # Load model
                    loader = ModelLoader(self.config.model_path, self.settings)
                    model, processor, generation_config = loader.load()

                    # Run sequential processing
                    self.run_sequential_processing(
                        model, processor, generation_config, self.settings
                    )

                    end_time = time.time()
                    self.logger.info(
                        f"Sequential processing completed in {end_time - start_time:.2f} seconds"
                    )
            else:
                # Run sequential processing on a single device
                self.logger.info(
                    f"Using sequential processing on {self.settings['device_map']}"
                )
                start_time = time.time()

                # Load model
                loader = ModelLoader(self.config.model_path, self.settings)
                model, processor, generation_config = loader.load()

                # Run sequential processing
                self.run_sequential_processing(
                    model, processor, generation_config, self.settings
                )

                end_time = time.time()
                self.logger.info(
                    f"Sequential processing completed in {end_time - start_time:.2f} seconds"
                )

        except Exception as e:
            self.logger.exception("Error during demo execution")
            raise

        finally:
            # Final memory cleanup
            clear_memory()
            self.logger.info("Demo completed")
