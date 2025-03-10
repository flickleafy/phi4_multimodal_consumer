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
from utils.file_utils import get_image, get_audio, save_result_to_file, process_layer_result, merge_layer_results
from utils.logging_utils import set_task_context, setup_logging
from utils.long_context_fix import apply_long_context_fixes

USE_CACHE = True  # Always use cache for generation
PHI4_CONTEXT_WINDOW = 131072  # Target context window size for Phi-4 multimodal


def format_context_data(context_results: List[Dict[str, Any]]) -> str:
    """
    Format context data from previous layers for inclusion in prompts.

    Args:
        context_results: List of processed layer results to use as context

    Returns:
        str: Formatted context data string for inclusion in prompts

    Time Complexity: O(n*m) where n is number of context results, m is average result size
    """
    if not context_results:
        return ""

    context_sections = []

    for context_result in context_results:
        layer_name = context_result.get('layer_name', 'unknown_layer')

        # Format the layer context
        context_section = f"### Context from {layer_name}:\n"

        # Add summary information if available
        if 'summary' in context_result:
            summary = context_result['summary']
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if isinstance(value, (str, int, float, bool)):
                        context_section += f"- {key}: {value}\n"
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        context_section += f"- {key}: {', '.join(value)}\n"
            elif isinstance(summary, str):
                context_section += f"- General: {summary}\n"

        # Add key findings from regions if available
        if 'regions' in context_result and context_result['regions']:
            # Limit to first 3 regions to avoid overwhelming context
            regions = context_result['regions'][:3]
            context_section += f"- Detected regions: {len(context_result['regions'])} total\n"
            for i, region in enumerate(regions):
                if 'objects' in region and region['objects']:
                    objects = [obj.get('label', obj.get('category', 'unknown'))
                               for obj in region['objects'][:2]]
                    context_section += f"  - Region {i+1}: {', '.join(objects)}\n"

        # Add other relevant top-level information
        for key, value in context_result.items():
            if key not in ['layer_name', 'summary', 'regions'] and isinstance(value, (str, int, float, bool)):
                context_section += f"- {key}: {value}\n"

        context_sections.append(context_section)

    # Combine all context sections
    formatted_context = "\n".join(context_sections)

    return f"""
## CONTEXT DATA (Previous Analysis Results)
**Note: This is reference data from previous analysis layers. Use this information to inform your analysis, but focus on the user prompt below as your primary instruction.**

{formatted_context}

---
"""


def split_layered_prompts(layered_prompts: List[Tuple[str, Any]]) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Any]]]:
    """
    Split layered prompts into context-gathering and main processing phases.

    Args:
        layered_prompts: Complete list of layered prompts

    Returns:
        Tuple containing (context_prompts, main_prompts)

    Time Complexity: O(n) where n is number of layered prompts
    """
    context_layers = {
        "global_synopsis_main_subject",
        "global_synopsis_foreground",
        "global_synopsis_background"
    }

    context_prompts = []
    main_prompts = []

    for label, prompt_pair in layered_prompts:
        if label in context_layers:
            context_prompts.append((label, prompt_pair))
        else:
            main_prompts.append((label, prompt_pair))

    return context_prompts, main_prompts


# Global lock for thread-safe operations
print_lock = Lock()


class ModelRunner:
    """Main class for running the Phi-4 multimodal demonstrations."""

    def __init__(self, config: AppConfig):
        """Initialize with configuration settings."""
        self.config = config

        # Apply long context fixes FIRST, with model path for config patching
        apply_long_context_fixes(model_path=config.model_path)

        # Set up logging
        self.logger = setup_logging(debug=config.debug)

        self.gpu_info = []
        self.settings = {}

        # Context reset mechanism - track generation calls to reset model context
        self.generation_count = 0
        self.max_generations_before_reset = 3  # Reset every 3 generations
        self.current_model = None
        self.current_processor = None
        self.current_generation_config = None

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

    def reload_model(self):
        """
        Completely reload the model from scratch to eliminate all cache issues.

        This method dumps the current model from memory and reloads it fresh,
        which is more reliable than trying to reset internal caches.

        Time Complexity: O(model_loading_time) - Model loading operation
        """
        if self.current_model is None:
            self.logger.warning("No model to reload")
            return

        try:
            self.logger.info(
                "🔄 Reloading model from scratch to clear all cache...")

            # Store the current loader settings
            model_path = self.config.model_path
            settings = self.settings

            # Clear CUDA cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Delete the current model from memory
            del self.current_model
            del self.current_processor
            del self.current_generation_config

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache again after deletion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(5)

            # Reload the model fresh
            from .model_utils import ModelLoader

            # Determine if audio should be disabled (same logic as in run_sequential_processing_layered_scan)
            disable_audio = not (
                hasattr(self.config, "audio_url_provided") and self.config.audio_url_provided)

            loader = ModelLoader(
                model_path,
                settings,
                disable_audio=disable_audio,
                target_context_window=PHI4_CONTEXT_WINDOW
            )

            # Load fresh model
            self.current_model, self.current_processor, self.current_generation_config = loader.load()

            # Reset generation count
            self.generation_count = 0

            self.logger.info(
                "✅ Model reloaded successfully - fresh state with no cache")

        except Exception as e:
            self.logger.error(f"❌ Failed to reload model: {e}")
            # Still reset the counter even if reload failed
            self.generation_count = 0
            raise

    def should_reset_context(self) -> bool:
        """
        Check if model context should be reset based on generation count.

        Returns:
            bool: True if context should be reset

        Time Complexity: O(1) - Simple comparison
        """
        return self.generation_count >= self.max_generations_before_reset

    def track_generation(self):
        """
        Track a generation call and reload model if needed.

        This method counts generations and reloads the entire model
        every 3 generations to avoid cache-related issues.

        Time Complexity: O(1) unless reload is triggered (then O(model_loading))
        """
        self.generation_count += 1
        use_cache = USE_CACHE

        # Set environment variable to indicate cache usage (always True now)
        import os
        os.environ['PHI4_GENERATION_COUNT'] = str(self.generation_count)
        os.environ['PHI4_USE_CACHE'] = str(use_cache).lower()

        self.logger.info(
            f"📊 Generation tracking: count={self.generation_count}, use_cache={use_cache}, should_reload={self.should_reset_context()}")

        if self.should_reset_context():
            self.logger.info(
                f"🔄 Generation limit reached ({self.generation_count}), reloading model from scratch...")
            self.reload_model()

    def set_current_model(self, model, processor, generation_config):
        """
        Set the current model references for context management.

        Args:
            model: The loaded model
            processor: The model processor
            generation_config: Generation configuration

        Time Complexity: O(1) - Simple assignment
        """
        # Only reset generation count if this is actually a different model instance
        if self.current_model is not model:
            self.logger.info(
                f"🔄 New model instance detected, resetting generation count (was {self.generation_count})")
            self.generation_count = 0

        self.current_model = model
        self.current_processor = processor
        self.current_generation_config = generation_config

    def run_image_demo(self, model, processor, generation_config, settings, task_id=""):
        """Run the image processing demonstration with context reset management."""
        # Set task context for this thread
        set_task_context(task_id or "image")

        # Set current model references for context management
        self.set_current_model(model, processor, generation_config)

        # Track generation and reset context if needed
        self.track_generation()

        self.logger.info(
            f"Starting image processing (generation #{self.generation_count})")

        prompt = (
            f"{self.config.user_tag}<|image_1|>{self.config.image_prompt}"
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
                generation_config
            )

            self.logger.info(f"Image processing response: {response}")
            return response

        except Exception as e:
            self.logger.exception(f"Error in image processing")
            return None

    def run_image_inference(self, model, processor, generation_config, settings, task_id="", system_prompt=None, user_prompt=None):
        """Run the image processing demonstration with context reset management."""
        # Set task context for this thread
        set_task_context(task_id or "image")

        # Set current model references for context management
        self.set_current_model(model, processor, generation_config)

        # Track generation and reset context if needed
        self.track_generation()

        self.logger.info(
            f"Starting image processing (generation #{self.generation_count})")

        prompt = (
            f"{self.config.system_tag}{system_prompt}{self.config.prompt_suffix}"
            f"{self.config.user_tag}<|image_1|>{user_prompt}"
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
                generation_config
            )

            self.logger.info(f"Image processing response: {response}")
            return response

        except Exception as e:
            self.logger.exception(f"Error in image processing")
            return None

    def run_audio_demo(self, model, processor, generation_config, settings, task_id=""):
        """Run the audio processing demonstration with context reset management."""
        # Set task context for this thread
        set_task_context(task_id or "audio")

        # Set current model references for context management
        self.set_current_model(model, processor, generation_config)

        # Track generation and reset context if needed
        self.track_generation()

        self.logger.info(
            f"Starting audio processing (generation #{self.generation_count})")

        try:
            # Use the speech prompt from config instead of a hardcoded one
            prompt = (
                f"{self.config.user_tag}<|audio_1|>{self.config.speech_prompt}"
                f"{self.config.prompt_suffix}{self.config.assistant_prompt}"
            )
            self.logger.debug(f"Using prompt: {prompt}")

            # Get the audio file
            audio, samplerate = get_audio(
                self.config.audio_url, self.config.cache_dir)

            # Process with the model
            response = process_audio(
                processor,
                model,
                prompt,
                audio,
                samplerate,
                settings["max_new_tokens"],
                generation_config
            )

            self.logger.info(f"Audio processing initial response: {response}")

            # Evaluate need for refinement
            if len(response.split(".")) < 3:  # Heuristic for poor punctuation
                self.logger.info(
                    "Performing second pass to improve transcription quality"
                )

                # Track another generation for refinement
                self.track_generation()

                refined_response = refine_transcription(
                    processor,
                    model,
                    response,
                    settings["max_new_tokens"],
                    generation_config,
                    self.config.user_tag,
                    self.config.assistant_prompt,
                    self.config.prompt_suffix,
                    self.config.refinement_instruction
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

        image_result = None
        audio_result = None

        # Determine which tasks to run based on config and which URLs were explicitly provided
        run_image = False
        run_audio = False

        # Check if image_url was explicitly provided by the user
        if (
            hasattr(self.config, "image_url_provided")
            and self.config.image_url_provided
        ):
            run_image = True
        # Check if image_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.image_url
        ):
            run_image = True

        # Check if audio_url was explicitly provided by the user
        if (
            hasattr(self.config, "audio_url_provided")
            and self.config.audio_url_provided
        ):
            run_audio = True
        # Check if audio_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.audio_url
        ):
            run_audio = True

        self.logger.info(f"Tasks to run: image={run_image}, audio={run_audio}")

        # If model is not provided, load it now with appropriate options
        if model is None:
            # Decide whether to disable audio components based on whether we need audio processing
            disable_audio = not run_audio

            if disable_audio:
                self.logger.info(
                    "Audio processing not needed - disabling audio components to save memory"
                )

            # Load model with audio components disabled if not needed
            loader = ModelLoader(

                self.config.model_path, settings, disable_audio=disable_audio,

                target_context_window=PHI4_CONTEXT_WINDOW  # Enable full 128K context window

            )
            model, processor, generation_config = loader.load()

        # Process image if needed
        if run_image:
            set_task_context("image")
            image_result = self.run_image_demo(
                model, processor, generation_config, settings
            )

            # Save image result
            if image_result:
                save_result_to_file(
                    image_result,
                    "image_analysis.txt",
                    "Image Analysis Result",
                    self.config.results_dir,
                    source_file=self.config.image_url,
                )

            # Clear memory between tasks
            clear_memory()

        # Process audio if needed
        if run_audio:
            set_task_context("audio")
            audio_result = self.run_audio_demo(
                model, processor, generation_config, settings
            )

            # Save audio result
            if audio_result:
                save_result_to_file(
                    audio_result,
                    "audio_transcript.txt",
                    "Audio Transcript Result",
                    self.config.results_dir,
                    source_file=self.config.audio_url,
                )

        return image_result, audio_result

    def run_sequential_processing_layered_scan(self, model, processor, generation_config, settings):
        """Run image and audio processing sequentially on a single device."""
        self.logger.info("Starting sequential processing")

        image_result = None
        audio_result = None

        # Determine which tasks to run based on config and which URLs were explicitly provided
        run_image = False
        run_audio = False

        # Check if image_url was explicitly provided by the user
        if (
            hasattr(self.config, "image_url_provided")
            and self.config.image_url_provided
        ):
            run_image = True
        # Check if image_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.image_url
        ):
            run_image = True

        # Check if audio_url was explicitly provided by the user
        if (
            hasattr(self.config, "audio_url_provided")
            and self.config.audio_url_provided
        ):
            run_audio = True
        # Check if audio_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.audio_url
        ):
            run_audio = True

        self.logger.info(f"Tasks to run: image={run_image}, audio={run_audio}")

        # If model is not provided, load it now with appropriate options
        if model is None:
            # Decide whether to disable audio components based on whether we need audio processing
            disable_audio = not run_audio

            if disable_audio:
                self.logger.info(
                    "Audio processing not needed - disabling audio components to save memory"
                )

            # Load model with audio components disabled if not needed
            loader = ModelLoader(

                self.config.model_path, settings, disable_audio=disable_audio,

                target_context_window=PHI4_CONTEXT_WINDOW  # Enable full 128K context window

            )
            model, processor, generation_config = loader.load()

        # Process image if needed
        if run_image:
            set_task_context("image")

            layered_prompts = [
                ("global_synopsis_main_subject",
                 self.config.image_prompt_global_synopsis_main_subject),
                ("global_synopsis_foreground",
                 self.config.image_prompt_global_synopsis_foreground),
                ("global_synopsis_background",
                 self.config.image_prompt_global_synopsis_background),
                ("global_synopsis",
                 self.config.image_prompt_global_synopsis),
                ("regional_semantics",
                 self.config.image_prompt_regional_semantics),
                ("micro_detail_forensics",
                 self.config.image_prompt_micro_detail_forensics),
                ("ocr_marks_sweep",
                 self.config.image_prompt_ocr_marks_sweep),
                ("error_seeking_audit",
                 self.config.image_prompt_error_seeking_audit),
            ]

            system_prompt = self.config.image_sytem_prompt

            # Split layered prompts into context-gathering and main processing phases
            context_prompts, main_prompts = split_layered_prompts(
                layered_prompts)

            # Collect all layer results for merging
            layer_results = []
            context_results = []

            # Phase 1: Process context-gathering layers
            self.logger.info("Phase 1: Processing context-gathering layers")
            for label, prompt_pair in context_prompts:
                schema_str = str(prompt_pair['schema'])
                prompt = prompt_pair['prompt']

                full_prompt = f"{prompt}\nSchema:\n```json\n{schema_str}\n```"
                self.logger.info(f"Context layer [{label}]")

                image_result = self.run_image_inference(
                    model=model, processor=processor, generation_config=generation_config,
                    settings=settings, task_id=label, system_prompt=system_prompt, user_prompt=full_prompt
                )

                # Process layer result: extract, fix, and parse JSON
                if image_result:
                    processed_result = process_layer_result(
                        image_result, label)
                    if processed_result:
                        layer_results.append(processed_result)
                        context_results.append(processed_result)
                        self.logger.info(
                            f"Successfully processed context layer '{label}' - JSON extracted and validated")
                    else:
                        self.logger.warning(
                            f"Failed to process context layer '{label}' - skipping from context")

                # Clear memory between tasks
                clear_memory()

            # Prepare context data for main processing layers (if enabled)
            context_data = ""
            if self.config.enable_memory_context and context_results:
                context_data = format_context_data(context_results)
                self.logger.info(
                    f"Generated context data from {len(context_results)} context layers")

            # Phase 2: Process main layers with context data
            self.logger.info(
                "Phase 2: Processing main analysis layers with context")
            for label, prompt_pair in main_prompts:
                schema_str = str(prompt_pair['schema'])
                prompt = prompt_pair['prompt']

                # Add context data if enabled and available
                if context_data:
                    full_prompt = f"{context_data}\n## USER PROMPT (Primary Instruction)\n{prompt}\n\nSchema:\n```json\n{schema_str}\n```"
                else:
                    full_prompt = f"{prompt}\nSchema:\n```json\n{schema_str}\n```"

                self.logger.info(
                    f"Main layer [{label}] {'with context' if context_data else 'without context'}")

                image_result = self.run_image_inference(
                    model=model, processor=processor, generation_config=generation_config,
                    settings=settings, task_id=label, system_prompt=system_prompt, user_prompt=full_prompt
                )

                # Process layer result: extract, fix, and parse JSON
                if image_result:
                    processed_result = process_layer_result(
                        image_result, label)
                    if processed_result:
                        layer_results.append(processed_result)
                        self.logger.info(
                            f"Successfully processed main layer '{label}' - JSON extracted and validated")
                    else:
                        self.logger.warning(
                            f"Failed to process main layer '{label}' - skipping from merge")

                # Clear memory between tasks
                clear_memory()

            # Merge all layer results into a single JSON structure
            if layer_results:
                merged_result = merge_layer_results(layer_results)

                # Save the merged result
                import json
                merged_json_str = json.dumps(
                    merged_result, indent=2, ensure_ascii=False)
                save_result_to_file(
                    result=merged_json_str,
                    filename="layered_image_analysis.txt",
                    description="Layered Image Analysis Result (All Layers)",
                    results_dir=self.config.results_dir,
                    task_id="",
                    source_file=self.config.image_url,
                )
                self.logger.info(
                    f"Merged results from {len(layer_results)} layers into single JSON file")
            else:
                self.logger.warning("No valid layer results to merge")

        # Process audio if needed
        if run_audio:
            set_task_context("audio")
            audio_result = self.run_audio_demo(
                model, processor, generation_config, settings
            )

            # Save audio result
            if audio_result:
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
        # Determine which tasks to run based on config and explicitly provided URLs
        run_image = False
        run_audio = False

        # Check if image_url was explicitly provided by the user
        if (
            hasattr(self.config, "image_url_provided")
            and self.config.image_url_provided
        ):
            run_image = True
        # Check if image_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.image_url
        ):
            run_image = True

        # Check if audio_url was explicitly provided by the user
        if (
            hasattr(self.config, "audio_url_provided")
            and self.config.audio_url_provided
        ):
            run_audio = True
        # Check if audio_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.audio_url
        ):
            run_audio = True

        self.logger.info(
            f"Parallel tasks to run: image={run_image}, audio={run_audio}")

        # If only one task, fall back to sequential
        if not (run_image and run_audio):
            self.logger.info(
                "Only one task requested, falling back to sequential processing"
            )
            return self.run_sequential_processing(
                None,
                None,
                None,
                settings,  # These will be loaded in sequential processing
            )

        # Check if we have enough GPUs for parallel
        if len(settings["available_gpus"]) < 2:
            self.logger.warning(
                "Not enough GPUs available for parallel processing")
            return self.run_sequential_processing(
                None,
                None,
                None,
                settings,  # These will be loaded in sequential processing
            )

        image_gpu = settings["available_gpus"][0]
        audio_gpu = settings["available_gpus"][1]

        self.logger.info(
            f"Running parallel processing: Image on GPU {image_gpu}, Audio on GPU {audio_gpu}"
        )

        # Create separate model loaders for different GPUs
        image_loader = ModelLoader(

            self.config.model_path, settings, specific_gpu=image_gpu,

            target_context_window=PHI4_CONTEXT_WINDOW  # Enable full 128K context window

        )
        audio_loader = ModelLoader(

            self.config.model_path, settings, specific_gpu=audio_gpu,

            target_context_window=PHI4_CONTEXT_WINDOW  # Enable full 128K context window

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

            image_result = None
            audio_result = None

            # Use ThreadPoolExecutor for parallel execution with custom workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit tasks
                futures = {}

                if run_image:
                    futures["image"] = executor.submit(worker_image)

                if run_audio:
                    futures["audio"] = executor.submit(worker_audio)

                # Wait for completion and get results
                if run_image:
                    image_result = futures["image"].result()

                if run_audio:
                    audio_result = futures["audio"].result()

            # Clear task context
            set_task_context("")

            # Save results using the function from file_utils.py
            if run_image and image_result:
                img_path = save_result_to_file(
                    image_result,
                    "image_analysis.txt",
                    f"Image Analysis Result (GPU {image_gpu})",
                    self.config.results_dir,
                    f"gpu{image_gpu}",
                    source_file=self.config.image_url,
                )

            if run_audio and audio_result:
                audio_path = save_result_to_file(
                    audio_result,
                    "audio_transcript.txt",
                    f"Audio Transcript Result (GPU {audio_gpu})",
                    self.config.results_dir,
                    f"gpu{audio_gpu}",
                    source_file=self.config.audio_url,
                )

            self.logger.info(f"Results saved successfully")
            return image_result, audio_result

        finally:
            # Clear memory on both GPUs
            clear_memory(image_gpu)
            clear_memory(audio_gpu)

            # Clear task context
            set_task_context("")

    def get_generation_kwargs(self, max_new_tokens: int, pad_token_id: int) -> dict:
        """
        Get generation kwargs with appropriate cache settings.

        Args:
            max_new_tokens: Maximum tokens to generate
            pad_token_id: Padding token ID

        Returns:
            dict: Generation kwargs with cache settings

        Time Complexity: O(1) - Function call and dictionary creation
        """
        from .long_context_fix import get_long_context_generation_kwargs
        return get_long_context_generation_kwargs(
            max_new_tokens,
            pad_token_id,
            use_cache=USE_CACHE
        )

    def run(self):
        """Main entry point to run the demo."""
        run_id = f"run_{int(time.time())}"
        set_task_context(run_id)
        self.logger.info(f"Starting Phi-4 multimodal demo ({run_id})")

        # Check which modes to run based on explicitly provided URLs
        run_image = False
        run_audio = False

        # Check if image_url was explicitly provided by the user
        if (
            hasattr(self.config, "image_url_provided")
            and self.config.image_url_provided
        ):
            run_image = True
        # Check if image_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.image_url
        ):
            run_image = True

        # Check if audio_url was explicitly provided by the user
        if (
            hasattr(self.config, "audio_url_provided")
            and self.config.audio_url_provided
        ):
            run_audio = True
        # Check if audio_url is available and we're in demo mode
        elif (
            hasattr(self.config, "demo_mode")
            and self.config.demo_mode
            and self.config.audio_url
        ):
            run_audio = True

        self.logger.info(
            f"Demo tasks to run: image={run_image}, audio={run_audio}")

        if not run_image and not run_audio:
            self.logger.warning(
                "No tasks to run. Provide image_url, audio_url, or enable demo_mode."
            )
            return

        try:
            # Determine whether to use parallel or sequential processing
            if (
                self.settings.get("multi_gpu", False)
                and len(self.settings.get("available_gpus", [])) > 1
                and not self.config.disable_parallel
                and run_image
                and run_audio  # Only use parallel if both tasks are needed
            ):
                # Run parallel processing across multiple GPUs
                self.logger.info("Using multiple GPUs for parallel processing")
                start_time = time.time()
                results = self.run_parallel_processing(self.settings)
                end_time = time.time()

                if (
                    results[0] is not None or results[1] is not None
                ):  # At least one result was successful
                    self.logger.info(
                        f"Parallel processing completed in {end_time - start_time:.2f} seconds"
                    )
                else:
                    self.logger.warning(
                        "Parallel processing failed, falling back to sequential"
                    )
                    # Fall back to sequential processing
                    start_time = time.time()

                    # Determine whether to disable audio components
                    disable_audio = not run_audio
                    if disable_audio:
                        self.logger.info(
                            "Audio processing not needed - disabling audio components to prevent gradient checkpointing messages"
                        )

                    # Load model with audio components disabled if not needed
                    loader = ModelLoader(

                        self.config.model_path,
                        self.settings,
                        disable_audio=disable_audio,

                        target_context_window=PHI4_CONTEXT_WINDOW  # Enable full 128K context window

                    )
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

                # Only load the model if we have tasks to run
                if run_image or run_audio:
                    # Determine whether to disable audio components
                    disable_audio = not run_audio
                    if disable_audio:
                        self.logger.info(
                            "Audio processing not needed - disabling audio components to prevent gradient checkpointing messages"
                        )

                    # Load model with audio components disabled if not needed
                    loader = ModelLoader(

                        self.config.model_path,
                        self.settings,
                        disable_audio=disable_audio,

                        target_context_window=PHI4_CONTEXT_WINDOW  # Enable full 128K context window

                    )
                    model, processor, generation_config = loader.load()

                    # Layered scan support for image processing
                    if run_image and getattr(self.config, "layered_scan", False):
                        self.run_sequential_processing_layered_scan(
                            model, processor, generation_config, self.settings)
                    else:
                        # Run sequential processing (default)
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
