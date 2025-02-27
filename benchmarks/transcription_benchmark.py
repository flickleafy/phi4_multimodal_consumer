"""
Benchmark script to compare Whisper and Phi-4 models for audio transcription.
"""

import os
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
from utils.file_utils import get_audio
from utils.processors import process_audio

# Define test audio files
TEST_FILES = [
    "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac",
    # Add more test files here
]

# Cache directory for downloaded files
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cached_files"
)

# Output directory for results
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define prompt structure for Phi-4
USER_PROMPT = "<|user|>"
ASSISTANT_PROMPT = "<|assistant|>"
PROMPT_SUFFIX = "<|end|>"


def load_whisper():
    """Load the Whisper large-v2 model."""
    print("Loading Whisper large-v2 model...")
    whisper = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v2",
        device=0 if torch.cuda.is_available() else "cpu",
    )
    return whisper


def load_phi4():
    """Load the Phi-4 multimodal model."""
    print("Loading Phi-4 multimodal model...")
    model_path = "Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        _attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def benchmark_whisper(whisper, audio_file):
    """Benchmark Whisper model on an audio file."""
    audio_data, sample_rate = get_audio(audio_file, CACHE_DIR)

    # Start timing
    start_time = time.time()

    # Transcribe with Whisper
    result = whisper(
        {"raw": audio_data, "sampling_rate": sample_rate}, return_timestamps=True
    )

    # End timing
    end_time = time.time()
    processing_time = end_time - start_time

    return {
        "transcription": result["text"],
        "processing_time": processing_time,
        "model": "whisper-large-v2",
    }


def benchmark_phi4(model, processor, audio_file):
    """Benchmark Phi-4 model on an audio file."""
    audio_data, sample_rate = get_audio(audio_file, CACHE_DIR)

    # Create transcription prompt
    speech_prompt = """Transcribe the audio to text with proper punctuation, 
    capitalization, and paragraph breaks. Format it as a professional transcript."""
    prompt = f"{USER_PROMPT}<|audio_1|>{speech_prompt}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"

    # Start timing
    start_time = time.time()

    # Setup for generation
    max_new_tokens = 200

    # Process with Phi-4
    with torch.no_grad():
        inputs = processor(
            text=prompt,
            audios=[(audio_data.astype(np.float32), sample_rate)],
            return_tensors="pt",
        )

        # Move inputs to the correct device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

        # Generate response
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        # Decode the response
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # End timing
    end_time = time.time()
    processing_time = end_time - start_time

    return {
        "transcription": response,
        "processing_time": processing_time,
        "model": "phi-4-multimodal-instruct",
    }


def run_benchmarks():
    """Run benchmarks on both models."""
    print("Starting transcription benchmark...")

    # Load the models
    whisper = load_whisper()
    phi4_model, phi4_processor = load_phi4()

    results = []

    # Test each audio file
    for i, audio_file in enumerate(TEST_FILES):
        print(
            f"\nProcessing test file {i+1}/{len(TEST_FILES)}: {os.path.basename(audio_file)}"
        )

        # Benchmark Whisper
        print("Running Whisper benchmark...")
        whisper_result = benchmark_whisper(whisper, audio_file)
        results.append(whisper_result)
        print(
            f"Whisper processing time: {whisper_result['processing_time']:.2f} seconds"
        )

        # Benchmark Phi-4
        print("Running Phi-4 benchmark...")
        phi4_result = benchmark_phi4(phi4_model, phi4_processor, audio_file)
        results.append(phi4_result)
        print(f"Phi-4 processing time: {phi4_result['processing_time']:.2f} seconds")

        # Save individual results
        filename = f"benchmark_{os.path.basename(audio_file).split('.')[0]}.txt"
        with open(os.path.join(RESULTS_DIR, filename), "w") as f:
            f.write(f"Audio file: {audio_file}\n\n")
            f.write(
                f"WHISPER LARGE-V2 (Time: {whisper_result['processing_time']:.2f}s):\n"
            )
            f.write(f"{whisper_result['transcription']}\n\n")
            f.write(
                f"PHI-4-MULTIMODAL (Time: {phi4_result['processing_time']:.2f}s):\n"
            )
            f.write(f"{phi4_result['transcription']}\n\n")

    # Save summary
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.txt"), "w") as f:
        f.write("AUDIO TRANSCRIPTION BENCHMARK SUMMARY\n")
        f.write("====================================\n\n")

        for i, audio_file in enumerate(TEST_FILES):
            whisper_result = results[i * 2]
            phi4_result = results[i * 2 + 1]

            f.write(f"Test {i+1}: {os.path.basename(audio_file)}\n")
            f.write(
                f"Whisper processing time: {whisper_result['processing_time']:.2f} seconds\n"
            )
            f.write(
                f"Phi-4 processing time: {phi4_result['processing_time']:.2f} seconds\n"
            )
            f.write(
                f"Speed difference: {phi4_result['processing_time']/whisper_result['processing_time']:.2f}x\n\n"
            )

    print(f"\nBenchmark completed. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run_benchmarks()
