"""Configuration utilities for the Phi-4 multimodal demo."""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class AppConfig:
    """Application configuration."""

    model_path: str = "Phi-4-multimodal-instruct"
    image_url: str = "https://www.ilankelman.org/stopsigns/australia.jpg"
    audio_url: str = (
        "https://upload.wikimedia.org/wikipedia/commons/b/b0/"
        "Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
    )
    cache_dir: str = "cached_files"
    results_dir: str = "results"
    user_prompt: str = "<|user|>"
    assistant_prompt: str = "<|assistant|>"
    prompt_suffix: str = "<|end|>"
    force_cpu: bool = False
    disable_parallel: bool = False
    multi_gpu_threshold_gb: float = 8.0
    debug: bool = False
    image_prompt: str = "What is shown in this image?"
    speech_prompt: str = """Transcribe the audio to text, paying special attention to:
1. Natural pauses in speech (use commas, periods, or other appropriate punctuation)
2. Changes in tone and intonation (questions, exclamations)
3. Emphasis and rhythm of the speaker's delivery
4. Natural paragraph breaks where topics shift
5. Maintain speaker's original cadence while ensuring readability

Format as a professional transcript that preserves the speaker's natural speech patterns."""
    refinement_instruction: str = (
        "Please add proper punctuation to this transcript while preserving the original meaning. The transcript is from a spoken lecture: "
    )
    demo_mode: bool = (
        False  # Run both image and audio processing regardless of URL availability
    )

    def __post_init__(self):
        """Validate and prepare configuration after initialization."""
        # Convert relative paths to absolute
        if not os.path.isabs(self.cache_dir):
            self.cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                self.cache_dir,
            )

        if not os.path.isabs(self.results_dir):
            self.results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                self.results_dir,
            )

        # Ensure necessary directories exist
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)
        Path(self.results_dir).mkdir(exist_ok=True, parents=True)


def load_config(config_file: str = None) -> AppConfig:
    """
    Load configuration from a file.

    Args:
        config_file: Path to the JSON configuration file

    Returns:
        AppConfig: Configuration object with loaded values
    """
    config = AppConfig()

    default_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )

    # Use default config if none specified
    if config_file is None and os.path.exists(default_config):
        config_file = default_config

    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Update config with file values
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    config.__post_init__()  # Ensure paths are properly set up
    return config


def load_config_from_args_and_file(args) -> AppConfig:
    """
    Load configuration from command-line arguments and file.

    Args:
        args: Parsed command-line arguments

    Returns:
        AppConfig: Configuration object with loaded values
    """
    # Load from config file first
    config_file = args.config_file if hasattr(args, "config_file") else None
    config = load_config(config_file)

    # Override with command-line arguments if provided
    for key in vars(config):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))

    return config


def to_dict(config: AppConfig) -> dict:
    """Convert config to dictionary."""
    return asdict(config)


def save_config(config: AppConfig, path: str) -> None:
    """Save configuration to a file."""
    with open(path, "w") as f:
        json.dump(to_dict(config), f, indent=2)
