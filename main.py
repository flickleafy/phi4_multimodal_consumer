"""
Main entry point for the Phi-4 multimodal model demo.

This script demonstrates the Phi-4 multimodal model's capabilities by:
1. Processing images and generating descriptions
2. Transcribing audio with proper punctuation
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_utils import load_config_from_args_and_file, AppConfig
from utils.model_utils import configure_environment
from utils.runner import ModelRunner
from utils.logging_utils import set_task_context


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="Phi-4 Multimodal Demo")

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory",
    )

    parser.add_argument(
        "--image-url",
        type=str,
        help="URL of the image to process",
    )

    parser.add_argument(
        "--audio-url",
        type=str,
        help="URL of the audio to process",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory to cache downloaded files",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory to save results",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPUs are available",
    )

    parser.add_argument(
        "--disable-parallel",
        action="store_true",
        help="Disable parallel processing even if multiple GPUs are available",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--config-file",
        type=str,
        default=str(project_root / "config.json"),
        help="Path to JSON configuration file (default: config.json in project root)",
    )

    parser.add_argument(
        "--image-prompt", type=str, help="Prompt to use for image analysis"
    )

    parser.add_argument(
        "--speech-prompt", type=str, help="Prompt to use for audio transcription"
    )

    parser.add_argument(
        "--refinement_instruction",
        type=str,
        help="Prompt to use for transcription refinement",
    )

    return parser


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Configure environment variables and warning filters
    configure_environment()

    # Load configuration from args and config file
    config = load_config_from_args_and_file(args)

    # Create and run the model runner
    runner = ModelRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
