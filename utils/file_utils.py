"""File download and caching utilities."""

import os
import requests
import io
from typing import Tuple, Optional, Union
from urllib.request import urlopen
from PIL import Image
import soundfile as sf
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("phi4_demo")


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure the specified directory exists.

    Args:
        directory: Directory path to create if it doesn't exist
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def is_url(path_or_url: str) -> bool:
    """
    Check if the given string is a URL.

    Args:
        path_or_url: String to check

    Returns:
        bool: True if it's a URL, False if it's a local path
    """
    return path_or_url.startswith(("http://", "https://"))


def get_image(path_or_url: str, cache_dir: str = "cached_files") -> Image.Image:
    """
    Get an image from URL, local path, or cache.

    Args:
        path_or_url: URL or local file path of the image
        cache_dir: Directory to cache downloaded files

    Returns:
        PIL.Image: Loaded image

    Raises:
        ValueError: If image download or loading fails
        FileNotFoundError: If the local file doesn't exist
    """
    # Check if it's a URL or local path
    if is_url(path_or_url):
        # Handle URL case
        url = path_or_url
        # Create filename from URL
        filename = os.path.join(cache_dir, f"image_{url.split('/')[-1]}")
        ensure_directory_exists(cache_dir)

        # Return cached file if it exists
        if os.path.exists(filename):
            logger.info(f"Loading image from cache: {filename}")
            return Image.open(filename)

        # Download the file if it doesn't exist
        logger.info(f"Downloading image from: {url}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise ValueError(
                f"Image download failed with status code {response.status_code}"
            )

        # Save and return the file
        with open(filename, "wb") as f:
            f.write(response.content)

        image = Image.open(filename)
        logger.info(
            f"Image cached and loaded successfully: {image.format} {image.size} {image.mode}"
        )
        return image
    else:
        # Handle local path case
        local_path = path_or_url

        # Check if file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Image file not found: {local_path}")

        # Load the image directly
        logger.info(f"Loading image from local path: {local_path}")
        image = Image.open(local_path)
        logger.info(
            f"Image loaded successfully: {image.format} {image.size} {image.mode}"
        )
        return image


def get_audio(
    path_or_url: str, cache_dir: str = "cached_files"
) -> Tuple[np.ndarray, int]:
    """
    Get audio data from URL, local path, or cache.

    Args:
        path_or_url: URL or local file path of the audio
        cache_dir: Directory to cache downloaded files

    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate

    Raises:
        ValueError: If audio download or loading fails
        FileNotFoundError: If the local file doesn't exist
    """
    # Check if it's a URL or local path
    if is_url(path_or_url):
        # Handle URL case
        url = path_or_url
        # Create filename from URL
        filename = os.path.join(cache_dir, f"audio_{url.split('/')[-1]}")
        ensure_directory_exists(cache_dir)

        # Return cached file if it exists
        if os.path.exists(filename):
            logger.info(f"Loading audio from cache: {filename}")
            audio_data, sample_rate = sf.read(filename)
            return audio_data, sample_rate

        # Download the file if it doesn't exist
        logger.info(f"Downloading audio from: {url}")
        audio_response = urlopen(url)
        if audio_response.status != 200:
            raise ValueError(
                f"Audio download failed with status code {audio_response.status}"
            )

        # Read the audio data
        audio_data, sample_rate = sf.read(io.BytesIO(audio_response.read()))

        # Save the audio file for future use
        sf.write(filename, audio_data, sample_rate)

        logger.info(
            f"Audio cached and loaded successfully: {audio_data.shape} {audio_data.dtype} {sample_rate}Hz"
        )
        return audio_data, sample_rate
    else:
        # Handle local path case
        local_path = path_or_url

        # Check if file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Audio file not found: {local_path}")

        # Load the audio directly
        logger.info(f"Loading audio from local path: {local_path}")
        audio_data, sample_rate = sf.read(local_path)
        logger.info(
            f"Audio loaded successfully: {audio_data.shape} {audio_data.dtype} {sample_rate}Hz"
        )
        return audio_data, sample_rate
