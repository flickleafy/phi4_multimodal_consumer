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


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure the specified directory exists.

    Args:
        directory: Directory path to create if it doesn't exist
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_image(url: str, cache_dir: str = "cached_files") -> Image.Image:
    """
    Get an image from URL or local cache.

    Args:
        url: URL of the image to download
        cache_dir: Directory to cache downloaded files

    Returns:
        PIL.Image: Loaded image

    Raises:
        ValueError: If image download or loading fails
    """
    # Create filename from URL
    filename = os.path.join(cache_dir, f"image_{url.split('/')[-1]}")
    ensure_directory_exists(cache_dir)

    # Return cached file if it exists
    if os.path.exists(filename):
        print(f"Loading image from cache: {filename}")
        return Image.open(filename)

    # Download the file if it doesn't exist
    print(f"Downloading image from: {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(
            f"Image download failed with status code {response.status_code}"
        )

    # Save and return the file
    with open(filename, "wb") as f:
        f.write(response.content)

    image = Image.open(filename)
    print(
        f"Image cached and loaded successfully: {image.format} {image.size} {image.mode}"
    )
    return image


def get_audio(url: str, cache_dir: str = "cached_files") -> Tuple[np.ndarray, int]:
    """
    Get audio data from URL or local cache.

    Args:
        url: URL of the audio file to download
        cache_dir: Directory to cache downloaded files

    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate

    Raises:
        ValueError: If audio download or loading fails
    """
    # Create filename from URL
    filename = os.path.join(cache_dir, f"audio_{url.split('/')[-1]}")
    ensure_directory_exists(cache_dir)

    # Return cached file if it exists
    if os.path.exists(filename):
        print(f"Loading audio from cache: {filename}")
        audio_data, sample_rate = sf.read(filename)
        return audio_data, sample_rate

    # Download the file if it doesn't exist
    print(f"Downloading audio from: {url}")
    audio_response = urlopen(url)
    if audio_response.status != 200:
        raise ValueError(
            f"Audio download failed with status code {audio_response.status}"
        )

    # Read the audio data
    audio_data, sample_rate = sf.read(io.BytesIO(audio_response.read()))

    # Save the audio file for future use
    sf.write(filename, audio_data, sample_rate)

    print(
        f"Audio cached and loaded successfully: {audio_data.shape} {audio_data.dtype} {sample_rate}Hz"
    )
    return audio_data, sample_rate
