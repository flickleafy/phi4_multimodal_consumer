"""File download and caching utilities."""

import os
import requests
import io
import datetime
import json
import re
from typing import Tuple, Optional, Union, Dict, Any, List
from urllib.request import urlopen
from PIL import Image
import soundfile as sf
import numpy as np
from pathlib import Path
import logging
from copy import deepcopy

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


def save_result_to_file(
    result: str,
    filename: str,
    description: str,
    results_dir: str,
    task_id: str = "",
    source_file: str = "unknown",
) -> Optional[str]:
    """
    Save processing result to a file, avoiding overwriting different content.

    Args:
        result: Text result to save
        filename: Name of the file to create
        description: Description of the content
        results_dir: Directory to save results
        task_id: Optional task identifier for the filename
        source_file: Name/path of the source file that was analyzed

    Returns:
        Optional[str]: Path to saved file or None if no result
    """
    if not result:
        return None

    # Get timestamp in a more readable format: YYYY-MM-DD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Extract just the filename if a full path is provided
    if "/" in source_file or "\\" in source_file:
        source_filename = os.path.basename(source_file)
    else:
        source_filename = source_file

    # Add task identifier to filename to avoid collisions in parallel processing
    if task_id:
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{task_id}{ext}"

    # Always include timestamp and source filename in the saved filename
    base, ext = os.path.splitext(filename)
    filename = f"{base}_{source_filename}_{timestamp}{ext}"

    # Create full filepath
    filepath = os.path.join(results_dir, filename)

    # Format the content with timestamp and analyzed file information
    content = f"# Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += f"# Source file: {source_file}\n\n"
    content += f"{description}:\n{result}"

    # Check if there's an identical content file already
    existing_files = [
        f for f in os.listdir(results_dir) if f.startswith(base) and f.endswith(ext)
    ]
    for existing_file in existing_files:
        try:
            with open(os.path.join(results_dir, existing_file), "r") as f:
                existing_content = f.read()

            # Skip the first few lines that contain the timestamp and source info
            existing_content_body = "\n".join(existing_content.split("\n")[3:])
            result_body = f"{description}:\n{result}"

            # If content is identical, don't create another file
            if existing_content_body == result_body:
                logger.info(
                    f"File already exists with identical content: {os.path.join(results_dir, existing_file)}"
                )
                return os.path.join(results_dir, existing_file)
        except Exception as e:
            logger.warning(f"Error comparing file contents: {e}")

    # Ensure the directory exists
    ensure_directory_exists(results_dir)

    # Save the file
    with open(filepath, "w") as f:
        f.write(content)

    logger.info(f"Result saved to {filepath}")
    return filepath


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON content from text that may contain markdown code blocks or other formatting.

    Args:
        text: Raw text containing JSON (possibly in markdown blocks)

    Returns:
        Optional[str]: Extracted JSON string or None if no valid JSON found

    Time Complexity: O(n) where n is the length of the text
    """
    if not text:
        return None

    # First, try to extract from markdown code blocks
    # Pattern: ```json ... ``` or ``` ... ```
    markdown_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```'
    ]

    for pattern in markdown_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match.strip():
                # Check if this looks like JSON (starts with { or [)
                stripped = match.strip()
                if stripped.startswith(('{', '[')):
                    return stripped

    # If no markdown blocks found, look for JSON-like content
    # Find content between outermost braces or brackets
    brace_pattern = r'\{.*\}'
    bracket_pattern = r'\[.*\]'

    for pattern in [brace_pattern, bracket_pattern]:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the longest match (most likely to be complete)
            return max(matches, key=len)

    return None


def fix_incomplete_json(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues and incomplete structures.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        str: Fixed JSON string

    Time Complexity: O(n) where n is the length of the JSON string
    """
    if not json_str:
        return "{}"

    # Remove any leading/trailing whitespace
    json_str = json_str.strip()

    # Handle common issues
    # 1. Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # 2. Ensure proper quote matching for strings
    # This is a simple fix - in production, you'd want more sophisticated parsing

    # 3. Check if JSON is incomplete (missing closing braces/brackets)
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    # Add missing closing braces
    while close_braces < open_braces:
        json_str += '}'
        close_braces += 1

    # Add missing closing brackets
    while close_brackets < open_brackets:
        json_str += ']'
        close_brackets += 1

    # 4. Handle incomplete string values (missing quotes)
    # This is complex - for now, we'll rely on json.loads to catch these

    return json_str


def parse_and_validate_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON string and validate it's complete and well-formed.

    Args:
        json_str: JSON string to parse

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON object or None if invalid

    Time Complexity: O(n) where n is the length of the JSON string
    """
    if not json_str:
        return None

    try:
        # First attempt - parse as-is
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except json.JSONDecodeError:
        try:
            # Second attempt - fix common issues and try again
            fixed_json = fix_incomplete_json(json_str)
            parsed = json.loads(fixed_json)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON after fixing: {e}")
            logger.debug(f"Problematic JSON: {json_str[:200]}...")
            return None


def merge_layer_results(layer_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from multiple analysis layers into a single JSON structure.

    Args:
        layer_results: List of parsed JSON objects from each layer

    Returns:
        Dict[str, Any]: Merged JSON structure with combined results

    Time Complexity: O(n*m) where n is number of layers and m is average size of each result
    """
    merged_result = {
        "summary": {},
        "regions": [],
        "metadata": {
            "layers_processed": [],
            "merge_timestamp": datetime.datetime.now().isoformat()
        }
    }

    for i, layer_result in enumerate(layer_results):
        if not layer_result:
            continue

        # Track which layer this came from
        layer_name = layer_result.get("layer_name", f"layer_{i}")
        merged_result["metadata"]["layers_processed"].append(layer_name)

        # Merge summary fields
        if "summary" in layer_result:
            summary_data = layer_result["summary"]
            if isinstance(summary_data, dict):
                # Merge summary dictionaries
                for key, value in summary_data.items():
                    if key not in merged_result["summary"]:
                        merged_result["summary"][key] = value
                    elif isinstance(value, list) and isinstance(merged_result["summary"][key], list):
                        # Merge lists, avoiding duplicates
                        merged_result["summary"][key].extend(
                            item for item in value if item not in merged_result["summary"][key]
                        )
                    elif isinstance(value, dict) and isinstance(merged_result["summary"][key], dict):
                        # Recursively merge nested dictionaries
                        merged_result["summary"][key].update(value)
                    else:
                        # For other types, create a list if different values exist
                        existing = merged_result["summary"][key]
                        if existing != value:
                            if not isinstance(existing, list):
                                merged_result["summary"][key] = [existing]
                            if value not in merged_result["summary"][key]:
                                merged_result["summary"][key].append(value)
            else:
                # If summary is not a dict, add it as a list item
                if "general" not in merged_result["summary"]:
                    merged_result["summary"]["general"] = []
                merged_result["summary"]["general"].append(summary_data)

        # Merge regions fields
        if "regions" in layer_result:
            regions_data = layer_result["regions"]
            if isinstance(regions_data, list):
                merged_result["regions"].extend(regions_data)
            elif isinstance(regions_data, dict):
                merged_result["regions"].append(regions_data)

        # Add other fields to root level
        for key, value in layer_result.items():
            if key not in ["summary", "regions", "layer_name"]:
                if key not in merged_result:
                    merged_result[key] = value
                elif isinstance(value, dict) and isinstance(merged_result[key], dict):
                    merged_result[key].update(value)
                elif isinstance(value, list) and isinstance(merged_result[key], list):
                    merged_result[key].extend(value)
                else:
                    # Create a list for conflicting values
                    if not isinstance(merged_result[key], list):
                        merged_result[key] = [merged_result[key]]
                    if value not in merged_result[key]:
                        merged_result[key].append(value)

    return merged_result


def process_layer_result(result_text: str, layer_name: str) -> Optional[Dict[str, Any]]:
    """
    Process a single layer result: extract, fix, and parse JSON.

    Args:
        result_text: Raw result text from the layer
        layer_name: Name/identifier of the analysis layer

    Returns:
        Optional[Dict[str, Any]]: Processed and validated JSON object

    Time Complexity: O(n) where n is the length of the result text
    """
    # Extract JSON from the result text
    json_str = extract_json_from_text(result_text)
    if not json_str:
        logger.warning(f"No JSON found in layer '{layer_name}' result")
        return None

    # Parse and validate the JSON
    parsed_json = parse_and_validate_json(json_str)
    if not parsed_json:
        logger.error(f"Failed to parse JSON for layer '{layer_name}'")
        return None

    # Add layer metadata
    parsed_json["layer_name"] = layer_name

    logger.info(f"Successfully processed layer '{layer_name}' result")
    return parsed_json
