"""Logging utilities for the Phi-4 multimodal demo."""

import logging
import threading
from logging.handlers import RotatingFileHandler
from threading import Lock

# Thread-local storage for task-specific context
thread_local = threading.local()

# Global lock for thread-safe operations
log_lock = Lock()


class TaskAwareFormatter(logging.Formatter):
    """Custom formatter that includes task ID in log records if available."""

    def format(self, record):
        # Add task_id to the record if it exists in thread_local
        if hasattr(thread_local, "task_id") and thread_local.task_id:
            record.task_id = thread_local.task_id
            original_msg = record.msg
            record.msg = f"[{record.task_id}] {original_msg}"
        else:
            record.task_id = ""

        return super().format(record)


def setup_logging(log_file="phi4_demo.log", debug=False):
    """Set up logging with task awareness and thread safety."""

    # Create formatters
    console_formatter = TaskAwareFormatter("%(asctime)s - %(levelname)s - %(message)s")
    file_formatter = TaskAwareFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set up handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Use rotating file handler to prevent logs from growing too large
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,  # 10MB per file, keep 5 backups
    )
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Create a logger for this module
    logger = logging.getLogger("phi4_demo")

    return logger


def set_task_context(task_id=""):
    """Set the current thread's task context for logging."""
    thread_local.task_id = task_id


def get_task_context():
    """Get the current thread's task context."""
    return getattr(thread_local, "task_id", "")
