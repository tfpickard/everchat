#!/usr/bin/env python

import logging
import os
from datetime import datetime

import yaml
from rich.console import Console

# Console setup for pretty printing
console = Console()

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Generate an ISO 8601 timestamped log filename (Windows-friendly)
log_filename = os.path.join(log_dir, f"conversation_{datetime.now().isoformat().replace(':', '_')}.txt")

# Configure logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s\n" + "-" * 100,
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)


def log_init() -> None:
    """Initializes logging system."""
    logging.info("=== Conversation Session Started ===")


def log_conversation(speaker: str, message: str, elapsed_time: float) -> None:
    """Logs AI conversation with improved formatting for readability."""
    logging.info(f"🗣️ {speaker} (⏳ {elapsed_time:.2f} sec):\n{message}\n" + "-" * 100)
