#!/usr/bin/env python

import json
import random
import time
from datetime import datetime
from typing import List

from rich.console import Console

from agent import Agent
from utils import log_conversation, log_init

console = Console()

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Create AI agents from list
agents = [Agent(**ai_model) for ai_model in config["ai_models"]]

# Generate identities for each agent
for agent in agents:
    agent.generate_identity()


def choose_starting_agent() -> Agent:
    """Randomly selects which agent starts the conversation."""
    return random.choice(agents)


def get_random_topic() -> str:
    """Selects a random topic from the config file."""
    topics = config.get("topics", [])
    return random.choice(topics) if topics else "Let's start with something unexpected."


def chat_loop():
    """Handles the conversation flow between AI agents."""
    log_init()
    chat_history = []

    # Determine which AI starts and what topic is used
    if config.get("enable_topic_seeding", False):
        starting_agent = choose_starting_agent()
        topic = get_random_topic()
        intro_message = starting_agent.generate_intro(topic)
        chat_history.append({"role": "assistant", "content": intro_message})
        log_conversation(starting_agent.name, intro_message, elapsed_time=0)
        console.print(f"[bold cyan]Starting Topic:[/bold cyan] {topic}")
        console.print(
            f"[bold yellow]{starting_agent.name}[/bold yellow] opens: {intro_message}"
        )

    # Continue conversation
    for _ in range(config["conversation"]["max_turns"]):
        current_agent = agents[_ % len(agents)]
        response = current_agent.respond(chat_history)
        chat_history.append({"role": "assistant", "content": response})
        log_conversation(current_agent.name, response, elapsed_time=0)
        console.print(f"[bold green]{current_agent.name}[/bold green]: {response}")


if __name__ == "__main__":
    chat_loop()
