#!/usr/bin/env python

import time
from datetime import datetime
from typing import Dict, List

import yaml
from rich.console import Console
from rich.panel import Panel

from agent import Agent
from utils import log_conversation, log_init

console = Console()

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Create AI agents
agent1 = Agent(
    model=config["ai_models"]["model1"],
    provider=config["ai_models"]["provider1"],
    traits=["cynical", "intelligent", "terse"],
)
agent2 = Agent(
    model=config["ai_models"]["model2"],
    provider=config["ai_models"]["provider2"],
    traits=["curious", "imaginative", "playful"],
)

agent1.generate_identity()
agent2.generate_identity()


def chat_loop():
    """Handles the conversation flow between two AI agents."""
    log_init()
    chat_history = []

    for _ in range(config["conversation"]["max_turns"]):
        current_agent = agent1 if _ % 2 == 0 else agent2
        response = current_agent.respond(chat_history)
        chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat_loop()
