#!/usr/bin/env python

from typing import Dict, List

import ollama
import openai
import yaml
from rich.console import Console
from rich.panel import Panel

# Console setup for pretty printing
console = Console()


def generate_response(
    model_name: str, provider: str, messages: List[Dict[str, str]]
) -> str:
    """Generates an AI response using the selected model (OpenAI or Ollama)."""
    if provider == "openai":
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.9
        )
        return response.choices[0].message.content
    else:
        response = ollama.chat(model=model_name, messages=messages)
        return response["message"]["content"]


class Agent:
    """Represents an AI agent with a unique identity, traits, and conversation capabilities."""

    def __init__(self, model: str, provider: str = "ollama", traits: List[str] = None):
        self.model = model
        self.provider = provider
        self.traits = traits if traits else []
        self.name = None
        self.backstory = None
        self.identity_generated = False

    def generate_identity(self) -> None:
        """Generates a unique name and backstory using the configured AI provider."""
        console.print(
            Panel(f"🧠 Generating identity for {self.model}...", style="yellow")
        )

        trait_description = ", ".join(self.traits) if self.traits else "neutral"
        prompt = (
            f"Generate a unique human-like identity for an AI with the following traits: {trait_description}. "
            "Choose a random name, a unique occupation, and an interesting quirk. "
            "Then, write a short, vivid backstory. Keep it conversational."
        )

        response = generate_response(
            self.model, self.provider, [{"role": "system", "content": prompt}]
        )
        identity_lines = response.split("\n")
        self.name = identity_lines[0].strip() if identity_lines else "Unnamed AI"
        self.backstory = (
            "\n".join(identity_lines[1:]).strip()
            if len(identity_lines) > 1
            else "No backstory provided."
        )
        self.identity_generated = True

        console.print(
            Panel(
                f"🤖 AI Identity Created:\n\n[bold]Name:[/bold] {self.name}\n\n[italic]{self.backstory}[/italic]",
                style="cyan",
            )
        )

    def respond(self, chat_history: List[Dict[str, str]]) -> str:
        """Generates a response based on chat history while maintaining context."""
        if not self.identity_generated:
            self.generate_identity()

        console.print(Panel(f"🤖 Sending to [bold]{self.name}[/bold]...", style="blue"))

        response = generate_response(
            self.model,
            self.provider,
            [{"role": "system", "content": self.backstory}] + chat_history,
        )

        console.print(
            Panel(f"💬 [bold]{self.name}:[/bold] {response}", style="magenta")
        )

        return response
