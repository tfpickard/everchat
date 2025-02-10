#!/usr/bin/env python

import json
import logging
import os
import traceback
from typing import Dict, List

import openai
from rich.console import Console
from rich.logging import RichHandler

console = Console()

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

VERBOSE = config.get("verbose", False)

# Configure logging with RichHandler
logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("agent")


class Agent:
    def __init__(self, model: str, provider: str, traits: List[str]):
        self.model = model
        self.provider = provider
        self.traits = traits
        self.name = "Unknown"
        self.personality = ""
        self.backstory = ""
        self.openai_client = None

        if self.provider == "openai":
            api_key = config.get("openai", {}).get("api_key") or os.environ.get(
                "OPENAI_API_KEY"
            )
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
            else:
                logger.error("No OpenAI API key found in config or environment!")

        logger.debug(
            f"Initializing Agent: {self.model} ({self.provider}) with traits: {self.traits}"
        )

        self.generate_identity()
        self.generate_backstory()

    def generate_intro(self, topic: str) -> str:
        """Generates an opening statement about the given topic."""
        intro_prompt = (
            f"You are {self.name}, an AI with the traits: {', '.join(self.traits)}. "
            f"Initiate a conversation about the topic: '{topic}'. "
            "Ensure your response is engaging and relevant to the topic."
        )
        return self.generate_llm_response(intro_prompt, "(Introduction Context)")

    def generate_identity(self) -> None:
        """Requests the LLM to generate a unique name and personality separately."""
        name_prompt = (
            "Generate a unique name for an AI character. The name should always follow the format 'First Last'. "
            "Ensure the name is creative but still sounds natural."
        )
        logger.debug("Calling LLM for name generation...")
        generated_name = self.generate_llm_response(
            name_prompt, "(Name Generation Context)"
        ).strip()

        if generated_name:
            self.name = generated_name
        else:
            logger.error("Name generation failed! Using default 'Unknown'.")

        logger.info(f"Generated Name: {self.name}")

        personality_prompt = (
            f"Generate a personality description for an AI character named {self.name}. "
            "The personality should align with the following traits: "
            + ", ".join(self.traits)
            + "."
        )
        logger.info("Calling LLM for personality generation...")
        self.personality = self.generate_llm_response(
            personality_prompt, "(Personality Generation Context)"
        )
        logger.info(f"Generated Personality: {self.personality}")

    def generate_backstory(self) -> None:
        """Requests the LLM to generate a unique backstory for the agent."""
        prompt = (
            f"Generate a unique and detailed backstory for an AI character named {self.name}. "
            "The backstory should be creative and original, providing insight into their personality, "
            "past experiences, and how they developed their worldview. Avoid repeating generic AI tropes."
        )
        logger.info("Calling LLM for backstory generation...")
        self.backstory = self.generate_llm_response(
            prompt, "(Backstory Generation Context)"
        )
        logger.info(f"Generated Backstory for {self.name}")

    def respond(self, chat_history: List[Dict[str, str]]) -> str:
        """Generates a response while using the backstory as internal context."""
        context = (
            f"Name: {self.name}\n"
            f"Personality Traits: {', '.join(self.traits)}\n"
            f"Personality: {self.personality}\n"
            f"(Internal Backstory: Not shared but used for context)\n"
        )

        last_message = (
            chat_history[-1]["content"] if chat_history else "Hello, who are you?"
        )
        logger.info("Calling LLM for response generation...")
        response = self.generate_llm_response(last_message, context)
        return response

    def generate_llm_response(self, user_input: str, context: str) -> str:
        """Generates a response using the configured AI provider."""
        if self.provider == "openai":
            return self.call_openai_api(user_input, context, self.model)
        else:
            logger.error("Unsupported AI provider!")
            return ""

    def call_openai_api(self, prompt: str, context: str, model: str) -> str:
        """Calls OpenAI API to generate responses using the correct client method."""
        if not self.openai_client:
            logger.error("OpenAI client is not initialized!")
            return ""

        try:
            logger.debug("Sending request to OpenAI API...")
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt},
                ],
            )
            logger.debug(f"OpenAI Response: {response}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}\n{traceback.format_exc()}")
            return ""
