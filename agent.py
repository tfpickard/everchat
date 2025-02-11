#!/usr/bin/env python

import json
import logging
import os
import random
import re  # Extract only the JSON portion from the LLM response, in case additional text is included
import time
import traceback
from datetime import datetime
from typing import Dict, List

import openai
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from textblob import TextBlob

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
        self.model: str = model
        self.provider: str = provider
        self.traits: list[str] = traits
        self.name: str = "unknown"
        self.personality: str = ""
        self.backstory: str = ""
        self.country: str = ""
        self.city: str = ""
        self.birth_date: str = ""
        self.age: str = ""
        self.gender: str = ""
        self.occupation: str = ""
        self.hobbies: str = ""
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
        self.topics = config.get("conversation", {}).get("topics", [])
        self.generate_backstory()

    def generate_response(self) -> str:
        """Generates a response while strictly enforcing sentence length distribution."""
        sentence_distribution = {
            1: 0.2,
            2: 0.5,
            3: 0.15,
            4: 0.15,
        }
        sentence_count = random.choices(
            list(sentence_distribution.keys()), list(sentence_distribution.values())
        )[0]
        sentences = ["This is a concise response." for _ in range(sentence_count)]
        return " ".join(sentences)

    #
    # def respond(self, chat_history: List[Dict[str, str]]) -> Dict[str, str]:
    #     start_time = time.time()
    #
    #     # Generate response with controlled sentence distribution
    #     response = self.generate_response()  # <- Called here
    #     sentence_count = len(response.split(". "))
    #     sentiment = self.analyze_sentiment(response)
    #     emotions = self.analyze_emotion(response)
    #
    #     elapsed_time = time.time() - start_time
    #
    #     log_entry = {
    #         "role": "assistant",
    #         "name": self.name,
    #         "content": response,
    #         "timestamp": datetime.utcnow().isoformat(),
    #         "word_count": len(response.split()),
    #         "sentence_count": sentence_count,
    #         "sentiment": sentiment,
    #         "emotion": emotions,
    #         "elapsed_time": elapsed_time,
    #     }
    #
    #     write_log(setup_logging(), log_entry)
    #
    #     return log_entry
    #
    def respond(self, chat_history: List[Dict[str, str]]) -> str:
        start_time = time.time()

        # Generate response with controlled sentence distribution
        context = (
            f"Name: {self.name} "
            f"Age: {self.age} "
            f"Gender: {self.gender} "
            f"Occupation: {self.occupation} "
            f"Location: {self.city}, {self.country} "
            f"Backstory: {self.backstory} "
            "Conversation History: "
            + " ".join([msg["content"] for msg in chat_history[-5:]])
        )
        last_message = (
            chat_history[-1]["content"] if chat_history else "Hello, who are you?"
        )
        try:
            sentence_distribution = {
                1: 0.2,
                2: 0.5,
                3: 0.15,
                4: 0.15,
            }
            sentence_count = random.choices(
                list(sentence_distribution.keys()), list(sentence_distribution.values())
            )[0]
            prompt = f"Generate a response in {sentence_count} sentences. Keep it concise and on-topic."
            response = self.generate_llm_response(last_message + " " + prompt, context)
        except Exception as e:
            logger.warning(f"LLM response failed, using fallback: {e}")
            response = self.generate_response()
        sentence_count = len(response.split(". "))
        sentiment = self.analyze_sentiment(response)
        emotions = self.analyze_emotion(response)

        elapsed_time = time.time() - start_time

        log_entry = {
            "role": "assistant",
            "name": self.name,
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
            "word_count": len(response.split()),
            "sentence_count": sentence_count,
            "sentiment": sentiment,
            "emotion": emotions,
            "elapsed_time": elapsed_time,
        }

        write_log(log_entry)
        return response

    # def respond(self, chat_history: List[Dict[str, str]]) -> str:
    #     """Generates a response based on conversation history and agent identity."""
    #     context = (
    #         f"Name: {self.name} "
    #         f"Age: {self.age} "
    #         f"Gender: {self.gender} "
    #         f"Occupation: {self.occupation} "
    #         f"Location: {self.city}, {self.country} "
    #         f"Backstory: {self.backstory} "
    #         "Conversation History: "
    #         + " ".join([msg["content"] for msg in chat_history[-5:]])
    #     )
    #     last_message = (
    #         chat_history[-1]["content"] if chat_history else "Hello, who are you?"
    #     )
    #     return self.generate_llm_response(last_message, context)

    def generate_intro(self, topics: List[str]) -> str:
        """Generates an opening statement about a unique topic based on provided topics."""
        intro_prompt = (
            f"You are {self.name}, a {self.age}-year-old {self.occupation} from {self.city}, {self.country}. "
            f"Using the following topics as inspiration: {', '.join(topics)}, generate a new, unique topic for discussion. "
            "Ensure your topic is engaging and thought-provoking."
        )
        return self.generate_llm_response(
            intro_prompt, "(Introduction Topic Generation)"
        )

    def generate_identity(self) -> None:
        """Requests the LLM to generate a full human-like identity in JSON format."""
        identity_prompt = (
            "Generate a realistic human identity for the year 2025 in JSON format. "
            "The JSON object should include: "
            "'name' (full name appropriate for their country), 'birth_date', 'age', 'gender', "
            "'country', 'city', 'occupation', and 'hobbies'. "
            "Ensure the name aligns with the cultural and linguistic background of the selected country."
            "Geography selection should follow these weightings: 70% English-speaking countries and Western Europe, 20% East, Southeast, and South Asia, and 10% all other locations."
            "The birth date should follow a normal distribution centered on 1990."
            "Hobbies should likely include at least one related to science and/or technology with an academic focus."
            "Output must be strictly JSON without any additional text."
        )
        logger.debug("Calling LLM for identity generation...")
        # Extract only the JSON portion from the LLM response, in case additional text is included

        generated_identity = self.generate_llm_response(
            identity_prompt, "(Identity Generation Context)"
        )
        match = re.search(r"{.*}", generated_identity, re.DOTALL)
        generated_identity = (
            match.group(0) if match else generated_identity.strip().strip("`")
        )

        # Retry JSON decoding up to 3 times if the initial response is malformed
        attempts = 3
        for _ in range(attempts):
            try:
                parsed_identity = json.loads(generated_identity)
                console.print_json(data=parsed_identity)
                break
            except json.JSONDecodeError:
                if _ < attempts - 1:
                    print(generated_identity)
                    logger.warning("Invalid JSON received, retrying...")
                    generated_identity = (
                        self.generate_llm_response(
                            identity_prompt, "(Identity Generation Context)"
                        )
                        .strip()
                        .strip("`")
                    )
                    continue
                logger.error("Invalid JSON received from LLM")

        if generated_identity:
            try:
                details = json.loads(generated_identity)
                self.name = details.get("name", "Unknown")
                self.birth_date = details.get("birth_date", "")
                self.age = details.get("age", "")
                self.gender = details.get("gender", "")
                self.country = details.get("country", "")
                self.city = details.get("city", "")
                self.occupation = details.get("occupation", "")
                self.hobbies = details.get("hobbies", "")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse identity JSON: {e}")
                self.name = "Unknown"
        else:
            logger.error("Identity generation failed! Using default values.")

        logger.info(
            f"Generated Identity: {self.name}, {self.age} years old from {self.city}, {self.country}"
        )

    def generate_backstory(self) -> None:
        """Requests the LLM to generate a unique backstory for the agent."""
        prompt = (
            f"Generate a unique backstory for a person named {self.name}, "
            f"a {self.age}-year-old {self.occupation} from {self.city}, {self.country}. "
            "The backstory should focus on transformational life events which would"
            "influence this person's worldview. Keep it concise and don't elaborate much."
            "This story is just a beginning and more details will be added later."
        )
        logger.info("Calling LLM for backstory generation...")
        self.backstory = self.generate_llm_response(
            prompt, "(Backstory Generation Context)"
        )
        logger.info(f"Generated Backstory for {self.name}: {self.backstory}")

    def analyze_sentiment(self, text: str) -> dict[str, float]:
        analysis = TextBlob(text)
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity,
            "detailed": {
                "very_positive": 1.0 if analysis.sentiment.polarity > 0.75 else 0.0,
                "positive": 1.0 if 0.25 < analysis.sentiment.polarity <= 0.75 else 0.0,
                "neutral": 1.0 if -0.25 <= analysis.sentiment.polarity <= 0.25 else 0.0,
                "negative": (
                    1.0 if -0.75 <= analysis.sentiment.polarity < -0.25 else 0.0
                ),
                "very_negative": 1.0 if analysis.sentiment.polarity < -0.75 else 0.0,
            },
        }

    def analyze_emotion(self, text: str) -> dict[str, float]:
        emotions = {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "neutral": 1.0,
            "fearful": 0.0,
            "surprised": 0.0,
        }
        polarity = TextBlob(text).sentiment.polarity

        if polarity > 0.5:
            emotions["happy"] = polarity
            emotions["neutral"] = 1.0 - polarity
        elif polarity < -0.5:
            emotions["sad"] = abs(polarity)
            emotions["neutral"] = 1.0 - abs(polarity)
        elif -0.5 <= polarity <= -0.1:
            emotions["angry"] = abs(polarity)
            emotions["neutral"] = 1.0 - abs(polarity)

        # Adding more nuance
        if abs(polarity) > 0.75:
            emotions["fearful"] = abs(polarity) / 2
        if 0.1 < abs(polarity) < 0.3:
            emotions["surprised"] = abs(polarity)

        return emotions

    def count_sentences(self, text: str) -> int:
        return len(re.split(r"[.!?]+", text)) - 1

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


log_filename: str = ""


def setup_logging():
    global log_filename
    log_dir = os.path.abspath("logs")
    os.makedirs("logs", exist_ok=True)
    log_filename = f"{log_dir}/conversation_{datetime.now().isoformat()}.json"
    symlink_path = f"{log_dir}/current.json"

    if os.path.islink(symlink_path):
        print(f"path exists: {symlink_path}")
        # os.unlink(symlink_path)
        os.remove(symlink_path)
    os.symlink(log_filename, symlink_path)

    return log_filename


def write_log(log_entry: dict[str, str]):
    global log_filename
    if log_filename == "":
        log_filename = setup_logging()
    with open(log_filename, "a") as log_file:
        json.dump(log_entry, log_file)
        _ = log_file.write("\n")
