#!/usr/bin/env python3
"""
Test script for checking Russian command recognition.
Sends requests to the API for recognizing Russian commands.
"""

import asyncio
import json
import logging
import os
import sys
import time
import re

import aiohttp
import requests

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
OLLAMA_BASE_URL = "http://localhost:11434"

# Path to prompts directory
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Тестовые русские команды
TEST_COMMANDS = [
    "передвинься вперед",
    "иди вперед",
    "двигайся влево",
    "повернись направо",
    "атаковать волка",
    "использовать меч",
    "поговори с торговцем",
    "открой дверь",
    "подними ключ",
    "Подписывайтесь на наш канал, ставьте лайки и пишите комментарии.  на  мой канал, ставьте лайки и подписывайтесь на мои видео.  на мой канал, ставьте лайки и подписывайтесь на мои видео. Передвинуть этот  предмет  на  пять  метров.  вправо."
]


def load_prompt(prompt_name):
    """Load a prompt from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
            return prompt_data.get("prompt_template", "")
    except Exception as e:
        logger.error(f"Error loading prompt {prompt_name}: {str(e)}")
        return ""


def check_api_health():
    """Check if the API is up and running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API is healthy")
            return True
        else:
            logger.error(
                f"❌ API health check failed with status {response.status_code}"
            )
            return False
    except Exception as e:
        logger.error(f"❌ API health check failed: {str(e)}")
        return False


def check_ollama_status():
    """Check if Ollama is running and the model is loaded."""
    try:
        # Check if Ollama server is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error(
                f"❌ Ollama server check failed with status {response.status_code}"
            )
            return False

        models = response.json().get("models", [])
        if not models:
            logger.error("❌ No models found in Ollama")
            return False

        # Check if qwen3:0.6b is loaded
        model_names = [model.get("name") for model in models]
        if "qwen3:0.6b" not in model_names:
            logger.error("❌ qwen3:0.6b model not found in Ollama")
            logger.info(f"Available models: {', '.join(model_names)}")

            # Try to pull the model
            logger.info("Attempting to pull qwen3:0.6b model...")
            os.system("ollama pull qwen3:0.6b")

            # Check again
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if "qwen3:0.6b" in model_names:
                    logger.info("✅ qwen3:0.6b model successfully pulled")
                    return True

            return False

        logger.info("✅ Ollama is running with qwen3:0.6b model")
        return True

    except Exception as e:
        logger.error(f"❌ Ollama check failed: {str(e)}")
        return False


async def test_command_recognition():
    """Test command recognition for Russian commands."""
    print("\n🔍 Testing Russian command recognition...")
    
    # Test each command
    for command in TEST_COMMANDS:
        print(f"\nTesting command: {command}")
        
        # Test with each prompt type
        prompt_types = [
            "movement_commands",
            "object_interactions",
            "combat_commands",
            "dialog_commands",
            "base_commands"
        ]
        
        for prompt_type in prompt_types:
            # Load and prepare the prompt
            prompt_template = load_prompt(prompt_type)
            if not prompt_template:
                print(f"❌ Failed to load {prompt_type}.json prompt")
                continue
                
            # Replace the placeholder for user_command
            prompt = prompt_template.replace("{user_command}", command)
            
            # Send request to Ollama
            payload = {
                "model": "qwen3:0.6b",
                "prompt": prompt,
                "stream": False
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get("response", "")
                            
                            # Extract JSON from response
                            try:
                                # Find content between <answer> tags
                                answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
                                if answer_match:
                                    command_data = json.loads(answer_match.group(1))
                                    confidence = command_data.get("confidence", 0.0)
                                    
                                    print(f"✅ {prompt_type}: Confidence {confidence:.2f}")
                                    print(f"Command data: {json.dumps(command_data, indent=2, ensure_ascii=False)}")
                                else:
                                    print(f"❌ {prompt_type}: No valid response format")
                            except json.JSONDecodeError:
                                print(f"❌ {prompt_type}: Failed to parse JSON response")
                        else:
                            print(f"❌ {prompt_type}: API error {response.status}")
            except Exception as e:
                print(f"❌ {prompt_type}: Error {str(e)}")


async def test_response_format():
    """Test the raw response format from Ollama for a command recognition prompt."""
    print("\n🔍 Testing Ollama response format...")

    # Load the movement_commands prompt and replace placeholders
    prompt_template = load_prompt("movement_commands")
    if not prompt_template:
        print("❌ Failed to load movement_commands.json prompt")
        return False

    # Replace the placeholder for user_command
    prompt = prompt_template.replace("{user_command}", "передвинься назад")

    payload = {"model": "qwen3:0.6b", "prompt": prompt, "stream": False}
    url = f"{OLLAMA_BASE_URL}/api/generate"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get("response", "")
                    print("\nRaw response:")
                    print(response_text)
                    return True
                else:
                    print(f"❌ API error: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Russian command recognition tests...")
    
    # Test command recognition
    await test_command_recognition()
    
    # Test response format
    await test_response_format()
    
    print("\n✨ Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
