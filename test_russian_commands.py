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

# Тестовые русские команды по категориям
TEST_COMMANDS = {
    "movement_commands": [
        "передвинься вперед",
        "иди вперед на два шага",
        "двигайся влево",
        "повернись направо",
        "беги вперёд",
        "прыгай вверх",
        "шагай назад"
    ],
    "combat_commands": [
        "атаковать волка",
        "защищайся от врага",
        "используй меч",
        "блокируй удар",
        "уворачивайся от атаки",
        "колдуй огненный шар"
    ],
    "dialog_commands": [
        "поговори с торговцем",
        "спроси о задании",
        "поприветствуй стражника",
        "попрощайся с торговцем",
        "скажи привет",
        "спроси про товары"
    ],
    "object_interactions": [
        "открой дверь",
        "подними ключ",
        "осмотри сундук",
        "возьми меч",
        "используй зелье",
        "брось камень"
    ]
}

# Смешанные команды для тестирования
MIXED_COMMANDS = [
    "go вперед",
    "attack монстра",
    "talk to торговец",
    "поднять key",
    "use меч",
    "двигайся forward",
    "cast огненный шар",
    "open дверь"
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


async def test_command_recognition(command: str) -> dict:
    """Test command recognition for a single command."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/api/commands/recognize",
                json={"text": command}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"API error: {response.status} - {error}")
    except Exception as e:
        logger.error(f"Error testing command: {str(e)}")
        return {
            "recognized": False,
            "error": str(e),
            "confidence": 0.0
        }


async def test_command_category(category: str, commands: list):
    """Test a category of commands."""
    print(f"\n🔍 Testing {category}...")
    
    results = {
        "total": len(commands),
        "recognized": 0,
        "correct_type": 0,
        "avg_confidence": 0.0
    }
    
    for command in commands:
        print(f"\nTesting: {command}")
        result = await test_command_recognition(command)
        
        if result["recognized"]:
            results["recognized"] += 1
            command_type = result["command"]["type"]
            confidence = result["confidence"]
            results["avg_confidence"] += confidence
            
            if command_type == category:
                results["correct_type"] += 1
                print(f"✅ Correct type ({confidence:.2f}): {command_type}")
                print(f"Details: {json.dumps(result['command']['details'], indent=2, ensure_ascii=False)}")
            else:
                print(f"❌ Wrong type ({confidence:.2f}): got {command_type}, expected {category}")
        else:
            print(f"❌ Not recognized: {result.get('error', 'Unknown error')}")
    
    if results["recognized"] > 0:
        results["avg_confidence"] /= results["recognized"]
    
    print(f"\n📊 Category Results for {category}:")
    print(f"Total commands: {results['total']}")
    print(f"Recognized: {results['recognized']} ({results['recognized']/results['total']*100:.1f}%)")
    print(f"Correct type: {results['correct_type']} ({results['correct_type']/results['total']*100:.1f}%)")
    print(f"Average confidence: {results['avg_confidence']:.2f}")
    
    return results


async def test_mixed_language_commands():
    """Test mixed language command handling."""
    print("\n🔍 Testing mixed language commands...")
    
    results = {
        "total": len(MIXED_COMMANDS),
        "recognized": 0,
        "avg_confidence": 0.0
    }
    
    for command in MIXED_COMMANDS:
        print(f"\nTesting: {command}")
        result = await test_command_recognition(command)
        
        if result["recognized"]:
            results["recognized"] += 1
            confidence = result["confidence"]
            results["avg_confidence"] += confidence
            
            print(f"✅ Recognized ({confidence:.2f}): {result['command']['type']}")
            print(f"Details: {json.dumps(result['command']['details'], indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ Not recognized: {result.get('error', 'Unknown error')}")
    
    if results["recognized"] > 0:
        results["avg_confidence"] /= results["recognized"]
    
    print(f"\n📊 Mixed Language Results:")
    print(f"Total commands: {results['total']}")
    print(f"Recognized: {results['recognized']} ({results['recognized']/results['total']*100:.1f}%)")
    print(f"Average confidence: {results['avg_confidence']:.2f}")
    
    return results


async def main():
    """Main test function."""
    print("🚀 Starting Russian command recognition tests...")
    
    # Check services
    if not check_api_health() or not check_ollama_status():
        print("❌ Service checks failed. Please ensure all services are running.")
        return
    
    # Test each category
    overall_results = {}
    for category, commands in TEST_COMMANDS.items():
        overall_results[category] = await test_command_category(category, commands)
    
    # Test mixed language commands
    overall_results["mixed"] = await test_mixed_language_commands()
    
    # Print overall summary
    print("\n📈 Overall Test Summary:")
    total_commands = sum(r["total"] for r in overall_results.values())
    total_recognized = sum(r["recognized"] for r in overall_results.values())
    total_correct = sum(r.get("correct_type", 0) for r in overall_results.values())
    avg_confidence = sum(r["avg_confidence"] * r["recognized"] for r in overall_results.values()) / total_recognized if total_recognized > 0 else 0
    
    print(f"Total commands tested: {total_commands}")
    print(f"Total recognized: {total_recognized} ({total_recognized/total_commands*100:.1f}%)")
    print(f"Total correct type: {total_correct} ({total_correct/total_commands*100:.1f}%)")
    print(f"Overall average confidence: {avg_confidence:.2f}")
    
    print("\n✨ Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
