import json
import logging
import os
import re
import traceback
from typing import Any, Dict, Optional

import aiohttp

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class CommandRecognitionService:
    """Service for recognizing game commands using Ollama."""

    def __init__(self):
        """Initialize the command recognition service."""
        self.prompts_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "prompts"
        )
        self.prompts = {}
        self.prompt_data = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load command recognition prompts from JSON files."""
        self.prompts = {}
        self.prompt_data = {}

        # Create prompts directory if it doesn't exist
        os.makedirs(self.prompts_dir, exist_ok=True)

        # Load prompts from files
        for prompt_file in os.listdir(self.prompts_dir):
            if prompt_file.endswith(".json"):
                prompt_name = os.path.splitext(prompt_file)[0]
                prompt_path = os.path.join(self.prompts_dir, prompt_file)
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt_data = json.load(f)
                        self.prompt_data[prompt_name] = prompt_data
                        prompt_template = prompt_data.get("prompt_template", "")
                        self.prompts[prompt_name] = prompt_template
                        logger.info(f"Loaded prompt from {prompt_file}")
                except Exception as e:
                    logger.error(f"Error loading prompt {prompt_file}: {str(e)}")

    def _prepare_prompt(self, prompt_name: str, transcription: str) -> str:
        """Prepare a prompt with the user command."""
        if prompt_name not in self.prompts:
            logger.warning(f"Prompt '{prompt_name}' not found, using base_commands")
            prompt_name = "base_commands"

        prompt_template = self.prompts[prompt_name]
        return prompt_template.replace("{user_command}", transcription)

    async def recognize_command(self, transcription: str, game_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recognize a command from transcribed text.

        Args:
            transcription: Transcribed text
            game_state: Current game state information (not used in this version)

        Returns:
            Dictionary with recognized command information
        """
        logger.info(f"Recognizing command: {transcription}")

        # Step 1: Use the base_commands router to determine command type
        router_prompt = self._prepare_prompt("base_commands", transcription)
        router_result = await self._query_ollama(router_prompt, is_router=True)
        
        logger.info(f"Router result: {router_result}")
        
        if not router_result or "command_type" not in router_result:
            logger.error(f"Router failed to determine command type: {router_result}")
            return {
                "recognized": False,
                "command": None,
                "confidence": 0.0,
                "error": "Failed to determine command type"
            }

        command_type = router_result["command_type"]
        router_confidence = float(router_result.get("confidence", 0.0))
        logger.info(f"Router identified command type: {command_type} with confidence: {router_confidence}")

        if router_confidence < 0.3:
            return {
                "recognized": False,
                "command": router_result,
                "confidence": router_confidence,
                "error": "Low confidence in command type"
            }

        # Step 2: Use the specialized handler to process the command
        handler_prompt = self._prepare_prompt(command_type, transcription)
        handler_result = await self._query_ollama(handler_prompt, is_router=False)
        
        logger.info(f"Handler result: {handler_result}")
        
        if not handler_result:
            return {
                "recognized": False,
                "command": router_result,
                "confidence": router_confidence,
                "error": f"Failed to process command with {command_type} handler"
            }

        # Step 3: Format the final result
        handler_confidence = float(handler_result.get("confidence", 0.0))
        final_confidence = min(router_confidence, handler_confidence)

        return {
            "recognized": final_confidence >= 0.5,
            "command": {
                "type": command_type,
                "details": handler_result,
                "alternatives": router_result.get("alternative_types", []),
            },
            "confidence": final_confidence,
            "error": None
        }

    async def _query_ollama(self, prompt: str, is_router: bool = False) -> Optional[Dict[str, Any]]:
        """Query Ollama with a prompt and extract the JSON response.
        
        Args:
            prompt: The formatted prompt to send to Ollama
            is_router: Whether this is a router prompt (base_commands)
            
        Returns:
            Parsed JSON response or None if parsing fails
        """
        try:
            logger.info(f"Querying Ollama at {settings.OLLAMA_HOST}")
            logger.info(f"Request type: {'router' if is_router else 'handler'}")
            
            # Special handling for Russian movement commands
            # if is_router and any(cmd in prompt.lower() for cmd in ["иди вперед", "идти вперед", "двигайся вперед"]):
            #     logger.info("Direct router response for Russian movement command")
            #     return {
            #         "command_type": "movement_commands",
            #         "confidence": 0.9,
            #         "explanation": "This is clearly a movement command in Russian",
            #         "alternative_types": [],
            #         "reasoning": "The command contains a movement verb in Russian"
            #     }
                
            # if not is_router and "movement_commands" in prompt and any(cmd in prompt.lower() for cmd in ["иди вперед", "идти вперед", "двигайся вперед"]):
            #     logger.info("Direct handler response for Russian movement command")
            #     return {
            #         "command": "move",
            #         "direction": "forward",
            #         "parameters": {"distance": 1},
            #         "confidence": 0.9
            #     }
            
            # Send request to Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{settings.OLLAMA_HOST}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                    },
                ) as response:
                    if response.status != 200:
                        logger.error(f"Ollama API error: {response.status}")
                        error_text = await response.text()
                        logger.error(f"Error details: {error_text}")
                        return None

                    result = await response.json()
                    response_text = result.get("response", "")
                    logger.info(f"Received response from Ollama (length: {len(response_text)})")
                    
                    # Extract JSON from between <answer> tags
                    match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
                    if not match:
                        logger.error("No answer tags found in response")
                        logger.error(f"Response content: {response_text}")
                        return None

                    # Parse the JSON response
                    try:
                        json_text = match.group(1).strip()
                        logger.info(f"Extracted JSON text: {json_text}")
                        parsed_result = json.loads(json_text)
                        
                        # Fix common issues with response structure for router requests
                        if is_router and "command" in parsed_result and "command_type" not in parsed_result:
                            command = parsed_result["command"]
                            logger.info(f"Converting 'command' to 'command_type': {command}")
                            
                            # Map common commands to command types
                            command_type_map = {
                                "move": "movement_commands",
                                "go": "movement_commands",
                                "attack": "combat_commands",
                                "fight": "combat_commands",
                                "talk": "dialog_commands",
                                "speak": "dialog_commands",
                                "use": "object_interactions",
                                "take": "object_interactions",
                                "open": "object_interactions"
                            }
                            command_type = command_type_map.get(command, "unknown")
                            return {
                                "command_type": command_type,
                                "confidence": parsed_result.get("confidence", 0.7),
                                "explanation": f"Command mapped from '{command}' to '{command_type}'",
                                "alternative_types": [],
                                "reasoning": "Converted from command response"
                            }
                        
                        return parsed_result
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {str(e)}")
                        logger.error(f"Invalid JSON: {json_text}")
                        return None

        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            logger.error(traceback.format_exc())
            return None
