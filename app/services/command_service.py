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
    """Service for recognizing game commands using Ollama with qwen3:0.6b.

    This service handles the recognition of game commands from transcribed text
    using specialized prompts for different command types.
    """

    def __init__(self):
        """Initialize the command recognition service."""
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        self.prompts = {}
        self.prompt_data = {}  # Store full prompt data including metadata
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load command recognition prompts from JSON files.

        Reads all JSON files in the prompts directory and loads them into
        the prompts dictionary. Creates the directory if it doesn't exist.
        """
        self.prompts = {}
        self.prompt_data = {}

        # Create prompts directory if it doesn't exist
        os.makedirs(self.prompts_dir, exist_ok=True)

        # Load prompts from files or use defaults
        for prompt_file in os.listdir(self.prompts_dir):
            if prompt_file.endswith(".json"):
                prompt_name = os.path.splitext(prompt_file)[0]
                prompt_path = os.path.join(self.prompts_dir, prompt_file)
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt_data = json.load(f)
                        
                        # Store the full prompt data
                        self.prompt_data[prompt_name] = prompt_data
                        
                        # Extract the template
                        prompt_template = prompt_data.get("prompt_template", "")
                        
                        # Validate that the template contains all required placeholders
                        required_placeholders = ["user_command"]
                        missing_placeholders = [
                            p
                            for p in required_placeholders
                            if f"{{{p}}}" not in prompt_template
                        ]

                        if missing_placeholders:
                            logger.error(
                                f"Prompt file {prompt_file} is missing required placeholders: {missing_placeholders}"
                            )
                            continue

                        # Check if the template contains any formatting errors
                        try:
                            # Test format with dummy values
                            prompt_template.format(
                                user_command="test",
                                commands="test",
                                objects="test",
                                interactions="test",
                                weapons="test",
                                targets="test",
                                npcs="test",
                                dialog_options="test",
                            )

                            # Store the template if validation passes
                            self.prompts[prompt_name] = prompt_template
                            logger.info(f"Loaded prompt from {prompt_file}")
                        except KeyError as e:
                            logger.error(
                                f"Prompt file {prompt_file} contains an invalid placeholder: {str(e)}"
                            )
                            # Remove the offending placeholder with a regex
                            placeholder_pattern = (
                                r"\{([^{}]*"
                                + str(e).replace("'", "")
                                + r"[^{}]*)\}"
                            )
                            cleaned_prompt = re.sub(
                                placeholder_pattern, "", prompt_template
                            )
                            logger.info(
                                "Removed invalid placeholder, continuing with cleaned prompt"
                            )
                            self.prompts[prompt_name] = cleaned_prompt
                except Exception as e:
                    logger.error(
                        f"Error loading prompt file {prompt_file}: {str(e)}"
                    )
                    logger.error(traceback.format_exc())

    def _prepare_prompt(
        self, prompt_name: str, transcription: str, game_state: Dict[str, Any]
    ) -> str:
        """Prepare a prompt for command recognition.

        Args:
            prompt_name: Name of the prompt template to use
            transcription: Transcribed text from ASR
            game_state: Current game state information

        Returns:
            Formatted prompt string with game state variables inserted
        """
        if prompt_name not in self.prompts:
            logger.warning(
                f"Prompt '{prompt_name}' not found, using base_commands"
            )
            prompt_name = "base_commands"

        prompt_template = self.prompts[prompt_name]
        prompt_metadata = self.prompt_data[prompt_name]

        # Pre-process the template to escape curly braces in the JSON example
        # This prevents format() from treating parts of the JSON example as placeholders
        lines = prompt_template.split("\n")
        processed_lines = []
        in_answer_block = False

        for line in lines:
            if "<answer>" in line:
                in_answer_block = True
                processed_lines.append(line)
            elif "</answer>" in line:
                in_answer_block = False
                processed_lines.append(line)
            elif in_answer_block and ("{" in line or "}" in line):
                # Replace all curly braces with doubled versions to escape them
                escaped_line = line.replace("{", "{{").replace("}", "}}")
                # But restore the actual placeholders if any
                actual_placeholders = [
                    "commands",
                    "objects",
                    "interactions",
                    "user_command",
                    "weapons",
                    "targets",
                    "npcs",
                    "dialog_options",
                ]
                for placeholder in actual_placeholders:
                    escaped_line = escaped_line.replace(
                        f"{{{{{placeholder}}}}}", f"{{{placeholder}}}"
                    )
                processed_lines.append(escaped_line)
            else:
                processed_lines.append(line)

        processed_template = "\n".join(processed_lines)

        # Extract information from game state
        commands = game_state.get("commands", [])
        objects = game_state.get("objects", [])
        interactions = game_state.get("interactions", [])
        weapons = game_state.get("weapons", [])
        targets = game_state.get("targets", [])
        npcs = game_state.get("npcs", [])
        dialog_options = game_state.get("dialog_options", [])

        # Prepare format arguments with default values
        commands_str = ", ".join(commands) or "move, interact, attack, talk"
        objects_str = ", ".join(objects) or "door, key, book, chest"
        interactions_str = (
            ", ".join(interactions) or "open, close, take, use, examine"
        )
        weapons_str = ", ".join(weapons) or "sword, bow, axe"
        targets_str = ", ".join(targets) or "enemy, monster, target"
        npcs_str = ", ".join(npcs) or "merchant, guard, villager"
        dialog_options_str = (
            ", ".join(dialog_options) or "greet, ask, buy, sell"
        )

        try:
            # Format the prompt with game state information
            formatted_prompt = processed_template.format(
                user_command=transcription,
                commands=commands_str,
                objects=objects_str,
                interactions=interactions_str,
                weapons=weapons_str,
                targets=targets_str,
                npcs=npcs_str,
                dialog_options=dialog_options_str,
            )
            logger.debug(
                f"Prepared prompt for {prompt_name}:\n{formatted_prompt[:200]}..."
            )
            return formatted_prompt
        except KeyError as e:
            # If we get a KeyError, log it and attempt to fix the template
            error_msg = f"Error formatting prompt {prompt_name}: Missing placeholder {e}"
            logger.error(error_msg)

            # Last resort - return a minimal working prompt
            return f'Analyze this command: "{transcription}". Respond with JSON in <answer> tags.'
        except Exception as e:
            # For any other exceptions, log and return a minimal prompt
            logger.error(f"Unexpected error formatting prompt: {str(e)}")
            logger.error(traceback.format_exc())
            return f'Analyze this command: "{transcription}". Respond with JSON in <answer> tags.'

    async def recognize_command(
        self, transcription: str, game_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Recognize a command from transcribed text.

        Processes the transcribed text through specialized prompt types to
        find the best matching command interpretation.

        Args:
            transcription: Transcribed text from ASR
            game_state: Current game state information

        Returns:
            Dictionary with recognized command information including:
            - recognized: Whether a command was successfully recognized
            - command: The structured command data if recognized
            - confidence: Confidence score of the recognition
            - error: Error message if recognition failed
        """
        if game_state is None:
            game_state = {}

        logger.info(f"Recognizing command from: {transcription}")

        # Initialize response
        command_response = {
            "recognized": False,
            "command": None,
            "confidence": 0.0,
            "error": None,
        }

        # Skip processing if transcription is empty
        if not transcription or transcription.strip() == "":
            return command_response

        # Process with each prompt type to find the best match
        prompt_types = [
            "movement_commands",
            "object_interactions",
            "combat_commands",
            "dialog_commands",
            "base_commands",
        ]

        best_match = None
        highest_confidence = 0.0

        for prompt_type in prompt_types:
            try:
                # Prepare the prompt
                prompt = self._prepare_prompt(prompt_type, transcription, game_state)

                # Get response from Ollama
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "qwen3:0.6b",
                            "prompt": prompt,
                            "stream": False,
                        },
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get("response", "")

                            # Extract JSON from response
                            try:
                                # Find content between <answer> tags
                                answer_match = re.search(
                                    r"<answer>(.*?)</answer>", response_text, re.DOTALL
                                )
                                if answer_match:
                                    command_data = json.loads(answer_match.group(1))
                                    
                                    # Get confidence from response
                                    confidence = command_data.get("confidence", 0.0)
                                    
                                    # Check if this is a better match
                                    if confidence > highest_confidence:
                                        highest_confidence = confidence
                                        best_match = command_data
                            except json.JSONDecodeError:
                                logger.error(
                                    f"Failed to parse JSON from response: {response_text}"
                                )
                                continue
                        else:
                            logger.error(
                                f"Error from Ollama API: {response.status} - {await response.text()}"
                            )
            except Exception as e:
                logger.error(f"Error processing prompt type {prompt_type}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Update response with best match if found
        if best_match and highest_confidence > 0.0:
            command_response.update(
                {
                    "recognized": True,
                    "command": best_match,
                    "confidence": highest_confidence,
                }
            )
        else:
            command_response["error"] = "No valid command recognized"

        return command_response
