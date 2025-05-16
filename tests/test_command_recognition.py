import sys
from pathlib import Path

import pytest

# Add parent directory to path to import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.command_service import CommandRecognitionService


@pytest.mark.asyncio
async def test_command_recognition_service_init():
    """Test that the CommandRecognitionService initializes properly."""
    service = CommandRecognitionService()
    assert service is not None
    assert service.prompts is not None
    assert service.prompt_data is not None
    assert "base_commands" in service.prompts
    assert "movement_commands" in service.prompts
    assert "object_interactions" in service.prompts
    assert "combat_commands" in service.prompts
    assert "dialog_commands" in service.prompts
    
    # Check prompt data structure
    for prompt_name in service.prompt_data:
        prompt_data = service.prompt_data[prompt_name]
        assert "description" in prompt_data
        assert "version" in prompt_data
        assert "language_support" in prompt_data
        assert "confidence_guidelines" in prompt_data
        assert "prompt_template" in prompt_data


@pytest.mark.asyncio
async def test_prepare_movement_prompt():
    """Test preparing a movement prompt."""
    service = CommandRecognitionService()

    transcription = "go forward"
    game_state = {
        "commands": ["go", "take", "use"],
        "objects": ["sword", "potion"],
        "interactions": ["open", "close"],
    }

    prompt = service._prepare_prompt(
        "movement_commands", transcription, game_state
    )

    assert transcription in prompt
    assert "forward" in prompt
    assert "backward" in prompt
    assert "<answer>" in prompt
    assert "confidence" in prompt


@pytest.mark.asyncio
async def test_prepare_object_prompt():
    """Test preparing an object interaction prompt."""
    service = CommandRecognitionService()

    transcription = "take the sword"
    game_state = {
        "commands": ["go", "take", "use"],
        "objects": ["sword", "potion"],
        "interactions": ["open", "close"],
    }

    prompt = service._prepare_prompt(
        "object_interactions", transcription, game_state
    )

    assert transcription in prompt
    assert "sword, potion" in prompt
    assert "take" in prompt
    assert "examine" in prompt
    assert "<answer>" in prompt
    assert "confidence" in prompt


@pytest.mark.asyncio
async def test_recognize_command():
    """Test command recognition with different command types."""
    service = CommandRecognitionService()
    
    # Test movement command
    movement_result = await service.recognize_command(
        "go forward",
        {
            "commands": ["go", "take", "use"],
            "objects": ["sword", "potion"],
            "interactions": ["open", "close"],
        }
    )
    assert isinstance(movement_result, dict)
    assert "recognized" in movement_result
    assert "confidence" in movement_result
    assert "command" in movement_result
    
    # Test object interaction command
    object_result = await service.recognize_command(
        "take the sword",
        {
            "commands": ["go", "take", "use"],
            "objects": ["sword", "potion"],
            "interactions": ["open", "close"],
        }
    )
    assert isinstance(object_result, dict)
    assert "recognized" in object_result
    assert "confidence" in object_result
    assert "command" in object_result
    
    # Test invalid command
    invalid_result = await service.recognize_command(
        "this is not a command",
        {
            "commands": ["go", "take", "use"],
            "objects": ["sword", "potion"],
            "interactions": ["open", "close"],
        }
    )
    assert isinstance(invalid_result, dict)
    assert not invalid_result["recognized"]
    assert invalid_result["confidence"] == 0.0
    assert invalid_result["command"] is None
