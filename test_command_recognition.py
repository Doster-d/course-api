import argparse
import asyncio
import json
from typing import Any, Dict

import aiohttp


async def test_command_recognition(
    text: str, session_id: str = "test_session"
) -> Dict[str, Any]:
    """Test the command recognition API with a text input.

    Args:
        text: The text to recognize commands from
        session_id: Session ID for game state tracking

    Returns:
        Response from the API
    """
    url = "http://localhost:8000/api/commands/recognize"

    payload = {"text": text, "session_id": session_id}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"Error: {response.status} - {error_text}")


async def populate_game_state(session_id: str = "test_session") -> None:
    """Populate the game state with some test data.

    Args:
        session_id: Session ID for game state tracking
    """
    base_url = f"http://localhost:8000/api/game/state/{session_id}"

    # Add some objects
    objects = [
        {
            "id": "sword_1",
            "name": "enchanted sword",
            "type": "weapon",
            "actions": ["take", "use", "examine", "attack"],
            "properties": {"damage": 15, "magical": True},
        },
        {
            "id": "key_1",
            "name": "rusty key",
            "type": "key",
            "actions": ["take", "use", "examine"],
            "properties": {"unlocks": "dungeon_door"},
        },
        {
            "id": "potion_1",
            "name": "health potion",
            "type": "consumable",
            "actions": ["take", "use", "examine"],
            "properties": {"health_restore": 50},
        },
    ]

    # Add some NPCs
    npcs = [
        {
            "id": "shopkeeper_1",
            "name": "village shopkeeper",
            "dialog_options": ["greet", "trade", "gossip", "farewell"],
            "properties": {"friendly": True, "has_rare_items": True},
        },
        {
            "id": "guard_1",
            "name": "castle guard",
            "dialog_options": ["greet", "question", "bribe", "farewell"],
            "properties": {"friendly": False, "bribable": True},
        },
    ]

    async with aiohttp.ClientSession() as session:
        # Add objects
        for obj in objects:
            async with session.post(
                f"{base_url}/add-object", json=obj
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error adding object: {error_text}")

        # Add NPCs
        for npc in npcs:
            async with session.post(
                f"{base_url}/add-npc", json=npc
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error adding NPC: {error_text}")

        # Update player position
        position = {"x": 10.0, "y": 0.0, "z": 5.0}
        async with session.post(
            f"{base_url}/update-position", json=position
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Error updating position: {error_text}")

    print(f"Game state populated for session {session_id}")


async def main():
    parser = argparse.ArgumentParser(description="Test command recognition")
    parser.add_argument("text", help="Text to recognize commands from")
    parser.add_argument(
        "--session",
        default="test_session",
        help="Session ID for game state tracking",
    )
    parser.add_argument(
        "--populate",
        action="store_true",
        help="Populate game state before testing",
    )

    args = parser.parse_args()

    try:
        if args.populate:
            print("Populating game state...")
            await populate_game_state(args.session)

        print(f"Testing command recognition for: '{args.text}'")
        result = await test_command_recognition(args.text, args.session)

        print("\nCommand Recognition Result:")
        print(json.dumps(result, indent=2))

        if result.get("recognized", False):
            print("\nSuccess! Command recognized.")
            print(f"Confidence: {result.get('confidence', 0)}")

            command = result.get("command", {})
            print(f"Command: {command.get('command')}")
            print(f"Object: {command.get('object')}")
            if "parameters" in command and command["parameters"]:
                print("Parameters:")
                for key, value in command["parameters"].items():
                    print(f"  {key}: {value}")
        else:
            print("\nNo command recognized.")
            if result.get("error"):
                print(f"Error: {result.get('error')}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
