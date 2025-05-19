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


async def test_command_types():
    """Test recognition of different command types in both English and Russian."""
    test_cases = [
        # Movement commands
        {
            "text": "go forward",
            "expected_type": "movement_commands",
            "language": "en",
        },
        {
            "text": "иди вперед",
            "expected_type": "movement_commands",
            "language": "ru",
        },
        # Combat commands
        {
            "text": "attack the enemy",
            "expected_type": "combat_commands",
            "language": "en",
        },
        {
            "text": "атаковать врага",
            "expected_type": "combat_commands",
            "language": "ru",
        },
        # Dialog commands
        {
            "text": "talk to merchant",
            "expected_type": "dialog_commands",
            "language": "en",
        },
        {
            "text": "поговорить с торговцем",
            "expected_type": "dialog_commands",
            "language": "ru",
        },
        # Object interaction commands
        {
            "text": "pick up the key",
            "expected_type": "object_interactions",
            "language": "en",
        },
        {
            "text": "поднять ключ",
            "expected_type": "object_interactions",
            "language": "ru",
        },
    ]

    print("\n🔍 Testing command type recognition...")

    for case in test_cases:
        print(f"\nTesting {case['language'].upper()} command: {case['text']}")
        try:
            result = await test_command_recognition(case["text"])

            if result["recognized"]:
                command = result["command"]
                actual_type = command["type"]
                confidence = result["confidence"]
                alternatives = command.get("alternatives", [])

                if actual_type == case["expected_type"]:
                    print(f"✅ Correct command type: {actual_type}")
                    print(f"Confidence: {confidence:.2f}")
                    if alternatives:
                        print(f"Alternative types: {', '.join(alternatives)}")
                    print(
                        f"Command details: {json.dumps(command['details'], indent=2, ensure_ascii=False)}"
                    )
                else:
                    print(
                        f"❌ Wrong command type: got {actual_type}, expected {case['expected_type']}"
                    )
            else:
                print(
                    f"❌ Command not recognized: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            print(f"❌ Test error: {str(e)}")


async def test_mixed_language_commands():
    """Test handling of commands that mix English and Russian."""
    mixed_commands = [
        "go вперед",
        "attack монстра",
        "talk to торговец",
        "поднять key",
    ]

    print("\n🔍 Testing mixed language command handling...")

    for command in mixed_commands:
        print(f"\nTesting mixed command: {command}")
        try:
            result = await test_command_recognition(command)

            if result["recognized"]:
                print(
                    f"✅ Command recognized with confidence: {result['confidence']:.2f}"
                )
                print(f"Command type: {result['command']['type']}")
                print(
                    f"Details: {json.dumps(result['command']['details'], indent=2, ensure_ascii=False)}"
                )
            else:
                print(
                    f"❌ Command not recognized: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            print(f"❌ Test error: {str(e)}")


async def main():
    parser = argparse.ArgumentParser(description="Test command recognition")
    parser.add_argument(
        "--text",
        help="Text to recognize commands from (optional)",
    )
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
    parser.add_argument(
        "--test-types",
        action="store_true",
        help="Run command type recognition tests",
    )
    parser.add_argument(
        "--test-mixed",
        action="store_true",
        help="Run mixed language command tests",
    )

    args = parser.parse_args()

    try:
        if args.populate:
            print("Populating game state...")
            await populate_game_state(args.session)

        if args.text:
            print(f"\nTesting specific command: '{args.text}'")
            result = await test_command_recognition(args.text, args.session)
            print("\nCommand Recognition Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.test_types:
            await test_command_types()

        if args.test_mixed:
            await test_mixed_language_commands()

        if not any([args.text, args.test_types, args.test_mixed]):
            print(
                "No test specified. Use --text, --test-types, or --test-mixed"
            )

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
