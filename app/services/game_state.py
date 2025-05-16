import logging
from typing import Any, Dict

from app.models.game_models import (
    NPC,
    GameContext,
    InteractionObject,
    Position,
)

logger = logging.getLogger(__name__)


class GameStateService:
    """Service for managing game state."""

    def __init__(self):
        """Initialize the game state service."""
        self.game_states: Dict[str, GameContext] = {}
        self.default_state = self._create_default_state()

    def _create_default_state(self) -> GameContext:
        """Create a default game state."""
        return GameContext(
            player_position=Position(x=0, y=0, z=0),
            available_objects=[
                InteractionObject(
                    id="sword_1",
                    name="sword",
                    type="weapon",
                    actions=["take", "use", "examine"],
                    properties={"damage": 10, "weight": 5},
                ),
                InteractionObject(
                    id="potion_1",
                    name="health potion",
                    type="consumable",
                    actions=["take", "use", "examine"],
                    properties={"health_restore": 50},
                ),
                InteractionObject(
                    id="door_1",
                    name="wooden door",
                    type="interactive",
                    actions=["open", "close", "examine", "lock", "unlock"],
                    properties={"locked": True},
                ),
            ],
            available_npcs=[
                NPC(
                    id="merchant_1",
                    name="merchant",
                    dialog_options=["greet", "trade", "farewell"],
                    properties={
                        "friendly": True,
                        "items_for_sale": ["map", "torch"],
                    },
                ),
                NPC(
                    id="guard_1",
                    name="guard",
                    dialog_options=["greet", "quest", "farewell"],
                    properties={"friendly": True, "has_quest": True},
                ),
            ],
            commands=["go", "take", "use", "examine", "talk", "attack"],
            interactions=["open", "close", "activate", "push", "pull"],
            weapons=["sword", "bow", "staff"],
            targets=["goblin", "troll", "dragon"],
        )

    def get_state(self, session_id: str) -> GameContext:
        """Get the game state for a session.

        Args:
            session_id: Session identifier

        Returns:
            Game context for the session
        """
        if session_id not in self.game_states:
            # Create new state for this session
            self.game_states[session_id] = self._create_default_state()
            logger.info(f"Created new game state for session {session_id}")

        return self.game_states[session_id]

    def update_state(
        self, session_id: str, updated_state: GameContext
    ) -> GameContext:
        """Update the game state for a session.

        Args:
            session_id: Session identifier
            updated_state: Updated game context

        Returns:
            Updated game context
        """
        self.game_states[session_id] = updated_state
        logger.info(f"Updated game state for session {session_id}")
        return self.game_states[session_id]

    def update_player_position(
        self, session_id: str, position: Position
    ) -> GameContext:
        """Update the player's position.

        Args:
            session_id: Session identifier
            position: New player position

        Returns:
            Updated game context
        """
        state = self.get_state(session_id)
        state.player_position = position
        return state

    def add_object(
        self, session_id: str, obj: InteractionObject
    ) -> GameContext:
        """Add an object to the game state.

        Args:
            session_id: Session identifier
            obj: Object to add

        Returns:
            Updated game context
        """
        state = self.get_state(session_id)

        # Check if the object already exists
        for i, existing_obj in enumerate(state.available_objects):
            if existing_obj.id == obj.id:
                # Replace existing object
                state.available_objects[i] = obj
                logger.info(f"Updated object {obj.id} in session {session_id}")
                return state

        # Add new object
        state.available_objects.append(obj)
        logger.info(f"Added object {obj.id} to session {session_id}")
        return state

    def remove_object(self, session_id: str, object_id: str) -> GameContext:
        """Remove an object from the game state.

        Args:
            session_id: Session identifier
            object_id: ID of the object to remove

        Returns:
            Updated game context
        """
        state = self.get_state(session_id)

        # Remove the object
        state.available_objects = [
            obj for obj in state.available_objects if obj.id != object_id
        ]
        logger.info(f"Removed object {object_id} from session {session_id}")
        return state

    def add_npc(self, session_id: str, npc: NPC) -> GameContext:
        """Add an NPC to the game state.

        Args:
            session_id: Session identifier
            npc: NPC to add

        Returns:
            Updated game context
        """
        state = self.get_state(session_id)

        # Check if the NPC already exists
        for i, existing_npc in enumerate(state.available_npcs):
            if existing_npc.id == npc.id:
                # Replace existing NPC
                state.available_npcs[i] = npc
                logger.info(f"Updated NPC {npc.id} in session {session_id}")
                return state

        # Add new NPC
        state.available_npcs.append(npc)
        logger.info(f"Added NPC {npc.id} to session {session_id}")
        return state

    def remove_npc(self, session_id: str, npc_id: str) -> GameContext:
        """Remove an NPC from the game state.

        Args:
            session_id: Session identifier
            npc_id: ID of the NPC to remove

        Returns:
            Updated game context
        """
        state = self.get_state(session_id)

        # Remove the NPC
        state.available_npcs = [
            npc for npc in state.available_npcs if npc.id != npc_id
        ]
        logger.info(f"Removed NPC {npc_id} from session {session_id}")
        return state

    def clear_state(self, session_id: str) -> None:
        """Clear the game state for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.game_states:
            del self.game_states[session_id]
            logger.info(f"Cleared game state for session {session_id}")

    def to_dict(self, session_id: str) -> Dict[str, Any]:
        """Convert game state to dictionary for command recognition.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary representation of game state
        """
        state = self.get_state(session_id)

        # Extract object and NPC names
        objects = [obj.name for obj in state.available_objects]
        npcs = [npc.name for npc in state.available_npcs]

        return {
            "commands": state.commands,
            "objects": objects,
            "interactions": state.interactions,
            "weapons": state.weapons,
            "targets": state.targets,
            "npcs": npcs,
            "dialog_options": [
                option
                for npc in state.available_npcs
                for option in npc.dialog_options
            ],
        }
