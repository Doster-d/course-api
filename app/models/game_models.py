from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    """Enumeration of possible movement directions in the game.

    Used to standardize direction values for movement commands.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class ActionType(str, Enum):
    """Enumeration of different command action types.

    Categorizes commands by their functional purpose in the game.
    """

    MOVEMENT = "movement"
    INTERACTION = "interaction"
    COMBAT = "combat"
    DIALOG = "dialog"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class Position(BaseModel):
    """Represents a 3D position in the game world.

    Attributes:
        x: X-coordinate (lateral position)
        y: Y-coordinate (vertical position/height)
        z: Z-coordinate (depth position)
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class InteractionObject(BaseModel):
    """Represents an interactive object in the game world.

    Attributes:
        id: Unique identifier for the object
        name: Display name of the object
        type: Object type category
        actions: List of possible actions that can be performed with this object
        properties: Additional custom properties for the object
    """

    id: str
    name: str
    type: str
    actions: List[str] = []
    properties: Dict[str, Any] = {}


class NPC(BaseModel):
    """Represents a non-player character in the game.

    Attributes:
        id: Unique identifier for the NPC
        name: Display name of the NPC
        dialog_options: Available dialog options when interacting with this NPC
        properties: Additional custom properties for the NPC
    """

    id: str
    name: str
    dialog_options: List[str] = []
    properties: Dict[str, Any] = {}


class CommandParameter(BaseModel):
    """Parameter for a game command.

    Attributes:
        name: Name of the parameter
        value: Value of the parameter (can be any type)
    """

    name: str
    value: Any


class Command(BaseModel):
    """Represents a structured command to be executed in the game.

    Attributes:
        action: The action to perform (verb)
        action_type: Category of action
        target: Optional target of the action
        parameters: Additional parameters for the command
        confidence: Confidence score of the command recognition (0.0-1.0)
    """

    action: str
    action_type: ActionType = ActionType.UNKNOWN
    target: Optional[str] = None
    parameters: Dict[str, Any] = {}
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GameContext(BaseModel):
    """Current state of the game world relevant to command recognition.

    Attributes:
        player_position: Current position of the player
        available_objects: Objects that can be interacted with
        available_npcs: NPCs that can be interacted with
        commands: Available generic commands
        interactions: Available interaction types
        weapons: Available weapons for combat
        targets: Available targets for actions
    """

    player_position: Position = Field(default_factory=Position)
    available_objects: List[InteractionObject] = []
    available_npcs: List[NPC] = []
    commands: List[str] = []
    interactions: List[str] = []
    weapons: List[str] = []
    targets: List[str] = []


class RecognizedCommand(BaseModel):
    """Result of command recognition from text input.

    Attributes:
        recognized: Whether a command was successfully recognized
        command: The structured command if recognized, otherwise None
        raw_text: The original text input that was processed
        error: Error message if recognition failed
    """

    recognized: bool = False
    command: Optional[Command] = None
    raw_text: str
    error: Optional[str] = None
