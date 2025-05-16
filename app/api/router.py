import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel

from app.models.game_models import NPC, InteractionObject, Position
from app.services.command_service import CommandRecognitionService
from app.services.game_state import GameStateService

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api")


# Models for request/response
class GameState(BaseModel):
    """Game state data for command recognition.

    Attributes:
        commands: List of available commands
        objects: List of interactive objects in the game world
        interactions: List of possible interaction types
        weapons: List of available weapons
        targets: List of possible targets
        npcs: List of non-player characters
        dialog_options: List of available dialog options
    """

    commands: List[str] = []
    objects: List[str] = []
    interactions: List[str] = []
    weapons: List[str] = []
    targets: List[str] = []
    npcs: List[str] = []
    dialog_options: List[str] = []


class CommandRecognitionRequest(BaseModel):
    """Request model for command recognition endpoint.

    Attributes:
        text: The text to analyze for commands
        game_state: Optional game state data
        session_id: Optional session ID to use stored game state
        return_unrecognized: Flag for returning unrecognized commands
    """

    text: str
    game_state: Optional[GameState] = None
    session_id: Optional[str] = None
    return_unrecognized: bool = False


class CommandRecognitionResponse(BaseModel):
    """Response model for command recognition results.

    Attributes:
        recognized: Whether a command was successfully recognized
        command: The structured command data if recognized
        confidence: Confidence score of the recognition
        error: Error message if recognition failed
    """

    recognized: bool
    command: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    error: Optional[str] = None


# Dependency functions for services
async def get_command_service() -> CommandRecognitionService:
    """Dependency for command recognition service.

    Returns:
        Initialized command recognition service
    """
    service = CommandRecognitionService()
    return service


async def get_game_state_service() -> GameStateService:
    """Dependency for game state service.

    Returns:
        Initialized game state service
    """
    service = GameStateService()
    return service


@api_router.post(
    "/commands/recognize", response_model=CommandRecognitionResponse
)
async def recognize_command(
    request: CommandRecognitionRequest,
    command_service: CommandRecognitionService = Depends(get_command_service),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Recognize a command from text input.

    Analyzes the provided text to identify game commands based on either the
    provided game state or stored session state.

    Parameters:
        request: Command recognition request containing text and game state
        command_service: Service for command recognition
        game_state_service: Service for game state management

    Returns:
        Information about the recognized command
    """
    try:
        logger.info(
            f"Received command recognition request for: {request.text}"
        )

        # Determine the game state to use
        game_state_dict = {}

        if request.session_id:
            # Use stored game state for this session
            game_state_dict = game_state_service.to_dict(request.session_id)
            logger.debug(
                f"Using stored game state for session {request.session_id}"
            )
        elif request.game_state:
            # Use provided game state
            game_state_dict = request.game_state.model_dump()
            logger.debug("Using provided game state")

        # Process the command
        result = await command_service.recognize_command(
            request.text, game_state_dict
        )

        # Всегда возвращаем ответ, даже если команда не распознана, но есть best_match
        if (
            not result["recognized"]
            and result["command"] is not None
            and result["confidence"] > 0
        ):
            logger.info(f"Returning low confidence command: {result}")
            if request.return_unrecognized:
                result["recognized"] = True
                logger.info(
                    "Forcing recognition for unrecognized command with low confidence"
                )

        # Return the result
        return CommandRecognitionResponse(**result)

    except Exception as e:
        logger.error(f"Error processing command recognition request: {str(e)}")
        return CommandRecognitionResponse(
            recognized=False, confidence=0.0, error=str(e)
        )


# Game state management endpoints
class GameStateResponse(BaseModel):
    """Response model for game state endpoints.

    Attributes:
        state: The current game state
    """

    state: Dict[str, Any]


@api_router.get("/game/state/{session_id}", response_model=GameStateResponse)
async def get_game_state(
    session_id: str = Path(..., description="Session identifier"),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Get the current game state for a session.

    Retrieves all game state data for the specified session.

    Parameters:
        session_id: Session identifier
        game_state_service: Service for game state management

    Returns:
        Current game state
    """
    state = game_state_service.get_state(session_id)
    return GameStateResponse(state=state.model_dump())


@api_router.post(
    "/game/state/{session_id}/update-position",
    response_model=GameStateResponse,
)
async def update_player_position(
    position: Position,
    session_id: str = Path(..., description="Session identifier"),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Update the player's position.

    Updates the player's position in the game world for the specified session.

    Parameters:
        position: New player position (x, y, z)
        session_id: Session identifier
        game_state_service: Service for game state management

    Returns:
        Updated game state
    """
    state = game_state_service.update_player_position(session_id, position)
    return GameStateResponse(state=state.model_dump())


@api_router.post(
    "/game/state/{session_id}/add-object", response_model=GameStateResponse
)
async def add_game_object(
    obj: InteractionObject,
    session_id: str = Path(..., description="Session identifier"),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Add an object to the game state.

    Adds a new interactive object to the game world for the specified session.

    Parameters:
        obj: Object to add
        session_id: Session identifier
        game_state_service: Service for game state management

    Returns:
        Updated game state
    """
    state = game_state_service.add_object(session_id, obj)
    return GameStateResponse(state=state.model_dump())


@api_router.post(
    "/game/state/{session_id}/remove-object/{object_id}",
    response_model=GameStateResponse,
)
async def remove_game_object(
    session_id: str = Path(..., description="Session identifier"),
    object_id: str = Path(..., description="Object identifier"),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Remove an object from the game state.

    Removes an interactive object from the game world for the specified session.

    Parameters:
        session_id: Session identifier
        object_id: ID of the object to remove
        game_state_service: Service for game state management

    Returns:
        Updated game state
    """
    state = game_state_service.remove_object(session_id, object_id)
    return GameStateResponse(state=state.model_dump())


@api_router.post(
    "/game/state/{session_id}/add-npc", response_model=GameStateResponse
)
async def add_npc(
    npc: NPC,
    session_id: str = Path(..., description="Session identifier"),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Add an NPC to the game state.

    Adds a new non-player character to the game world for the specified session.

    Parameters:
        npc: NPC to add
        session_id: Session identifier
        game_state_service: Service for game state management

    Returns:
        Updated game state
    """
    state = game_state_service.add_npc(session_id, npc)
    return GameStateResponse(state=state.model_dump())


@api_router.post(
    "/game/state/{session_id}/remove-npc/{npc_id}",
    response_model=GameStateResponse,
)
async def remove_npc(
    session_id: str = Path(..., description="Session identifier"),
    npc_id: str = Path(..., description="NPC identifier"),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Remove an NPC from the game state.

    Removes a non-player character from the game world for the specified session.

    Parameters:
        session_id: Session identifier
        npc_id: ID of the NPC to remove
        game_state_service: Service for game state management

    Returns:
        Updated game state
    """
    state = game_state_service.remove_npc(session_id, npc_id)
    return GameStateResponse(state=state.model_dump())


@api_router.post(
    "/game/state/{session_id}/clear", response_model=Dict[str, str]
)
async def clear_game_state(
    session_id: str = Path(..., description="Session identifier"),
    game_state_service: GameStateService = Depends(get_game_state_service),
):
    """Clear the game state for a session.

    Resets all game state data for the specified session.

    Parameters:
        session_id: Session identifier
        game_state_service: Service for game state management

    Returns:
        Status message
    """
    game_state_service.clear_state(session_id)
    return {"status": "Game state cleared"}


@api_router.get("/health")
async def api_health_check():
    """API health check endpoint.

    Returns:
        Health status information
    """
    return {"status": "healthy"}
