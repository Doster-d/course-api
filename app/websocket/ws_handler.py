import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.asr_service import ASRService
from app.services.command_service import CommandRecognitionService
from app.services.game_state import GameStateService

logger = logging.getLogger(__name__)

websocket_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and associated services.

    Handles connection lifecycle for WebSocket clients including setup of
    associated services (ASR, command recognition) for each client, message
    passing, and cleanup on disconnect.

    Attributes:
        active_connections: Dictionary of active WebSocket connections by client ID
        asr_services: Dictionary of ASR services by client ID
        command_services: Dictionary of command recognition services by client ID
        game_state_service: Shared game state service for all connections
    """

    def __init__(self):
        """Initialize the connection manager with empty collections."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.asr_services: Dict[str, ASRService] = {}
        self.command_services: Dict[str, CommandRecognitionService] = {}
        self.game_state_service = GameStateService()

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection and initialize client services.

        Args:
            websocket: The WebSocket connection to accept
            client_id: Unique identifier for the client
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.asr_services[client_id] = ASRService()
        self.command_services[client_id] = CommandRecognitionService()
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        """Clean up resources when a client disconnects.

        Removes the client's WebSocket connection and cleans up associated
        services to prevent resource leaks.

        Args:
            client_id: Unique identifier for the client to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.asr_services:
            self.asr_services[client_id].cleanup()
            del self.asr_services[client_id]
        if client_id in self.command_services:
            del self.command_services[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: str):
        """Send a text message to a specific client.

        Args:
            client_id: Unique identifier for the client
            message: Text message to send
        """
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def send_json(self, client_id: str, data: Dict[str, Any]):
        """Send JSON data to a specific client.

        Args:
            client_id: Unique identifier for the client
            data: JSON-serializable data to send
        """
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)


manager = ConnectionManager()


@websocket_router.websocket("/ws/asr/{client_id}")
async def websocket_asr_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time speech recognition.

    Receives audio data chunks from clients, processes them through ASR,
    and sends back transcription results and recognized commands.

    Args:
        websocket: The WebSocket connection
        client_id: Unique identifier for the client
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive audio chunk as binary data
            audio_chunk = await websocket.receive_bytes()

            # Process audio with ASR service
            asr_service = manager.asr_services[client_id]
            asr_service.insert_audio_chunk(audio_chunk)

            # Get transcription
            transcription = asr_service.process_iter()

            if transcription:
                logger.debug(f"Transcription: {transcription}")

                # Get current game state for this client
                game_state_dict = manager.game_state_service.to_dict(client_id)

                # Process command if transcription is available
                command_service = manager.command_services[client_id]
                command_result = await command_service.recognize_command(
                    transcription, game_state=game_state_dict
                )

                # Send result back to client
                await manager.send_json(
                    client_id,
                    {
                        "transcription": transcription,
                        "command": command_result,
                    },
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        manager.disconnect(client_id)


@websocket_router.websocket("/ws/game-state/{client_id}")
async def websocket_game_state_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time game state updates.

    Allows clients to send and receive game state updates including
    player position, objects, and NPCs.

    Args:
        websocket: The WebSocket connection
        client_id: Unique identifier for the client
    """
    await websocket.accept()
    try:
        while True:
            # Receive game state update as JSON
            data = await websocket.receive_json()

            # Update game state
            if "action" in data:
                action = data["action"]

                if action == "update_position" and "position" in data:
                    position_data = data["position"]
                    from app.models.game_models import Position

                    position = Position(**position_data)
                    manager.game_state_service.update_player_position(
                        client_id, position
                    )

                elif action == "add_object" and "object" in data:
                    object_data = data["object"]
                    from app.models.game_models import InteractionObject

                    obj = InteractionObject(**object_data)
                    manager.game_state_service.add_object(client_id, obj)

                elif action == "remove_object" and "object_id" in data:
                    object_id = data["object_id"]
                    manager.game_state_service.remove_object(
                        client_id, object_id
                    )

                elif action == "add_npc" and "npc" in data:
                    npc_data = data["npc"]
                    from app.models.game_models import NPC

                    npc = NPC(**npc_data)
                    manager.game_state_service.add_npc(client_id, npc)

                elif action == "remove_npc" and "npc_id" in data:
                    npc_id = data["npc_id"]
                    manager.game_state_service.remove_npc(client_id, npc_id)

                elif action == "get_state":
                    # Get current state and send it back
                    current_state = manager.game_state_service.get_state(
                        client_id
                    )
                    await websocket.send_json(
                        {"state": current_state.model_dump()}
                    )

                elif action == "clear_state":
                    manager.game_state_service.clear_state(client_id)

            # Send acknowledgment
            await websocket.send_json(
                {"status": "ok", "action": data.get("action", "unknown")}
            )

    except WebSocketDisconnect:
        logger.info(
            f"Game state WebSocket disconnected for client {client_id}"
        )
    except Exception as e:
        logger.error(f"Error in game state WebSocket: {str(e)}")
        await websocket.close()
