# Game ASR and Command Recognition API

A FastAPI application that combines real-time speech recognition using Whisper Streaming with game command recognition using Ollama with qwen3:0.6b.

## Features

- Real-time audio streaming and speech recognition via WebSocket
- Command recognition from transcribed text
- REST API for direct command recognition from text
- Game state management with session support
- Custom prompts for different command types

## Requirements

- Python 3.8+
- Ollama with qwen3:0.6b model installed
- Whisper Streaming library

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/game-asr-api.git
cd game-asr-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama:
```bash
# Follow instructions at https://ollama.com/download
```

5. Pull the qwen3:0.6b model:
```bash
ollama pull qwen3:0.6b
```

## Configuration

Create a `.env` file in the project root with the following variables (or adjust as needed):

```
DEBUG=True
HOST=0.0.0.0
PORT=8000
WHISPER_MODEL=large-v2
USE_VAD=True
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
CMD_RECOGNITION_CONFIDENCE_THRESHOLD=0.6
```

## Usage

1. Start the Ollama service:
```bash
ollama serve
```

2. Start the FastAPI application:
```bash
python run.py
```

3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

### Command Recognition

- `POST /api/commands/recognize` - Recognize commands from text input
  ```json
  {
    "text": "go forward and pick up the sword",
    "session_id": "player123"  // Optional, uses stored game state
  }
  ```

### Game State Management

- `GET /api/game/state/{session_id}` - Get current game state for a session
- `POST /api/game/state/{session_id}/update-position` - Update player position
  ```json
  {
    "x": 10.5,
    "y": 0.0,
    "z": 5.2
  }
  ```
- `POST /api/game/state/{session_id}/add-object` - Add object to game state
  ```json
  {
    "id": "key_1",
    "name": "golden key",
    "type": "key",
    "actions": ["take", "use", "examine"],
    "properties": {
      "unlocks": "treasure_chest_1"
    }
  }
  ```
- `POST /api/game/state/{session_id}/remove-object/{object_id}` - Remove object from game state
- `POST /api/game/state/{session_id}/add-npc` - Add NPC to game state
- `POST /api/game/state/{session_id}/remove-npc/{npc_id}` - Remove NPC from game state
- `POST /api/game/state/{session_id}/clear` - Clear game state for a session

### Health Check

- `GET /api/health` - API health check endpoint
- `GET /health` - System health check endpoint

## WebSocket Interfaces

### ASR WebSocket

Connect to `ws://localhost:8000/ws/asr/{client_id}` to establish a speech recognition session.

```javascript
// JavaScript example
const clientId = "player123";
const ws = new WebSocket(`ws://localhost:8000/ws/asr/${clientId}`);

// Handle messages from server
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Transcription:', data.transcription);
    console.log('Command:', data.command);
};

// Send audio data
function sendAudioChunk(audioChunk) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(audioChunk);
    }
}
```

### Game State WebSocket

Connect to `ws://localhost:8000/ws/game-state/{client_id}` to establish a real-time game state management session.

```javascript
// JavaScript example
const clientId = "player123";
const gameStateWs = new WebSocket(`ws://localhost:8000/ws/game-state/${clientId}`);

// Handle messages from server
gameStateWs.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Game state update:', response);
};

// Update player position
function updatePlayerPosition(x, y, z) {
    const data = {
        action: "update_position",
        position: { x, y, z }
    };
    gameStateWs.send(JSON.stringify(data));
}

// Add object to game state
function addObject(obj) {
    const data = {
        action: "add_object",
        object: obj
    };
    gameStateWs.send(JSON.stringify(data));
}

// Get current game state
function getGameState() {
    const data = {
        action: "get_state"
    };
    gameStateWs.send(JSON.stringify(data));
}
```

## Real-world Examples

### 1. Process Voice Command

```python
import websockets
import json
import asyncio

async def process_voice_command():
    client_id = "player123"
    uri = f"ws://localhost:8000/ws/asr/{client_id}"
    
    async with websockets.connect(uri) as websocket:
        # Send audio data chunks
        with open("audio_sample.raw", "rb") as f:
            chunk = f.read(4096)  # Read 4KB chunks
            while chunk:
                await websocket.send(chunk)
                
                # Check for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(response)
                    
                    if data.get("command", {}).get("recognized", False):
                        print(f"Command recognized: {data['command']}")
                        # Process command
                        
                except asyncio.TimeoutError:
                    pass  # No response yet
                    
                chunk = f.read(4096)
        
        # Wait for final processing
        await asyncio.sleep(1)
        
# Run the example
asyncio.run(process_voice_command())
```

### 2. Recognize Text Command via REST API

```python
import requests
import json

def recognize_text_command(text, session_id):
    url = "http://localhost:8000/api/commands/recognize"
    
    payload = {
        "text": text,
        "session_id": session_id
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        if result["recognized"]:
            print(f"Command recognized with confidence {result['confidence']}")
            print(f"Command details: {json.dumps(result['command'], indent=2)}")
            return result["command"]
        else:
            print("No command recognized.")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example usage
command = recognize_text_command("pick up the sword and attack the goblin", "player123")
```

## Customizing Prompts

Prompts are stored in the `prompts/` directory as JSON files:
- `base_commands.json` - General command recognition
- `movement_commands.json` - Movement-specific commands
- `object_interactions.json` - Object interaction commands
- `combat_commands.json` - Combat commands
- `dialog_commands.json` - NPC dialog commands

You can edit these files to customize the prompt templates for your specific game commands.

## License

[MIT License](LICENSE)
