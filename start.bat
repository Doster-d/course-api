@echo off
setlocal

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ollama is not running. Please start Ollama manually.
    echo You can download it from https://ollama.com/download
    pause
    exit /b 1
)

REM Check if qwen3:0.6b model is available
curl -s http://localhost:11434/api/tags | findstr "qwen3:0.6b" > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo qwen3:0.6b model not found. Pulling model...
    ollama pull qwen3:0.6b
)

REM Check if uv is installed
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo uv is not installed. Installing uv...
    powershell -Command "Invoke-WebRequest -UseBasicParsing https://install.python-uv.org/windows-launcher | python"
)

REM Check if virtual environment exists
if not exist venv\ (
    echo Creating virtual environment with uv...
    uv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies with uv
echo Installing dependencies with uv...
uv sync

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    (
        echo DEBUG=True
        echo HOST=0.0.0.0
        echo PORT=8000
        echo WHISPER_MODEL=bzikst/faster-whisper-large-v3-russian
        echo USE_VAD=True
        echo OLLAMA_HOST=http://localhost:11434
        echo OLLAMA_MODEL=qwen3:0.6b
        echo CMD_RECOGNITION_CONFIDENCE_THRESHOLD=0.6
    ) > .env
)

REM Start the application
echo Starting the Game ASR and Command API...
uv run run.py

pause 