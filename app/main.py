import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.websocket.ws_handler import websocket_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Game ASR and Command Recognition API",
    description="API for real-time speech recognition and game command processing",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint that confirms the API is running.

    Returns:
        dict: A simple message indicating the API is operational
    """
    return {"message": "Game ASR and Command API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring API status.

    This endpoint can be used by load balancers and monitoring tools
    to verify the API is operational.

    Returns:
        dict: Health status information
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
