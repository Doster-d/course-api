import uvicorn

from app.config import settings

if __name__ == "__main__":
    print(
        f"Starting Game ASR and Command API on {settings.HOST}:{settings.PORT}"
    )
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug",
    )
