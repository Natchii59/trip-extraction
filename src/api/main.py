"""
Trip Parser REST API.

This is the main FastAPI application that provides endpoints
for trip information extraction using ML models.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .routers import trip_router
from .schemas import HealthResponse
from .services import TripParserService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.

    Handles startup and shutdown events.
    """
    # Startup: preload models
    logger.info("Starting Trip Parser API...")
    try:
        service = TripParserService()
        # Trigger model loading by making a test parse
        # This ensures models are ready before accepting requests
        logger.info("Preloading models...")
        try:
            service.parse_trip("Test de Paris à Lyon")
            logger.info("Models preloaded successfully")
        except Exception as e:
            logger.warning(f"Could not preload models: {e}")
            logger.warning("Models will be loaded on first request")

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)

    logger.info("Trip Parser API ready")

    yield

    # Shutdown
    logger.info("Shutting down Trip Parser API...")


# Create FastAPI app
app = FastAPI(
    title="Trip Parser API",
    description="API to extract departure and arrival cities from French text.",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(trip_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running.",
    tags=["health"],
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        API health status.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.

    Args:
        request: The request that caused the error.
        exc: The exception that was raised.

    Returns:
        JSON response with error details.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please try again later.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
