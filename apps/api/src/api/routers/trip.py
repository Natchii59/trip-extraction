"""
Trip parsing API routes.

This module defines the REST endpoints for trip extraction.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from trip_parser.exceptions import InvalidInputError, ModelNotFoundError, TripExtractionError

from ..schemas import TripParseRequest, TripParseResponse
from ..services import TripParserService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/trip",
    tags=["trip"],
)


# Initialize the service (singleton)
trip_service = TripParserService()


@router.post(
    "/parse",
    response_model=TripParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract departure and arrival cities",
    description="Analyzes French text and returns detected departure and arrival cities.",
    responses={
        200: {
            "description": "Request processed successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "both_cities_found": {
                            "summary": "Both cities extracted successfully",
                            "value": {
                                "departure": "Paris",
                                "arrival": "Lyon",
                                "success": True,
                                "message": None,
                            },
                        },
                        "no_cities_found": {
                            "summary": "No cities detected in text",
                            "value": {
                                "departure": None,
                                "arrival": None,
                                "success": False,
                                "message": "Could not extract departure and arrival cities from the text",
                            },
                        },
                    }
                }
            },
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "summary": "Text is empty",
                        "value": {
                            "detail": [
                                {
                                    "type": "string_too_short",
                                    "loc": ["body", "text"],
                                    "msg": "String should have at least 1 character",
                                    "input": "",
                                    "ctx": {"min_length": 1},
                                }
                            ]
                        },
                    }
                }
            },
        },
        500: {
            "description": "Internal server error or model loading failure",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred: [error message]"}
                }
            },
        },
    },
)
async def parse_trip(request: TripParseRequest) -> TripParseResponse:
    """
    Parse trip information from French text.

    Args:
        request: Request containing the text to parse.

    Returns:
        TripParseResponse with departure and arrival cities.

    Raises:
        HTTPException: For various error conditions.
    """
    try:
        logger.info(f"Processing trip parse request: {request.text[:50]}...")
        result = trip_service.parse_trip(request.text)
        logger.info(f"Trip parsed: {result.departure} → {result.arrival}")
        return result

    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except ModelNotFoundError as e:
        logger.error(f"Models not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    except TripExtractionError as e:
        logger.error(f"Trip extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract trip information: {str(e)}",
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        ) from e


@router.get(
    "/status",
    summary="Check service status",
    description="Checks if the service is ready to process requests.",
    response_model=dict,
)
async def get_status():
    """
    Get the status of the trip parsing service.

    Returns:
        Status information including whether models are loaded.
    """
    is_ready = trip_service.is_ready()
    return {
        "ready": is_ready,
        "message": "Models loaded and ready" if is_ready else "Models not yet loaded",
    }
