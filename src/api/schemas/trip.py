"""
Pydantic models for trip parsing API.

These models define the request/response schemas for trip extraction endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TripParseRequest(BaseModel):
    """Request model for trip parsing."""

    text: str = Field(
        ...,
        description="French text describing a trip",
        min_length=1,
        max_length=1000,
        examples=["Je veux aller de Paris à Lyon"],
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {"text": "Je veux prendre le train de Paris à Lyon demain matin"}
        }


class TripParseResponse(BaseModel):
    """Response model for successful trip parsing."""

    departure: Optional[str] = Field(
        None,
        description="Departure city extracted from text",
        examples=["Paris"],
    )
    arrival: Optional[str] = Field(
        None,
        description="Arrival city extracted from text",
        examples=["Lyon"],
    )
    success: bool = Field(
        ...,
        description="Whether the parsing was successful",
    )
    message: Optional[str] = Field(
        None,
        description="Additional information about the result",
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "departure": "Paris",
                "arrival": "Lyon",
                "success": True,
                "message": None,
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(
        ...,
        description="Error type or category",
    )
    detail: str = Field(
        ...,
        description="Detailed error message",
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Text cannot be empty",
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        ...,
        description="API health status",
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="API version",
        examples=["0.1.0"],
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
            }
        }
