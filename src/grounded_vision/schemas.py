"""Typed contracts shared by model adapters, pipeline, API, and benchmark."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class GroundingStatus(str, Enum):
    FOUND = "found"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


class TechnicalError(BaseModel):
    code: str
    message: str
    detail: str | None = None


class GroundingResult(BaseModel):
    status: GroundingStatus
    bbox_xyxy: tuple[float, float, float, float] | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    backend: str
    latency_ms: float = Field(ge=0)
    candidate_boxes: list[tuple[float, float, float, float]] | None = None
    proposal_failure: bool | None = None
    selection_failure: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_box(self) -> "GroundingResult":
        if (self.status == GroundingStatus.FOUND) != (self.bbox_xyxy is not None):
            raise ValueError("found requires one bbox; other statuses require bbox=null")
        return self
