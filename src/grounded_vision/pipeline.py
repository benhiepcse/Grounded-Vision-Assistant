"""Orchestrate direct and proposal grounding without conflating technical errors."""
from __future__ import annotations

import time
from typing import Protocol, Any
from PIL import Image, ImageDraw

from .image_io import validate_box
from .schemas import GroundingResult, GroundingStatus


class DirectBackend(Protocol):
    def ground(self, image: Image.Image, query: str) -> dict[str, Any]: ...


class Detector(Protocol):
    def propose(self, image: Image.Image, query: str) -> list[tuple[float, float, float, float]]: ...


class Selector(Protocol):
    def select(self, image: Image.Image, query: str, boxes: list[tuple[float, float, float, float]]) -> dict[str, Any]: ...


def _status(value: Any) -> GroundingStatus:
    try:
        return GroundingStatus(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown business status from model: {value!r}") from exc


def _result(status: GroundingStatus, box: Any, image: Image.Image, backend: str, started: float, **extra: Any) -> GroundingResult:
    valid = validate_box(tuple(box), *image.size) if status == GroundingStatus.FOUND else None
    return GroundingResult(status=status, bbox_xyxy=valid, image_width=image.width, image_height=image.height,
                           backend=backend, latency_ms=(time.perf_counter() - started) * 1000, **extra)


def draw_candidates(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for index, box in enumerate(boxes):
        draw.rectangle(box, outline="#ff2d55", width=max(2, min(image.size) // 200))
        draw.text((box[0] + 3, box[1] + 3), str(index), fill="white", stroke_width=2, stroke_fill="black")
    return marked


class GroundingPipeline:
    def __init__(self, direct: DirectBackend, detector: Detector | None = None, selector: Selector | None = None):
        self.direct, self.detector, self.selector = direct, detector, selector

    def run(self, image: Image.Image, query: str, backend: str) -> GroundingResult:
        query = query.strip()
        if not query:
            raise ValueError("Mô tả không được để trống.")
        started = time.perf_counter()
        if backend == "direct":
            output = self.direct.ground(image, query)
            return _result(_status(output.get("status")), output.get("bbox"), image, backend, started)
        if backend != "proposal":
            raise ValueError("Backend phải là 'direct' hoặc 'proposal'.")
        if self.detector is None or self.selector is None:
            raise RuntimeError("Proposal backend is not configured")
        boxes = [validate_box(box, *image.size) for box in self.detector.propose(image, query)]
        if not boxes:
            return _result(GroundingStatus.NOT_FOUND, None, image, backend, started,
                           candidate_boxes=[], proposal_failure=True, selection_failure=False)
        output = self.selector.select(draw_candidates(image, boxes), query, boxes)
        status = _status(output.get("status"))
        index = output.get("index")
        if status == GroundingStatus.FOUND:
            if not isinstance(index, int) or not 0 <= index < len(boxes):
                raise ValueError(f"Selector returned invalid candidate index: {index!r}")
            box = boxes[index]
        else:
            box = None
        return _result(status, box, image, backend, started, candidate_boxes=boxes,
                       proposal_failure=False, selection_failure=status != GroundingStatus.FOUND)
