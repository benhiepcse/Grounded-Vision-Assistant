"""Pure metric helpers; boxes use half-open pixel coordinates [x1,y1,x2,y2]."""
from __future__ import annotations

import math
from collections.abc import Iterable


def coco_xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(v) for v in box)
    if width <= 0 or height <= 0:
        raise ValueError("COCO width and height must be positive")
    return x, y, x + width, y + height


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
