"""Safe image decoding with EXIF normalization before coordinates are created."""
from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_PIXELS = 40_000_000


class ImageValidationError(ValueError):
    """The uploaded bytes are not an accepted, safely-sized image."""


def decode_image(data: bytes, *, max_bytes: int = 10 * 1024 * 1024) -> Image.Image:
    if not data:
        raise ImageValidationError("Tệp ảnh rỗng.")
    if len(data) > max_bytes:
        raise ImageValidationError(f"Ảnh vượt giới hạn {max_bytes // (1024 * 1024)} MB.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise ImageValidationError("Chỉ chấp nhận ảnh JPEG hoặc PNG.")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                raise ImageValidationError("Kích thước ảnh không hợp lệ hoặc quá lớn.")
            source.load()
            # All downstream boxes refer to this transposed pixel space.
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("Không đọc được ảnh JPEG/PNG.") from exc


def validate_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    if len(box) != 4 or not all(float("-inf") < float(v) < float("inf") for v in box):
        raise ValueError("bbox must contain four finite numbers")
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox is empty after clipping")
    return x1, y1, x2, y2
