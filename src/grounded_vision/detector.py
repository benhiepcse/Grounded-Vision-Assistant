"""Lazy Grounding DINO proposal generator."""
from __future__ import annotations

import re
from PIL import Image
from .vlm import ModelInferenceError, ModelLoadError


def detector_prompt(query: str) -> str:
    """Conservative noun-like prompt; full query remains available to VLM selection."""
    cleaned = re.sub(r"[^\w\s-]", " ", query.lower(), flags=re.UNICODE)
    stop = {"the", "a", "an", "on", "in", "at", "left", "right", "of", "beside", "with",
            "chiếc", "cái", "con", "bên", "trái", "phải", "của", "cạnh", "trên", "dưới", "màu"}
    tokens = [token for token in cleaned.split() if token not in stop]
    return " ".join(tokens) or cleaned.strip()


class GroundingDinoDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", box_threshold: float = 0.25, text_threshold: float = 0.20):
        self.model_id = model_id
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.model = self.processor = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        except Exception as exc:
            raise ModelLoadError(f"Cannot load {self.model_id}: {exc}") from exc

    def propose(self, image: Image.Image, query: str) -> list[tuple[float, float, float, float]]:
        self._load()
        try:
            import torch
            text = detector_prompt(query)
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            return [tuple(float(v) for v in box.tolist()) for box in result["boxes"]]
        except Exception as exc:
            raise ModelInferenceError(f"Grounding DINO inference failed: {exc}") from exc
