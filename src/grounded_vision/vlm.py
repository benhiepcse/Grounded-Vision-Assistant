"""Lazy Qwen2.5-VL adapter for direct grounding and numbered-box selection."""
from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image


class ModelLoadError(RuntimeError): pass
class ModelInferenceError(RuntimeError): pass
class ModelOutputError(RuntimeError): pass


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        raise ModelOutputError(f"No JSON object in model output: {text[:200]!r}")
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"Invalid JSON in model output: {text[:200]!r}") from exc
    if not isinstance(value, dict):
        raise ModelOutputError("Expected a JSON object")
    return value


class QwenGrounder:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto"):
        self.model_id, self.device = model_id, device
        self.model = self.processor = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype, device_map=self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:
            raise ModelLoadError(f"Cannot load {self.model_id}: {exc}") from exc

    def _generate(self, image: Image.Image, prompt: str) -> str:
        self._load()
        try:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
            generated = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        except Exception as exc:
            raise ModelInferenceError(str(exc)) from exc

    def ground(self, image: Image.Image, query: str) -> dict[str, Any]:
        # Qwen2.5-VL was trained to report the actual image-size coordinate scale,
        # unlike Qwen2-VL recipes that commonly used a normalized 0..1000 space.
        prompt = (f"Locate the single object described below in this {image.width}x{image.height} image. "
                  "Return JSON only: {\"status\":\"found\",\"bbox_2d\":[x1,y1,x2,y2]} using actual pixel coordinates, "
                  "or status ambiguous/not_found with bbox_2d null. Description: " + query)
        parsed = parse_json_object(self._generate(image, prompt))
        parsed["bbox"] = parsed.get("bbox_2d", parsed.get("bbox"))
        return parsed

    def select(self, image: Image.Image, query: str, boxes: list[tuple[float, float, float, float]]) -> dict[str, Any]:
        prompt = (f"Given candidate boxes numbered 0..{len(boxes)-1}, use the full description to select exactly one. "
                  "Return JSON only: {\"status\":\"found\",\"index\":0}, or ambiguous/not_found with index null. Description: " + query)
        return parse_json_object(self._generate(image, prompt))
