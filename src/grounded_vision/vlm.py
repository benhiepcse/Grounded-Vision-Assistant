"""Lazy Qwen2.5-VL adapter for direct grounding and numbered-box selection."""
from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image


class ModelLoadError(RuntimeError): pass
class ModelInferenceError(RuntimeError): pass
class ModelOutputError(RuntimeError): pass


def resize_for_model(image: Image.Image, max_side: int) -> tuple[Image.Image, tuple[float, float]]:
    """Resize deterministically before the processor; return x/y scales back to source."""
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    ratio = min(1.0, max_side / max(image.size))
    if ratio == 1.0:
        return image, (1.0, 1.0)
    resized = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))))
    return resized, (image.width / resized.width, image.height / resized.height)


def validate_business_output(value: dict[str, Any], *, selection: bool = False) -> dict[str, Any]:
    status = value.get("status")
    if status not in {"found", "ambiguous", "not_found"}:
        raise ModelOutputError(f"Unknown model status: {status!r}")
    field = "index" if selection else "bbox"
    payload = value.get(field)
    if status == "found":
        if selection and (not isinstance(payload, int) or isinstance(payload, bool)):
            raise ModelOutputError("found selection requires an integer index")
        if not selection and (not isinstance(payload, (list, tuple)) or len(payload) != 4 or
                              any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in payload)):
            raise ModelOutputError("found grounding requires a numeric four-value bbox")
    elif payload is not None:
        raise ModelOutputError(f"{status} requires {field}=null")
    return value


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
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto", max_image_side: int = 1024, max_gpu_memory_gib: float = 6.0, load_in_4bit: bool = True):
        self.model_id, self.device = model_id, device
        self.max_image_side = max_image_side
        self.max_gpu_memory_gib = max_gpu_memory_gib
        self.load_in_4bit = load_in_4bit
        self.model = self.processor = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            processor = AutoProcessor.from_pretrained(self.model_id)
            load_options: dict[str, Any] = {"dtype": dtype, "device_map": self.device, "low_cpu_mem_usage": True}
            if self.load_in_4bit and torch.cuda.is_available():
                load_options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            if self.device == "auto" and torch.cuda.is_available():
                # Keep headroom for vision tokens/KV cache on an 8 GB laptop GPU.
                load_options["max_memory"] = {0: f"{self.max_gpu_memory_gib}GiB", "cpu": "32GiB"}
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **load_options)
            self.processor, self.model = processor, model
        except Exception as exc:
            self.model = self.processor = None
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
        model_image, (scale_x, scale_y) = resize_for_model(image, self.max_image_side)
        prompt = (f"Locate the single object described below in this {model_image.width}x{model_image.height} image. "
                  "Return JSON only: {\"status\":\"found\",\"bbox_2d\":[x1,y1,x2,y2]} using actual pixel coordinates, "
                  "or status ambiguous/not_found with bbox_2d null. Description: " + query)
        parsed = parse_json_object(self._generate(model_image, prompt))
        parsed["bbox"] = parsed.get("bbox_2d", parsed.get("bbox"))
        validate_business_output(parsed)
        if parsed["status"] == "found":
            x1, y1, x2, y2 = parsed["bbox"]
            parsed["bbox"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        return parsed

    def select(self, image: Image.Image, query: str, boxes: list[tuple[float, float, float, float]]) -> dict[str, Any]:
        prompt = (f"Given candidate boxes numbered 0..{len(boxes)-1}, use the full description to select exactly one. "
                  "Return JSON only: {\"status\":\"found\",\"index\":0}, or ambiguous/not_found with index null. Description: " + query)
        return validate_business_output(parse_json_object(self._generate(image, prompt)), selection=True)
