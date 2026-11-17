"""YAML configuration with precedence: defaults < YAML < environment < CLI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def app_config(path: Path) -> dict[str, Any]:
    raw = load_yaml(path)
    vlm, inference = raw.get("vlm", {}), raw.get("inference", {})
    return {
        "model_id": os.getenv("GVA_MODEL_ID", vlm.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
        "device": os.getenv("GVA_DEVICE", vlm.get("device", "auto")),
        "max_image_side": int(os.getenv("GVA_MAX_IMAGE_SIDE", vlm.get("max_image_side", 1024))),
        "max_gpu_memory_gib": float(os.getenv("GVA_MAX_GPU_MEMORY_GIB", vlm.get("max_gpu_memory_gib", 6.0))),
        "load_in_4bit": os.getenv("GVA_LOAD_IN_4BIT", str(vlm.get("load_in_4bit", True))).lower() in {"1", "true", "yes"},
        "default_backend": os.getenv("GVA_BACKEND", inference.get("backend", "direct")),
        "max_upload_mb": int(os.getenv("GVA_MAX_UPLOAD_MB", inference.get("max_upload_mb", 10))),
    }
