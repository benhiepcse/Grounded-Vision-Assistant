"""FastAPI application and static local web UI."""
from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector import GroundingDinoDetector
from .image_io import ImageValidationError, decode_image
from .pipeline import GroundingPipeline
from .vlm import ModelInferenceError, ModelLoadError, ModelOutputError, QwenGrounder

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = int(os.getenv("GVA_MAX_UPLOAD_MB", "10")) * 1024 * 1024


def create_app(pipeline: GroundingPipeline | None = None) -> FastAPI:
    app = FastAPI(title="Grounded Vision Assistant", version="1.0.0")
    qwen = QwenGrounder()
    app.state.pipeline = pipeline or GroundingPipeline(qwen, GroundingDinoDetector(), qwen)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "backends": ["direct", "proposal"], "models_loaded": False}

    @app.post("/ground")
    async def ground(image: UploadFile = File(...), query: str = Form(...), backend: str = Form("direct")):
        if backend not in {"direct", "proposal"}:
            raise HTTPException(422, detail={"code": "invalid_backend", "message": "Backend không hợp lệ."})
        if not query.strip():
            raise HTTPException(422, detail={"code": "empty_query", "message": "Mô tả không được để trống."})
        data = await image.read(MAX_BYTES + 1)
        try:
            decoded = decode_image(data, max_bytes=MAX_BYTES)
            return app.state.pipeline.run(decoded, query, backend)
        except ImageValidationError as exc:
            raise HTTPException(400, detail={"code": "invalid_image", "message": str(exc)}) from exc
        except ModelLoadError as exc:
            raise HTTPException(503, detail={"code": "model_load_error", "message": str(exc)}) from exc
        except ModelInferenceError as exc:
            raise HTTPException(500, detail={"code": "inference_error", "message": str(exc)}) from exc
        except ModelOutputError as exc:
            raise HTTPException(502, detail={"code": "model_output_error", "message": str(exc)}) from exc

    web = ROOT / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")
    @app.get("/", include_in_schema=False)
    async def index(): return FileResponse(web / "index.html")
    return app


app = create_app()
