"""Reproducible benchmark; warm-ups and technical errors never enter metrics."""
from __future__ import annotations

import argparse, json, platform, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image, ImageOps  # noqa: E402
from grounded_vision.config import load_yaml  # noqa: E402
from grounded_vision.detector import GroundingDinoDetector  # noqa: E402
from grounded_vision.metrics import box_iou, percentile  # noqa: E402
from grounded_vision.pipeline import GroundingPipeline  # noqa: E402
from grounded_vision.vlm import QwenGrounder  # noqa: E402

STATUSES = {"found", "ambiguous", "not_found"}


def validate_sample(row: dict[str, Any], line: int) -> dict[str, Any]:
    required = ("id", "dataset", "split", "image_path", "query", "gold_status")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
        raise ValueError(f"Manifest line {line}: required string field is missing/empty")
    if row["gold_status"] not in STATUSES:
        raise ValueError(f"Manifest line {line}: invalid gold_status")
    box = row.get("gold_bbox_xyxy")
    if row["gold_status"] == "found":
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Manifest line {line}: found requires gold_bbox_xyxy")
    elif box is not None:
        raise ValueError(f"Manifest line {line}: non-found requires null gold_bbox_xyxy")
    if not (ROOT / row["image_path"]).is_file():
        raise ValueError(f"Manifest line {line}: image does not exist: {row['image_path']}")
    return row


def load_manifest(path: Path, split: str | None, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    rows = [validate_sample(json.loads(text), i) for i, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if text.strip()]
    rows = [row for row in rows if split is None or row["split"] == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Manifest selection is empty")
    return rows


def evaluate_sample(sample: dict[str, Any], backend: str, pipeline: GroundingPipeline, threshold: float) -> dict[str, Any]:
    row = {key: sample.get(key) for key in ("id", "dataset", "split", "gold_status", "gold_bbox_xyxy")}
    row["backend"] = backend
    try:
        with Image.open(ROOT / sample["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        result = pipeline.run(image, sample["query"], backend)
        row.update(result.model_dump(mode="json"))
        gold = sample.get("gold_bbox_xyxy")
        row["iou"] = box_iou(result.bbox_xyxy, gold) if result.bbox_xyxy is not None and gold is not None else None
        if backend == "proposal" and gold is not None:
            candidate_ious = [box_iou(candidate, gold) for candidate in result.candidate_boxes or []]
            recalled = max(candidate_ious, default=0.0) >= threshold
            row["proposal_recalled"] = recalled
            row["diagnosis"] = ("proposal_miss" if not recalled else
                                "final_correct" if (row["iou"] or 0) >= threshold else "selection_error")
        else:
            row["proposal_recalled"] = None
            row["diagnosis"] = "final_correct" if gold is not None and (row["iou"] or 0) >= threshold else None
    except Exception as exc:
        row["technical_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return row


def summarize(rows: list[dict[str, Any]], threshold: float = .5) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split"], row["backend"])].append(row)
    output = []
    for (dataset, split, backend), items in groups.items():
        valid = [r for r in items if not r.get("technical_error")]
        found = [r for r in valid if r["gold_status"] == "found"]
        absent = [r for r in valid if r["gold_status"] == "not_found"]
        ambiguous = [r for r in valid if r["gold_status"] == "ambiguous"]
        proposal = [r for r in found if r.get("proposal_recalled") is not None]
        output.append({
            "dataset": dataset, "split": split, "backend": backend,
            "available": bool(valid), "samples_total": len(items), "samples_valid": len(valid),
            "technical_errors": len(items) - len(valid),
            "grounding_acc_0_5": sum((r.get("iou") or 0) >= threshold for r in found) / len(found) if found else None,
            "grounding_acc_denominator": len(found),
            "proposal_recall_0_5": sum(bool(r["proposal_recalled"]) for r in proposal) / len(proposal) if proposal else None,
            "proposal_recall_denominator": len(proposal),
            "false_positive_rate": sum(r.get("status") == "found" for r in absent) / len(absent) if absent else None,
            "false_positive_denominator": len(absent),
            "ambiguity_recall": sum(r.get("status") == "ambiguous" for r in ambiguous) / len(ambiguous) if ambiguous else None,
            "ambiguity_denominator": len(ambiguous),
            "diagnoses": {name: sum(r.get("diagnosis") == name for r in found) for name in ("proposal_miss", "selection_error", "final_correct")},
            "latency_p50_ms": percentile([r["latency_ms"] for r in valid], .5),
            "latency_p95_ms": percentile([r["latency_ms"] for r in valid], .95),
        })
    return output


def run_warmups(samples: list[dict[str, Any]], backends: list[str], count: int, pipeline: GroundingPipeline, threshold: float) -> dict[str, list[dict[str, Any]]]:
    results = {backend: [] for backend in backends}
    for backend in backends:
        for index in range(count):
            results[backend].append(evaluate_sample(samples[index % len(samples)], backend, pipeline, threshold))
    return results


def main() -> int:
    config = load_yaml(ROOT / "configs/benchmark.yaml")
    evaluation = config.get("evaluation", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--backend", choices=["direct", "proposal", "both"], default="both")
    parser.add_argument("--split"); parser.add_argument("--limit", type=int); parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / evaluation.get("output_dir", "outputs")); parser.add_argument("--iou-threshold", type=float, default=float(evaluation.get("iou_threshold", .5)))
    args = parser.parse_args()
    if args.warmup < 0: raise ValueError("--warmup cannot be negative")
    samples = load_manifest(args.manifest, args.split, args.limit)
    qwen = QwenGrounder(); detector = GroundingDinoDetector(); pipeline = GroundingPipeline(qwen, detector, qwen)
    backends = ["direct", "proposal"] if args.backend == "both" else [args.backend]
    cold = run_warmups(samples, backends, args.warmup, pipeline, args.iou_threshold)
    rows: list[dict[str, Any]] = []
    vram: dict[str, float | None] = {}
    try: import torch
    except ImportError: torch = None
    for backend in backends:
        if torch is not None and torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        for sample in samples:
            row = evaluate_sample(sample, backend, pipeline, args.iou_threshold); rows.append(row)
            print(json.dumps({"id": sample["id"], "backend": backend, "status": row.get("status"), "error": row.get("technical_error")}, ensure_ascii=False))
        vram[backend] = torch.cuda.max_memory_allocated() / 1024**3 if torch is not None and torch.cuda.is_available() else None
    summaries = summarize(rows, args.iou_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True); stamp = time.strftime("%Y%m%d-%H%M%S")
    prediction_path = args.output_dir / f"predictions-{stamp}.jsonl"; report_path = args.output_dir / f"summary-{stamp}.json"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    metadata = {"protocol": {"seed": evaluation.get("seed", 42), "iou_threshold": args.iou_threshold, "warmup_per_backend": args.warmup},
                "models": {"vlm": {"id": qwen.model_id, "revision": "default"}, "detector": {"id": detector.model_id, "revision": "default", "box_threshold": detector.box_threshold, "text_threshold": detector.text_threshold}},
                "runtime": {"python": platform.python_version(), "platform": platform.platform(), "device": "cuda" if torch is not None and torch.cuda.is_available() else "cpu", "dtype": "float16" if torch is not None and torch.cuda.is_available() else "float32"},
                "cold_start_ms": {backend: [row.get("latency_ms") for row in cold[backend] if not row.get("technical_error")] for backend in backends}, "cuda_peak_allocated_gb": vram, "summaries": summaries}
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {prediction_path} and {report_path}")
    return 2 if any(not summary["available"] for summary in summaries) else 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
