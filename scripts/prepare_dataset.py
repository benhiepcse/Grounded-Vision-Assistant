"""Join trusted REFER pickles, COCO annotations, and images into JSONL."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grounded_vision.metrics import coco_xywh_to_xyxy  # noqa: E402

SPLIT_FILES = {("refcoco", "unc"): "refs(unc).p", ("refcocog", "google"): "refs(google).p"}


def load_refer(dataset: str, split_by: str, refer_root: Path, image_root: Path) -> list[dict[str, Any]]:
    key = (dataset, split_by)
    if key not in SPLIT_FILES:
        raise ValueError(f"Unsupported dataset/splitBy: {key}; expected one of {sorted(SPLIT_FILES)}")
    dataset_dir = refer_root / dataset
    pickle_path = (dataset_dir / SPLIT_FILES[key]).resolve()
    if dataset_dir.resolve() not in pickle_path.parents:
        raise ValueError("Refusing a pickle outside the configured REFER directory")
    with pickle_path.open("rb") as handle:  # trusted archive named explicitly above
        refs = pickle.load(handle)
    with (dataset_dir / "instances.json").open(encoding="utf-8") as handle:
        coco = json.load(handle)
    annotations = {int(item["id"]): item for item in coco["annotations"]}
    relative_image_root = image_root.resolve().relative_to(ROOT.resolve()).as_posix()
    records: list[dict[str, Any]] = []
    for ref in refs:
        ann = annotations.get(int(ref["ann_id"]))
        if not ann or int(ann["image_id"]) != int(ref["image_id"]):
            raise ValueError(f"Broken annotation join for ref_id={ref.get('ref_id')}")
        box = coco_xywh_to_xyxy(ann["bbox"])
        image_path = image_root / f"COCO_train2014_{int(ref['image_id']):012d}.jpg"
        relative_image_path = f"{relative_image_root}/{image_path.name}"
        for sentence in ref["sentences"]:
            records.append({
                "id": f"{dataset}:{split_by}:r{int(ref['ref_id'])}:s{int(sentence['sent_id'])}",
                "dataset": dataset,
                "split": str(ref["split"]),
                "image_id": int(ref["image_id"]),
                "ann_id": int(ref["ann_id"]),
                "image_path": relative_image_path,
                "query": str(sentence.get("sent") or sentence["raw"]).strip(),
                "gold_status": "found",
                "gold_bbox_xyxy": list(box),
                "is_test_data": False,
            })
    return records


def load_vietnamese(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"INFO: Vietnamese manifest is optional and missing: {path}", file=sys.stderr)
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(records: list[dict[str, Any]], split: str | None, limit: int | None, seed: int) -> list[dict[str, Any]]:
    chosen = [record for record in records if split is None or record["split"] == split]
    if limit is not None and limit < len(chosen):
        chosen = random.Random(seed).sample(chosen, limit)
    return sorted(chosen, key=lambda item: item["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("refcoco", "refcocog", "vietnamese"), required=True)
    parser.add_argument("--split-by", default=None)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=0, help="Print N validated records without writing")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifests" / "benchmark.jsonl")
    args = parser.parse_args()
    if args.dataset == "vietnamese":
        records = load_vietnamese(ROOT / "data/vietnamese/annotations.jsonl")
    else:
        split_by = args.split_by or ("unc" if args.dataset == "refcoco" else "google")
        records = load_refer(args.dataset, split_by, ROOT / "data/raw/refer", ROOT / "data/raw/coco/train2014")
    records = select(records, args.split, args.limit, args.seed)
    for record in records:
        if not (ROOT / record["image_path"]).is_file():
            raise FileNotFoundError(ROOT / record["image_path"])
    if args.preview:
        print(json.dumps(records[: args.preview], ensure_ascii=False, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
