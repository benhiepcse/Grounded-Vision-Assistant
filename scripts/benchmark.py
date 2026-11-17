"""Run one or both backends over the same stable manifest and write JSON/JSONL."""
from __future__ import annotations
import argparse, json, platform, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from PIL import Image, ImageOps  # noqa: E402
from grounded_vision.detector import GroundingDinoDetector  # noqa: E402
from grounded_vision.metrics import box_iou, percentile  # noqa: E402
from grounded_vision.pipeline import GroundingPipeline  # noqa: E402
from grounded_vision.vlm import QwenGrounder  # noqa: E402

def load_manifest(path: Path, split: str|None, limit: int|None) -> list[dict[str,Any]]:
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    rows=[r for r in rows if split is None or r['split']==split]
    return rows[:limit] if limit else rows

def summarize(rows: list[dict[str,Any]], threshold: float=.5) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for row in rows: groups[(row['dataset'],row['split'],row['backend'])].append(row)
    output=[]
    for (dataset,split,backend),items in groups.items():
        valid=[r for r in items if not r.get('technical_error')]; found_gold=[r for r in valid if r['gold_status']=='found']
        acc=sum(r.get('iou',0)>=threshold for r in found_gold)/len(found_gold) if found_gold else None
        recalls=[r['proposal_recalled'] for r in found_gold if r.get('proposal_recalled') is not None]
        absent=[r for r in valid if r['gold_status']=='not_found']; ambiguous=[r for r in valid if r['gold_status']=='ambiguous']
        output.append({'dataset':dataset,'split':split,'backend':backend,'samples':len(items),'technical_errors':len(items)-len(valid),
          'grounding_acc_0_5':acc,'proposal_recall_0_5':sum(recalls)/len(recalls) if recalls else None,
          'false_positive_rate':sum(r.get('status')=='found' for r in absent)/len(absent) if absent else None,
          'ambiguity_recall':sum(r.get('status')=='ambiguous' for r in ambiguous)/len(ambiguous) if ambiguous else None,
          'latency_p50_ms':percentile([r['latency_ms'] for r in valid],.5),'latency_p95_ms':percentile([r['latency_ms'] for r in valid],.95)})
    return output

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest',type=Path,required=True); parser.add_argument('--backend',choices=['direct','proposal','both'],default='both'); parser.add_argument('--split'); parser.add_argument('--limit',type=int); parser.add_argument('--warmup',type=int,default=0); parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs'); args=parser.parse_args()
    samples=load_manifest(args.manifest,args.split,args.limit); qwen=QwenGrounder(); pipeline=GroundingPipeline(qwen,GroundingDinoDetector(),qwen); backends=['direct','proposal'] if args.backend=='both' else [args.backend]
    rows=[]
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    except ImportError: torch=None
    for sample in samples:
      for backend in backends:
        row={k:sample.get(k) for k in ('id','dataset','split','gold_status','gold_bbox_xyxy')}; row['backend']=backend
        try:
          with Image.open(ROOT/sample['image_path']) as source: image=ImageOps.exif_transpose(source).convert('RGB')
          result=pipeline.run(image,sample['query'],backend); row.update(result.model_dump(mode='json'))
          row['iou']=box_iou(result.bbox_xyxy,sample['gold_bbox_xyxy']) if result.bbox_xyxy and sample.get('gold_bbox_xyxy') else 0.0
          row['proposal_recalled']=max((box_iou(b,sample['gold_bbox_xyxy']) for b in result.candidate_boxes or []),default=0)>=.5 if backend=='proposal' else None
        except Exception as exc: row['technical_error']={'type':type(exc).__name__,'message':str(exc)}
        rows.append(row); print(json.dumps({'id':sample['id'],'backend':backend,'status':row.get('status'),'error':row.get('technical_error')},ensure_ascii=False))
    args.output_dir.mkdir(parents=True,exist_ok=True); stamp=time.strftime('%Y%m%d-%H%M%S'); pred=args.output_dir/f'predictions-{stamp}.jsonl'; report=args.output_dir/f'summary-{stamp}.json'
    pred.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    peak=None
    if torch is not None and torch.cuda.is_available(): peak=torch.cuda.max_memory_allocated()/1024**3
    metadata={'seed':42,'samples':len(samples),'backends':backends,'models':{'vlm':qwen.model_id,'detector':'IDEA-Research/grounding-dino-tiny'},'precision':'float16 CUDA / float32 CPU','warmup':args.warmup,'python':platform.python_version(),'platform':platform.platform(),'cuda_peak_allocated_gb':peak,'summaries':summarize(rows)}
    report.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'Wrote {pred} and {report}')
if __name__=='__main__': main()
