import importlib.util
from pathlib import Path
from PIL import Image
from grounded_vision.schemas import GroundingResult

spec=importlib.util.spec_from_file_location('benchmark',Path(__file__).parents[1]/'scripts/benchmark.py')
benchmark=importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)

class FakePipeline:
    def __init__(self,result): self.result=result; self.calls=0
    def run(self,image,query,backend): self.calls+=1; return self.result

def sample(status='not_found',box=None):
    return {'id':'x','dataset':'fixture','split':'test','image_path':'data/raw/coco/train2014/COCO_train2014_000000559760.jpg','query':'missing object','gold_status':status,'gold_bbox_xyxy':box}

def result(status='not_found',box=None,candidates=None):
    return GroundingResult(status=status,bbox_xyxy=box,image_width=640,image_height=427,backend='proposal',latency_ms=1,candidate_boxes=candidates or [])

def test_absent_with_candidates_is_valid_not_technical_error():
    row=benchmark.evaluate_sample(sample(), 'proposal', FakePipeline(result(candidates=[(1,1,5,5)])), .5)
    assert row['status']=='not_found' and 'technical_error' not in row and row['iou'] is None
    summary=benchmark.summarize([row])[0]
    assert summary['false_positive_rate']==0 and summary['false_positive_denominator']==1

def test_proposal_diagnoses_miss_and_selection_error():
    gold=[10,10,30,30]
    miss=benchmark.evaluate_sample(sample('found',gold),'proposal',FakePipeline(result('found',(1,1,5,5),[(1,1,5,5)])),.5)
    wrong=benchmark.evaluate_sample(sample('found',gold),'proposal',FakePipeline(result('found',(1,1,5,5),[(10,10,30,30),(1,1,5,5)])),.5)
    assert miss['diagnosis']=='proposal_miss' and wrong['diagnosis']=='selection_error'

def test_warmup_calls_each_backend_n_times():
    pipeline=FakePipeline(result())
    warm=benchmark.run_warmups([sample()],['direct','proposal'],3,pipeline,.5)
    assert pipeline.calls==6 and len(warm['direct'])==len(warm['proposal'])==3

def test_all_technical_errors_have_zero_valid_denominator():
    row={'id':'x','dataset':'d','split':'s','backend':'direct','gold_status':'found','technical_error':{'type':'X'}}
    summary=benchmark.summarize([row])[0]
    assert not summary['available'] and summary['samples_valid']==0 and summary['grounding_acc_denominator']==0
