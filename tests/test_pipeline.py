import pytest
from PIL import Image
from grounded_vision.pipeline import GroundingPipeline
from grounded_vision.vlm import ModelOutputError, resize_for_model
from grounded_vision.detector import detector_prompt

class FakeVlm:
    def __init__(self, output): self.output=output
    def ground(self, image, query): return self.output
    def select(self, image, query, boxes): return self.output
class FakeDetector:
    def __init__(self, boxes): self.boxes=boxes
    def propose(self, image, query): return self.boxes

def test_direct_found():
    result=GroundingPipeline(FakeVlm({'status':'found','bbox':[1,2,9,8]})).run(Image.new('RGB',(10,10)),'cup','direct')
    assert result.status == 'found' and result.bbox_xyxy == (1,2,9,8)
def test_proposal_miss_is_not_found():
    result=GroundingPipeline(FakeVlm({}),FakeDetector([]),FakeVlm({})).run(Image.new('RGB',(10,10)),'cup','proposal')
    assert result.status == 'not_found' and result.proposal_failure
def test_proposal_selection():
    result=GroundingPipeline(FakeVlm({}),FakeDetector([(1,1,5,5)]),FakeVlm({'status':'found','index':0})).run(Image.new('RGB',(10,10)),'cup','proposal')
    assert result.bbox_xyxy == (1,1,5,5)
def test_empty_query():
    with pytest.raises(ValueError): GroundingPipeline(FakeVlm({})).run(Image.new('RGB',(10,10)),' ','direct')

@pytest.mark.parametrize('output',[{'status':'found'},{'status':'weird','bbox':None},{'status':'found','bbox':'bad'}])
def test_bad_model_output(output):
    with pytest.raises(ModelOutputError): GroundingPipeline(FakeVlm(output)).run(Image.new('RGB',(10,10)),'cup','direct')

def test_bad_selector_index():
    pipeline=GroundingPipeline(FakeVlm({}),FakeDetector([(1,1,5,5)]),FakeVlm({'status':'found','index':3}))
    with pytest.raises(ModelOutputError): pipeline.run(Image.new('RGB',(10,10)),'cup','proposal')

def test_resize_coordinate_scale():
    resized, scale=resize_for_model(Image.new('RGB',(2000,1000)),1000)
    assert resized.size==(1000,500) and scale==(2,2)

def test_vietnamese_detector_prompt_excludes_reference():
    assert detector_prompt('chiếc cốc trắng bên trái bàn phím')=='cốc trắng'
