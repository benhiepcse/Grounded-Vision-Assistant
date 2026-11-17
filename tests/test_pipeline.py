import pytest
from PIL import Image
from grounded_vision.pipeline import GroundingPipeline

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
