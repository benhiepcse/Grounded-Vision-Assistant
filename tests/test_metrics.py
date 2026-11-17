from grounded_vision.metrics import box_iou, coco_xywh_to_xyxy, percentile

def test_coco_conversion(): assert coco_xywh_to_xyxy([10, 20, 30, 40]) == (10, 20, 40, 60)
def test_iou(): assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
def test_empty_iou(): assert box_iou((0, 0, 0, 0), (0, 0, 2, 2)) == 0
def test_percentile(): assert percentile([1, 2, 3, 4], .5) == 2.5
