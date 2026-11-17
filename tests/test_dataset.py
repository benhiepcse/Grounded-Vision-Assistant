import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('prepare_dataset',Path(__file__).parents[1]/'scripts/prepare_dataset.py')
prepare=importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)

def test_vietnamese_legacy_fields_are_normalized():
    rows=prepare.load_vietnamese(Path(__file__).parent/'fixtures/vietnamese_manifest.jsonl')
    assert rows[0]['id']=='vi-fixture-001'
    assert rows[0]['gold_status']=='found' and rows[0]['gold_bbox_xyxy']==[56.67,63.65,304.3,383.64]
    assert rows[0]['is_test_data'] is True

def test_invalid_split_and_limit_are_rejected():
    rows=[{'id':'x','split':'val'}]
    import pytest
    with pytest.raises(ValueError): prepare.select(rows,'test',1,42)
    with pytest.raises(ValueError): prepare.select(rows,'val',0,42)
