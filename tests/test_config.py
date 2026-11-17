from pathlib import Path
from grounded_vision.config import app_config

def test_yaml_changes_runtime_config(tmp_path: Path, monkeypatch):
    path=tmp_path/'app.yaml'; path.write_text('vlm:\n  max_image_side: 333\ninference:\n  max_upload_mb: 2\n',encoding='utf-8')
    assert app_config(path)['max_image_side']==333 and app_config(path)['max_upload_mb']==2
    monkeypatch.setenv('GVA_MAX_IMAGE_SIDE','444')
    assert app_config(path)['max_image_side']==444
