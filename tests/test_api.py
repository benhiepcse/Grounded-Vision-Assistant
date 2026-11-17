from io import BytesIO
import pytest
from PIL import Image
from grounded_vision.api import create_app
from grounded_vision.image_io import ImageValidationError, decode_image
from grounded_vision.pipeline import GroundingPipeline
from fastapi import HTTPException
from grounded_vision.vlm import ModelOutputError

class Fake:
    def ground(self,image,query): return {'status':'found','bbox':[1,1,4,4]}
app=create_app(GroundingPipeline(Fake()))
def png():
    stream=BytesIO(); Image.new('RGB',(8,6),'white').save(stream,'PNG'); return stream.getvalue()
def test_routes_exist():
    methods={route.path:route.methods for route in app.routes if hasattr(route,'methods')}
    assert 'GET' in methods['/health'] and 'POST' in methods['/ground']
def test_openapi_multipart_contract():
    operation=app.openapi()['paths']['/ground']['post']
    assert 'multipart/form-data' in operation['requestBody']['content']
def test_valid_image(): assert decode_image(png()).size==(8,6)
def test_bad_image():
    with pytest.raises(ImageValidationError): decode_image(b'bad')

class BadFake:
    def ground(self,image,query): raise ModelOutputError('bad output')

def test_multipart_model_output_is_json_502():
    bad_app=create_app(GroundingPipeline(BadFake()))
    endpoint=next(route.endpoint for route in bad_app.routes if getattr(route,'path',None)=='/ground')
    class Upload:
        async def read(self, size=-1): return png()
    upload=Upload()
    with pytest.raises(HTTPException) as caught:
        endpoint(image=upload,query='cup',backend='direct').send(None)
    assert caught.value.status_code==502 and caught.value.detail['code']=='model_output_error'
