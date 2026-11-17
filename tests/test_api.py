from io import BytesIO
from PIL import Image
from grounded_vision.api import create_app
from grounded_vision.image_io import ImageValidationError, decode_image
from grounded_vision.pipeline import GroundingPipeline

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
    import pytest
    with pytest.raises(ImageValidationError): decode_image(b'bad')
