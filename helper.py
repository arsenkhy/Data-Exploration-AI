import base64
from PIL import Image
from io import BytesIO

def render_svg(svg_file):
    with open(svg_file, "r") as f:
        lines = f.readlines()
        svg = "".join(lines)

        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        return html
    
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))