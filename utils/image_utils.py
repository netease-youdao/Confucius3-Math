from PIL import Image
import io
import base64

# 创建utils目录并迁移工具函数
def encode_image(image_path):
    img = Image.open(image_path)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')