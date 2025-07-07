import requests
from PIL import Image
import io

API_URL = "https://api-inference.huggingface.co/models/causalml/image-upscaler"
API_TOKEN = "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"

def enhance_image(image_file):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    img_bytes = image_file.read()

    response = requests.post(API_URL, headers=headers, data=img_bytes)
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API failed: {response.text}")

    output = Image.open(io.BytesIO(response.content))
    path = "output/enhanced.png"
    output.save(path)
    return path
