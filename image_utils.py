from PIL import Image, ImageEnhance, ImageFilter
import io

def resize_image(file, width, height):
    img = Image.open(file)
    resized = img.resize((width, height))
    output = io.BytesIO()
    resized.save(output, format='PNG')
    output.seek(0)
    return output

def crop_image(file, x, y, w, h):
    img = Image.open(file)
    cropped = img.crop((x, y, x + w, y + h))
    output = io.BytesIO()
    cropped.save(output, format='PNG')
    output.seek(0)
    return output

def apply_effect(file, effect_type):
    img = Image.open(file)
    if effect_type == 'grayscale':
        img = img.convert('L').convert('RGBA')
    elif effect_type == 'blur':
        img = img.filter(ImageFilter.BLUR)
    elif effect_type == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    return output
