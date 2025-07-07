from PIL import Image, ImageFilter

def resize_image(image_file, width, height):
    image = Image.open(image_file)
    resized = image.resize((width, height))
    path = "output/resized.png"
    resized.save(path)
    return path

def crop_image(image_file, x, y, w, h):
    image = Image.open(image_file)
    cropped = image.crop((x, y, x + w, y + h))
    path = "output/cropped.png"
    cropped.save(path)
    return path

def apply_effect(image_file, effect_type):
    image = Image.open(image_file)
    if effect_type == "blur":
        image = image.filter(ImageFilter.BLUR)
    elif effect_type == "sharpen":
        image = image.filter(ImageFilter.SHARPEN)
    elif effect_type == "contour":
        image = image.filter(ImageFilter.CONTOUR)
    path = f"output/effect_{effect_type}.png"
    image.save(path)
    return path
