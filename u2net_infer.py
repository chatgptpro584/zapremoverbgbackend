import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

def remove_background(image_file, output_path):
    from models.u2net import U2NET  # import from your model
    model_path = './models/u2net.pth'

    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = model(input_tensor)[0][0]
        mask = pred.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask).resize(image.size, Image.BILINEAR)
        image.putalpha(mask)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
