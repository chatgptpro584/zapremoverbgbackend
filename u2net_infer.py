from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from models.u2net import U2NET
import os

def remove_background(image_file, output_path):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)

    model = U2NET(3, 1)
    model_path = './models/u2net.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        d1, *_ = model(img_tensor)
        mask = d1[0][0].detach().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)

        empty = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image.putalpha(mask)
        image.save(output_path)
