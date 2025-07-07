from flask import Flask, request, jsonify, send_file
from u2net_infer import remove_background
from enhancer import enhance_image
from image_utils import resize_image, crop_image, apply_effect
import os
import gdown

# Auto-download U²-Net model if missing
os.makedirs('models', exist_ok=True)
model_path = './models/u2net.pth'

try:
    if not os.path.exists(model_path):
        print("Downloading u2net.pth from Google Drive...")
        url = 'https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ'
        gdown.download(url, model_path, quiet=False)
except Exception as e:
    print("❌ Model download failed:", e)

app = Flask(__name__)

@app.route('/')
def home():
    return "ZAP Remover BG backend is live!"

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image = request.files['image']
    output_path = 'output/removed.png'
    os.makedirs('output', exist_ok=True)
    remove_background(image, output_path)
    return send_file(output_path, mimetype='image/png')

@app.route('/enhance', methods=['POST'])
def enhance():
    image = request.files['image']
    output = enhance_image(image)
    return send_file(output, mimetype='image/png')

@app.route('/resize', methods=['POST'])
def resize():
    image = request.files['image']
    width = int(request.form.get('width'))
    height = int(request.form.get('height'))
    output = resize_image(image, width, height)
    return send_file(output, mimetype='image/png')

@app.route('/crop', methods=['POST'])
def crop():
    image = request.files['image']
    x = int(request.form.get('x'))
    y = int(request.form.get('y'))
    w = int(request.form.get('width'))
    h = int(request.form.get('height'))
    output = crop_image(image, x, y, w, h)
    return send_file(output, mimetype='image/png')

@app.route('/effect', methods=['POST'])
def effect():
    image = request.files['image']
    effect_type = request.form.get('effect')
    output = apply_effect(image, effect_type)
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
