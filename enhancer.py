import requests

def enhance_image(image_file):
    image_bytes = image_file.read()
    headers = {"accept": "application/json"}
    files = {"image": image_bytes}
    response = requests.post("https://hf.space/embed/deepghs/image-enhance/+/api/predict", files=files, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        result_url = json_data['data'][0]
        result = requests.get(result_url)
        output_path = "output/enhanced.png"
        with open(output_path, "wb") as f:
            f.write(result.content)
        return output_path
    else:
        raise Exception("Enhancement failed")
