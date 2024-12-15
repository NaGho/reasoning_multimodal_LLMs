from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_image(image_path):
    image = Image.open(image_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(pixel_values, max_length=model.config.max_length)
    prediction = processor.decode(outputs[0], skip_special_tokens=True)
    return prediction

# ... other visual processing functions (CLIP, etc.)