import json
import os
import cv2

def load_mawps_data(data_path):
    # Load MAWPS data
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def load_visual_data(image_dir):
    visual_data = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)
            visual_data[filename] = image
    return visual_data

# ... other data loading functions