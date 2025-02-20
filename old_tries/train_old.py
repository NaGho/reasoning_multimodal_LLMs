import open_clip
import torch
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Configuration
MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'laion2b_s34b_b79k'
BATCH_SIZE = 32  # Adjust based on your GPU memory
EPOCHS = 10
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
DATA_DIR = "EduChat-Math" # Path to your EduChat-Math dataset

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset Class
class MathDataset(Dataset):
    def __init__(self, data_dir, preprocess):
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.image_paths = []
        self.texts = []

        # Load data (adapt to your dataset structure)
        for filename in os.listdir(data_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(data_dir, filename)
                text_filename = filename.split('.')[0] + '.txt' # Assumes text files have the same name
                text_path = os.path.join(data_dir, text_filename)
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f: # Handle potential UnicodeDecodeError
                        text = f.read().strip()
                        self.image_paths.append(image_path)
                        self.texts.append(text)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image)
        except Exception as e:
            print(f"Error loading image: {image_path}, error: {e}")
            return None, None
        
        return image, text

# Model and Preprocessing
model, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(device)

# Data Loading
dataset = MathDataset(DATA_DIR, preprocess)
dataset = [item for item in dataset if item != (None, None)] # Remove any None items
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 on windows

# Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(dataloader) * EPOCHS)

# Loss function
loss_fn = open_clip.loss.CLIPLoss(local_loss=False)

# Training Loop
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, texts in pbar:
        if images is None or texts is None: #Ensure no null inputs
            continue
        images = images.to(device)
        texts = open_clip.tokenize(texts).to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = loss_fn(image_features, text_features)

        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix({"loss": loss.item()})

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_openclip.pt")
print("Fine-tuning complete. Model saved as fine_tuned_openclip.pt")