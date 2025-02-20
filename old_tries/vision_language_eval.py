import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from my_datasets import load_dataset  # Assuming the GMS8k dataset is HuggingFace-compatible
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np

compute = True

# Check device compatibility
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_name = "Qwen/Qwen2-VL-2B-Instruct"
dataset_name = "deepcs233/Visual-CoT"
file_name = f"data/output/{dataset_name.split('/')[-1]}_{model_name.split('/')[-1]}.csv"
# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    device_map=None
)

# Initialize processor
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

# Load the dataset
dataset = load_dataset("Graphcore/vqa", split="validation[:50]", trust_remote_code=True, download_mode="force_redownload")

# Evaluation storage
results = []

if compute:
    # Iterate over dataset
    for sample in tqdm(dataset):
        image_path = None  # Adjust key to match GMS8k format
        prompt = sample["question"]    # Adjust key to match GMS8k format
        answer = sample["answer"]
        ground_truth = answer.split('#### ')[1]
        

        # Prepare input
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]} # {"type": "image", "image": image_path}, 
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the device
        inputs = inputs.to(device)
        model = model.to(device)

        # Perform inference
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Store results
        results.append({"image": image_path, "prompt": prompt, "generated_text": output_text, "ground_truth": ground_truth})

    pd.DataFrame(results).to_csv(file_name)
else:
    results = pd.read_csv(file_name)

def final_answer(text: str):
    text = text.lower()
    if 'answer is' not in text:
        return np.nan
    return text.split('answer is:').strip().replace('$', '')
# Example metric: String matching (very basic)
ground_truths = [result["ground_truth"] for result in results]
generated_texts = [final_answer(result["generated_text"]) for result in results]
exact_matches = [gt == gen for gt, gen in zip(ground_truths, generated_texts)]
accuracy = sum(exact_matches) / len(exact_matches)

print(f"Exact Match Accuracy: {accuracy:.4f}")

# Optionally save results
import json
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)
