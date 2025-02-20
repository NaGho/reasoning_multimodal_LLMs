from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    AutoModelForVision2Seq
)
from PIL import Image
import requests
import torch

class VisionLanguageProcessor:
    def __init__(self, model_type="qwen", optimize_memory=True):
        """
        Initialize the vision-language processor with specified model type.
        Args:
            model_type (str): One of "llava" or "qwen"
            optimize_memory (bool): Whether to apply memory optimizations
        """
        self.MODEL_TYPE = model_type.lower()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.use_half = False
        
        model_configs = {
            "llava": {
                "name": "llava-hf/llava-1.5-7b-hf",
                "model_class": LlavaForConditionalGeneration,
            },
            "qwen": {
                "name": "Qwen/Qwen2-VL-7B",
                "model_class": AutoModelForVision2Seq,
            }
        }
        
        if self.MODEL_TYPE not in model_configs:
            raise ValueError(f"Invalid model type. Choose from {list(model_configs.keys())}")
            
        config = model_configs[self.MODEL_TYPE]
        
        try:
            print(f"Loading {self.MODEL_TYPE} model...")
            
            if optimize_memory:
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.deterministic = False
            
            # Load components in correct order
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["name"],
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                config["name"],
                trust_remote_code=True
            )
            
            self.model = config["model_class"].from_pretrained(
                config["name"],
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map='auto'
            )
            
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                
        except Exception as e:
            raise Exception(f"Error loading {self.MODEL_TYPE} model: {e}")

    def load_image(self, image_url):
        """Load and preprocess image from URL."""
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        except Exception as e:
            raise Exception(f"Error processing image: {e}")

    def process(self, image_url, user_prompt, max_length=200):
        """Process image and text prompt with optimized settings."""
        try:
            # Load raw image
            image = self.load_image(image_url)
            
            if self.MODEL_TYPE == "llava":
                return self._process_llava(image, user_prompt, max_length)
            else:  # qwen
                return self._process_qwen(image, user_prompt, max_length)
                
        except Exception as e:
            print(f"Error during processing: {e}")
            return None

    def _process_llava(self, image, user_prompt, max_length):
        """Process using LLaVA model with optimized generation parameters."""
        inputs = self.processor(
            images=image,
            text=user_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=30,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _process_qwen(self, image, user_prompt, max_length):
        """Process using Qwen2-VL model with optimized generation parameters."""
        prompt = f"<image>\n{user_prompt}"
        
        # Process image and text together
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=30,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()

def main():
    # Test with a single model
    processor = VisionLanguageProcessor(model_type="qwen", optimize_memory=True)
    
    image_url = "https://www.thespruceeats.com/thmb/s11oj7aiRC0zcjIXuu80NmT-L4o=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/SES-basic-meat-lasagna-recipe-2097886-hero-01-cdd28f5b4aa940faa193e39a1629f89a.jpg"
    user_prompt = "Describe this food in detail and suggest a drink pairing."
    
    try:
        output_text = processor.process(image_url, user_prompt)
        if output_text:
            print("Generated Text:", output_text)
        else:
            print("Error in processing.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()