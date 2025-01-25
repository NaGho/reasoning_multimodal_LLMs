from datasets import load_dataset
import json
from tqdm import tqdm 

def convert_mathvision_to_json(split):
    ds = load_dataset("MathLLMs/MathVision")
    
    converted_data = []
    for item in tqdm(ds[split]):
        # Basic structure for conversation
        # system_promt = f"Answer the following questions about the image. \n Question: {item.get('question')} Detail your reasoning step by step and write the final answer as one number or one letter at the end after printing <Answer:>"
        # if len(item['options']) > 0:
        #     system_promt += f"\n The answer is one of {', '.join(item['options'])}. "
        conversation_entry = {
            "system_prompt": "Answer the following question given the image.",
            "image": item.get("image"),
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>{item.get('question', 'Describe this image.')}"
                },
                {
                    "from": "gpt",
                    "value": item.get('answer', 'No description available.')
                }
            ]
        }
        
        converted_data.append(conversation_entry)
    
    return converted_data

# # Convert test split
# test_json = convert_mathvision_to_json('test')
# with open('mathvision_test.json', 'w') as f:
#     json.dump(test_json, f, indent=2)
# print(f"Converted {len(test_json)} entries for test split")

# Convert testmini split
testmini_json = convert_mathvision_to_json('testmini')
with open('mathvision_testmini.json', 'w') as f:
    json.dump(testmini_json, f, indent=2)

print(f"Converted {len(testmini_json)} entries for testmini split")