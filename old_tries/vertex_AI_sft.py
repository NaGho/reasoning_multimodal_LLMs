import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.tuning import sft
# TODO : Set values as per your requirements
# Project and Storage Constants
PROJECT_ID = "<project_id>"
REGION = "<region>"
vertexai.init(project=PROJECT_ID, location=REGION)
# define training & eval dataset.
TRAINING_DATASET = "gs://cloud-samples-data/vertex-ai/model-evaluation/peft_train_sample.jsonl"
# set base model and specify a name for the tuned model
BASE_MODEL = "gemini-1.5-pro-002"
TUNED_MODEL_DISPLAY_NAME = "gemini-fine-tuning-v1"
# start the fine-tuning job
sft_tuning_job = sft.train(
source_model=BASE_MODEL,
train_dataset=TRAINING_DATASET,
# # Optional:
tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
)
# Get the tuning job info.
sft_tuning_job.to_dict()

# tuned model endpoint name
tuned_model_endpoint_name = sft_tuning_job.tuned_model_endpoint_name
# use the tuned model
tuned_genai_model = GenerativeModel(tuned_model_endpoint_name)
print(tuned_genai_model.generate_content(contents="What is a LLM?"))