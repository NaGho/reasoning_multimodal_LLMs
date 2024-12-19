import streamlit as st
from utils.data_loader import load_mawps_data, load_visual_data
from models.text_processor import process_text
from models.visual_processor import process_image, get_image_text_similarity
from solvers.math_solver import solve_math_problem
import open_clip
import torch
from PIL import Image

st.title("Multimodal Math Problem Solver Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], use_column_width=True)

# User input
if prompt := st.chat_input("Ask me a math question or upload an image"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"], key="image_upload")
    if uploaded_file:
        st.session_state.messages[-1]["image"] = uploaded_file
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        # Example usage of similarity function
        if prompt:
            similarity = get_image_text_similarity(uploaded_file, prompt)
            if similarity is not None:
                st.write(f"Image-Text Similarity: {similarity:.4f}")

    # Process inputs
    visual_representation = None
    if "image" in st.session_state.messages[-1]:
        visual_representation = process_image(st.session_state.messages[-1]["image"])

    text_representation = process_text(prompt)

    # Get the solution
    solution = solve_math_problem(text_representation, visual_representation)

    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": solution})
    # Display bot message in chat message container
    with st.chat_message("assistant"):
        st.markdown(solution)


try:
    model, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None) # Load without pretraining
    model.load_state_dict(torch.load("fine_tuned_openclip.pt", map_location=device)) # Load fine-tuned weights
    model.to(device)
    model.eval() # Set to evaluation mode
    print("Fine-tuned OpenCLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading fine-tuned OpenCLIP model: {e}")
    exit()