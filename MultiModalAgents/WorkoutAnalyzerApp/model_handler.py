import streamlit as st
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

@st.cache_resource
def load_vlm():
    """Caches the ChatOllama model so it doesn't reload on every interaction."""
    return ChatOllama(model="bonsai-2:4b", num_ctx=16384)

def encode_image_to_base64(path):
    """Encodes an image file into a base64 string for VLM payload."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_workout_frames(llm, extracted_frames, prompt):
    """Constructs the multi-image payload and queries the model."""
    message_content = [{"type": "text", "text": prompt}]
    
    for path in extracted_frames:
        b64_str = encode_image_to_base64(path)
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
        })

    try:
        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
    except Exception as e:
        return f"Inference error occurred: {str(e)}"