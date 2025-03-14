import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time
import numpy as np
import cv2
from streamlit_image_select import image_select
from google import genai
from google.genai import types

# --------------- FUNCTIONS START -------------------------
def img_proc(img):
    # Convert image to numpy array for YOLO processing
    image_np = np.array(image)

    # Run inference
    results = model2(image, conf=0.05)
    with st.status("Processing P&ID..."):
        st.write("Analyzing Data...")
        time.sleep(1)
        st.write("Initializing Trained Computer Vision Model")
        time.sleep(2)
        st.write("CV Model Processing...")
        time.sleep(1)

    return image_np, results

def bounding_box(image_np, results):
    # Draw bounding boxes
    for box in results[0].boxes:
        original_class = results[0].names[int(box.cls)]
        if selected_classes.get(str(int(box.cls)), False):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label = class_name_mapping.get(original_class, original_class)
            color = class_color_mapping.get(str(int(box.cls)), (0, 255, 0))

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

import streamlit as st
import google.generativeai as genai
import os

# Load API key from Streamlit secrets or environment variable
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Missing API key. Please set 'GEMINI_API_KEY' in Streamlit secrets.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)

# Define system prompt
context_prompt = """Objective:
You are a chatbot designed to assist engineers in understanding and working with architecture diagrams. Your primary function is to accurately label diagrams, identify symbols, provide explanations, and handle FAQs related to P&ID and other engineering schematics. 
If any out-of-topic or out-of-domain questions are asked, politely redirect users to industry-based information and relevant engineering queries. Keep responses under 500 words.

Core Capabilities:
1. Architecture Labeling
   - Accepts user-input diagrams.
   - Identifies and labels components automatically.
   - Provides explanations for each labeled component.
2. Symbol Detection and Filtering
   - Lists all symbols present in a diagram.
   - Allows users to filter specific symbols.
   - Highlights selected symbols within the diagram.
3. Chatbot for FAQs and Engineering Queries
   - Answers user questions based on P&ID and engineering concepts.
"""

# Function to generate response using Gemini API
def generate(chat_input):
    model = genai.GenerativeModel("gemini-pro")  # Use "gemini-pro" for accuracy
    response = model.generate_content(f"{context_prompt}\n\nUser: {chat_input}\n\nAssistant:")
    return response.text

# Streamlit UI
st.title("P&ID Chatbot - Powered by Gemini AI")

user_input = st.text_input("Ask your engineering question:")
if user_input:
    response = generate(user_input)
    st.write(response)


# --------------- FUNCTIONS END -------------------------
model2 = YOLO("finaltrain_best.pt")
# Class name mapping
class_name_mapping = {
    "0": "Ball Valve 1",
    "1": "Ball Valve 2",
    "2": "Ball Valve 3",
    "3": "Onsheet Connector",
    "4": "Centrifugal Fan",
    "5": "IHTL",
    "6": "Pneumatic Signal",
    "7": "NP"
}
# Color mapping for bounding boxes
class_color_mapping = {
    "0": (255, 0, 0),
    "1": (0, 255, 0),
    "2": (0, 0, 255),
    "3": (255, 255, 0),
    "4": (255, 0, 255),
    "5": (0, 255, 255),
    "6": (128, 0, 128),
    "7": (0, 128, 128)
}

selector_img_wide_screen = ["assets/PIDs/2.jpeg", "assets/PIDs/3.jpeg", "assets/PIDs/5.png", "assets/PIDs/6.png"]


# Streamlit UI
st.title("PIDetect - P&ID Tool")
st.write("Upload or Select a P&ID Image")



# Sidebar for class filtering
st.sidebar.title("Filter Classes")
selected_classes = {}
for class_id, class_name in class_name_mapping.items():
    selected_classes[class_id] = st.sidebar.checkbox(class_name, value=True)

with st.sidebar:
    # Display the chatbot's title on the page
    st.title("Chat with PIDgpt")
    user_input = st.chat_input("Ask Assistant...")
    if user_input:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_input)

        # Send user's message to Gemini and get the response
        response = generate(user_input)

        # Display Gemini's response
        with st.chat_message("assistant"):
            st.markdown(response)

# Image Selector
img_path = image_select("Select P&ID Image", selector_img_wide_screen)

# File Uploader
uploaded_file = st.file_uploader("Or Upload an Image", type=["png", "jpg", "jpeg"])

# Determine which image to process
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif img_path:
    image = Image.open(img_path)

# Process the selected image
if image is not None:
    st.image(image, caption="Selected Image", use_container_width=True)

    if st.button("Predict"):
        image_np, results = img_proc(image)
        bounding_box(image_np, results)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Processed Image", use_container_width=True)
