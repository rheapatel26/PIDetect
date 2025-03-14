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

def generate(chat_input):
    api_key = str(st.secrets["GEMINI_API_KEY"])  # Ensure it's a string
    client = genai.Client(api_key=api_key)

    context_prompt = """Your role is to assist engineers in understanding and working with architecture diagrams, 
    primarily focusing on P&ID diagrams. Ensure responses are industry-standard, clear, and relevant to engineering 
    professionals. Respond in a concise and technical manner within 500 words."""

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=context_prompt + "\n\nUser Query: " + chat_input),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=200,
        response_mime_type="text/plain",
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text
    return response_text


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
