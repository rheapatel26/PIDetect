import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time
import numpy as np
import cv2
from streamlit_image_select import image_select
import google.generativeai as genai

# -------------------- FUNCTIONS --------------------

def img_proc(img):
    image_np = np.array(img)

    results = model2(img, conf=0.05)

    with st.status("Processing P&ID..."):
        st.write("Analyzing Data...")
        time.sleep(1)
        st.write("Initializing Trained Computer Vision Model")
        time.sleep(2)
        st.write("CV Model Processing...")
        time.sleep(1)

    return image_np, results


def bounding_box(image_np, results):
    for box in results[0].boxes:
        cls_id = str(int(box.cls))

        if selected_classes.get(cls_id, False):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label = class_name_mapping.get(cls_id, cls_id)
            color = class_color_mapping.get(cls_id, (0, 255, 0))

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def generate(chat_input):
    api_key = st.secrets["GEMINI_API_KEY"]

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    context_prompt = """You are an engineering assistant for P&ID diagrams.
Answer clearly, in bullet points, within 300-500 words.
If out-of-domain, redirect to engineering topics."""

    try:
        response = model.generate_content(
            context_prompt + "\n\nUser Query: " + chat_input
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


# -------------------- MODEL --------------------

model2 = YOLO("finaltrain_best.pt")

# Class mappings
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

# Sample images
selector_img_wide_screen = [
    "assets/PIDs/2.jpeg",
    "assets/PIDs/3.jpeg",
    "assets/PIDs/5.png",
    "assets/PIDs/6.png"
]

# -------------------- UI --------------------

st.title("PIDetect - P&ID Tool")
st.write("Upload or Select a P&ID Image")

# Sidebar filters
st.sidebar.title("Filter Classes")
selected_classes = {}

for class_id, class_name in class_name_mapping.items():
    selected_classes[class_id] = st.sidebar.checkbox(class_name, value=True)

# Chatbot
with st.sidebar:
    st.title("Chat with PIDgpt")

    user_input = st.chat_input("Ask Assistant...")

    if user_input:
        st.chat_message("user").markdown(user_input)

        response = generate(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

# Image selection
img_path = image_select("Select P&ID Image", selector_img_wide_screen)

uploaded_file = st.file_uploader("Or Upload an Image", type=["png", "jpg", "jpeg"])

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif img_path:
    image = Image.open(img_path)

# Prediction
if image is not None:
    st.image(image, caption="Selected Image", use_container_width=True)

    if st.button("Predict"):
        image_np, results = img_proc(image)
        bounding_box(image_np, results)

        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Processed Image", use_container_width=True)
