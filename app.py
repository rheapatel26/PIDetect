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

    context_prompt = """Objective:
You are a chatbot designed to assist engineers in understanding and working with architecture diagrams. Your primary function is to accurately label diagrams, identify symbols, provide explanations, and handle FAQs related to P&ID and other engineering schematics. You must ensure precise responses for engineers working in industries similar to engineering service companies.

If any out-of-topic or out-of-domain questions are asked, politely redirect users to industry-based information and relevant engineering queries. Complete your answer and do not exceed 500 words

Core Capabilities:
1. Architecture Labeling
   - Accepts user-input diagrams.
   - Identifies and labels components automatically.
   - Provides explanations for each labeled component.
2. Symbol Detection and Filtering
   - Lists all symbols present in a diagram.
   - Allows users to filter specific symbols.
   - Highlights selected symbols within the diagram.
3. ChatGPT-4o-mini for FAQs and Doubts
   - Answers technical queries about architecture and symbols.
   - Provides industry-standard explanations for components.
   - Assists in troubleshooting and best practices.
 General Engineering Knowledge for the Chatbot
# Common Engineering Symbols and Their Meanings:
1. Pipelines and Flow Indicators
   - Solid Line: General Process Flow
   - Dashed Line: Instrumentation Signal Flow
   - Arrow: Flow Direction
   - Double Line: Insulated Pipeline
2. Valves
   - Gate Valve: Controls flow by lifting a barrier.
   - Check Valve: Allows unidirectional flow, prevents backflow.
   - Ball Valve: Quick shut-off using a spherical disc.
   - Globe Valve: Regulates flow precisely.
3. Instruments
   - PT (Pressure Transmitter): Measures pressure.
   - FT (Flow Transmitter): Measures fluid flow.
   - TT (Temperature Transmitter): Measures temperature.
   - LT (Level Transmitter): Measures liquid levels.
4. Pumps and Compressors
   - Centrifugal Pump: Used for fluid transport.
   - Reciprocating Pump: Uses pistons for high-pressure applications.
   - Rotary Compressor: Continuous compression mechanism.
   - Screw Compressor: Used in gas transport and HVAC systems.
5. Tanks and Vessels
   - Storage Tank: Holds liquids/gases.
   - Pressure Vessel: Stores high-pressure substances.
   - Reactors: Used for chemical processing.
6. Electrical Components
   - Motor Symbols: Represent different motor types.
   - Transformers: Step-up/down voltage conversion.
   - Circuit Breakers: Protect against short circuits.

 15 Key FAQs for Fine-Tuning (with Answers):
1. What do the different pipeline line types indicate in a P&ID?  

   Pipeline line types represent different functions: solid lines indicate general process flow, dashed lines represent instrumentation signal flow, and double lines denote insulated pipelines.

2. How can I filter and view only the control valves in my architecture?  

   Use the filter function to select 'Control Valves' from the available categories. The system will highlight only those valves in your diagram.

3. What is the difference between a centrifugal and reciprocating pump?  

   A centrifugal pump uses rotational energy to move fluids, while a reciprocating pump uses pistons to generate pressure-driven flow.

4. How does a pressure transmitter (PT) function?  

   A pressure transmitter converts pressure measurements into an electrical signal that can be read by monitoring systems.

5. Can you explain how a check valve prevents backflow?  

   A check valve allows fluid to flow in one direction and automatically closes to prevent backflow when the flow stops.

6. What is the significance of different arrow types in pipeline diagrams?  

   Arrows indicate flow direction. Single arrows show standard flow, while double arrows may indicate bidirectional flow.

7. How do I identify insulated pipelines in my diagram?  

   Insulated pipelines are typically represented by double lines in P&ID diagrams.

8. What is the purpose of a globe valve in a piping system?  

   A globe valve regulates fluid flow with precise control by raising or lowering a disk.

9. How does the filter function help in isolating components in my diagram?  

   The filter function enables users to highlight and isolate specific symbols, making it easier to focus on particular components.

10. What does a dashed line mean in a control system architecture?  

    Dashed lines represent signal flows, typically from instrumentation or control systems.

11. How do I ensure that my architecture diagram follows industry standards?  

    Use standardized symbols and adhere to P&ID conventions such as ISA S5.1 and ISO 14617.

12. What is the function of a motor in a process system?  

    A motor converts electrical energy into mechanical energy to drive pumps, compressors, or conveyors.

13. Why are reactors used in chemical processing plants?  

    Reactors facilitate chemical reactions under controlled conditions for efficient production.

14. How does a pressure vessel differ from a storage tank?  

    Pressure vessels store substances under high pressure, whereas storage tanks hold liquids or gases at near-atmospheric pressure.

15. Can you provide recommendations for improving pipeline efficiency?  

    Optimize pipe sizing, minimize bends, maintain proper insulation, and use high-efficiency pumps and valves.

 Response Format for the Chatbot:

- Clear and concise responses.

- Diagrams where applicable.

- References to industry standards.

- Step-by-step guidance for troubleshooting.

- User-friendly and technical explanations.

"""
    MAX_TOKENS = 100


    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=chat_input),
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
