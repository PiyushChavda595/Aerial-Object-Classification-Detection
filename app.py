import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
from ultralytics import YOLO

# --- Page Configuration (Must be first) ---
st.set_page_config(
    page_title="Aerial Surveillance AI",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117; /* Dark Background */
        color: #E0E0E0;
    }

    /* --- FIX FOR SIDEBAR TOGGLE --- */
    /* Keep the header visible so the toggle button exists */
    header {
        background-color: transparent !important;
    }
    /* Style the sidebar toggle button (the arrow) */
    [data-testid="collapsedControl"] {
        color: #00E5FF !important;
    }

    /* Hide Default Streamlit Elements (Hamburger menu & Footer) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22; /* Slightly lighter dark */
        border-right: 1px solid #30363D;
    }

    /* Custom Headers */
    h1 {
        background: -webkit-linear-gradient(45deg, #00E5FF, #2979FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 0px;
    }
    
    h2, h3 {
        color: #FAFAFA !important;
        font-weight: 600;
    }

    /* Card-like Containers */
    .metric-card {
        background-color: #1C2128;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00E5FF;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00E5FF 0%, #2979FF 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.5);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00E5FF, #2979FF);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models (Cached) ---
@st.cache_resource
def load_classifier():
    model_path = os.path.join("models", "best_classification_model.h5")
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_yolo():
    model_path = os.path.join("models", "best_detection_model.pt")
    return YOLO(model_path)

@st.cache_data
def load_indices():
    with open(os.path.join("models", "class_indices.json"), "r") as f:
        indices = json.load(f)
    return {v: k for k, v in indices.items()}

# --- Preprocessing ---
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/drone.png", width=80)
    st.title("Aerial AI")
    st.markdown("`v2.0.0 | ResNet50V2`")
    st.markdown("---")
    
    # Navigation using radio buttons with custom formatting
    app_mode = st.radio(
        "Navigate",
        ["Home", "Classifier", "Detector"],
        captions=["Project Overview", "Bird vs Drone", "YOLOv8 Object Detection"]
    )
    
    st.markdown("---")
    st.markdown("###### System Status: 🟢 Online")

# --- MAIN PAGES ---

# 1. HOME PAGE
if app_mode == "Home":
    # Hero Section
    st.markdown("<h1>AERIAL SURVEILLANCE AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #8B949E;'>Intelligent Airspace Monitoring System</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Intro Text
    st.markdown("""
    <div style='text-align: center; max-width: 800px; margin: 0 auto;'>
        This AI system leverages advanced Deep Learning to solve the critical challenge of 
        distinguishing between biological targets (<b>Birds</b>) and technological threats (<b>Drones</b>) 
        in restricted airspace.
    </div>
    <br>
    """, unsafe_allow_html=True)

    # Key Stats/Features Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Deep Learning</h3>
            <p style="color: #8B949E;">Powered by <b>ResNet50V2</b> & Custom CNNs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p style="color: #8B949E;">Achieved <b>98% Accuracy</b> on Test Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-Time</h3>
            <p style="color: #8B949E;"><b>YOLOv8</b> Integration for Object Detection</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Cleaner Instruction
    st.info("👆 **Get Started:** Use the sidebar menu to access the **Classifier** or **Detector** modules.")


# 2. CLASSIFIER PAGE
elif app_mode == "Classifier":
    st.markdown("## 🔍 Binary Classification")
    st.markdown("Identify objects in an image as **Bird** or **Drone**.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 1. Upload Image")
        uploaded_file = st.file_uploader("Supported formats: JPG, PNG, JPEG", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Preview", use_column_width=True)
    
    with col2:
        st.markdown("### 2. Analysis")
        
        if uploaded_file:
            if st.button("Analyze Target"):
                with st.spinner("Processing..."):
                    model = load_classifier()
                    class_map = load_indices()
                    
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img)
                    
                    confidence = np.max(prediction)
                    class_idx = np.argmax(prediction)
                    label = class_map[class_idx].upper()
                    
                    # UI Logic for Result
                    if label == "DRONE":
                        color = "#FF4B4B"
                        icon = "🚁"
                    else:
                        color = "#00E5FF"
                        icon = "🦅"
                    
                    # Result Card
                    st.markdown(f"""
                    <div style="background-color: #1C2128; padding: 25px; border-radius: 12px; border: 1px solid #30363D; text-align: center;">
                        <p style="color: #8B949E; margin-bottom: 5px;">DETECTED CLASS</p>
                        <h1 style="color: {color} !important; font-size: 3rem !important; margin: 0;">{icon} {label}</h1>
                        <p style="font-size: 1.2rem; margin-top: 10px;">Confidence: <b>{confidence*100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("###")
                    st.progress(float(confidence))
        else:
            st.info("Please upload an image to begin analysis.")


# 3. DETECTOR PAGE
elif app_mode == "Detector":
    st.markdown("## 🎯 Object Detection (YOLOv8)")
    st.markdown("Locate and classify multiple objects in complex scenes.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        if st.button("Run Detection"):
            with st.spinner("Scanning image..."):
                model = load_yolo()
                results = model(image)
                
                res_plotted = results[0].plot()
                res_image = Image.fromarray(res_plotted[..., ::-1])
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Input", use_column_width=True)
                with col2:
                    st.image(res_image, caption="AI Detection Output", use_column_width=True)
                
                # Metrics
                count = len(results[0].boxes)
                st.success(f"Successfully detected **{count}** object(s).")