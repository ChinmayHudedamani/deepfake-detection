import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Deepfake Detector", page_icon="🧠", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #ffffff;
    text-align: center;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
MODEL_PATH = "models/best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = 224

# ================= FUNCTIONS =================
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ================= HEADER =================
st.title("🧠 AI Deepfake Detection System")
st.write("### Upload an image and let AI decide whether it's REAL or FAKE")

# ================= LAYOUT =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("📊 Prediction Result")

# ================= PROCESS =================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing Image with AI Model..."):
        time.sleep(1.5)
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]

    confidence = float(prediction)

    if confidence > 0.5:
        col2.markdown(f'<div class="result-box" style="background-color:#ffcccc;color:#900;">⚠ FAKE IMAGE DETECTED<br>Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
    else:
        col2.markdown(f'<div class="result-box" style="background-color:#ccffcc;color:#060;">✅ REAL IMAGE<br>Confidence: {(1-confidence)*100:.2f}%</div>', unsafe_allow_html=True)

    st.progress(int(confidence * 100))

# ================= SIDEBAR =================
st.sidebar.title("🔍 About Project")
st.sidebar.write("""
This system uses a **Deep Learning CNN Model** to detect whether an image is real or AI-generated.

### Technologies Used:
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- CNN (Transfer Learning)

### Use Cases:
- Social Media Verification
- Fake News Detection
- Digital Forensics
- AI Image Validation
""")

st.sidebar.success("Development in progress")

# ================= FOOTER =================
st.markdown("---")
st.markdown("<center>Chinmay V Hudedamani | Deepfake Detection System</center>", unsafe_allow_html=True)

