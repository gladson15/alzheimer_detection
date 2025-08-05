import streamlit as st
import torch
from PIL import Image
from model.alzheimer_model import AlzheimerModel
from utils.age_utils import estimate_age_from_stage

# --- Configuration ---
st.set_page_config(page_title="🧠 Alzheimer's MRI Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Alzheimer's MRI Analyzer</h1>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🖼️ Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("ℹ️ **Supported**: Preprocessed axial brain MRI (T1-weighted).")
st.sidebar.markdown("💡 **Note**: This is a demo app. Predictions are not for clinical use.")

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerModel("model/alzeimer_best_model.pth", device=device)
classes = ['No Impairment', 'Very Mild Impairment', 'Mild Impairment', 'Moderate Impairment']

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 🧠 Uploaded MRI", unsafe_allow_html=True)
    st.image(image, use_column_width=False, width=350)

    # Predict
    pred_class, confidence = model.predict(image)
    stage = classes[pred_class]
    age = estimate_age_from_stage(stage)
    confidence_pct = round(confidence * 100, 2)

    st.markdown("---")
    st.markdown("### 🔍 Prediction Summary")
    st.markdown(f"<span style='font-size:22px;'>🧠 <b>Stage:</b> {stage}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size:22px;'>📅 <b>Estimated Age:</b> {age} years</span>", unsafe_allow_html=True)

    st.markdown("#### 🔘 Prediction Confidence")
    st.progress(confidence)

    st.markdown("---")
    with st.expander("🧠 What do these stages mean?"):
        st.markdown("""
        - **No Impairment**: Healthy brain.
        - **Very Mild Impairment**: Slight memory loss.
        - **Mild Impairment**: Noticeable confusion or memory issues.
        - **Moderate Impairment**: Clear signs of Alzheimer’s, daily support needed.
        """)
