"""
Pneumonia X-Ray Pipeline — Stage 6: Simple Web Interface (Streamlit)

- Upload X-ray image.
- Show prediction, confidence score, triage level.
- Display Grad-CAM heatmap overlay (infected area in red).
"""

import streamlit as st
from pathlib import Path
import tempfile
import os

from predict import predict
from config import MODEL_SAVE_PATH


def run_app(model_path: str = None):
    model_path = model_path or MODEL_SAVE_PATH
    if not Path(model_path).exists():
        st.warning(f"Model not found at {model_path}. Train first with: python train.py")
        st.stop()

    st.set_page_config(page_title="Pneumonia Second Opinion", layout="wide")
    st.title("Pneumonia X-Ray Second Opinion")
    st.markdown("Upload a chest X-ray to get prediction, confidence, triage level, and Grad-CAM heatmap.")

    uploaded = st.file_uploader("Choose a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    try:
        result = predict(tmp_path, model_path=model_path, gradcam_output_path=tmp_path + ".gradcam.png")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction")
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Confidence:** {result['confidence_pct']:.1f}%")
            st.write(f"**Triage:** {result['triage']}")
            st.image(tmp_path, caption="Uploaded X-ray", use_container_width=True)

        with col2:
            st.subheader("Grad-CAM (infected area highlighted)")
            if Path(result["gradcam_overlay_path"]).exists():
                st.image(result["gradcam_overlay_path"], caption="Heatmap overlay", use_container_width=True)
            else:
                st.write("Grad-CAM overlay not saved.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if os.path.exists(tmp_path + ".gradcam.png"):
            os.unlink(tmp_path + ".gradcam.png")


if __name__ == "__main__":
    run_app()
