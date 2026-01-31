"""
Pneumonia X-Ray Pipeline — Stage 5: Inference

- Load image, preprocess, predict label and confidence.
- Assign triage level: ≥0.8 High Risk, 0.4–0.8 Medium Risk, <0.4 Low Risk.
- Generate Grad-CAM visualization (heatmap overlay).
"""

import argparse
import numpy as np
from pathlib import Path
from tensorflow import keras

from preprocess import load_and_preprocess
from config import MODEL_SAVE_PATH, TRIAGE_HIGH, TRIAGE_MED
from gradcam import gradcam_on_path


def confidence_to_triage(prob: float) -> str:
    """Convert sigmoid probability to triage level."""
    if prob >= TRIAGE_HIGH:
        return "High Risk"
    if prob >= TRIAGE_MED:
        return "Medium Risk"
    return "Low Risk"


def predict(
    image_path: str,
    model_path: str = None,
    gradcam_output_path: str = None,
) -> dict:
    """
    Run inference on one image.
    Returns dict: label, confidence_pct, triage, gradcam_overlay_path.
    """
    model_path = model_path or MODEL_SAVE_PATH
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with: python train.py")
    model = keras.models.load_model(model_path)

    img = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img, axis=0)
    prob = float(model.predict(img_batch, verbose=0)[0, 0])

    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence_pct = prob * 100.0 if label == "PNEUMONIA" else (1.0 - prob) * 100.0
    triage = confidence_to_triage(prob)

    if gradcam_output_path is None:
        gradcam_output_path = str(Path(image_path).with_suffix(".gradcam.png"))

    try:
        overlay, _, _ = gradcam_on_path(model_path, image_path, output_path=gradcam_output_path)
    except Exception as e:
        print(f"Grad-CAM skipped: {e}")
        overlay = None
        gradcam_output_path = ""

    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence_pct:.1f}%")
    print(f"Triage: {triage}")
    print(f"Grad-CAM saved: {gradcam_output_path}")

    return {
        "label": label,
        "confidence_pct": confidence_pct,
        "triage": triage,
        "probability": prob,
        "gradcam_overlay_path": gradcam_output_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pneumonia X-Ray inference")
    parser.add_argument("image_path", help="Path to chest X-ray image")
    parser.add_argument("--model", default=MODEL_SAVE_PATH, help="Path to saved model")
    parser.add_argument("--gradcam-out", default=None, help="Path for Grad-CAM overlay image")
    args = parser.parse_args()
    predict(args.image_path, model_path=args.model, gradcam_output_path=args.gradcam_out)
