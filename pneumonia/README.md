# Pneumonia X-Ray Second Opinion Pipeline

Binary classification (Normal / Pneumonia) with transfer learning, triage levels, and Grad-CAM explainability.

## Data layout

Place chest X-rays in:

```
data/chest_xray/
  NORMAL/    -> label 0
  PNEUMONIA/ -> label 1
```

(JPEG/PNG; corrupted images are dropped automatically.)

## Stages (short explanations)

1. **preprocess.py** — Load images from folder structure, remove corrupted files. Preprocess: grayscale → resize 224×224 → median blur → CLAHE → normalize [0,1] → 3-channel. Augmentation: rotation, horizontal flip, zoom. Stratified train/val/test split.

2. **train.py** — ResNet50 (ImageNet), frozen base; GlobalAveragePooling + Dense(1, sigmoid). Binary Cross Entropy, Adam. Metrics: accuracy, precision, **recall** (best model saved by val_recall).

3. **evaluate.py** — Confusion matrix and classification report on test set; recall emphasized.

4. **gradcam.py** — Grad-CAM from last conv layer; overlay heatmap on X-ray (infected area in red) for doctor verification.

5. **predict.py** — Single-image inference: label, confidence %, triage (≥0.8 High Risk, 0.4–0.8 Medium, <0.4 Low), and Grad-CAM image.

6. **app.py** — Optional Streamlit UI: upload X-ray → prediction, confidence, triage, Grad-CAM overlay.

## Commands

```bash
cd pneumonia
pip install -r requirements.txt
python check_setup.py   # verify dependencies and data folder
```

**Train** (requires `data/chest_xray/NORMAL` and `data/chest_xray/PNEUMONIA`):

```bash
python train.py
```

**Evaluate**:

```bash
python evaluate.py
```

**Inference**:

```bash
python predict.py path/to/xray.jpg [--model models/pneumonia_resnet50.keras] [--gradcam-out out.png]
```

**Web UI**:

```bash
streamlit run app.py
```

## Config

Edit `config.py` for paths, batch size, epochs, and triage thresholds (`TRIAGE_HIGH`, `TRIAGE_MED`).
