"""
Shared config for pneumonia pipeline: paths, image size, training defaults.
"""

from pathlib import Path

# Base dir: folder containing this config (pneumonia/)
_BASE = Path(__file__).resolve().parent

# Data: pneumonia/data/chest_xray/NORMAL and PNEUMONIA
DATA_ROOT = _BASE / "data" / "chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training
EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

# Model: pneumonia/models/pneumonia_resnet50.keras
MODEL_SAVE_PATH = str(_BASE / "models" / "pneumonia_resnet50.keras")
NUM_CLASSES = 1  # binary: single sigmoid output

# Triage (confidence thresholds)
TRIAGE_HIGH = 0.8   # >= High Risk
TRIAGE_MED  = 0.4   # 0.4--0.8 Medium, <0.4 Low
