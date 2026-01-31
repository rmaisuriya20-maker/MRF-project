"""Pneumonia X-Ray Pipeline — Stage 1: Dataset Handling & Preprocessing.

- Load chest X-rays from directory structure (Normal / Pneumonia).
- Remove corrupted images.
- Preprocessing: grayscale → resize 224×224 → median blur → CLAHE → normalize → 3-channel.
- Data augmentation: rotation, horizontal flip, zoom.
- Train / validation / test split.
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import cv2
except ImportError:
    cv2 = None  # pip install opencv-python

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
CLASSES = ["NORMAL", "PNEUMONIA"]  # binary: 0 = Normal, 1 = Pneumonia


def load_image_paths_and_labels(data_root: str) -> tuple[list[str], list[int]]:
    """
    Load image paths and binary labels from directory structure:
        data_root/
            NORMAL/   -> label 0
            PNEUMONIA/ -> label 1
    Returns (list of paths, list of labels).
    """
    data_root = Path(data_root)
    paths, labels = [], []
    for class_idx, class_name in enumerate(CLASSES):
        folder = data_root / class_name
        if not folder.exists():
            continue
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for p in folder.glob(ext):
                paths.append(str(p))
                labels.append(class_idx)
    return paths, labels


def is_corrupted(path: str) -> bool:
    """Try to open image; return True if corrupted or invalid."""
    try:
        img = Image.open(path)
        img.verify()
        return False
    except Exception:
        return True


def remove_corrupted(paths: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    """Drop paths that cannot be loaded as valid images."""
    valid_paths, valid_labels = [], []
    for p, l in zip(paths, labels):
        if not is_corrupted(p):
            valid_paths.append(p)
            valid_labels.append(l)
    return valid_paths, valid_labels


# ---------------------------------------------------------------------------
# Preprocessing (single image)
# ---------------------------------------------------------------------------

def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = IMG_SIZE,
    median_blur_ksize: int = 3,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Pipeline: grayscale → resize → median blur → CLAHE → normalize [0,1] → 3-channel.
    Input: BGR (OpenCV) or RGB numpy array; output: float32 (H, W, 3) in [0, 1].
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for preprocessing. Run: pip install opencv-python")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.asarray(image, dtype=np.uint8)

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.medianBlur(resized, median_blur_ksize)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(blurred)

    normalized = (enhanced.astype(np.float32) / 255.0).clip(0.0, 1.0)
    # 3-channel for ResNet50
    three_channel = np.stack([normalized, normalized, normalized], axis=-1)
    return three_channel


def load_and_preprocess(path: str, **preprocess_kw) -> np.ndarray:
    """Load image from path and run preprocessing pipeline."""
    if cv2 is None:
        raise ImportError("opencv-python is required. Run: pip install opencv-python")
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return preprocess_image(img, **preprocess_kw)


# ---------------------------------------------------------------------------
# Data augmentation (for training)
# ---------------------------------------------------------------------------

def augment_image(image: np.ndarray, rotation_range: float = 15, zoom_range: tuple = (0.9, 1.1), flip: bool = True) -> np.ndarray:
    """
    Apply random: rotation (degrees), zoom (scale), horizontal flip.
    image: (H, W, C) float in [0, 1].
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for augmentation. Run: pip install opencv-python")
    h, w = image.shape[:2]
    out = image.copy()

    if rotation_range and np.random.rand() > 0.5:
        angle = np.random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if zoom_range and np.random.rand() > 0.5:
        scale = np.random.uniform(zoom_range[0], zoom_range[1])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        out = cv2.resize(out, (w, h))

    if flip and np.random.rand() > 0.5:
        out = np.fliplr(out).copy()

    return out


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def train_val_test_split(
    paths: list[str],
    labels: list[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Stratified split into train / val / test (stratify only if both classes present).
    Returns (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    paths, labels = np.array(paths), np.array(labels)
    stratify_arg = labels if len(np.unique(labels)) >= 2 else None

    train_p, rest_p, train_l, rest_l = train_test_split(
        paths, labels, train_size=train_ratio, stratify=stratify_arg, random_state=random_state
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    stratify_rest = rest_l if len(np.unique(rest_l)) >= 2 else None
    val_p, test_p, val_l, test_l = train_test_split(
        rest_p, rest_l, train_size=val_frac, stratify=stratify_rest, random_state=random_state
    )
    return (
        train_p.tolist(), train_l.tolist(),
        val_p.tolist(), val_l.tolist(),
        test_p.tolist(), test_l.tolist(),
    )


# ---------------------------------------------------------------------------
# Data generator (for training with augmentation)
# ---------------------------------------------------------------------------

def batch_generator(
    paths: list[str],
    labels: list[int],
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = True,
    preprocess_kw: dict = None,
):
    """Yield (X_batch, y_batch). X: (B, H, W, 3), y: (B,) int."""
    preprocess_kw = preprocess_kw or {}
    indices = np.arange(len(paths))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        X, y = [], []
        for i in idx:
            img = load_and_preprocess(paths[i], **preprocess_kw)
            if augment:
                img = augment_image(img)
            X.append(img)
            y.append(labels[i])
        yield np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
