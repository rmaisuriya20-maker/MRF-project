"""
Run this to verify the pneumonia pipeline setup.
Reports missing packages and data so you know what to install/create.
"""

import sys
from pathlib import Path

def check():
    errors = []
    # 1. Dependencies
    try:
        import cv2
    except ImportError:
        errors.append("opencv-python not installed -> pip install opencv-python")
    try:
        import numpy
    except ImportError:
        errors.append("numpy not installed -> pip install numpy")
    try:
        from PIL import Image
    except ImportError:
        errors.append("Pillow not installed -> pip install Pillow")
    try:
        import sklearn
    except ImportError:
        errors.append("scikit-learn not installed -> pip install scikit-learn")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        errors.append("tensorflow not installed -> pip install tensorflow")

    # 2. Data dir (required for train)
    try:
        from config import DATA_ROOT
    except Exception:
        errors.append("config failed (check config.py)")
        DATA_ROOT = None
    data_path = Path(DATA_ROOT) if DATA_ROOT is not None else Path("data/chest_xray")
    if not data_path.exists():
        errors.append(f"Data folder not found: {data_path}. Create 'data/chest_xray/NORMAL/' and 'data/chest_xray/PNEUMONIA/' and add X-ray images.")
    else:
        try:
            from preprocess import load_image_paths_and_labels
            paths, labels = load_image_paths_and_labels(str(data_path))
            if not paths:
                errors.append(f"No images in {data_path}. Add JPEG/PNG under NORMAL/ and PNEUMONIA/.")
        except Exception as e:
            errors.append(f"Loading data failed: {e}")

    if errors:
        print("Setup issues found:")
        for e in errors:
            print("  -", e)
        print("\nFix with: pip install -r requirements.txt")
        return False
    print("Setup OK. You can run: python train.py")
    return True

if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
