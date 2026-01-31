"""
Pneumonia X-Ray Pipeline — Stage 3: Evaluation

- Confusion matrix and classification report.
- Emphasize recall (minimize false negatives in healthcare).
"""

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from preprocess import load_image_paths_and_labels, remove_corrupted, train_val_test_split, load_and_preprocess
from config import DATA_ROOT, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, MODEL_SAVE_PATH


def evaluate_model(
    model_path: str = None,
    data_root: str = None,
) -> dict:
    """
    Load model and test set; compute predictions, confusion matrix, classification report.
    Returns dict with metrics and report text.
    """
    model_path = model_path or MODEL_SAVE_PATH
    data_root = data_root or str(DATA_ROOT)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with: python train.py")
    model = keras.models.load_model(model_path)
    paths, labels = load_image_paths_and_labels(data_root)
    paths, labels = remove_corrupted(paths, labels)
    (_, _, _, _, test_p, test_l) = train_val_test_split(paths, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    X_test = np.array([load_and_preprocess(p) for p in test_p], dtype=np.float32)
    y_true = np.array(test_l, dtype=np.int32)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(np.int32)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        digits=4,
    )

    print("Confusion matrix (True \\ Pred):")
    print(cm)
    print("\nClassification report (Recall emphasized for healthcare):")
    print(report)
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


if __name__ == "__main__":
    evaluate_model()
