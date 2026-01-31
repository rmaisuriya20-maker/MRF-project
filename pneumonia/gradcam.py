"""
Pneumonia X-Ray Pipeline — Stage 4: Explainability (Grad-CAM)

- Generate heatmap from last convolution layer of ResNet50.
- Overlay heatmap on original X-ray for doctor verification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    import cv2
except ImportError:
    cv2 = None


def get_gradcam_model(model: keras.Model, last_conv_layer_name: str = None) -> keras.Model:
    """
    Build a model that outputs (last_conv_output, final_output).
    ResNet50 last conv is typically 'conv5_block3_out' or similar.
    """
    last_conv = None
    if last_conv_layer_name is not None:
        try:
            last_conv = model.get_layer(last_conv_layer_name)
        except Exception:
            pass
    if last_conv is None:
        for layer in reversed(model.layers):
            if "conv" in layer.name and "out" in layer.name:
                last_conv = layer
                break
        if last_conv is None:
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    last_conv = layer
                    break
        if last_conv is None:
            last_conv = model.layers[-2]
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output],
    )
    return grad_model


def compute_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str = None,
    pred_index: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Compute Grad-CAM heatmap for the positive class (Pneumonia).
    img_array: (1, H, W, 3) float in [0, 1].
    Returns (heatmap 2D, prediction probability).
    """
    grad_model = get_gradcam_model(model, last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_val = predictions[0, pred_index]
        tape.watch(conv_outputs)

    grads = tape.gradient(pred_val, conv_outputs)
    if grads is None:
        return np.zeros(img_array.shape[1:3]), float(predictions.numpy()[0, pred_index])

    # Global average of gradients (weights)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam).numpy()[0]

    # Normalize to [0, 1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h, w = img_array.shape[1], img_array.shape[2]
    if cv2 is not None:
        cam = cv2.resize(cam, (w, h))
    else:
        from PIL import Image
        cam_uint8 = (cam * 255).astype(np.uint8)
        cam_img = Image.fromarray(cam_uint8).resize((w, h), Image.BILINEAR)
        cam = np.array(cam_img, dtype=np.float32) / 255.0
    prob = float(predictions.numpy()[0, pred_index])
    return cam, prob


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = None,
) -> np.ndarray:
    """
    Overlay heatmap on image. image: (H,W,3) [0,1] or [0,255]; heatmap: (H,W) [0,1].
    Returns RGB overlay (0-255) with infected area in red.
    """
    if cv2 is not None and colormap is None:
        colormap = cv2.COLORMAP_JET
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.stack([image.squeeze()] * 3, axis=-1)
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    if cv2 is not None and colormap is not None:
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: red channel = heatmap, green/blue = 0
        heatmap_colored = np.zeros((*heatmap_uint8.shape, 3), dtype=np.uint8)
        heatmap_colored[..., 0] = heatmap_uint8
    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    return overlay


def gradcam_on_path(
    model_path: str,
    image_path: str,
    output_path: str = None,
    last_conv_layer_name: str = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Load image, preprocess, run Grad-CAM, overlay and optionally save.
    Returns (overlay RGB image, probability, raw heatmap).
    """
    from preprocess import load_and_preprocess

    model = keras.models.load_model(model_path)
    img = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img, axis=0)

    heatmap, prob = compute_heatmap(model, img_batch, last_conv_layer_name)
    overlay = overlay_heatmap_on_image(img, heatmap, alpha=0.5)

    if output_path:
        from pathlib import Path
        from PIL import Image as PILImage
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        else:
            PILImage.fromarray(overlay).save(output_path)
    return overlay, prob, heatmap
