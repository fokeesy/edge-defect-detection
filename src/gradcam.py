"""Grad-CAM (Gradient-weighted Class Activation Mapping): shows *which
pixels* drove a DEFECT prediction, not just the prediction itself - the
difference between "trust me" and "here's what I actually looked at".

This needs gradients through the real network, so it runs on the full Keras
model, not the quantized INT8 TFLite model used for deployed inference.
That's not a shortcut - it's how this is meant to work in practice:
explainability is a development/validation tool you run on your own
machine to sanity-check the model, not something you'd run on the
resource-constrained edge device itself.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2


def load_matching_keras_model(tflite_path):
    """Grad-CAM needs gradients, which the quantized TFLite model can't
    give - loads the full Keras model saved alongside it (see edge_eval.py
    / train_on_folder.py, which save a `*_keras.h5` next to each
    `*_int8.tflite` / `*_float32.tflite`). Returns None if there isn't a
    matching one, e.g. for a model trained before this feature existed."""
    tflite_path = str(tflite_path)
    for suffix in ("_int8.tflite", "_float32.tflite"):
        if tflite_path.endswith(suffix):
            keras_path = Path(tflite_path[: -len(suffix)] + "_keras.h5")
            if keras_path.exists():
                return tf.keras.models.load_model(str(keras_path))
    return None


def find_last_conv_layer(model, min_spatial=6):
    """The last Conv2D whose feature map is at least `min_spatial` x
    `min_spatial`. The very last conv of a deep network like MobileNetV2 is
    only 3x3 on a 96px image - a heatmap that coarse is a few giant blobs -
    so this steps back to the last layer that still has spatial detail."""
    fallback = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            fallback = fallback or layer.name
            if layer.output.shape[1] >= min_spatial:
                return layer.name
    if fallback is None:
        raise ValueError("model has no Conv2D layer - can't compute Grad-CAM")
    return fallback


def make_gradcam_heatmap(model, image, last_conv_layer_name=None):
    """`image`: a single HxWx3 uint8 array (same raw format the model's own
    Rescaling layer expects). Returns a HxW float32 heatmap in [0, 1]."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_array = image[None, ...].astype(np.float32)
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        # single sigmoid output = P(defect); its gradient w.r.t. the last
        # conv feature map tells us which spatial regions pushed that
        # probability up
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # per-channel importance

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)  # ReLU: only positive (defect-supporting) evidence
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy().astype(np.float32)


def overlay_heatmap(image, heatmap, alpha=0.45):
    """`image`: HxWx3 uint8 RGB. Returns an HxWx3 uint8 RGB image with the
    heatmap blended on top."""
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    blended = image.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
