import numpy as np
import pytest

from src.model import build_model
from src.data_gen import generate_dataset
from src.gradcam import make_gradcam_heatmap, overlay_heatmap, find_last_conv_layer, load_matching_keras_model


@pytest.fixture(scope="module")
def quick_model():
    """A minimally-trained model - Grad-CAM needs BatchNorm to have real
    running statistics (see the quantize tests for the same reasoning),
    not accuracy."""
    rng = np.random.default_rng(0)
    x, y, _ = generate_dataset(30, 30, rng)
    model = build_model("tiny")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=2, batch_size=16, verbose=0)
    return model, x[0]


def test_find_last_conv_layer_returns_a_real_conv_layer():
    model = build_model("small")
    name = find_last_conv_layer(model)
    layer = model.get_layer(name)
    assert "conv2d" in name.lower()
    assert layer.__class__.__name__ == "Conv2D"


def test_heatmap_shape_and_range(quick_model):
    model, img = quick_model
    heatmap = make_gradcam_heatmap(model, img)
    assert heatmap.ndim == 2
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_overlay_matches_original_image_size(quick_model):
    model, img = quick_model
    heatmap = make_gradcam_heatmap(model, img)
    overlay = overlay_heatmap(img, heatmap)
    assert overlay.shape == img.shape
    assert overlay.dtype == np.uint8


def test_load_matching_keras_model_returns_none_when_absent(tmp_path):
    fake_tflite = tmp_path / "nope_int8.tflite"
    fake_tflite.write_bytes(b"not a real model")
    assert load_matching_keras_model(fake_tflite) is None


def test_load_matching_keras_model_finds_a_real_one(tmp_path, quick_model):
    model, _ = quick_model
    model.save(str(tmp_path / "mymodel_keras.h5"))
    fake_tflite = tmp_path / "mymodel_int8.tflite"
    fake_tflite.write_bytes(b"placeholder")
    loaded = load_matching_keras_model(fake_tflite)
    assert loaded is not None
    assert loaded.count_params() == model.count_params()


def test_gradcam_works_on_the_transfer_model_with_usable_resolution():
    """MobileNetV2's very last conv is only 3x3 on a 96px image; Grad-CAM
    should step back to a layer with real spatial detail (and still work
    through the flat-wired backbone)."""
    model = build_model("mobilenet", pretrained=False)
    name = find_last_conv_layer(model)
    assert model.get_layer(name).output.shape[1] >= 6
    img = np.random.default_rng(1).integers(0, 256, size=(96, 96, 3)).astype(np.uint8)
    heatmap = make_gradcam_heatmap(model, img)
    assert heatmap.ndim == 2 and heatmap.shape[0] >= 6
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6
