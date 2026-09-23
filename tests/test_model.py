import numpy as np
import pytest

from src.model import build_model, PRESETS, INPUT_SHAPE


@pytest.mark.parametrize("preset", list(PRESETS.keys()))
def test_model_builds_and_predicts_right_shape(preset):
    model = build_model(preset)
    batch = np.random.randint(0, 255, size=(4, *INPUT_SHAPE), dtype=np.uint8)
    out = model.predict(batch, verbose=0)
    assert out.shape == (4, 1)
    assert np.all((out >= 0) & (out <= 1)), "sigmoid output must be a valid probability"


def test_preset_sizes_are_actually_ordered():
    """The whole point of having presets is a real size/capacity gradient -
    a regression here would quietly break the size-vs-accuracy story."""
    tiny = build_model("tiny").count_params()
    small = build_model("small").count_params()
    medium = build_model("medium").count_params()
    assert tiny < small < medium


def test_transfer_model_builds_frozen_backbone_with_trainable_head():
    """pretrained=False so the test needs no network - the wiring is what's
    under test here, not the ImageNet weights."""
    model = build_model("mobilenet", pretrained=False)
    batch = np.random.randint(0, 255, size=(3, *INPUT_SHAPE), dtype=np.uint8)
    out = model.predict(batch, verbose=0)
    assert out.shape == (3, 1)
    assert np.all((out >= 0) & (out <= 1))
    # backbone frozen -> only the new Dense head (kernel + bias) trains
    assert len(model.trainable_weights) == 2


def test_unknown_preset_error_lists_the_valid_ones():
    with pytest.raises(ValueError, match="mobilenet"):
        build_model("nope")
