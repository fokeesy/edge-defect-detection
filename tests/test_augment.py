import numpy as np
import tensorflow as tf

from src.train import augment_image, _balanced_class_weights


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(96, 96, 3)).astype(np.uint8)


def test_augment_keeps_shape_range_and_label():
    out, label = augment_image(tf.constant(_img()), tf.constant(1.0))
    out = out.numpy()
    assert out.shape == (96, 96, 3)
    assert out.min() >= 0.0 and out.max() <= 255.0
    assert float(label) == 1.0


def test_augment_is_actually_random():
    img = tf.constant(_img())
    outs = [augment_image(img, tf.constant(0.0))[0].numpy() for _ in range(6)]
    assert any(not np.array_equal(outs[0], o) for o in outs[1:]), "augmentation returned identical images every time"


def test_balanced_class_weights_upweight_the_rare_class():
    y = np.array([0] * 90 + [1] * 10)
    w = _balanced_class_weights(y)
    assert w[1] > w[0]
    assert abs(w[1] - 5.0) < 1e-9
    # each class contributes the same total weight
    assert abs(w[0] * 90 - w[1] * 10) < 1e-9


def test_balanced_class_weights_are_even_for_a_balanced_set():
    w = _balanced_class_weights(np.array([0, 0, 1, 1]))
    assert w == {0: 1.0, 1: 1.0}


def test_balanced_class_weights_skip_single_class_data():
    assert _balanced_class_weights(np.zeros(10)) is None
