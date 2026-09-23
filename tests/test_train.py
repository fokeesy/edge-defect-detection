import numpy as np
import pytest

import src.train as train_mod
from src.model import build_model


def test_training_does_not_collapse_to_a_single_class(monkeypatch):
    """Regression test: early stopping on a noisy val_auc signal used to
    restore weights before the classifier's probability output had actually
    calibrated around the decision threshold, producing a degenerate
    'always predict one class' model despite good underlying AUC (up to
    0.999 in the case that motivated this test). See README pitfalls."""
    monkeypatch.setattr(train_mod, "N_TRAIN_PER_CLASS", 150)
    monkeypatch.setattr(train_mod, "N_VAL_PER_CLASS", 40)
    monkeypatch.setattr(train_mod, "N_TEST_PER_CLASS", 100)

    _, metrics, _ = train_mod.train_model("medium", seed=0, epochs=40, verbose=0)
    cm = np.array(metrics["confusion_matrix"])
    predicted_positive = cm[:, 1].sum()
    predicted_negative = cm[:, 0].sum()
    assert predicted_positive > 0, "model predicted DEFECT for zero test images - degenerate collapse"
    assert predicted_negative > 0, "model predicted OK for zero test images - degenerate collapse"
    assert metrics["test_accuracy"] > 0.7


def test_augmented_balanced_training_runs_and_returns_valid_metrics():
    """Smoke test for the small-dataset path (augmentation + class weights +
    centered threshold) on the from-scratch model - no network needed."""
    x_tr, y_tr, _ = train_mod.make_split(20, seed=1)
    x_va, y_va, _ = train_mod.make_split(8, seed=2)
    x_te, y_te, _ = train_mod.make_split(10, seed=3)
    _, metrics, _ = train_mod.train_on_arrays(
        "tiny", x_tr, y_tr, x_va, y_va, x_te, y_te, epochs=2,
        augment=True, balance_classes=True, centered_threshold=True,
    )
    assert 0.0 <= metrics["threshold"] <= 1.0
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["epochs_trained"] >= 1


def test_small_photo_set_recipe_learns_from_only_40_images_per_class():
    """The headline claim for training on your own photos: a few dozen
    images per class is enough, using the pretrained backbone + augmentation
    + centered threshold. The old from-scratch/no-augmentation recipe scored
    ~63% accuracy on exactly this kind of set. Needs the ImageNet weights
    (downloaded once, cached) - skipped, not failed, if offline."""
    try:
        build_model("mobilenet")
    except Exception as e:  # download failure surfaces as various error types
        pytest.skip(f"pretrained MobileNetV2 weights unavailable: {e}")

    x_tr, y_tr, _ = train_mod.make_split(40, seed=10)
    x_va, y_va, _ = train_mod.make_split(6, seed=11)
    x_te, y_te, _ = train_mod.make_split(60, seed=12)
    _, metrics, _ = train_mod.train_on_arrays(
        "mobilenet", x_tr, y_tr, x_va, y_va, x_te, y_te, epochs=30,
        augment=True, balance_classes=True, centered_threshold=True,
    )
    assert metrics["test_auc"] > 0.9
    assert metrics["test_accuracy"] > 0.8
