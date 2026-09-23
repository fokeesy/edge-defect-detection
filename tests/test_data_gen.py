import numpy as np
import pytest

from src.data_gen import generate_sample, generate_dataset, DEFECT_TYPES, IMG_SIZE


def test_clean_sample_shape_and_label():
    rng = np.random.default_rng(0)
    img, meta = generate_sample(rng, defective=False)
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert img.dtype == np.uint8
    assert meta["defective"] is False
    assert meta["defect_type"] is None


@pytest.mark.parametrize("defect_type", DEFECT_TYPES)
def test_defective_sample_differs_from_a_clean_baseline(defect_type):
    """The whole point of a defect is that it's visible - regression guard
    against a defect-drawing bug that silently draws nothing."""
    rng = np.random.default_rng(1)
    clean, _ = generate_sample(np.random.default_rng(1), defective=False)
    defective, meta = generate_sample(rng, defective=True, defect_type=defect_type, severity=0.9)
    assert meta["defective"] is True
    assert meta["defect_type"] == defect_type
    diff = np.abs(defective.astype(int) - clean.astype(int)).sum()
    assert diff > 500, f"{defect_type} at high severity should be clearly visible"


@pytest.mark.parametrize("defect_type", DEFECT_TYPES)
def test_higher_severity_is_more_visible(defect_type):
    """Severity should be a real dial, not a cosmetic no-op - this is what
    the whole evaluation sweep depends on."""
    rng_low = np.random.default_rng(5)
    rng_high = np.random.default_rng(5)
    clean, _ = generate_sample(np.random.default_rng(5), defective=False)

    low, _ = generate_sample(rng_low, defective=True, defect_type=defect_type, severity=0.1)
    high, _ = generate_sample(rng_high, defective=True, defect_type=defect_type, severity=1.0)

    diff_low = np.abs(low.astype(int) - clean.astype(int)).sum()
    diff_high = np.abs(high.astype(int) - clean.astype(int)).sum()
    assert diff_high > diff_low


def test_generate_dataset_is_balanced_and_shuffled():
    rng = np.random.default_rng(2)
    images, labels, metas = generate_dataset(20, 20, rng)
    assert images.shape == (40, IMG_SIZE, IMG_SIZE, 3)
    assert labels.sum() == 20
    assert len(metas) == 40
    # shuffled: labels shouldn't be perfectly grouped (0*20 then 1*20)
    assert not np.array_equal(labels, sorted(labels))
