import numpy as np

from src.train import select_threshold, select_threshold_centered


def test_threshold_selection_on_perfectly_separable_probs():
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    thr = select_threshold(y, probs)
    preds = (probs >= thr).astype(int)
    assert np.array_equal(preds, y)


def test_threshold_selection_recovers_signal_from_a_compressed_distribution():
    """Regression test: the exact failure mode that motivated this function
    - a model whose sigmoid output never crosses 0.5 at all, even though the
    two classes are cleanly separated within a compressed band. A fixed 0.5
    threshold would predict a single class for every example here."""
    y = np.array([0] * 5 + [1] * 5)
    probs = np.array([0.21, 0.22, 0.23, 0.24, 0.25, 0.30, 0.31, 0.32, 0.33, 0.34])
    thr = select_threshold(y, probs)
    assert 0.5 not in (thr,)  # sanity: this is exactly the case 0.5 fails on
    preds = (probs >= thr).astype(int)
    assert (preds == y).mean() == 1.0


def test_threshold_selection_handles_no_separation_gracefully():
    """When there's genuinely no signal, it should still return *some*
    valid threshold rather than crashing."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    probs = np.full(200, 0.5)  # a degenerate constant model
    thr = select_threshold(y, probs)
    assert isinstance(thr, float)


def test_centered_threshold_sits_in_the_middle_of_the_gap_not_at_its_edge():
    """`select_threshold` returns the lowest tied-best threshold - i.e. hugs
    the lowest positive it saw. On a tiny validation set that's fragile;
    the centered version should land between the two classes instead."""
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    edge = select_threshold(y, probs)
    centered = select_threshold_centered(y, probs)
    assert edge == 0.8
    assert abs(centered - 0.5) < 1e-6
    assert np.array_equal((probs >= centered).astype(int), y)


def test_centered_threshold_still_optimal_on_overlapping_classes():
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    probs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9])
    thr = select_threshold_centered(y, probs)
    best = select_threshold(y, probs)
    acc = lambda t: ((probs >= t).astype(int) == y).mean()
    assert acc(thr) == acc(best)


def test_centered_threshold_handles_degenerate_inputs():
    assert isinstance(select_threshold_centered(np.array([0, 1, 0]), np.full(3, 0.5)), float)
    y = np.array([0, 1])
    assert isinstance(select_threshold_centered(y, np.array([0.2, 0.8])), float)
