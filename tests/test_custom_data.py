import cv2
import numpy as np
import pytest

from src.data_gen import generate_sample
from src.custom_data import load_labeled_folders, split_dataset, load_dataset


@pytest.fixture
def tiny_dataset(tmp_path):
    """A small real folder-of-images dataset, built from the synthetic
    generator but saved and reloaded as actual image files - exercising the
    real disk-loading path, not just in-memory arrays."""
    rng = np.random.default_rng(0)
    ok_dir = tmp_path / "ok"
    defect_dir = tmp_path / "defect"
    ok_dir.mkdir()
    defect_dir.mkdir()

    for i in range(5):
        img, _ = generate_sample(rng, defective=False)
        cv2.imwrite(str(ok_dir / f"ok_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for i in range(4):
        img, _ = generate_sample(rng, defective=True, severity=0.8)
        cv2.imwrite(str(defect_dir / f"defect_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return tmp_path


def test_load_labeled_folders_reads_all_images_with_correct_labels(tiny_dataset):
    images, labels, paths = load_labeled_folders(tiny_dataset)
    assert images.shape == (9, 96, 96, 3)
    assert images.dtype == np.uint8
    assert (labels == 0).sum() == 5
    assert (labels == 1).sum() == 4
    assert len(paths) == 9


def test_load_labeled_folders_raises_a_clear_error_for_missing_subfolder(tmp_path):
    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    rng = np.random.default_rng(0)
    img, _ = generate_sample(rng, defective=False)
    cv2.imwrite(str(ok_dir / "ok_0.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # no "defect" subfolder created - should fail clearly, not silently
    with pytest.raises(FileNotFoundError, match="defect"):
        load_labeled_folders(tmp_path)


def test_split_dataset_covers_every_example_exactly_once(tiny_dataset):
    images, labels, paths = load_labeled_folders(tiny_dataset)
    (x_tr, y_tr, p_tr), (x_va, y_va, p_va), (x_te, y_te, p_te) = split_dataset(
        images, labels, paths, val_frac=0.2, test_frac=0.2, seed=0
    )
    total = len(x_tr) + len(x_va) + len(x_te)
    assert total == len(images)
    assert set(p_tr) | set(p_va) | set(p_te) == set(paths)
    assert not (set(p_tr) & set(p_va))
    assert not (set(p_tr) & set(p_te))


def test_load_dataset_end_to_end(tiny_dataset):
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_dataset(tiny_dataset, seed=0)
    assert len(x_tr) + len(x_va) + len(x_te) == 9
    assert x_tr.shape[1:] == (96, 96, 3)


def test_too_few_photos_gives_a_clear_error_not_a_crash(tmp_path):
    """Regression test: 3 photos per class used to die deep inside numpy with
    a cryptic 'arrays used as indices must be of integer type' IndexError."""
    rng = np.random.default_rng(0)
    for sub, defective in (("ok", False), ("defect", True)):
        (tmp_path / sub).mkdir()
        for i in range(3):
            img, _ = generate_sample(rng, defective=defective, severity=0.8)
            cv2.imwrite(str(tmp_path / sub / f"{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with pytest.raises(ValueError, match="at least 4"):
        load_dataset(tmp_path)
