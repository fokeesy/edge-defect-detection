"""Loads a real, labeled image dataset from disk, in place of the synthetic
generator - so the same model/train/quantize pipeline can be pointed at your
own photos instead of procedurally-rendered parts.

Expected folder layout:

    my_dataset/
      ok/
        part001.jpg
        part002.png
        ...
      defect/
        part050.jpg
        ...

Any common image format cv2 can read works (.jpg, .png, .bmp, ...). Images
are resized (not cropped) to the model's input size and converted to RGB.
"""
from pathlib import Path

import cv2
import numpy as np

from .model import INPUT_SHAPE

IMG_SIZE = INPUT_SHAPE[0]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_PHOTOS_PER_CLASS = 4  # one each for validation and test, two left to train on


def load_image(path, size=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def load_labeled_folders(root, ok_dirname="ok", defect_dirname="defect", size=IMG_SIZE):
    """Returns (images [N,H,W,3] uint8, labels [N] int, paths list[str]),
    label 0 = ok_dirname, label 1 = defect_dirname."""
    root = Path(root)
    images, labels, paths = [], [], []
    for label, dirname in ((0, ok_dirname), (1, defect_dirname)):
        folder = root / dirname
        if not folder.is_dir():
            raise FileNotFoundError(
                f"expected a '{dirname}' subfolder under {root}, found none. "
                f"Layout should be {root}/{ok_dirname}/*.jpg and {root}/{defect_dirname}/*.jpg"
            )
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
        if not files:
            raise ValueError(f"no images found in {folder} - paste your "
                             f"{'good' if label == 0 else 'defective'}-part photos there "
                             f"({', '.join(sorted(VALID_EXTENSIONS))})")
        for p in files:
            images.append(load_image(p, size))
            labels.append(label)
            paths.append(str(p))

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64), paths


def split_dataset(images, labels, paths, val_frac=0.15, test_frac=0.15, seed=0):
    """A simple random split, stratified by label so small defect classes
    aren't accidentally left out of one split entirely."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n < MIN_PHOTOS_PER_CLASS:
            raise ValueError(
                f"only {n} {'defective' if label == 1 else 'good'}-part photo(s) found - need at least "
                f"{MIN_PHOTOS_PER_CLASS} per class just to split into train/validation/test, "
                f"and realistically 30 or more to learn anything useful"
            )
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        val_idx.extend(idx[:n_val])
        test_idx.extend(idx[n_val:n_val + n_test])
        train_idx.extend(idx[n_val + n_test:])

    def _take(idx):
        idx = np.array(sorted(idx))
        return images[idx], labels[idx], [paths[i] for i in idx]

    return _take(train_idx), _take(val_idx), _take(test_idx)


def load_dataset(root, val_frac=0.15, test_frac=0.15, seed=0, size=IMG_SIZE,
                  ok_dirname="ok", defect_dirname="defect"):
    """One-call convenience: load + split. Returns
    (x_train,y_train), (x_val,y_val), (x_test,y_test)."""
    images, labels, paths = load_labeled_folders(root, ok_dirname, defect_dirname, size)
    (x_tr, y_tr, _), (x_va, y_va, _), (x_te, y_te, _) = split_dataset(
        images, labels, paths, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    print(f"Loaded {len(images)} images from {root}: "
          f"{len(x_tr)} train / {len(x_va)} val / {len(x_te)} test "
          f"({int(labels.sum())} defect, {int((labels == 0).sum())} ok)")
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)
