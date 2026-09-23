"""Trains one defect-classifier preset and evaluates it with real metrics
(ROC-AUC, precision/recall, confusion matrix) on a held-out test set - not
just training accuracy.

`train_model` trains on the synthetic generator; `train_on_arrays` is the
underlying core that works on any (images, labels) arrays - real datasets
loaded via `custom_data.py` use this directly.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
from tqdm import tqdm

from .data_gen import generate_dataset
from .model import build_model

N_TRAIN_PER_CLASS = 1200
N_VAL_PER_CLASS = 300
N_TEST_PER_CLASS = 300
MIN_STEPS_PER_EPOCH = 20


def make_split(n_per_class, seed, severity_range=(0.25, 1.0)):
    rng = np.random.default_rng(seed)
    images, labels, metas = generate_dataset(n_per_class, n_per_class, rng,
                                              severity_range=severity_range)
    return images, labels, metas


def select_threshold(y_true, probs):
    """Picks the probability threshold that maximizes validation accuracy,
    instead of assuming 0.5. A model can have excellent ROC-AUC (it ranks
    defective parts above clean ones correctly) while its raw sigmoid output
    is uncalibrated - e.g. every probability landing in a compressed
    0.2-0.4 band, never crossing 0.5 at all. AUC is threshold-independent so
    it doesn't catch this; accuracy at a fixed 0.5 does, silently, by
    collapsing to "predict one class always". See README pitfalls."""
    candidates = np.unique(probs)
    best_thr, best_acc = 0.5, -1.0
    for thr in candidates:
        acc = ((probs >= thr).astype(int) == y_true).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr


def select_threshold_centered(y_true, probs):
    """Like `select_threshold`, but returns the *middle* of the widest range
    of thresholds that tie for best accuracy, instead of the edge of it.
    With a large validation set the two barely differ; with a tiny one (a
    dozen photos) the edge hugs the few positives it happened to see, and
    real defects that score slightly lower slip under it - the middle is the
    safe choice."""
    y_true = np.asarray(y_true).astype(int)
    p = np.unique(probs)
    if len(p) < 2:
        return float(p[0])
    cuts = (p[:-1] + p[1:]) / 2.0
    accs = np.array([((probs >= c).astype(int) == y_true).mean() for c in cuts])
    good = np.where(accs >= accs.max() - 1e-12)[0]
    runs = np.split(good, np.where(np.diff(good) > 1)[0] + 1)
    run = max(runs, key=lambda r: p[r[-1] + 1] - p[r[0]])
    return float((p[run[0]] + p[run[-1] + 1]) / 2.0)


class _TqdmEpochCallback(tf.keras.callbacks.Callback):
    """One line, live-updating progress bar with the current loss/accuracy/
    AUC in the postfix - instead of either dead silence (verbose=0) or a
    fresh multi-line bar spammed per batch for every epoch (verbose=1)."""

    def __init__(self, epochs, desc):
        super().__init__()
        self.pbar = tqdm(total=epochs, desc=desc, unit="epoch", leave=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.pbar.set_postfix({k: f"{v:.3f}" for k, v in logs.items()})
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()


def augment_image(img, label):
    """Random flips / 90-degree rotations / small shifts / brightness and
    contrast jitter, on a raw 0-255 image. Turns a few dozen photos into
    effectively endless variations, so the model has to learn the defect
    itself rather than memorize the exact training pictures - the single
    biggest thing that makes tiny real datasets trainable."""
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    h, w = img.shape[0], img.shape[1]
    pad = h // 12
    img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, (h, w, 3))
    img = tf.image.random_brightness(img, 25.0)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 255.0), label


def _balanced_class_weights(y):
    """Weights so a rare class (usually defects) counts as much as a common
    one - otherwise 'always say OK' looks 95% accurate on a 95%-OK dataset."""
    y = np.asarray(y).astype(int)
    n, n_pos = len(y), int(y.sum())
    if n_pos == 0 or n_pos == n:
        return None
    return {0: n / (2.0 * (n - n_pos)), 1: n / (2.0 * n_pos)}


def train_on_arrays(preset, x_train, y_train, x_val, y_val, x_test, y_test,
                     seed=0, epochs=30, verbose=0, progress_bar=False, desc=None,
                     augment=False, balance_classes=False, centered_threshold=False):
    tf.keras.utils.set_random_seed(seed)

    model = build_model(preset)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    # Deliberately the simplest setup that works: plain early stopping on
    # val_auc, nothing fancier - see README pitfalls for why "stability"
    # additions (val_loss monitoring, LR scheduling, gradient clipping) were
    # tried and reverted.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        )
    ]
    if progress_bar:
        callbacks.append(_TqdmEpochCallback(epochs, desc or f"training [{preset}]"))

    class_weight = _balanced_class_weights(y_train) if balance_classes else None
    if augment:
        train_data = (tf.data.Dataset.from_tensor_slices((x_train, y_train.astype(np.float32)))
                      .shuffle(len(x_train), seed=seed, reshuffle_each_iteration=True)
                      .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(32).repeat().prefetch(tf.data.AUTOTUNE))
        # tiny datasets would otherwise be just 1-2 gradient steps per
        # epoch; with fresh augmentations every pass it's fine to run more
        steps = max(int(np.ceil(len(x_train) / 32)), MIN_STEPS_PER_EPOCH)
        fit_kwargs = {"x": train_data, "steps_per_epoch": steps}
    else:
        fit_kwargs = {"x": x_train, "y": y_train, "batch_size": 32}

    history = model.fit(
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose,
        **fit_kwargs,
    )

    y_val_prob = model.predict(x_val, verbose=0).ravel()
    pick_threshold = select_threshold_centered if centered_threshold else select_threshold
    threshold = pick_threshold(y_val, y_val_prob)

    y_prob = model.predict(x_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "preset": preset,
        "threshold": threshold,
        "test_auc": float(roc_auc_score(y_test, y_prob)),
        "test_accuracy": float((y_pred == y_test).mean()),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_params": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
    }
    return model, metrics, (x_test, y_test)


def train_model(preset="small", seed=0, epochs=30, verbose=0, progress_bar=False):
    """Trains on the synthetic procedural dataset. For a real dataset, load
    it with `custom_data.load_dataset(...)` and call `train_on_arrays`
    directly instead - see README "Training on your own images"."""
    x_train, y_train, _ = make_split(N_TRAIN_PER_CLASS, seed=seed)
    x_val, y_val, _ = make_split(N_VAL_PER_CLASS, seed=seed + 1)
    x_test, y_test, _ = make_split(N_TEST_PER_CLASS, seed=seed + 2)
    return train_on_arrays(preset, x_train, y_train, x_val, y_val, x_test, y_test,
                            seed=seed, epochs=epochs, verbose=verbose, progress_bar=progress_bar)


if __name__ == "__main__":
    for preset in ("tiny", "small", "medium"):
        print(f"\n=== training {preset} ===")
        _, metrics, _ = train_model(preset, progress_bar=True)
        print(metrics)
