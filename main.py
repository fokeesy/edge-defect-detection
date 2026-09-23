"""Quick demo: runs a deployed (INT8-quantized) model on a handful of fresh
example parts and saves an annotated grid showing what it predicted vs. the
ground truth - to outputs/demo_predictions.png.

Uses the strongest model already trained and saved by `python -m
src.edge_eval` if present (medium, then tiny, then small - see README "Run-
to-run variance"); otherwise trains one on the spot (takes a minute or two).

    python main.py
"""
from pathlib import Path

import cv2
import numpy as np

from src.data_gen import generate_sample, DEFECT_TYPES
from src.quantize import to_tflite_int8, predict_tflite
from src.train import train_model

OUT_DIR = Path(__file__).resolve().parent / "outputs"
MODELS_DIR = OUT_DIR / "models"
# preference order when several trained presets are on disk: medium and tiny
# both trained perfectly in this project's own evaluation runs, small did
# not (see README "run-to-run variance") - no point demoing the weak one
# when a stronger one is sitting right there
PRESET_PREFERENCE = ("medium", "tiny", "small")


def _get_model():
    for preset in PRESET_PREFERENCE:
        model_path = MODELS_DIR / f"{preset}_int8.tflite"
        threshold_path = MODELS_DIR / f"{preset}_threshold.txt"
        if model_path.exists() and threshold_path.exists():
            print(f"Loading existing quantized model from {model_path}")
            return model_path.read_bytes(), float(threshold_path.read_text())

    print("No trained model found - training one now (this takes a minute or two)...")
    model, metrics, (x_test, _) = train_model("small", seed=0, verbose=0, progress_bar=True)
    print(f"Trained: test AUC={metrics['test_auc']:.3f}, accuracy={metrics['test_accuracy']:.3f}")
    tflite_int8 = to_tflite_int8(model, x_test[:200])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "small_int8.tflite").write_bytes(tflite_int8)
    (MODELS_DIR / "small_threshold.txt").write_text(str(metrics["threshold"]))
    return tflite_int8, metrics["threshold"]


def _annotate(img, predicted_label, prob, true_label):
    canvas = np.full((img.shape[0] + 26, img.shape[1], 3), 20, dtype=np.uint8)
    canvas[26:, :, :] = img
    correct = predicted_label == true_label
    # canvas is RGB at this point (converted to BGR for imwrite only at the
    # end, in main()) - so colors here must be given as RGB, not BGR
    color = (60, 200, 60) if correct else (230, 50, 50)
    text = f"{predicted_label} {prob:.2f}"
    cv2.putText(canvas, text, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def main():
    tflite_int8, threshold = _get_model()
    print(f"Using calibrated decision threshold: {threshold:.3f}")

    rng = np.random.default_rng(42)
    samples = [("OK", None, 0.0)]
    for dt in DEFECT_TYPES:
        samples.append((dt, dt, rng.uniform(0.4, 0.8)))
    samples.append(("OK", None, 0.0))

    tiles = []
    for label, defect_type, severity in samples:
        defective = defect_type is not None
        img, meta = generate_sample(rng, defective=defective, defect_type=defect_type, severity=severity)
        prob = float(predict_tflite(tflite_int8, img[None, ...])[0])
        predicted = "DEFECT" if prob >= threshold else "OK"
        true_label = "DEFECT" if defective else "OK"
        tiles.append(_annotate(img, predicted, prob, true_label))

    grid = np.hstack(tiles)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "demo_predictions.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
