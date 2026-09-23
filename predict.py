"""Classifies a real image (or a folder of them) as OK or DEFECT using a
trained model - the synthetic-trained default, or one you trained yourself
with train_on_folder.py. Add --heatmap to see *why* (Grad-CAM), not just the
verdict.

    python predict.py --image path/to/photo.jpg
    python predict.py --folder path/to/photos/ --save-annotated results.png
    python predict.py --image photo.jpg --heatmap --save-annotated result.png
    python predict.py --image photo.jpg --model outputs/custom_models/mydata_int8.tflite
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.custom_data import load_image
from src.quantize import predict_tflite
from src.gradcam import make_gradcam_heatmap, overlay_heatmap, load_matching_keras_model

DEFAULT_MODEL = Path(__file__).resolve().parent / "outputs" / "models" / "medium_int8.tflite"


def _load_threshold(model_path):
    for suffix in ("_int8.tflite", "_float32.tflite"):
        if str(model_path).endswith(suffix):
            thr_path = Path(str(model_path)[: -len(suffix)] + "_threshold.txt")
            if thr_path.exists():
                return float(thr_path.read_text())
    print("(no matching *_threshold.txt found next to the model - using 0.5, "
          "which may be miscalibrated; see README pitfalls)")
    return 0.5


def classify(model_bytes, threshold, image_path):
    img = load_image(image_path)
    prob = float(predict_tflite(model_bytes, img[None, ...])[0])
    label = "DEFECT" if prob >= threshold else "OK"
    return label, prob, img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="path to a single image")
    group.add_argument("--folder", help="path to a folder of images")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a *_int8.tflite or *_float32.tflite model")
    parser.add_argument("--save-annotated", default=None,
                         help="path to save an annotated image (single tile for --image, a grid for --folder)")
    parser.add_argument("--heatmap", action="store_true",
                         help="show/save a Grad-CAM heatmap of what drove the prediction, not just the verdict")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}\n"
                          f"Run `python -m src.edge_eval` first, or point --model at one you trained "
                          f"with train_on_folder.py.")
    model_bytes = model_path.read_bytes()
    threshold = _load_threshold(model_path)
    print(f"Model: {model_path}  |  calibrated threshold: {threshold:.3f}\n")

    keras_model = None
    if args.heatmap:
        keras_model = load_matching_keras_model(model_path)
        if keras_model is None:
            print("(no matching *_keras.h5 found next to this model - can't compute a heatmap for it)")

    def _tile(img, label, prob):
        display = img
        if keras_model is not None:
            display = overlay_heatmap(img, make_gradcam_heatmap(keras_model, img))
        canvas = np.full((img.shape[0] + 22, img.shape[1], 3), 20, dtype=np.uint8)
        canvas[22:, :, :] = display
        color = (230, 50, 50) if label == "DEFECT" else (60, 200, 60)
        cv2.putText(canvas, f"{label} {prob:.2f}", (2, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return canvas

    if args.image:
        label, prob, img = classify(model_bytes, threshold, args.image)
        print(f"{args.image}: {label}  (confidence {prob:.3f})")
        if args.save_annotated:
            cv2.imwrite(args.save_annotated, cv2.cvtColor(_tile(img, label, prob), cv2.COLOR_RGB2BGR))
            print(f"Saved: {args.save_annotated}")
        return

    folder = Path(args.folder)
    paths = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise SystemExit(f"no images found in {folder}")

    tiles = []
    for p in paths:
        label, prob, img = classify(model_bytes, threshold, p)
        print(f"{p.name}: {label}  (confidence {prob:.3f})")
        if args.save_annotated:
            tiles.append(_tile(img, label, prob))

    if args.save_annotated and tiles:
        grid = np.hstack(tiles)
        cv2.imwrite(args.save_annotated, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved annotated grid: {args.save_annotated}")


if __name__ == "__main__":
    main()
