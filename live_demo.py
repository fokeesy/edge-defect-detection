"""Live visual demo: opens a window and classifies parts one at a time,
in real time, like watching an inspection camera on a production line -
with a live Grad-CAM heatmap showing *why*, not just the verdict.

By default it generates fresh synthetic parts forever, using the `medium`
model (the strongest of the three trained presets); point it at a real
folder of images and/or a different model with --folder / --model.

    python live_demo.py
    python live_demo.py --folder my_photos/
    python live_demo.py --model outputs/custom_models/my_dataset_int8.tflite
    python live_demo.py --no-heatmap   # faster, verdict only

Press 'q' or close the window to stop.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.custom_data import load_image
from src.data_gen import generate_sample, DEFECT_TYPES
from src.quantize import predict_tflite
from src.gradcam import make_gradcam_heatmap, overlay_heatmap, load_matching_keras_model

DEFAULT_MODEL = Path(__file__).resolve().parent / "outputs" / "models" / "medium_int8.tflite"
DISPLAY_SIZE = 480
WINDOW_NAME = "Edge AI Defect Detection - Live"


def _load_threshold(model_path):
    for suffix in ("_int8.tflite", "_float32.tflite"):
        if str(model_path).endswith(suffix):
            thr_path = Path(str(model_path)[: -len(suffix)] + "_threshold.txt")
            if thr_path.exists():
                return float(thr_path.read_text())
    return 0.5




def _synthetic_stream(rng):
    """Infinite generator of (image, source_label) - cycles through clean
    parts and every defect type/severity for visual variety."""
    i = 0
    while True:
        if i % 3 == 0:
            img, meta = generate_sample(rng, defective=False)
        else:
            dt = rng.choice(DEFECT_TYPES)
            severity = rng.uniform(0.3, 1.0)
            img, meta = generate_sample(rng, defective=True, defect_type=dt, severity=severity)
        yield img, ("DEFECT" if meta["defective"] else "OK")
        i += 1


def _folder_stream(folder):
    paths = sorted(p for p in Path(folder).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise SystemExit(f"no images found in {folder}")
    for p in paths:
        yield load_image(p), None  # ground truth unknown for real photos


def _draw_frame(display_img, phase, label=None, prob=None, true_label=None,
                 idx=0, total=None, has_heatmap=False):
    big = cv2.resize(display_img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((DISPLAY_SIZE + 70, DISPLAY_SIZE, 3), (18, 18, 20), dtype=np.uint8)
    canvas[60:60 + DISPLAY_SIZE, :, :] = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)

    header = f"part {idx + 1}" + (f"/{total}" if total else "")
    cv2.putText(canvas, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    if phase == "scanning":
        cv2.putText(canvas, "inspecting...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    else:
        color = (60, 60, 230) if label == "DEFECT" else (60, 200, 60)
        text = f"{label}  ({prob:.2f} confidence)"
        cv2.putText(canvas, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if has_heatmap:
            cv2.putText(canvas, "red = what the AI is looking at", (10, DISPLAY_SIZE + 90 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        if true_label is not None:
            match = "correct" if true_label == label else "WRONG"
            mcolor = (60, 200, 60) if match == "correct" else (60, 60, 230)
            cv2.putText(canvas, f"(ground truth: {true_label}, {match})",
                        (10, DISPLAY_SIZE + 90 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mcolor, 1, cv2.LINE_AA)
    return canvas


def _scan_animation(img, idx, total, steps=10, step_delay=0.03):
    for s in range(steps):
        frame = _draw_frame(img, "scanning", idx=idx, total=total)
        y = 60 + int((s + 1) / steps * DISPLAY_SIZE)
        cv2.line(frame, (0, y), (DISPLAY_SIZE, y), (0, 220, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(int(step_delay * 1000)) & 0xFF == ord("q"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default=None, help="classify real images instead of synthetic ones")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--pause", type=float, default=1.2, help="seconds to show each result before moving on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-heatmap", action="store_true",
                         help="skip Grad-CAM (faster; just shows the verdict)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}\nRun `python -m src.edge_eval` first.")
    model_bytes = model_path.read_bytes()
    threshold = _load_threshold(model_path)

    keras_model = None if args.no_heatmap else load_matching_keras_model(model_path)
    if not args.no_heatmap and keras_model is None:
        print("(no matching *_keras.h5 found next to this model - showing verdicts without a heatmap)")

    if args.folder:
        stream = _folder_stream(args.folder)
        total = len(list(Path(args.folder).iterdir()))
    else:
        rng = np.random.default_rng(args.seed)
        stream = _synthetic_stream(rng)
        total = None

    print(f"Model: {model_path}  |  threshold: {threshold:.3f}")
    print("Press 'q' in the window to stop.")

    cv2.namedWindow(WINDOW_NAME)
    for idx, (img, true_label) in enumerate(stream):
        if not _scan_animation(img, idx, total):
            break
        prob = float(predict_tflite(model_bytes, img[None, ...])[0])
        label = "DEFECT" if prob >= threshold else "OK"

        display_img = img
        has_heatmap = False
        if keras_model is not None:
            heatmap = make_gradcam_heatmap(keras_model, img)
            display_img = overlay_heatmap(img, heatmap)
            has_heatmap = True

        frame = _draw_frame(display_img, "result", label=label, prob=prob, true_label=true_label,
                             idx=idx, total=total, has_heatmap=has_heatmap)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(int(args.pause * 1000)) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
