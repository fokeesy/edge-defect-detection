"""Trains a defect classifier on YOUR OWN images - any product, any kind of
defect. Paste photos into two folders:

    my_dataset/
      ok/       <- photos of good parts       (jpg / png / bmp ...)
      defect/   <- photos of defective parts

then just run:

    python train_on_folder.py

(or point --data at any other folder laid out the same way). A few dozen
photos per class is enough - by default it fine-tunes a MobileNetV2 that
was already pretrained on ImageNet, and randomly flips / shifts / re-lights
your photos during training so it learns the defect, not the exact pictures.

Saves the model (float32 + INT8 TFLite, plus the full Keras model for
Grad-CAM) and its calibrated decision threshold to outputs/custom_models/.
Then use `predict.py` or `live_demo.py` to classify new images with it.
"""
import argparse
from pathlib import Path

from src.custom_data import load_dataset
from src.model import ALL_PRESETS, TRANSFER_PRESET
from src.train import train_on_arrays, select_threshold_centered
from src.quantize import to_tflite_float32, to_tflite_int8, predict_tflite

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "my_dataset"
OUT_DIR = ROOT / "outputs" / "custom_models"
MIN_RELIABLE_TEST_IMAGES = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                         help="dataset folder with ok/ and defect/ subfolders (default: my_dataset/)")
    parser.add_argument("--preset", default=TRANSFER_PRESET, choices=list(ALL_PRESETS),
                         help=f"'{TRANSFER_PRESET}' (default) starts from a pretrained network and works "
                              "with few photos; tiny/small/medium train from scratch and need far more")
    parser.add_argument("--ok-dirname", default="ok")
    parser.add_argument("--defect-dirname", default="defect")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true",
                         help="don't randomly flip/rotate/shift/re-light photos while training "
                              "(use this if orientation matters, e.g. a label that must not be upside-down)")
    parser.add_argument("--name", default=None,
                         help="output filename prefix (default: the dataset folder's name)")
    args = parser.parse_args()

    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset(
            args.data, seed=args.seed, ok_dirname=args.ok_dirname, defect_dirname=args.defect_dirname,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Can't load the dataset: {e}")

    model, metrics, _ = train_on_arrays(
        args.preset, x_train, y_train, x_val, y_val, x_test, y_test,
        seed=args.seed, epochs=args.epochs, progress_bar=True,
        desc=f"training [{args.preset}] on {Path(args.data).name}",
        augment=not args.no_augment, balance_classes=True, centered_threshold=True,
    )

    print(f"\nTest AUC: {metrics['test_auc']:.3f}  "
          f"Accuracy: {metrics['test_accuracy']:.3f}  "
          f"Precision: {metrics['test_precision']:.3f}  "
          f"Recall: {metrics['test_recall']:.3f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")
    if len(x_test) < MIN_RELIABLE_TEST_IMAGES:
        print(f"Note: only {len(x_test)} images were held out for testing, so these numbers are a rough "
              f"guide at best - add more photos for a trustworthy score.")

    name = args.name or Path(args.data).name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tflite_int8 = to_tflite_int8(model, x_train[:min(200, len(x_train))])
    (OUT_DIR / f"{name}_int8.tflite").write_bytes(tflite_int8)
    (OUT_DIR / f"{name}_float32.tflite").write_bytes(to_tflite_float32(model))
    model.save(str(OUT_DIR / f"{name}_keras.h5"))  # needed for Grad-CAM - see gradcam.py

    # The INT8 file is what actually gets deployed, and quantization nudges
    # its scores slightly, so the decision threshold is re-calibrated on the
    # INT8 model's own validation outputs (the float model's threshold cost
    # a few accuracy points when reused as-is).
    threshold = select_threshold_centered(y_val, predict_tflite(tflite_int8, x_val))
    (OUT_DIR / f"{name}_threshold.txt").write_text(str(threshold))
    int8_pred = (predict_tflite(tflite_int8, x_test) >= threshold).astype(int)
    print(f"\nDeployed INT8 model accuracy on the same test images: {float((int8_pred == y_test).mean()):.3f} "
          f"(float model: {metrics['test_accuracy']:.3f})")

    print(f"Saved model to {OUT_DIR / f'{name}_int8.tflite'}")
    print(f"Try it: python live_demo.py --model {OUT_DIR / f'{name}_int8.tflite'} --folder path/to/new_photos")


if __name__ == "__main__":
    main()
