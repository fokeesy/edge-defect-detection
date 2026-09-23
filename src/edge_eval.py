"""The real deliverable of this project: for each model-size preset, train
it, quantize it to INT8, and measure the actual tradeoffs an edge deployment
has to live with - accuracy vs. model size vs. inference latency, and how
detection accuracy degrades as defects get more subtle (the same style of
robustness sweep as the peg-in-hole-assembly project's clearance/vision-noise
sweep).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .train import train_model
from .quantize import to_tflite_float32, to_tflite_int8, predict_tflite, benchmark_latency_ms
from .data_gen import generate_severity_bucket
from .model import PRESETS

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODELS_DIR = OUT_DIR / "models"
SEVERITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def evaluate_preset(preset, seed=0, n_severity_per_class=60, verbose=0, progress_bar=True):
    model, metrics, (x_test, y_test) = train_model(
        preset, seed=seed, verbose=verbose, progress_bar=progress_bar,
    )
    # calibrated on the float model's validation set inside train_model() -
    # reused as-is for the quantized variants too, the same way a real
    # deployment would calibrate once and ship a fixed threshold rather than
    # assuming a bare 0.5 (see README pitfalls: a model can have excellent
    # AUC while its raw sigmoid output never actually crosses 0.5)
    threshold = metrics["threshold"]

    rep_images = x_test[:200]
    tflite_f32 = to_tflite_float32(model)
    tflite_int8 = to_tflite_int8(model, rep_images)

    size_f32_kb = len(tflite_f32) / 1024.0
    size_int8_kb = len(tflite_int8) / 1024.0

    probs_f32 = predict_tflite(tflite_f32, x_test)
    probs_int8 = predict_tflite(tflite_int8, x_test)

    auc_f32 = roc_auc_score(y_test, probs_f32)
    auc_int8 = roc_auc_score(y_test, probs_int8)
    acc_f32 = float(((probs_f32 >= threshold).astype(int) == y_test).mean())
    acc_int8 = float(((probs_int8 >= threshold).astype(int) == y_test).mean())

    lat_f32_mean, lat_f32_std = benchmark_latency_ms(tflite_f32, x_test[0])
    lat_int8_mean, lat_int8_std = benchmark_latency_ms(tflite_int8, x_test[0])

    summary = {
        "preset": preset,
        "threshold": threshold,
        "n_params": metrics["n_params"],
        "keras_test_auc": metrics["test_auc"],
        "keras_test_accuracy": metrics["test_accuracy"],
        "tflite_f32_size_kb": size_f32_kb,
        "tflite_int8_size_kb": size_int8_kb,
        "compression_ratio": size_f32_kb / size_int8_kb if size_int8_kb else float("nan"),
        "tflite_f32_auc": float(auc_f32),
        "tflite_int8_auc": float(auc_int8),
        "tflite_f32_accuracy": acc_f32,
        "tflite_int8_accuracy": acc_int8,
        "accuracy_drop_from_quantization": acc_f32 - acc_int8,
        "tflite_f32_latency_ms": lat_f32_mean,
        "tflite_int8_latency_ms": lat_int8_mean,
    }
    tqdm.write(f"[{preset}] keras_auc={metrics['test_auc']:.3f} "
               f"int8_size={size_int8_kb:.1f}KB int8_latency={lat_int8_mean:.3f}ms "
               f"int8_acc={acc_int8:.3f} (drop from f32: {acc_f32 - acc_int8:+.3f})")

    severity_rows = []
    rng = np.random.default_rng(seed + 100)
    sev_bar = tqdm(SEVERITIES, desc=f"severity sweep [{preset}]", unit="bucket", leave=True)
    for sev in sev_bar:
        imgs, labels, _ = generate_severity_bucket(n_severity_per_class, n_severity_per_class, sev, rng)
        probs = predict_tflite(tflite_int8, imgs)
        preds = (probs >= threshold).astype(int)
        acc = float((preds == labels).mean())
        defect_mask = labels == 1
        recall = float((preds[defect_mask] == 1).mean()) if defect_mask.sum() else float("nan")
        severity_rows.append({"preset": preset, "severity": sev, "accuracy": acc, "defect_recall": recall})
        sev_bar.set_postfix({"severity": f"{sev:.1f}", "acc": f"{acc:.3f}", "recall": f"{recall:.3f}"})

    return summary, severity_rows, tflite_f32, tflite_int8, model


def save_csvs(summaries, severity_rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "model_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    with open(OUT_DIR / "severity_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(severity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(severity_rows)


def make_plots(summaries, severity_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    presets = [s["preset"] for s in summaries]
    colors = {"tiny": "#2b6cb0", "small": "#38a169", "medium": "#c53030"}

    # 1. size vs preset (f32 vs int8)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(presets))
    width = 0.35
    f32_sizes = [s["tflite_f32_size_kb"] for s in summaries]
    int8_sizes = [s["tflite_int8_size_kb"] for s in summaries]
    ax.bar(x - width / 2, f32_sizes, width, label="float32", color="#a0aec0")
    ax.bar(x + width / 2, int8_sizes, width, label="INT8 quantized", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(presets)
    ax.set_ylabel("Model size (KB)")
    ax.set_title("Model size: float32 vs. INT8 quantized")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "size_comparison.png", dpi=150)
    plt.close(fig)

    # 2. latency vs preset
    fig, ax = plt.subplots(figsize=(6, 4.5))
    f32_lat = [s["tflite_f32_latency_ms"] for s in summaries]
    int8_lat = [s["tflite_int8_latency_ms"] for s in summaries]
    ax.bar(x - width / 2, f32_lat, width, label="float32", color="#a0aec0")
    ax.bar(x + width / 2, int8_lat, width, label="INT8 quantized", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(presets)
    ax.set_ylabel("Inference latency (ms, single image, CPU)")
    ax.set_title("Inference latency: float32 vs. INT8 quantized")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_comparison.png", dpi=150)
    plt.close(fig)

    # 3. accuracy vs preset (keras float, tflite f32, tflite int8)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    width = 0.25
    keras_acc = [s["keras_test_accuracy"] for s in summaries]
    f32_acc = [s["tflite_f32_accuracy"] for s in summaries]
    int8_acc = [s["tflite_int8_accuracy"] for s in summaries]
    ax.bar(x - width, keras_acc, width, label="Keras (float32, pre-conversion)", color="#718096")
    ax.bar(x, f32_acc, width, label="TFLite float32", color="#a0aec0")
    ax.bar(x + width, int8_acc, width, label="TFLite INT8", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(presets)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Accuracy: does quantization actually cost you anything?")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)

    # 4. accuracy/recall vs severity, per preset (the robustness curve)
    fig, ax = plt.subplots(figsize=(7, 5))
    for preset in presets:
        rows = [r for r in severity_rows if r["preset"] == preset]
        rows.sort(key=lambda r: r["severity"])
        sevs = [r["severity"] for r in rows]
        recalls = [r["defect_recall"] for r in rows]
        ax.plot(sevs, recalls, marker="o", label=preset, color=colors.get(preset))
    ax.set_xlabel("Defect severity (0 = barely perceptible, 1 = obvious)")
    ax.set_ylabel("Defect recall (INT8 quantized model)")
    ax.set_title("Can the edge-deployed model catch subtle defects?")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall_vs_severity.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=list(PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-severity-per-class", type=int, default=60)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summaries, all_severity_rows = [], []
    for i, preset in enumerate(args.presets):
        tqdm.write(f"\n=== preset {i + 1}/{len(args.presets)}: {preset} ===")
        summary, severity_rows, tflite_f32, tflite_int8, model = evaluate_preset(
            preset, seed=args.seed, n_severity_per_class=args.n_severity_per_class,
        )
        summaries.append(summary)
        all_severity_rows.extend(severity_rows)
        (MODELS_DIR / f"{preset}_float32.tflite").write_bytes(tflite_f32)
        (MODELS_DIR / f"{preset}_int8.tflite").write_bytes(tflite_int8)
        (MODELS_DIR / f"{preset}_threshold.txt").write_text(str(summary["threshold"]))
        # the full Keras model, kept alongside the deployment artifacts
        # above purely so Grad-CAM (which needs gradients - the quantized
        # TFLite model can't provide them) can explain this model's
        # predictions later; see gradcam.py
        model.save(str(MODELS_DIR / f"{preset}_keras.h5"))

    save_csvs(summaries, all_severity_rows)
    make_plots(summaries, all_severity_rows)

    print("\n=== summary ===")
    for s in summaries:
        print(json.dumps(s, indent=2))
    print(f"\nModels saved to {MODELS_DIR}")
    print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
