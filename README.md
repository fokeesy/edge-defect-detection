# Edge AI Defect Detection

A from-scratch pipeline that trains a defect classifier for manufactured
parts and then makes it actually *deployable on edge hardware*: quantized to
INT8, benchmarked for real size/latency/accuracy tradeoffs, not just trained
and left as a laptop-only demo. This is the "TinyML" / edge-AI skill set
manufacturers increasingly want alongside classical robotics: a camera on
the production line has to run inference locally, in real time, on hardware
with a fraction of a laptop's memory and compute.

## What it does

1. **Generates a synthetic QC dataset**: a procedurally-rendered manufactured
   part (a lathe-turned metal disc) in "OK" and "DEFECT" variants - four
   defect types (scratch, dent, discoloration, crack), each with a
   controllable *severity* dial from barely-perceptible to obvious. Synthetic
   rather than a downloaded benchmark (like MVTec AD) so the whole project is
   reproducible with just `pip install`, no multi-GB external download - and
   so severity can be swept directly, the same way the peg-in-hole project
   swept vision noise.
2. **Trains three model sizes** (tiny / small / medium - the same CNN
   architecture at three capacities) and evaluates each with real metrics:
   ROC-AUC, precision, recall, confusion matrix on a held-out test set.
3. **Quantizes each to INT8** via TensorFlow Lite - the actual format that
   runs on real edge/embedded targets (Coral, Raspberry Pi, and the same op
   kernels TensorFlow Lite for Microcontrollers uses on a bare
   microcontroller).
4. **Benchmarks the real tradeoffs**: model size (float32 vs. INT8),
   inference latency, and how much accuracy quantization actually costs -
   not assumed, measured.
5. **Sweeps defect severity** through the deployed INT8 model specifically,
   producing the key result: at what point do defects become too subtle for
   the edge-deployed model to catch?
6. **Explains its own predictions with Grad-CAM**: a heatmap showing which
   pixels actually drove a DEFECT verdict, not just the verdict itself - see
   `python live_demo.py` or `predict.py --heatmap`.
7. **Trains on your own photos**: paste pictures of good and defective parts
   of *any* product into two folders and get a deployable INT8 model back -
   from as few as a couple dozen photos per class (see "Using your own
   images").

## Quickstart

```bash
pip install -r requirements.txt

# quick demo: predictions on a handful of example parts -> outputs/demo_predictions.png
python main.py

# watch it classify parts live, one at a time, in a window - with a
# Grad-CAM heatmap showing what it's actually looking at
python live_demo.py

# train on YOUR OWN photos: paste them into my_dataset/ok and
# my_dataset/defect first (see "Using your own images" below)
python train_on_folder.py

# full pipeline: train all 3 presets, quantize, sweep severity, save plots
python -m src.edge_eval

# run the test suite
python -m pytest tests/ -v
```

Training shows a live progress bar (loss/accuracy/AUC updating every epoch)
instead of either silence or per-batch spam.

## Explainability: seeing what the model is looking at

A "DEFECT, 0.86 confidence" verdict on its own isn't very trustworthy - it
could be looking at the actual scratch, or it could be picking up on
something spurious (like lighting) that happens to correlate with defects
in the training data. **Grad-CAM** answers that by computing which pixels
the model's own gradients say mattered most for that specific prediction,
and overlaying them as a heatmap - `live_demo.py` shows this by default,
and `predict.py --heatmap` saves it for a single image or a folder.

This needs the full trainable model, not the quantized INT8 one (quantized
models can't provide gradients), so `edge_eval.py` and `train_on_folder.py`
both save a `*_keras.h5` file alongside each deployment `.tflite` file
purely for this - which is realistic, not a shortcut: explainability is a
development/validation tool you'd run on your own machine to sanity-check
the model, not something that needs to run on the constrained edge device
itself.

The pretrained MobileNet model (the default for your own photos) gives a
coarser heatmap - 6x6 cells instead of 24x24 - because its network is much
deeper and shrinks the image more before the last layer Grad-CAM can read;
it still lands on the defect, just with bigger blobs.

One honest finding from building this: heatmap quality tracks model
quality. Run it against a weak/undertrained model and the "explanation" is
a vague blob covering half the part - not wrong, exactly, but not useful
either. Run it against a well-trained model (`medium` or `tiny` in this
project's results) and the heatmap tracks the actual scratch/dent/crack
shape closely. A blurry Grad-CAM heatmap is itself a diagnostic signal that
the model hasn't learned a spatially precise concept yet.

## Using your own images

The synthetic-disc model above only knows metal discs. To teach it *your*
product and *your* kind of defect, paste photos into two folders that are
already waiting in the project:

```
my_dataset/
  ok/       <- photos of good parts       (.jpg .png .bmp .tif)
  defect/   <- photos of defective parts
```

```bash
# 1. train (fine-tunes a pretrained MobileNetV2 - works with few photos)
python train_on_folder.py

# 2. classify new photos with the model you just trained
python live_demo.py --model outputs/custom_models/my_dataset_int8.tflite --folder new_parts/
python predict.py --model outputs/custom_models/my_dataset_int8.tflite --image new_part.jpg --heatmap
python predict.py --model outputs/custom_models/my_dataset_int8.tflite \
                   --folder new_parts/ --save-annotated results.png
```

Point `--data` at any other folder with the same `ok/` + `defect/` layout to
train several products side by side (`--name` sets the output filename).
Photos you put in `my_dataset/` are git-ignored, so they're never committed.

**What the training recipe does, and why** (the default `--preset mobilenet`):

- **Starts from a pretrained network.** MobileNetV2 (width 0.35, ~0.4M
  params - an actual edge-deployment architecture) already knows edges,
  textures and shapes from ImageNet, so a few dozen photos are enough to
  teach it good-vs-defect. The from-scratch `tiny/small/medium` presets are
  still selectable (`--preset small`) but need far more data. The
  ImageNet weights (~2 MB) download automatically on first use.
- **Augments your photos on the fly**: random flips, 90-degree rotations,
  small shifts, brightness/contrast jitter - so a few dozen photos become
  endless variations and the model has to learn the defect itself, not
  memorize the pictures. Pass `--no-augment` if orientation matters (e.g. a
  label that must never be upside-down).
- **Balances classes**, so 10 defect photos among 200 good ones aren't
  ignored (otherwise "always say OK" looks 95% accurate).
- **Calibrates the decision threshold on the deployed INT8 model itself**,
  at the middle of the best-scoring range rather than its edge.
- Checks the INT8 file (the one you'd actually ship) against the held-out
  test images and prints that accuracy next to the float model's.

**How well does it work?** Measured by training on a small folder of
JPEGs and testing on 400 unseen images, averaged over several seeds - and
the from-scratch pipeline this replaced was a coin flip at these sizes:

| Photos per class | Recipe | ROC-AUC | Accuracy |
|---|---|---|---|
| 40 | small CNN from scratch, no augmentation (old pipeline) | 0.69 | 63% |
| 40 | tiny CNN from scratch + augmentation | 0.84 | 79% |
| 40 | **pretrained MobileNetV2 + augmentation (default)** | **0.996** | **98%** |
| 15 | tiny CNN from scratch + augmentation | 0.75 | 68% |
| 15 | **pretrained MobileNetV2 + augmentation (default)** | **0.999** | **96%** |

INT8 quantization of the MobileNet model cost 0.5 accuracy points on average
(97.8% float -> 97.3% INT8, 4 seeds) once its threshold was calibrated on
the INT8 outputs; reusing the float model's threshold instead dropped it to
95.4%. The INT8 model is ~0.6 MB and runs in ~0.5 ms per image on a laptop
CPU.

**Honest limits** - read these before trusting it on real products:

- Those numbers come from folders of *this project's synthetic parts* saved
  as JPEGs - the only labeled data available without an external download.
  Real photos with cluttered backgrounds, changing lighting and very subtle
  defects are harder; expect to need more photos than the table suggests.
- **Images are resized to 96x96** (non-square photos are squashed to a
  square, not cropped). A small defect on a large, high-res photo can shrink
  to nothing. Crop tightly around the part (or the region being inspected)
  before training, so defects stay a few pixels wide at 96x96.
- **It learns whatever separates your two folders** - which should be the
  defect, but could be lighting, background or camera angle if the good and
  defective photos were taken differently. Keep capture conditions the same
  for both classes, and check with `--heatmap`: the hot spots should sit on
  the defect.
- One model = one product and one yes/no question (OK vs DEFECT). It
  doesn't say *which* defect, and a different product needs its own model.
- With very few photos the held-out test set is tiny (a dozen images or
  fewer), so the printed accuracy is only a rough guide - the script says so
  when that's the case. More photos give a trustworthy score. (The hard
  minimum is 4 photos per class, just so there's something to train,
  validate and test on; aim for 30+.)

## Results

See `outputs/accuracy_comparison.png`, `outputs/size_comparison.png`,
`outputs/latency_comparison.png`, `outputs/recall_vs_severity.png`, and
`outputs/model_summary.csv` / `outputs/severity_sweep.csv` for the full
sweep. `outputs/demo_predictions.png` shows a handful of example
predictions from the deployed model.

**Headline numbers** (one run; see the note on run-to-run variance below):

| preset | params | INT8 size | INT8 latency | INT8 accuracy | quantization cost |
|---|---|---|---|---|---|
| tiny   | 5,433  | 11.7 KB | 0.39 ms | 100.0% | -0.2% (i.e. *improved*) |
| small  | 20,849 | 28.0 KB | 0.49 ms | 70.0%  | +0.5% |
| medium | 81,633 | 89.7 KB | 1.39 ms | 99.8%  | -0.2% |

- **Quantization to INT8 barely costs any accuracy** - every preset lands
  within half a percentage point of its pre-quantization number, and two of
  the three actually score marginally *higher* after quantization (noise at
  this sample size, but the headline finding - "INT8 is not a meaningful
  accuracy tax here" - holds).
- **INT8 is not automatically faster on a general-purpose x86 CPU.** For the
  `tiny` and `small` presets, the INT8 model was actually *slower* than
  float32 (0.39ms vs 0.23ms, 0.49ms vs 0.43ms) - only `medium` showed the
  expected win (1.39ms vs 1.58ms). This CPU doesn't have the same INT8 SIMD
  path that real edge silicon (Coral's Edge TPU, ARM's NPUs) has purpose-built
  for quantized inference; INT8's real, reliable win here is model size
  (2.1x-3.6x smaller), not latency, unless you're targeting hardware that
  actually accelerates it.
- **Bigger is not straightforwardly better.** `medium` (81K params) narrowly
  beat `tiny` (5K params) on accuracy in this run, but `tiny` was smaller,
  faster, *and* tied on accuracy - and in other runs during development,
  `tiny` was the best performer outright. See the next point.
- **Run-to-run variance is real and worth stating plainly.** Training a
  small CNN on ~2,400 images with random initialization does not converge to
  the same quality every time - across development runs, `small` in
  particular ranged from AUC 0.79 to a perfect 1.0 depending on the random
  seed, despite identical code and data. A production pipeline would train
  multiple seeds per preset and select/ensemble, not ship a single run. This
  project reports one full run's numbers honestly rather than the best one
  observed, but the variance itself is real, not hidden - see the pitfalls
  below.

## Architecture

```
src/
  data_gen.py      Procedural synthetic part/defect image generator
  custom_data.py   Loads a real, labeled image folder in the same shape
  model.py         CNN architecture: three from-scratch presets (tiny/small/
                    medium) + a pretrained MobileNetV2 transfer model
  train.py         Core training loop (train_on_arrays) + the synthetic-data
                    wrapper (train_model), live tqdm progress bar, and the
                    small-dataset toolkit: augmentation, class balancing,
                    centered threshold selection
  quantize.py      Keras -> TFLite float32 / INT8 conversion + inference +
                    latency benchmarking
  gradcam.py       Grad-CAM: which pixels drove a prediction, on the full
                    Keras model (quantized models can't give gradients)
  edge_eval.py     The real deliverable: trains all presets, quantizes,
                    sweeps defect severity, saves CSVs + plots
main.py            Quick demo -> annotated prediction grid
train_on_folder.py Train on your own photos (my_dataset/ok + defect/)
predict.py         Classify a new image or folder with a trained model
                    (--heatmap for Grad-CAM)
live_demo.py       Live window: classifies parts one at a time as you watch,
                    with a live Grad-CAM heatmap
my_dataset/       Empty ok/ + defect/ folders - paste your photos here
tests/             pytest suite (data generation, custom data loading, model,
                    quantization, threshold calibration, augmentation,
                    training, Grad-CAM)
```

## Simplifications (stated up front)

- **Synthetic data, not a real camera feed or the real MVTec AD benchmark.**
  This keeps the project fully reproducible and lets severity be swept
  directly; the tradeoff is that real manufacturing defects have more
  visual variety than four procedural defect types can capture. Plugging in
  MVTec AD instead is a documented extension below.
- **INT8 benchmarking uses the standard TFLite interpreter, not a flashed
  microcontroller.** This is the same runtime family that ships on
  Raspberry Pi, Coral, and Android, and shares its op kernels with
  TensorFlow Lite for Microcontrollers - a legitimate and standard way to
  validate "this model is quantization-ready and edge-deployable." It is
  not the same as measuring latency on an actual $5 MCU, which would be
  slower; see Possible Extensions.

## Engineering pitfalls hit along the way

- **A fatal LLVM crash during INT8 conversion.** `TFLiteConverter` calling
  into MLIR crashed with `LLVM ERROR: Failed to infer result type(s)` on a
  model containing `BatchNormalization`, whenever TensorFlow 2.16+ was
  installed. TF 2.16 switched to Keras 3 by default, and its TFLite INT8
  converter has a compatibility bug with BatchNorm layers as of writing.
  Fix: pin `tensorflow==2.15.1`, which still defaults to Keras 2 and uses
  the mature, stable conversion path. Caught by actually running the
  quantization step during development, not just training and stopping.
- **A classifier that "learned nothing" actually had excellent AUC.** The
  first full sweep produced a `medium` model stuck at exactly 50% accuracy,
  predicting the same class for every single test image (100% recall, 0%
  the other way). That looked like a training failure - but ROC-AUC on the
  same model was 0.83-0.999, and inspecting the raw probabilities showed why:
  every single prediction sat in a compressed band like 0.21-0.42, real
  separation between classes, just never crossing the 0.5 threshold. AUC is
  threshold-independent, so it doesn't catch this; a fixed 0.5 cutoff does,
  silently. **Fix: calibrate the decision threshold on the validation set**
  (`train.select_threshold`, sweeping candidate thresholds for the one that
  maximizes validation accuracy) instead of assuming 0.5, and reuse that
  same calibrated threshold for the quantized models too. Also fixed
  **TensorFlow's own RNG never being seeded** (only the synthetic-data
  generator was) - identical code was silently producing different results
  every run.
- **The misdiagnosis that came before the real fix.** Before finding the
  calibration explanation above, the compressed-probability symptom looked
  like training instability, so `val_loss`-based early stopping,
  `ReduceLROnPlateau`, and gradient clipping (`clipnorm=1.0`) were all added
  as "stability" measures. Gradient clipping in particular backfired badly:
  the `tiny` preset, which had trained to a perfect AUC of 1.0 with the
  original simple setup, dropped to AUC 0.51 - random chance - with clipping
  enabled, isolated by retraining the identical setup with only that one
  line changed. None of these changes were actually needed once threshold
  calibration was in place; **the final code is the original plain
  `val_auc`-based early stopping**, unchanged, plus calibration. Lesson: a
  plausible-sounding stability fix is still a hypothesis, not a fix, until
  verified against a known-good baseline - and it's worth pausing to ask
  whether a symptom is actually what it looks like before reaching for more
  machinery to suppress it.
- **"Works on your own images" was technically true and practically a
  coin flip.** The first version of `train_on_folder.py` accepted any photo
  folder - but trained the small from-scratch CNN on it with no
  augmentation, and on a realistic 40-photos-per-class folder it scored
  50-63% on unseen images (chance is 50%). It only became visible by
  actually dropping a small folder in and testing on fresh images instead of
  trusting that the code path ran. Fixes, each measured on its own:
  augmentation (63% -> 85%), a pretrained MobileNetV2 backbone (-> 98%), and
  a centered decision threshold (a threshold picked from a 12-image
  validation set was landing at the *edge* of the best range and cost up to
  25 accuracy points on some seeds despite near-perfect AUC). A last catch:
  INT8 quantization shifted the MobileNet's output scores enough that the
  float model's threshold lost ~4 accuracy points, so the threshold is now
  calibrated on the INT8 model itself.
- **An orphaned `jax`/`jaxlib` install broke `import tensorflow` entirely**
  with an unrelated `ml_dtypes` version conflict (`jaxlib` needed a newer
  `ml_dtypes` than `tensorflow-intel` pins). Neither package is a real
  dependency of anything in this project - `pip show jax` confirmed nothing
  required it. Fix: uninstall both; the conflict disappeared because there
  was nothing to conflict over.

## Possible extensions

- **Training on real images is already supported** (`train_on_folder.py` +
  `custom_data.py`) - the remaining gap for the actual MVTec AD benchmark
  specifically is a thin adapter script to reshape its per-category folder
  layout into the flat `ok/`/`defect/` structure this project expects, and
  ideally exploiting its pixel-level ground truth masks (MVTec AD ships
  those) for real defect localization rather than just image-level labels.
- Actually flash the INT8 model to a real microcontroller via TensorFlow
  Lite for Microcontrollers (Arduino/ESP32) and measure real latency and
  memory footprint on the target hardware, instead of the x86 TFLite
  interpreter proxy used here.
- Grad-CAM already gives a coarse *where* (see above), but a proper defect
  *localization* head (bounding box or segmentation) would be sharper and
  wouldn't need gradients at inference time - closer to what a real QC
  system needs to show a human precisely where to look.
