"""Converts a trained Keras model to TFLite (float32 baseline) and to a
fully INT8-quantized TFLite model - the actual format that runs on real
edge/embedded targets (Coral, Raspberry Pi, and the same op kernels used by
TensorFlow Lite for Microcontrollers on a bare microcontroller). Also
benchmarks size, latency, and accuracy for each, since "quantized" and
"quantized without losing too much accuracy" are very different claims.
"""
import time

import numpy as np
import tensorflow as tf


def to_tflite_float32(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def to_tflite_int8(model, representative_images):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data_gen():
        for img in representative_images:
            yield [img.astype(np.float32)[None, ...]]

    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()


def _prepare_input(img, input_details):
    dtype = input_details["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = input_details["quantization"]
        if not scale:
            scale = 1.0
        x = img.astype(np.float32) / scale + zero_point
        info = np.iinfo(dtype)
        x = np.clip(np.round(x), info.min, info.max).astype(dtype)
    else:
        x = img.astype(np.float32)
    return x[None, ...]


def _read_output(raw, output_details):
    dtype = output_details["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = output_details["quantization"]
        if not scale:
            scale = 1.0
        return (raw.astype(np.float32) - zero_point) * scale
    return raw.astype(np.float32)


def predict_tflite(tflite_bytes, images):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    probs = np.zeros(len(images), dtype=np.float32)
    for i, img in enumerate(images):
        interpreter.set_tensor(input_details["index"], _prepare_input(img, input_details))
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"])
        probs[i] = _read_output(raw, output_details).ravel()[0]
    return probs


def benchmark_latency_ms(tflite_bytes, sample_image, n_warmup=10, n_runs=100):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    x = _prepare_input(sample_image, input_details)

    for _ in range(n_warmup):
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))
