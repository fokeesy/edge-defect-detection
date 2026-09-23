import numpy as np

from src.model import build_model
from src.data_gen import generate_dataset
from src.quantize import to_tflite_float32, to_tflite_int8, predict_tflite, benchmark_latency_ms


def _quick_trained_model():
    """A minimally-trained (not accurate, just not freshly-initialized)
    model - enough for BatchNorm to have real running statistics, which
    matters for whether quantization conversion behaves realistically."""
    rng = np.random.default_rng(0)
    x, y, _ = generate_dataset(40, 40, rng)
    model = build_model("tiny")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=2, batch_size=16, verbose=0)
    return model, x


def test_tflite_conversion_produces_valid_smaller_int8_model():
    model, x = _quick_trained_model()
    tflite_f32 = to_tflite_float32(model)
    tflite_int8 = to_tflite_int8(model, x[:20])

    assert len(tflite_f32) > 0
    assert len(tflite_int8) > 0
    assert len(tflite_int8) < len(tflite_f32), "INT8 quantization should shrink the model"


def test_tflite_predictions_are_valid_probabilities():
    model, x = _quick_trained_model()
    tflite_int8 = to_tflite_int8(model, x[:20])
    probs = predict_tflite(tflite_int8, x[:10])
    assert probs.shape == (10,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_latency_benchmark_returns_positive_numbers():
    model, x = _quick_trained_model()
    tflite_f32 = to_tflite_float32(model)
    mean_ms, std_ms = benchmark_latency_ms(tflite_f32, x[0], n_warmup=2, n_runs=5)
    assert mean_ms > 0
    assert std_ms >= 0
