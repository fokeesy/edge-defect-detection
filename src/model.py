"""Small CNN classifiers (OK vs DEFECT), in three capacity presets so we can
sweep accuracy vs. model size vs. inference latency - the actual tradeoff an
edge deployment has to make, not just "here is one model that works."
"""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

INPUT_SHAPE = (96, 96, 3)

PRESETS = {
    "tiny": {"filters": (8, 16, 24), "dense": 16},
    "small": {"filters": (16, 32, 48), "dense": 32},
    "medium": {"filters": (32, 64, 96), "dense": 64},
}


TRANSFER_PRESET = "mobilenet"
ALL_PRESETS = (*PRESETS, TRANSFER_PRESET)


def build_transfer_model(input_shape=INPUT_SHAPE, pretrained=True):
    """MobileNetV2 (width 0.35) pretrained on ImageNet, with a small new
    classification head. For small real-photo datasets this is the right
    tool: the backbone already knows edges/textures/shapes from ~1M images,
    so a few dozen of your own photos are enough to teach it OK vs DEFECT,
    where the from-scratch presets above need hundreds or thousands.

    Built "flat" (backbone wired straight into this graph via input_tensor,
    not nested as a sub-model) so Grad-CAM and TFLite conversion see plain
    layers, exactly like the from-scratch presets.
    """
    inputs = layers.Input(shape=input_shape)
    # MobileNetV2 was trained on inputs scaled to [-1, 1]; the from-scratch
    # presets use [0, 1] - both start from the same raw uint8 pixels.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    backbone = MobileNetV2(input_tensor=x, alpha=0.35, include_top=False,
                            weights="imagenet" if pretrained else None)
    backbone.trainable = False  # BatchNorm layers stay in inference mode too
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="defect_net_mobilenet")


def build_model(preset="small", input_shape=INPUT_SHAPE, pretrained=True):
    if preset == TRANSFER_PRESET:
        return build_transfer_model(input_shape, pretrained=pretrained)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, choose from {list(ALL_PRESETS)}")
    cfg = PRESETS[preset]

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in cfg["filters"]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg["dense"], activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name=f"defect_net_{preset}")
    return model
