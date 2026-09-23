"""Procedural synthetic dataset of a manufactured circular part ("OK" vs
"DEFECT"), stood in for a real industrial QC camera feed.

Why synthetic instead of downloading a real benchmark (e.g. MVTec AD): this
keeps the whole project reproducible with a single `pip install` and no
multi-GB external download, and - crucially for the evaluation story - lets
us control defect *severity* directly, so we can measure "how subtle can a
defect be before the model misses it" the same way the peg-in-hole project
measured "how much vision error before the search fails."
"""
import numpy as np
import cv2

IMG_SIZE = 96
DEFECT_TYPES = ("scratch", "dent", "discoloration", "crack")


def _smooth_noise(size, strength, rng, blur_ksize=7):
    noise = rng.normal(0, strength, size=(size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (blur_ksize, blur_ksize), 0)
    return noise


def _part_mask(size, center, radius):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def generate_clean_part(rng, size=IMG_SIZE):
    """A synthetic lathe-turned metal disc on a darker background, with
    mild per-sample lighting/position/texture variation (representing
    normal manufacturing + camera variance that a real inspector - human
    or model - has to be robust to even on a non-defective part)."""
    bg_level = rng.uniform(40, 55)
    img = np.full((size, size, 3), bg_level, dtype=np.float32)
    img += _smooth_noise(size, 4.0, rng)[..., None]

    radius = int(size * rng.uniform(0.34, 0.40))
    cx = size // 2 + rng.integers(-3, 4)
    cy = size // 2 + rng.integers(-3, 4)
    mask = _part_mask(size, (cx, cy), radius) > 0

    part_level = rng.uniform(120, 150)
    part = np.full((size, size), part_level, dtype=np.float32)
    part += _smooth_noise(size, 6.0, rng, blur_ksize=3)

    # concentric lathe-turning rings
    n_rings = rng.integers(5, 9)
    for i in range(n_rings):
        r = int(radius * (i + 1) / (n_rings + 1))
        ring_shade = rng.uniform(-8, 8)
        cv2.circle(part, (cx, cy), r, part_level + ring_shade, 1)
    part = cv2.GaussianBlur(part, (3, 3), 0)

    # simple directional lighting gradient across the part
    yy, xx = np.mgrid[0:size, 0:size]
    angle = rng.uniform(0, 2 * np.pi)
    grad = (xx * np.cos(angle) + yy * np.sin(angle)) / size
    part += grad * rng.uniform(-10, 10)

    for c in range(3):
        channel = img[:, :, c]
        channel[mask] = part[mask]
        img[:, :, c] = channel

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, (cx, cy), radius


def _soft_blend(img, patch_mask, color_delta, feather=5):
    """Alpha-blend `color_delta` into `img` where `patch_mask` is set, with
    a feathered (blurred) edge so defects don't look like paste-on stickers."""
    alpha = cv2.GaussianBlur(patch_mask.astype(np.float32), (feather * 2 + 1,) * 2, 0)
    alpha = np.clip(alpha, 0, 1)[..., None]
    out = img.astype(np.float32) + alpha * color_delta
    return np.clip(out, 0, 255).astype(np.uint8)


def add_defect(img, center, radius, defect_type, severity, rng):
    """`severity` in [0, 1]: 0 is barely perceptible, 1 is obvious."""
    size = img.shape[0]
    patch = np.zeros((size, size), dtype=np.uint8)
    severity = float(np.clip(severity, 0.02, 1.0))

    if defect_type == "scratch":
        length = int(radius * rng.uniform(0.5, 1.3))
        angle = rng.uniform(0, np.pi)
        cx, cy = center
        ox = int(rng.uniform(-radius * 0.4, radius * 0.4))
        oy = int(rng.uniform(-radius * 0.4, radius * 0.4))
        p1 = (cx + ox - int(length / 2 * np.cos(angle)), cy + oy - int(length / 2 * np.sin(angle)))
        p2 = (cx + ox + int(length / 2 * np.cos(angle)), cy + oy + int(length / 2 * np.sin(angle)))
        thickness = max(1, int(1 + severity * 2))
        cv2.line(patch, p1, p2, 255, thickness)
        delta_mag = 40 + 90 * severity
        color_delta = np.array([-delta_mag, -delta_mag, -delta_mag])

    elif defect_type == "crack":
        cx, cy = center
        x, y = cx + rng.integers(-radius // 2, radius // 2), cy + rng.integers(-radius // 2, radius // 2)
        n_segs = rng.integers(4, 8)
        for _ in range(n_segs):
            angle = rng.uniform(0, 2 * np.pi)
            step = radius * rng.uniform(0.08, 0.18)
            nx, ny = int(x + step * np.cos(angle)), int(y + step * np.sin(angle))
            cv2.line(patch, (x, y), (nx, ny), 255, 1)
            x, y = nx, ny
        delta_mag = 50 + 100 * severity
        color_delta = np.array([-delta_mag, -delta_mag, -delta_mag])

    elif defect_type == "dent":
        cx, cy = center
        ox = int(rng.uniform(-radius * 0.5, radius * 0.5))
        oy = int(rng.uniform(-radius * 0.5, radius * 0.5))
        r = max(2, int(radius * rng.uniform(0.1, 0.22) * (0.4 + severity)))
        cv2.circle(patch, (cx + ox, cy + oy), r, 255, -1)
        delta_mag = 30 + 70 * severity
        color_delta = np.array([-delta_mag, -delta_mag, -delta_mag])

    elif defect_type == "discoloration":
        cx, cy = center
        ox = int(rng.uniform(-radius * 0.5, radius * 0.5))
        oy = int(rng.uniform(-radius * 0.5, radius * 0.5))
        r = max(3, int(radius * rng.uniform(0.15, 0.3) * (0.4 + severity)))
        cv2.circle(patch, (cx + ox, cy + oy), r, 255, -1)
        delta_mag = 20 + 60 * severity
        color_delta = np.array([delta_mag * 0.6, delta_mag * 0.1, -delta_mag * 0.5])

    else:
        raise ValueError(f"unknown defect_type {defect_type}")

    part_mask = _part_mask(size, center, radius) > 0
    patch = patch * part_mask.astype(np.uint8)
    return _soft_blend(img, patch, color_delta)


def generate_sample(rng, defective, defect_type=None, severity=None, size=IMG_SIZE):
    img, center, radius = generate_clean_part(rng, size=size)
    meta = {"defective": bool(defective)}
    if defective:
        defect_type = defect_type or rng.choice(DEFECT_TYPES)
        severity = rng.uniform(0.25, 1.0) if severity is None else severity
        img = add_defect(img, center, radius, defect_type, severity, rng)
        meta.update({"defect_type": defect_type, "severity": float(severity)})
    else:
        meta.update({"defect_type": None, "severity": 0.0})
    return img, meta


def generate_dataset(n_ok, n_defect, rng, severity_range=(0.25, 1.0), size=IMG_SIZE):
    """Balanced-ish dataset with defect type and severity drawn uniformly
    from `severity_range`. Returns (images uint8 [N,H,W,3], labels int [N],
    meta list[dict])."""
    images, labels, metas = [], [], []
    for _ in range(n_ok):
        img, meta = generate_sample(rng, defective=False, size=size)
        images.append(img)
        labels.append(0)
        metas.append(meta)
    for _ in range(n_defect):
        severity = rng.uniform(*severity_range)
        defect_type = rng.choice(DEFECT_TYPES)
        img, meta = generate_sample(rng, defective=True, defect_type=defect_type,
                                     severity=severity, size=size)
        images.append(img)
        labels.append(1)
        metas.append(meta)
    idx = rng.permutation(len(images))
    images = np.array(images, dtype=np.uint8)[idx]
    labels = np.array(labels, dtype=np.int64)[idx]
    metas = [metas[i] for i in idx]
    return images, labels, metas


def generate_severity_bucket(n_ok, n_defect, severity, rng, size=IMG_SIZE):
    """A fixed-severity evaluation set: n_ok clean parts + n_defect defective
    parts all at exactly this severity level, for the accuracy-vs-severity
    sweep."""
    return generate_dataset(n_ok, n_defect, rng, severity_range=(severity, severity), size=size)
