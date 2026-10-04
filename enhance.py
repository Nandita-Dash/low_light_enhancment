"""Enhancement pipeline: autoencoder -> histogram matching -> metrics."""
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

BASE = Path(__file__).parent
WEIGHTS_PATH = BASE / "weights" / "autoencoder.pt"
REF_CDF_PATH = BASE / "weights" / "reference_cdf.npy"
MAX_SIDE = 768
MAX_MATCH_SLOPE = 1.5  # max contrast gain of lightness matching
DEFAULT_BRIGHTNESS = 0.6  # target mean lightness (0-1) of the enhanced image

_model = None
_model_checked = False


def get_model():
    """Load the trained autoencoder once. Returns None if weights or torch are missing."""
    global _model, _model_checked
    if _model_checked:
        return _model
    _model_checked = True
    if not WEIGHTS_PATH.exists():
        return None
    try:
        import torch
        from model import AutoEncoder

        m = AutoEncoder()
        m.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        m.eval()
        _model = m
    except Exception as exc:  # torch missing or bad weights
        print("Could not load autoencoder:", exc)
    return _model


def decode_image(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("Could not read the image file")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def limit_size(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def run_autoencoder(img: np.ndarray) -> np.ndarray:
    import torch

    model = get_model()
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = model(x)[0].permute(1, 2, 0).numpy()
    y = (y[:h, :w] * 255.0).clip(0, 255).astype(np.uint8)
    return y


def noise_sigma(img: np.ndarray) -> float:
    """Fast estimate of the noise standard deviation (Immerkaer's method)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    resp = np.abs(cv2.filter2D(gray, -1, k))[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * resp.mean() / 6)


def demo_stage(img: np.ndarray, brightness: float = DEFAULT_BRIGHTNESS) -> np.ndarray:
    """Stand-in used only when no trained weights exist: denoise, then lift lightness.
    Works in LAB. Lightness gets edge-preserving NLM scaled to the measured noise level, so
    fine structure survives; colour channels are smoothed hard, since colour noise carries no detail.
    The lift curve 1 - (1 - L)^p has a finite slope at black, so near-black pixels are not blown into blotches."""
    sigma = noise_sigma(img)
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    L = cv2.fastNlMeansDenoising(L, None, float(np.clip(sigma, 3, 25)), 7, 35)
    if sigma > 4:
        a = cv2.bilateralFilter(cv2.GaussianBlur(a, (0, 0), 2), 9, 20, 9)
        b = cv2.bilateralFilter(cv2.GaussianBlur(b, (0, 0), 2), 9, 20, 9)
    Lf = L.astype(np.float32) / 255.0
    mean = float(np.clip(Lf.mean(), 1e-3, brightness))
    p = np.log(1 - brightness) / np.log(1 - mean)  # maps the mean lightness to the target
    lifted = 1 - np.power(1 - Lf, p)
    # Chroma shrinks with exposure, so scale it back up with the (smoothed) brightness gain
    gain = cv2.GaussianBlur(lifted, (0, 0), 3) / np.maximum(cv2.GaussianBlur(Lf, (0, 0), 3), 1e-3)
    gain = np.clip(np.sqrt(gain), 1.0, 1.8)  # softened so strong lifts do not oversaturate
    a = (128 + (a.astype(np.float32) - 128) * gain).clip(0, 255).astype(np.uint8)
    b = (128 + (b.astype(np.float32) - 128) * gain).clip(0, 255).astype(np.uint8)
    # White point: stretch so the brightest areas (paper, lit walls) reach near-white
    white = max(float(np.percentile(lifted, 99.5)), 1e-3)
    lifted = lifted * min(0.98 / white, 1.6)
    L = (lifted * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2RGB)


def keep_color_balance(out: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Remove any colour cast the enhancement added. The output's mean colour (a/b in LAB) is
    turned back to the source's hue, keeping its strength, so a neutral image stays neutral and a
    warm one stays warm instead of drifting blue."""
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_cast = ref[..., 1:].reshape(-1, 2).mean(0) - 128
    out_cast = lab[..., 1:].reshape(-1, 2).mean(0) - 128
    n_src = float(np.hypot(*src_cast))
    target = src_cast if n_src < 1 else src_cast / n_src * max(float(np.hypot(*out_cast)), n_src)
    lab[..., 1:] += target - out_cast
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def channel_cdf(img: np.ndarray) -> np.ndarray:
    """Per-channel cumulative distribution, shape (256, 3)."""
    out = np.zeros((256, 3), np.float64)
    for c in range(3):
        hist = np.bincount(img[..., c].ravel(), minlength=256).astype(np.float64)
        out[:, c] = np.cumsum(hist) / hist.sum()
    return out


def default_reference_cdf(brightness: float = DEFAULT_BRIGHTNESS) -> np.ndarray:
    """Reference distribution: learned from LOL high-light images if available,
    otherwise a smooth bell-shaped distribution centred on the target brightness."""
    if REF_CDF_PATH.exists():
        return np.load(REF_CDF_PATH)
    x = np.arange(256)
    pdf = np.exp(-0.5 * ((x - 255 * brightness) / 55.0) ** 2)
    cdf = np.cumsum(pdf) / pdf.sum()
    return np.stack([cdf] * 3, axis=1)


def match_histogram(src: np.ndarray, ref_cdf: np.ndarray, strength: float = 1.0) -> np.ndarray:
    out = np.empty_like(src)
    levels = np.arange(256)
    src_cdf = channel_cdf(src)
    for c in range(3):
        lut = np.interp(src_cdf[:, c], ref_cdf[:, c], levels)
        matched = lut[src[..., c]]
        blended = strength * matched + (1 - strength) * src[..., c]
        out[..., c] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def match_lightness(src: np.ndarray, ref_cdf: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Histogram matching on LAB lightness only, so the image keeps its own colours.
    Used with the default reference, which describes exposure but carries no colour information."""
    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    src_cdf = np.cumsum(np.bincount(L.ravel(), minlength=256)) / L.size
    lut = np.interp(src_cdf, ref_cdf.mean(axis=1), np.arange(256))
    # Cap the contrast gain so flat areas (sky, walls) do not get leftover noise stretched into blotches,
    # then restore the mean brightness the full matching aimed for
    steps = np.minimum(np.diff(lut), MAX_MATCH_SLOPE)
    capped = np.concatenate([[lut[0]], lut[0] + np.cumsum(steps)])
    capped += lut[L].mean() - capped[L].mean()
    capped = np.maximum(capped, np.arange(256))  # only brighten; never pull white paper or lights down to grey
    lab[..., 0] = np.clip(strength * capped[L] + (1 - strength) * L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def luminance_hist(img: np.ndarray, bins: int = 64):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h = np.histogram(gray, bins=bins, range=(0, 256))[0]
    return (h / h.sum()).round(5).tolist()


def stats(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {"brightness": round(float(gray.mean()), 1), "contrast": round(float(gray.std()), 1)}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return {
        "psnr": round(float(peak_signal_noise_ratio(b, a, data_range=255)), 2),
        "ssim": round(float(structural_similarity(b, a, channel_axis=2, data_range=255)), 4),
    }


def run_pipeline(img, ref_img=None, truth_img=None, use_matching=True, strength=1.0,
                 brightness=DEFAULT_BRIGHTNESS):
    brightness = float(np.clip(brightness, 0.3, 0.8))
    img = limit_size(img)
    using_model = get_model() is not None
    stage1 = keep_color_balance(run_autoencoder(img) if using_model else demo_stage(img, brightness), img)

    final = stage1
    if use_matching:
        if ref_img is not None:
            final = match_histogram(stage1, channel_cdf(ref_img), strength)
        else:
            final = match_lightness(stage1, default_reference_cdf(brightness), strength)

    result = {
        "mode": "autoencoder" if using_model else "demo",
        "images": {"original": img, "autoencoder": stage1, "final": final},
        "hist": {"original": luminance_hist(img), "final": luminance_hist(final)},
        "stats": {"original": stats(img), "autoencoder": stats(stage1), "final": stats(final)},
    }
    if truth_img is not None:
        truth = limit_size(truth_img)
        result["metrics"] = {
            "original": compare(img, truth),
            "autoencoder": compare(stage1, truth),
            "final": compare(final, truth),
        }
    return result
