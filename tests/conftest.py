import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import enhance  # noqa: E402


@pytest.fixture
def dark_image():
    """Small noisy, dark RGB image with some structure."""
    rng = np.random.default_rng(0)
    h, w = 96, 128
    grad = np.linspace(5, 60, w)[None, :, None] * np.ones((h, 1, 3))
    grad[20:50, 30:70] += 30  # a brighter object
    return np.clip(grad + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)


@pytest.fixture
def demo_mode(monkeypatch):
    monkeypatch.setattr(enhance, "_model", None)
    monkeypatch.setattr(enhance, "_model_checked", True)


@pytest.fixture
def model_mode(monkeypatch):
    """Trained-model mode with a randomly initialised autoencoder (no weights needed)."""
    torch = pytest.importorskip("torch")
    from model import AutoEncoder

    torch.manual_seed(0)
    monkeypatch.setattr(enhance, "_model", AutoEncoder().eval())
    monkeypatch.setattr(enhance, "_model_checked", True)


@pytest.fixture
def learned_reference(monkeypatch, tmp_path):
    """A stand-in for weights/reference_cdf.npy: a bright-ish, skewed distribution."""
    x = np.arange(256)
    pdf = np.exp(-0.5 * ((x - 170) / 40.0) ** 2) + 0.3 * np.exp(-0.5 * ((x - 60) / 20.0) ** 2)
    cdf = np.cumsum(pdf) / pdf.sum()
    path = tmp_path / "reference_cdf.npy"
    np.save(path, np.stack([cdf] * 3, axis=1))
    monkeypatch.setattr(enhance, "REF_CDF_PATH", path)
