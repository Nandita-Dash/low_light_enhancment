"""Build the example images shown in the app and the README.

Takes well-lit sample photos bundled with scikit-image (public domain / CC0) and darkens them like an
underexposed, noisy capture. Writes, per example, to static/examples/:
  <name>.jpg        low-light input (the app loads this when the example is clicked)
  <name>_truth.jpg  the original, used as ground truth for PSNR / SSIM
  <name>_compare.jpg  low-light input | enhanced result, for the README

Run:  python make_examples.py
"""
from pathlib import Path

import cv2
import numpy as np
from skimage import data

import enhance

OUT = Path(__file__).parent / "static" / "examples"
EXAMPLES = ["coffee", "chelsea", "astronaut"]


def darken(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """Underexpose in linear light (to 12%), then add shot and read noise."""
    rng = np.random.default_rng(seed)
    lin = (img / 255.0) ** 2.2 * 0.12
    lin = rng.poisson(lin * 600) / 600 + rng.normal(0, 0.002, lin.shape)
    return (np.clip(lin, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def save(path: Path, img_rgb: np.ndarray):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLES:
        truth = enhance.limit_size(getattr(data, name)()[..., :3].copy())
        dark = darken(truth)
        save(OUT / f"{name}.jpg", dark)
        save(OUT / f"{name}_truth.jpg", truth)
        res = enhance.run_pipeline(dark, truth_img=truth)
        gap = np.zeros((dark.shape[0], 8, 3), np.uint8)
        save(OUT / f"{name}_compare.jpg", np.hstack([dark, gap, res["images"]["final"]]))
        m = res["metrics"]
        print(f"{name:10s} PSNR {m['original']['psnr']:5.2f} -> {m['final']['psnr']:5.2f} dB   "
              f"SSIM {m['original']['ssim']:.3f} -> {m['final']['ssim']:.3f}")


if __name__ == "__main__":
    main()
