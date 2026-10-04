"""Build the example images shown in the app and the README from the LOL test set.

LOL (Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018) pairs genuine low-light
captures with normal-exposure shots of the same scenes. Examples come from eval15, which train.py does not
train on. Writes, per example, to static/examples/:
  <name>.png          low-light capture (the app loads this when the example is clicked)
  <name>_truth.png    normal-exposure shot, used as ground truth for PSNR / SSIM
  <name>_compare.jpg  low-light capture | enhanced result, for the README
It also prints the scores on every eval15 pair, so the chosen examples can be read against the average.

Run:  python make_examples.py --data path/to/LOL
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import enhance

OUT = Path(__file__).parent / "static" / "examples"
EXAMPLES = {"79": "kitchen", "111": "window", "669": "bowling"}  # eval15 file stem -> example name


def load(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def save(path: Path, img_rgb: np.ndarray, *params):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), list(params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="LOL folder containing eval15/low and eval15/high")
    args = ap.parse_args()
    eval_dir = Path(args.data) / "eval15"
    OUT.mkdir(parents=True, exist_ok=True)

    scores = []
    for low_path in sorted((eval_dir / "low").glob("*.png"), key=lambda p: int(p.stem)):
        low, high = load(low_path), load(eval_dir / "high" / low_path.name)
        res = enhance.run_pipeline(low, truth_img=high)
        m = res["metrics"]
        scores.append([m["original"]["psnr"], m["final"]["psnr"], m["original"]["ssim"], m["final"]["ssim"]])
        name = EXAMPLES.get(low_path.stem)
        print(f"{low_path.name:8s} PSNR {scores[-1][0]:5.2f} -> {scores[-1][1]:5.2f} dB   "
              f"SSIM {scores[-1][2]:.3f} -> {scores[-1][3]:.3f}" + (f"   example: {name}" if name else ""))
        if name:
            save(OUT / f"{name}.png", low)
            save(OUT / f"{name}_truth.png", high)
            gap = np.zeros((low.shape[0], 8, 3), np.uint8)
            save(OUT / f"{name}_compare.jpg", np.hstack([low, gap, res["images"]["final"]]), cv2.IMWRITE_JPEG_QUALITY, 90)
    s = np.mean(scores, axis=0)
    print(f"mean of {len(scores)}: PSNR {s[0]:.2f} -> {s[1]:.2f} dB   SSIM {s[2]:.3f} -> {s[3]:.3f}")


if __name__ == "__main__":
    main()
