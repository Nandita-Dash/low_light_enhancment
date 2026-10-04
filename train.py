"""Train the autoencoder on the LOL dataset.

Expected folder layout (standard LOL download):
  LOL/our485/low/*.png   LOL/our485/high/*.png
  LOL/eval15/low/*.png   LOL/eval15/high/*.png

Run:  python train.py --data path/to/LOL --epochs 100
Outputs: weights/autoencoder.pt and weights/reference_cdf.npy
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from enhance import channel_cdf
from model import AutoEncoder

SIZE = 256


def load(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


class LOL(Dataset):
    def __init__(self, root, train=True):
        self.low = sorted((Path(root) / "low").glob("*.png"))
        self.high = [Path(root) / "high" / p.name for p in self.low]
        self.train = train
        self.cache = [(load(l), load(h)) for l, h in zip(self.low, self.high)]

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        lo, hi = self.cache[i]
        if self.train and np.random.rand() < 0.5:
            lo, hi = lo[:, ::-1], hi[:, ::-1]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
        return t(lo), t(hi)


def ssim(a, b, win=7):
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = F.avg_pool2d(a, win, 1, win // 2), F.avg_pool2d(b, win, 1, win // 2)
    va = F.avg_pool2d(a * a, win, 1, win // 2) - mu_a**2
    vb = F.avg_pool2d(b * b, win, 1, win // 2) - mu_b**2
    cov = F.avg_pool2d(a * b, win, 1, win // 2) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return s.mean()


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    train_ds, val_ds = LOL(root / "our485", True), LOL(root / "eval15", False)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, 1)

    Path("weights").mkdir(exist_ok=True)
    # Reference histogram: average distribution of the well-lit training targets
    cdfs = [channel_cdf(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            for p in train_ds.high]
    np.save("weights/reference_cdf.npy", np.mean(cdfs, axis=0))

    model = AutoEncoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for lo, hi in train_dl:
            lo, hi = lo.to(dev), hi.to(dev)
            out = model(lo)
            loss = F.l1_loss(out, hi) + 0.2 * (1 - ssim(out, hi))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            val = np.mean([psnr(model(l.to(dev)), h.to(dev)) for l, h in val_dl])
        print(f"epoch {ep:3d}  loss {total / len(train_dl):.4f}  val PSNR {val:.2f} dB")
        if val > best:
            best = val
            torch.save(model.state_dict(), "weights/autoencoder.pt")
    print(f"Best val PSNR {best:.2f} dB. Weights saved to weights/autoencoder.pt")


if __name__ == "__main__":
    main()
