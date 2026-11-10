#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
CATEGORIES = ('breakfast_box', 'juice_bottle', 'pushpins', 'splicing_connectors', 'screw_bag')

def read_rgb(path, size=256):
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0

class LogicALTest(Dataset):

    def __init__(self, data_root, size=256):
        (self.size, self.items) = (size, [])
        for cat in CATEGORIES:
            base = Path(data_root) / cat / 'test'
            for (label, sub) in [(0, 'good'), (1, 'logical_anomalies')]:
                for p in sorted((base / sub).glob('*')):
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        self.items.append((cat, label, p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        (cat, label, path) = self.items[i]
        x = torch.from_numpy(read_rgb(path, self.size).transpose(2, 0, 1).copy())
        return (x, label, cat, str(path))

def auc(labels, scores):
    (labels, scores) = (np.asarray(labels), np.asarray(scores))
    (pos, neg) = (labels == 1, labels == 0)
    if not pos.any() or not neg.any():
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        ii = np.flatnonzero(scores == value)
        if len(ii) > 1:
            ranks[ii] = ranks[ii].mean()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))

def ssim_loss(x, y):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sx = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sy = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sxy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim = (2 * mu_x * mu_y + 0.01 ** 2) * (2 * sxy + 0.03 ** 2) / ((mu_x ** 2 + mu_y ** 2 + 0.01 ** 2) * (sx + sy + 0.03 ** 2) + 1e-08)
    return 1 - ssim.mean()

def focal_loss(logits, target, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    alpha = 0.75 * target + 0.25 * (1 - target)
    return (alpha * (1 - pt).pow(gamma) * bce).mean()

def jnd(x):
    lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    local = F.avg_pool2d(lum, 5, 1, 2)
    contrast = F.avg_pool2d(torch.abs(lum - local), 5, 1, 2)
    return torch.clamp(0.6 * local + 1.8 * contrast, 0, 1)

def ssim_ground_truth(a, b):
    mux = F.avg_pool2d(a, 11, 1, 5)
    muy = F.avg_pool2d(b, 11, 1, 5)
    vx = F.avg_pool2d(a * a, 11, 1, 5) - mux * mux
    vy = F.avg_pool2d(b * b, 11, 1, 5) - muy * muy
    vxy = F.avg_pool2d(a * b, 11, 1, 5) - mux * muy
    s = (2 * mux * muy + 0.01 ** 2) * (2 * vxy + 0.03 ** 2) / ((mux * mux + muy * muy + 0.01 ** 2) * (vx + vy + 0.03 ** 2) + 1e-08)
    return torch.clamp(1 - s.mean(1, keepdim=True), 0, 1)

def annotator_robust_edge_loss(pred, target, eta=0.3, beta=1.1):
    pred = pred.clamp(1e-06, 1 - 1e-06)
    neg = target <= 1 / 255
    pos = target >= eta
    denom = (neg.sum() + pos.sum()).clamp_min(1)
    return -(torch.log1p(-pred)[neg].sum() + beta * torch.log(pred)[pos].sum()) / denom

@torch.no_grad()
def evaluate(model, loader, device, out_dir, epoch):
    model.eval()
    rows = []
    for (x, y, c, p) in loader:
        x = x.to(device)
        (_, _, _, logit) = model(x)
        prob = torch.sigmoid(logit)
        s = prob.flatten(1).topk(prob[0].numel() // 100, 1).values.mean(1).cpu().numpy()
        for (cc, yy, ss, pp) in zip(c, y.numpy(), s, p):
            rows.append((cc, int(yy), float(ss), pp))
    per = {c: auc([r[1] for r in rows if r[0] == c], [r[2] for r in rows if r[0] == c]) for c in CATEGORIES}
    z = {'epoch': epoch, 'logical_image_auroc_macro': float(np.mean(list(per.values()))), 'per_category': per}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'scores_epoch_{epoch:03d}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'label', 'score', 'path'])
        w.writerows(rows)
    return z

def write_history(path, history):
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
