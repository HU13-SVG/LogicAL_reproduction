#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'deepsim'))
from deepsim_models import networks
from common import CATEGORIES, LogicALTest, annotator_robust_edge_loss, evaluate, focal_loss, jnd, ssim_ground_truth, ssim_loss, write_history
from logical_net import LogicALNet
from synthesis import arbitrary_mask, augment_edge, choose_semantic, gray, manipulate, rgb

class OnlineTrain(Dataset):

    def __init__(self, data_root, edges_root, mask_root, size=256):
        self.size = size
        self.edges_root = Path(edges_root)
        self.mask_root = Path(mask_root)
        self.items = []
        for c in CATEGORIES:
            ps = sorted((Path(data_root) / c / 'train' / 'good').glob('*'))
            for p in ps:
                self.items.append((c, p))
        self.bycat = {c: [x for x in self.items if x[0] == c] for c in CATEGORIES}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        (c, p) = self.items[i]
        normal = rgb(p, self.size)
        edge_target = gray(self.edges_root / c / (p.stem + '.png'), self.size)
        edge = 1 - edge_target
        (_, q) = random.choice(self.bycat[c])
        cand = 1 - gray(self.edges_root / c / (q.stem + '.png'), self.size)
        sm = np.asarray(Image.open(self.mask_root / c / (p.stem + '.png')).convert('RGB').resize((self.size, self.size), Image.NEAREST))
        mask = choose_semantic(sm) if random.random() < 0.5 else arbitrary_mask()
        if mask is None or mask.sum() < 64:
            mask = arbitrary_mask()
        k = random.choice([3, 5, 7])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        cand = augment_edge(cand)
        op = random.randrange(3)
        (modified, actual) = manipulate(edge, cand, mask, op)
        return (torch.from_numpy(normal.transpose(2, 0, 1).copy()), torch.from_numpy(edge_target[None].copy()), torch.from_numpy(modified[None].copy()), torch.from_numpy(actual[None].copy()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(ROOT / 'data' / 'mvtec_loco'))
    ap.add_argument('--edges-root', default=str(ROOT / 'work' / 'pidinet_edges'))
    ap.add_argument('--mask-root', default=str(ROOT / 'data' / 'csad_component_masks'))
    ap.add_argument('--deepsim-checkpoint', default=str(ROOT / 'work' / 'deepsim_unified' / 'latest_net_G.pth'))
    ap.add_argument('--output', default=str(ROOT / 'work' / 'localization'))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=12)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--eval-workers', type=int, default=8)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    gp = Path(a.deepsim_checkpoint)
    if not gp.exists():
        raise RuntimeError(f'Fresh unified DeepSIM checkpoint missing: {gp}')
    gen = networks.define_G(3, 3, 64, 'global', 4, 9, 1, 3, 'instance', [])
    gen.load_state_dict(torch.load(gp, map_location='cpu'))
    gen.cuda().eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    net = LogicALNet().cuda()
    raw = net
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        net = nn.DataParallel(net)
    opt = torch.optim.Adam(net.parameters(), lr=0.0001)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [200, 260], gamma=0.1)
    tr = DataLoader(OnlineTrain(a.data_root, a.edges_root, a.mask_root), a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    te = DataLoader(LogicALTest(str(a.data_root), 256), a.batch_size, num_workers=a.eval_workers, pin_memory=True)
    hist = []
    start = time.time()
    print(f'ONLINE_LOGICAL_FROM_SCRATCH train={len(tr.dataset)} test={len(te.dataset)} gpus={torch.cuda.device_count()}', flush=True)
    for ep in range(1, a.epochs + 1):
        net.train()
        total = 0.0
        for (target, edge, mod, mask) in tr:
            (target, edge, mod, mask) = [z.cuda(non_blocking=True) for z in (target, edge, mod, mask)]
            with torch.no_grad():
                synth = (gen(mod.repeat(1, 3, 1, 1) * 2 - 1) + 1) / 2
                normal_edge = (1 - edge).repeat(1, 3, 1, 1) * 2 - 1
                generated_normal = (gen(normal_edge) + 1) / 2
                mask = ssim_ground_truth(generated_normal, synth)
            (rec, jnd_pred, edge_pred, seg) = net(synth)
            loss = F.mse_loss(rec, target) + ssim_loss(rec, target) + F.mse_loss(jnd_pred, jnd(target)) + ssim_loss(jnd_pred, jnd(target)) + annotator_robust_edge_loss(edge_pred, edge) + focal_loss(seg, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        print(f'epoch={ep:03d} loss={total / len(tr):.6f} lr={sch.get_last_lr()[0]:.2e} elapsed_h={(time.time() - start) / 3600:.2f}', flush=True)
        if ep % a.eval_every == 0 or ep == a.epochs:
            z = evaluate(net, te, torch.device('cuda'), out, ep)
            hist.append(z)
            print('EVAL ' + json.dumps(z), flush=True)
            torch.save({'epoch': ep, 'model': raw.state_dict(), 'result': z}, out / 'latest.pth')
            write_history(out / 'results.json', hist)
    print('LOGICAL_FROM_SCRATCH_DONE')
if __name__ == '__main__':
    main()
