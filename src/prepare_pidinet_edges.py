#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
CATS = ('breakfast_box', 'juice_bottle', 'pushpins', 'splicing_connectors', 'screw_bag')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'pidinet'))
import models

class Images(Dataset):

    def __init__(self, data_root):
        self.items = []
        for c in CATS:
            for p in sorted((Path(data_root) / c / 'train' / 'good').glob('*')):
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    self.items.append((c, p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        (c, p) = self.items[i]
        a = np.asarray(Image.open(p).convert('RGB').resize((256, 256), Image.BILINEAR), np.float32) / 255
        a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return (torch.from_numpy(a.transpose(2, 0, 1)), c, p.stem)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(ROOT / 'data' / 'mvtec_loco'))
    ap.add_argument('--output', default=str(ROOT / 'work' / 'pidinet_edges'))
    ap.add_argument('--checkpoint', default=str(ROOT / 'checkpoints' / 'table5_pidinet.pth'))
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    net = models.pidinet(SimpleNamespace(config='carv4', sa=True, dil=True))
    ck = torch.load(a.checkpoint, map_location='cpu')['state_dict']
    net.load_state_dict({k.replace('module.', '', 1): v for (k, v) in ck.items()}, strict=True)
    net.cuda().eval()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = Images(a.data_root)
    dl = DataLoader(ds, a.batch_size, num_workers=a.workers, pin_memory=True)
    done = 0
    with torch.no_grad():
        for (x, cats, stems) in dl:
            edge = net(x.cuda(non_blocking=True))[0].cpu().numpy()[:, 0]
            for (e, c, s) in zip(edge, cats, stems):
                d = out / c
                d.mkdir(exist_ok=True)
                Image.fromarray(np.uint8(np.clip(e, 0, 1) * 255)).save(d / f'{s}.png')
                done += 1
            print(f'edges {done}/{len(ds)}', flush=True)
    print(f'PIDINET_EDGES_DONE={done}')
if __name__ == '__main__':
    main()
