#!/usr/bin/env python3
import argparse
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'deepsim'))
from deepsim_models import networks
from data.base_dataset import get_params, get_transform
CATS = ('breakfast_box', 'juice_bottle', 'pushpins', 'splicing_connectors', 'screw_bag')

class Pairs(Dataset):

    def __init__(self, data_root, edges_root, no_flip=False):
        self.items = []
        for c in CATS:
            for p in sorted((Path(data_root) / c / 'train' / 'good').glob('*')):
                e = Path(edges_root) / c / (p.stem + '.png')
                if e.exists():
                    self.items.append((p, e))
        self.opt = SimpleNamespace(fineSize=256, affine_transforms={}, affine_aug='none', tps_percent=0.99, tps_aug=1, tps_points_per_dim=3, cutmix_aug=0, canny_aug=0, no_flip=no_flip, resize_or_crop='none', isTrain=True, canny_color=1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        (ip, ep) = self.items[i]
        rgb = Image.open(ip).convert('RGB').resize((256, 256), Image.BICUBIC)
        edge_np = 255 - np.asarray(Image.open(ep).convert('L').resize((256, 256), Image.BILINEAR))
        edge = Image.fromarray(edge_np).convert('RGB')
        params = get_params(self.opt, rgb.size, rgb)
        x = get_transform(self.opt, params, normalize=True, is_primitive=True, is_edges=False)(edge)
        y = get_transform(self.opt, params, normalize=True, is_primitive=False, is_edges=False)(rgb)
        return (x, y)

def gan_loss(pred, real):
    loss = 0.0
    for scale in pred:
        p = scale[-1] if isinstance(scale, list) else scale
        loss = loss + torch.mean((p - (1.0 if real else 0.0)) ** 2)
    return loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(ROOT / 'data' / 'mvtec_loco'))
    ap.add_argument('--edges-root', default=str(ROOT / 'work' / 'pidinet_edges'))
    ap.add_argument('--output', default=str(ROOT / 'work' / 'deepsim_unified'))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=56)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--save-every', type=int, default=50, help='Checkpoint interval; 0 saves only the final model.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-flip', action='store_true', help='Disable the official DeepSIM random horizontal flip.')
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = Pairs(a.data_root, a.edges_root, no_flip=a.no_flip)
    dl = DataLoader(ds, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    g = networks.define_G(3, 3, 64, 'global', 4, 9, 1, 3, 'instance', []).cuda()
    d = networks.define_D(6, 64, 3, 'instance', False, 2, True, []).cuda()
    if torch.cuda.device_count() > 1:
        g = nn.DataParallel(g)
        d = nn.DataParallel(d)
    rawg = g.module if isinstance(g, nn.DataParallel) else g
    rawd = d.module if isinstance(d, nn.DataParallel) else d
    og = torch.optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
    l1 = nn.L1Loss()
    vgg = networks.Vgg19().cuda().eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    if torch.cuda.device_count() > 1:
        vgg = nn.DataParallel(vgg)
    vgg_weights = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0)
    start = time.time()
    print(f'DEEPSIM_UNIFIED_FROM_SCRATCH images={len(ds)} batch={a.batch_size} gpus={torch.cuda.device_count()} loss=GAN+FM+VGG', flush=True)
    for epoch in range(1, a.epochs + 1):
        sg = sd = 0.0
        lr = 0.0002 if epoch <= 150 else 0.0002 * (a.epochs - epoch) / 150
        for q in (og, od):
            for pg in q.param_groups:
                pg['lr'] = max(lr, 0.0)
        for (edge, rgb) in dl:
            edge = edge.cuda(non_blocking=True)
            rgb = rgb.cuda(non_blocking=True)
            fake = g(edge)
            od.zero_grad(set_to_none=True)
            pr = d(torch.cat((edge, rgb), 1))
            pf = d(torch.cat((edge, fake.detach()), 1))
            ld = 0.5 * (gan_loss(pr, True) + gan_loss(pf, False))
            ld.backward()
            od.step()
            og.zero_grad(set_to_none=True)
            pf = d(torch.cat((edge, fake), 1))
            pr = d(torch.cat((edge, rgb), 1))
            fake_vgg = vgg(fake)
            real_vgg = vgg(rgb)
            perceptual = sum((w * l1(x, y.detach()) for (w, x, y) in zip(vgg_weights, fake_vgg, real_vgg)))
            lg = gan_loss(pf, True) + 10 * perceptual
            fm = 0.0
            for (x, y) in zip(pf, pr):
                for (xx, yy) in zip(x[:-1], y[:-1]):
                    fm = fm + l1(xx, yy.detach())
            lg = lg + 5 * fm
            lg.backward()
            og.step()
            sg += lg.item()
            sd += ld.item()
        print(f'epoch={epoch:03d} G={sg / len(dl):.5f} D={sd / len(dl):.5f} lr={lr:.2e} elapsed_h={(time.time() - start) / 3600:.2f}', flush=True)
        save_now = a.save_every > 0 and epoch % a.save_every == 0 or epoch == a.epochs
        if save_now:
            torch.save({'epoch': epoch, 'generator': rawg.state_dict(), 'discriminator': rawd.state_dict(), 'opt_g': og.state_dict(), 'opt_d': od.state_dict()}, out / f'epoch_{epoch:03d}.pth')
            torch.save(rawg.state_dict(), out / 'latest_net_G.pth')
    print('DEEPSIM_TRAINING_DONE')
if __name__ == '__main__':
    main()
