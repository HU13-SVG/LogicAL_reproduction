#!/usr/bin/env python3
import argparse
import csv
import random
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'deepsim'))
from deepsim_models import networks
from synthesis import arbitrary_mask, augment_edge, choose_semantic, gray, manipulate
from common import CATEGORIES
OPS = ('remove', 'replace', 'merge')

class GeneratePairs(Dataset):

    def __init__(self, data_root, edges_root, mask_root):
        self.edges_root = Path(edges_root)
        self.mask_root = Path(mask_root)
        self.items = []
        self.bycat = {}
        for category in CATEGORIES:
            paths = sorted((Path(data_root) / category / 'train' / 'good').glob('*'))
            paths = [p for p in paths if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            self.bycat[category] = paths
            self.items.extend(((category, p) for p in paths))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        seed = 20260929 + index
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        (category, source) = self.items[index]
        edge = 1 - gray(self.edges_root / category / f'{source.stem}.png')
        candidate = random.choice(self.bycat[category])
        candidate_edge = 1 - gray(self.edges_root / category / f'{candidate.stem}.png')
        semantic = np.asarray(Image.open(self.mask_root / category / f'{source.stem}.png').convert('RGB').resize((256, 256), Image.NEAREST))
        region_kind = 'semantic' if random.random() < 0.5 else 'arbitrary'
        mask = choose_semantic(semantic) if region_kind == 'semantic' else arbitrary_mask()
        if mask is None or mask.sum() < 64:
            region_kind = 'arbitrary'
            mask = arbitrary_mask()
        kernel = random.choice((3, 5, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
        candidate_edge = augment_edge(candidate_edge)
        op_id = random.randrange(3)
        (modified, _) = manipulate(edge, candidate_edge, mask, op_id)
        tensor = torch.from_numpy(modified[None].copy()).repeat(3, 1, 1) * 2 - 1
        return (tensor, category, source.stem, candidate.stem, OPS[op_id], region_kind)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(ROOT / 'data' / 'mvtec_loco'))
    ap.add_argument('--edges-root', default=str(ROOT / 'work' / 'pidinet_edges'))
    ap.add_argument('--mask-root', default=str(ROOT / 'data' / 'csad_component_masks'))
    ap.add_argument('--checkpoint', default=str(ROOT / 'work' / 'deepsim_unified' / 'latest_net_G.pth'))
    ap.add_argument('--output', default=str(ROOT / 'work' / 'qualitative'))
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    random.seed(20260929)
    np.random.seed(20260929)
    torch.manual_seed(20260929)
    checkpoint = Path(a.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    generator = networks.define_G(3, 3, 64, 'global', 4, 9, 1, 3, 'instance', [])
    generator.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    generator.cuda().eval()
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
    dataset = GeneratePairs(a.data_root, a.edges_root, a.mask_root)
    loader = DataLoader(dataset, a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    (rows, done) = ([], 0)
    with torch.no_grad():
        for (edge, categories, stems, candidates, operations, region_kinds) in loader:
            generated = ((generator(edge.cuda(non_blocking=True)) + 1) / 2).clamp(0, 1)
            generated = (generated.cpu().numpy().transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)
            for (image, category, stem, candidate, operation, region_kind) in zip(generated, categories, stems, candidates, operations, region_kinds):
                category_dir = output / category
                category_dir.mkdir(exist_ok=True)
                filename = f'{stem}.png'
                Image.fromarray(image).save(category_dir / filename)
                rows.append((category, filename, f'{stem}.png', f'{candidate}.png', operation, region_kind))
                done += 1
            print(f'generated {done}/{len(dataset)}', flush=True)
    with open(output / 'metadata.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('category', 'generated_file', 'normal_source', 'candidate_source', 'edge_operation', 'region_selection'))
        writer.writerows(rows)
    print(f'GENERATION_DONE total={done} output={output}')
if __name__ == '__main__':
    main()
