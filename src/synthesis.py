#!/usr/bin/env python3
import random
import cv2
import numpy as np
from PIL import Image

def rgb(path, size=256):
    return np.asarray(Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR), np.float32) / 255

def gray(path, size=256):
    return np.asarray(Image.open(path).convert('L').resize((size, size), Image.BILINEAR), np.float32) / 255

def choose_semantic(a):
    (cols, cnt) = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
    ids = [i for (i, (c, n)) in enumerate(zip(cols, cnt)) if tuple(c) != (0, 0, 0) and n > 150]
    if not ids:
        return None
    c = cols[random.choice(ids)]
    return np.all(a == c, axis=2).astype(np.float32)

def arbitrary_mask():
    m = np.zeros((256, 256), np.float32)
    for _ in range(3):
        typ = random.randrange(2)
        x = random.randint(10, 210)
        y = random.randint(10, 210)
        w = random.randint(20, 70)
        h = random.randint(20, 70)
        if typ:
            cv2.ellipse(m, (x, y), (w // 2, h // 2), random.randint(0, 180), 0, 360, 1, -1)
        else:
            cv2.rectangle(m, (x, y), (min(255, x + w), min(255, y + h)), 1, -1)
    return m

def augment_edge(edge):
    if random.random() < 0.5:
        edge = np.fliplr(edge).copy()
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.85, 1.15)
    mat = cv2.getRotationMatrix2D((127.5, 127.5), angle, scale)
    mat[:, 2] += [random.uniform(-16, 16), random.uniform(-16, 16)]
    return cv2.warpAffine(edge, mat, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1)

def manipulate(edge, candidate, mask, op):
    modified = edge.copy()
    selected = mask > 0
    if op == 0:
        modified[selected] = 1
    elif op == 1:
        modified[selected] = candidate[selected]
    else:
        modified[selected] = np.clip(candidate[selected] + edge[selected] - 1, 0, 1)
    actual = (np.abs(modified - edge) > 0.03).astype(np.float32) * mask
    if actual.sum() < 32:
        actual = mask
    return (modified, actual)
