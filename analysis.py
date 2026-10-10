import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = 'data/widerface_homework'
os.makedirs('results', exist_ok=True)


def parse_label(txt_path):
    img_dir = os.path.join(os.path.dirname(txt_path), 'images')
    imgs, anns = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                imgs.append(os.path.join(img_dir, line[1:].strip()))
                anns.append([])
            else:
                anns[-1].append([float(x) for x in line.split()])
    anns = [np.array(a) if len(a) else np.zeros((0, 20)) for a in anns]
    return imgs, anns


def get_sizes(imgs):
    sizes = []
    for path in imgs:
        with Image.open(path) as image:
            sizes.append(image.size)
    return np.array(sizes)


def basic_stats(anns, split):
    boxes = np.concatenate([a[:, :4] for a in anns if len(a)], axis=0)
    w, h = boxes[:, 2], boxes[:, 3]
    print(f'[{split}] images: {len(anns)}, GT: {len(boxes)}')
    print(f'[{split}] mean w: {w.mean():.2f}, mean h: {h.mean():.2f}, mean area: {(w * h).mean():.2f}')
    print(f'[{split}] degenerate boxes (w<=0 or h<=0): {((w <= 0) | (h <= 0)).sum()}')


def face_distribution(anns, sizes, split):
    cx, cy, cnt = [], [], []
    for a, (W, H) in zip(anns, sizes):
        cnt.append(len(a))
        if len(a) == 0:
            continue
        cx.append((a[:, 0] + a[:, 2] / 2) / W)
        cy.append((a[:, 1] + a[:, 3] / 2) / H)
    cx, cy = np.concatenate(cx), np.concatenate(cy)

    center = ((cx > 0.25) & (cx < 0.75) & (cy > 0.25) & (cy < 0.75)).mean()
    print(f'[{split}] fraction of centers in the middle 50%x50%: {center:.3f} (uniform: 0.25)')
    print(f'[{split}] faces per image: mean {np.mean(cnt):.2f}, median {np.median(cnt)}, max {np.max(cnt)}')

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    hh = ax[0].hist2d(cx, cy, bins=20, range=[[0, 1], [0, 1]])
    ax[0].invert_yaxis()
    ax[0].set_title(f'{split}: face center distribution')
    ax[0].set_xlabel('Center x / image width')
    ax[0].set_ylabel('Center y / image height')
    plt.colorbar(hh[3], ax=ax[0], label='Number of faces')
    ax[1].hist(cnt, bins=50)
    ax[1].set_yscale('log')
    ax[1].set_title(f'{split}: faces per image')
    ax[1].set_xlabel('Faces per image')
    ax[1].set_ylabel('Number of images (log scale)')
    plt.tight_layout()
    plt.savefig(f'results/dist_{split}.png', dpi=150)
    plt.close()


def out_of_range(anns, sizes, split):
    bad_boxes, bad_imgs, total = 0, 0, 0
    for a, (W, H) in zip(anns, sizes):
        if len(a) == 0:
            continue
        total += len(a)
        bad = (a[:, 0] < 0) | (a[:, 1] < 0) | (a[:, 0] + a[:, 2] > W) | (a[:, 1] + a[:, 3] > H)
        bad_boxes += bad.sum()
        bad_imgs += int(bad.any())
    print(f'[{split}] out-of-range GT: {bad_boxes}/{total} ({bad_boxes / total:.4%}), images affected: {bad_imgs}')


def overlap_ratio(anns, split):
    total, overlapped = 0, 0
    for a in anns:
        n = len(a)
        total += n
        if n < 2:
            continue
        x1, y1 = a[:, 0], a[:, 1]
        x2, y2 = x1 + a[:, 2], y1 + a[:, 3]
        iw = np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None])
        ih = np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        np.fill_diagonal(inter, 0)
        overlapped += (inter > 0).any(axis=1).sum()
    print(f'[{split}] overlapped GT (IoU>0): {overlapped}/{total} ({overlapped / total:.2%})')

def visualize(imgs, anns, n=4):
    import cv2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate(axes.ravel()[:n]):
        img = cv2.imread(imgs[i])
        for a in anns[i]:
            x1, y1 = int(a[0]), int(a[1])
            x2, y2 = int(a[0] + a[2]), int(a[1] + a[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for k in range(5):
                lx, ly = a[4 + 3 * k], a[5 + 3 * k]
                if lx >= 0:
                    cv2.circle(img, (int(lx), int(ly)), 3, (0, 0, 255), -1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('results/vis_first4.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    print('start')
    for split in ['train', 'val']:
        path = f'{ROOT}/{split}/label.txt'
        if not os.path.exists(path):
            print(f'[{split}] label.txt not found, skipped')
            continue
        imgs, anns = parse_label(path)
        sizes = get_sizes(imgs)
        basic_stats(anns, split)
        face_distribution(anns, sizes, split)
        out_of_range(anns, sizes, split)
        overlap_ratio(anns, split)
        if split == 'train':
            visualize(imgs, anns)
        print()
