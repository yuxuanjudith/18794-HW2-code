from __future__ import print_function
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import argparse
import math
import json
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import cfg
from loss_and_anchor import anchor
from detector.mydetector import mydetector
from utils.nms import nms
from utils.box_utils import decode, decode_landm
from utils.misc import load_model

parser = argparse.ArgumentParser(description='18794 detection')
parser.add_argument('--ckpt', default='./weights/weights_cn20_2gpuResnet50_Final.pth', type=str)
parser.add_argument('--cpu', action="store_true", default=False)
parser.add_argument('--dataset_folder', default='data/widerface_homework/val/images/', type=str)
parser.add_argument('--confidence_threshold', default=0.1, type=float)
parser.add_argument('--top_k', default=5000, type=int)
parser.add_argument('--nms_threshold', default=0.4, type=float)
parser.add_argument('--keep_top_k', default=750, type=int)
parser.add_argument('--vis_thres', default=0.5, type=float)
parser.add_argument('--num_vis', default=6, type=int)
parser.add_argument('--output', default='results/test_results.png')
args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    if args.num_vis < 1:
        parser.error('--num_vis must be positive')

    net = mydetector(cfg=cfg, phase='test')
    net = load_model(net, args.ckpt, args.cpu)
    net.eval()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    testset_folder = args.dataset_folder
    label_path = os.path.join(os.path.dirname(testset_folder.rstrip('/\\')), 'label.txt')
    with open(label_path, 'r') as fr:
        test_dataset = [l[1:].strip() for l in fr if l.startswith('#')]

    results = []
    counts = []
    for i, img_name in enumerate(test_dataset[:args.num_vis]):
        image_path = os.path.join(testset_folder, img_name)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            raise FileNotFoundError(image_path)
        img = np.float32(img_raw)

        target_size = 1200
        max_size = 1600
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = net(img)
        priorbox = anchor(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)

        vis = img_raw.copy()
        counts.append(int((dets[:, 4] >= args.vis_thres).sum()))
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, '{:.2f}'.format(b[4]), (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for k in range(5):
                cv2.circle(vis, (int(b[5 + 2 * k]), int(b[6 + 2 * k])), 2, (0, 0, 255), -1)
        results.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        print('done', i + 1, img_name, 'faces:', counts[-1], flush=True)

    if not results:
        raise ValueError('No validation images listed in ' + label_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows = math.ceil(len(results) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, im in zip(axes.ravel(), results):
        ax.imshow(im)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close(fig)
    with open(os.path.splitext(args.output)[0] + '.json', 'w') as f:
        json.dump({'args': vars(args), 'images': test_dataset[:args.num_vis],
                   'visible_faces': counts}, f, indent=2)

