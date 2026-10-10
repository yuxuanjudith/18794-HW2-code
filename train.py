import os
# Avoid loading a second Intel OpenMP runtime through Conda's MKL.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import time
import math
import torch
import argparse
import datetime
import csv
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import mydataloader, collate, preproc, cfg
from loss_and_anchor import Loss, anchor
from detector.mydetector import mydetector

parser = argparse.ArgumentParser(description='18794 detection')
parser.add_argument('--data_path', default='data/widerface_homework/train/label.txt')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--save_path', default='./weights/')
parser.add_argument('--max_iter', default=0, type=int)
parser.add_argument('--results_dir', default='results')
parser.add_argument('--seed', default=18794, type=int)

args = parser.parse_args()


def plot_loss_curves(hist, results_dir):
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11}):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), layout='constrained')
        for ax, key, title in zip(axes, ['loc', 'cls', 'landm'],
                                  ['Localization loss', 'Classification loss', 'Landmark loss']):
            ax.plot(hist['iter'], hist[key], color='#0072B2', linewidth=0.8)
            ax.set(title=title, xlabel='Iteration', ylabel='Loss', ylim=(0, None))
            ax.grid(alpha=0.2)
        fig.savefig(os.path.join(results_dir, 'loss_curves.png'), dpi=200)
        plt.close(fig)


def train():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    os.makedirs(args.results_dir, exist_ok=True)
    net = mydetector(cfg=cfg)
    if cfg['ngpu'] > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    net.train()
    torch.backends.cudnn.benchmark = False

    epoch = 0
    dataset = mydataloader(args.data_path, preproc(img_dim=cfg['image_size'],
                                                         rgb_means=(104, 117, 123)))
    traindata = torch.utils.data.DataLoader(dataset,
                                            cfg['batch_size'],
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate)

    epoch_size = math.ceil(len(dataset) / cfg['batch_size'])
    max_iter = cfg['epoch'] * epoch_size
    if args.max_iter > 0:
        max_iter = args.max_iter
    if epoch_size == 0:
        raise ValueError('No training images with valid boxes')
    with open(os.path.join(args.results_dir, 'run_config.json'), 'w') as f:
        json.dump({'config': cfg, 'args': vars(args), 'training_images': len(dataset),
                   'epoch_size': epoch_size, 'max_iter': max_iter,
                   'torch_version': torch.__version__,
                   'gpu': torch.cuda.get_device_name(0)}, f, indent=2)
    csv_path = os.path.join(args.results_dir, 'loss_history.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['iteration', 'loc', 'cls', 'landm'])
    hist = {'iter': [], 'loc': [], 'cls': [], 'landm': []}

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = Loss(2, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = anchor(cfg, image_size=(cfg['image_size'], cfg['image_size']))
    with torch.no_grad():
        priors = priorbox.forward().cuda()

    for iteration in range(0, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(traindata)
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        out = net(images)

        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss at iteration {}'.format(iteration + 1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration - 1))

        if (iteration + 1) % 20 == 0:
            hist['iter'].append(iteration + 1)
            hist['loc'].append(loss_l.item())
            hist['cls'].append(loss_c.item())
            hist['landm'].append(loss_landm.item())
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([iteration + 1, loss_l.item(), loss_c.item(), loss_landm.item()])
        if iteration == 0 or (iteration + 1) % 20 == 0 or iteration + 1 == max_iter:
            print('Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || Batchtime: {:.3f} s || ETA: {}'
                  .format(iteration + 1, max_iter, loss_l.item(), loss_c.item(),
                          loss_landm.item(), batch_time, str(datetime.timedelta(seconds=eta))), flush=True)

    os.makedirs(args.save_path, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_path, cfg['name'] + '_Final.pth'))

    plot_loss_curves(hist, args.results_dir)
    print('Training complete: {} iterations'.format(max_iter), flush=True)


def adjust_learning_rate(optimizer, epoch, step_index, iteration, epoch_size):
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = args.lr * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
