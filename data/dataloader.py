import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class mydataloader(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        img_dir = os.path.join(os.path.dirname(txt_path), 'images')
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    self.imgs_path.append(os.path.join(img_dir, line[1:].strip()))
                    self.words.append([])
                else:
                    self.words[-1].append([float(x) for x in line.split()])

        # The augmentation pipeline requires at least one non-degenerate face.
        valid = [i for i, labels in enumerate(self.words)
                 if any(label[2] > 0 and label[3] > 0 for label in labels)]
        print('Dataset: {} images; excluded {} without valid boxes'.format(
            len(valid), len(self.imgs_path) - len(valid)))
        self.imgs_path = [self.imgs_path[i] for i in valid]
        self.words = [self.words[i] for i in valid]


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        if img is None:
            raise FileNotFoundError(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        for idx, label in enumerate(labels):
            if label[2] <= 0 or label[3] <= 0:
                continue
            annotation = np.zeros((1, 15))
            annotation[0, 0] = label[0]
            annotation[0, 1] = label[1]
            annotation[0, 2] = label[0] + label[2]
            annotation[0, 3] = label[1] + label[3]
            annotation[0, 4:14] = [label[4], label[5], label[7], label[8], label[10],
                                   label[11], label[13], label[14], label[16], label[17]]
            annotation[0, 14] = -1 if label[4] < 0 else 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
