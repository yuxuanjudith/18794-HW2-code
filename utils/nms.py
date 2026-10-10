import numpy as np
import torch

from utils.box_utils import jaccard


def nms(dets, thresh):
    dets = torch.as_tensor(dets).clone()
    scores = dets[:, 4]
    order = scores.argsort(descending=True)
    chosen = []
    while order.numel() > 0:
        i = order[0].item()
        chosen.append(i)
        if order.numel() == 1:
            break
        ious = jaccard(dets[i:i + 1, :4], dets[order[1:], :4])[0]
        order = order[1:][ious <= thresh]
    return chosen
