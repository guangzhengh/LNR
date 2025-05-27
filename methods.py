import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    #print('mixing x with lam_x ', lam)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, n = None):
    #print(n,y_a,y_b)
    lam_y = []
    loss = 0
    for i, v in enumerate(y_a):
        if (n[y_a[i]] / n[y_b[i]]) >= 3 and lam < 0.5:
            lam_y.append(0)
        elif (n[y_a[i]] / n[y_b[i]]) < (1/3) and (1-lam) < 0.5:
            lam_y.append(1)
        else:
            lam_y.append(lam)
    lam_y = torch.tensor(np.array(lam_y)).cuda()
    #print('mixing y with lam_y mean', torch.mean(lam_y))
    return criterion(pred, y_a, lam_y) +  criterion(pred, y_b,(1-lam_y))


class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, smooth_head, smooth_tail, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, x, target, lam_y = [1]):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        if len(lam_y) == 1:
            loss = confidence * nll_loss + smoothing * smooth_loss
        else:
            loss = lam_y * confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x
