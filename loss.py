import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        focal_loss = alpha * (1. - pt) ** self.gamma * ce
        return focal_loss


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=0.11):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)
