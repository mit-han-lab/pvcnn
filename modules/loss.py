import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.functional import kl_loss

__all__ = ['KLLoss', 'discrepancy_loss']


class KLLoss(nn.Module):
    def forward(self, x, y):
        return kl_loss(x, y)


def discrepancy_loss(out1, out2):
    """discrepancy loss"""
    out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
    return out
