import torch
import torch.nn.functional as F

__all__ = ['KLLoss']


class KLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = F.softmax(x.detach(), dim=1)
        y = F.log_softmax(y, dim=1)
        return torch.mean(torch.sum(x * (torch.log(x) - y), dim=1))
