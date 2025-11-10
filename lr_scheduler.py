# lr_scheduler.py
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLRWarmup(_LRScheduler):
    """
    Linear warmup for T_warmup steps, then cosine annealing to eta_min over T_max steps (total).
    Usage is identical to the repo’s: CosineAnnealingLRWarmup(optim, T_max=1000, T_warmup=20, eta_min=0)
    """
    def __init__(self, optimizer, T_max, T_warmup=0, eta_min=0.0, last_epoch=-1):
        self.T_max = int(T_max)
        self.T_warmup = int(T_warmup)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if self.T_warmup > 0 and step <= self.T_warmup:
                # linear warmup from 0 -> base_lr
                scale = step / float(self.T_warmup)
                lrs.append(base_lr * scale)
            else:
                t = max(0, step - self.T_warmup)
                T = max(1, self.T_max - self.T_warmup)
                cos = 0.5 * (1 + math.cos(math.pi * t / T))
                lrs.append(self.eta_min + (base_lr - self.eta_min) * cos)
        return lrs
