import torch


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, list):
        for j, v in enumerate(x):
            x[j] = cuda(v)
    elif isinstance(x, dict):
        x = {k: cuda(v) for k, v in x.items()}
    return x