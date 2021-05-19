import torch
from torchmetrics.metric import Metric

class MeanAveragePrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("updates", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, correct_preds: torch.Tensor):
        map = compute_map(correct_preds)
        self.acc += map.sum()
        self.updates += len(map)

    def compute(self):
        return self.acc / self.updates

def compute_map(correct_preds: torch.Tensor):
    b, k = correct_preds.shape
    diver = torch.arange(1, k+1, device=correct_preds.device).expand(b, k)
    pk = torch.cumsum(correct_preds, dim=1) / diver
    scale = correct_preds.sum(dim=1).masked_fill_(~correct_preds.any(dim=1), 1)
    relevant = (pk * correct_preds).sum(dim=1) / scale
    return relevant
