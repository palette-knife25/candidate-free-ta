import torch
from torchmetrics.metric import Metric


class PrecisionK(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("updates", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, correct_preds: torch.Tensor):
        precision = correct_preds.float().mean(dim=1)
        self.acc += precision.sum()
        self.updates += len(precision)

    def compute(self):
        return self.acc / self.updates
