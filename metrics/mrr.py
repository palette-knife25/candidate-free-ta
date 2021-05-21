from typing import List, Any
import torch
from torchmetrics.metric import Metric


class MeanReciprocalRank(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("updates", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, correct_preds: torch.Tensor):
        reciprocal = reciprocal_rank(correct_preds)
        self.acc += reciprocal.sum()
        self.updates += len(reciprocal)

    def compute(self):
        return self.acc / self.updates


def first_nonzero(tensor: torch.Tensor, dim=0):
    """
        Returns 
            indices: the index of the first nonzero element along dimension,
            all_zero: a mask for all-zero sequences
    """
    nonzero = (tensor > 0)
    cumsum = torch.cumsum(nonzero, dim=dim)
    indices = ((cumsum == 1) & nonzero).max(dim=dim).indices
    all_zero = torch.sum(nonzero, dim=dim) == 0
    return indices, all_zero


def reciprocal_rank(correct_preds: torch.Tensor):
    """
        correct_preds: torch.Tensor of shape [b x k], where a non-zero value means
                       the prediction is among the gt labels.
        Returns
            reciprocal: torch.Tensor of shape [b] of the reciprocal rank. 
                        No match results in a value of 0.
    """
    indices, all_zero = first_nonzero(correct_preds, dim=1)
    reciprocal = torch.reciprocal(indices.float() + 1)
    reciprocal.masked_fill_(all_zero, 0)
    return reciprocal


def build_match_matrix(preds: List[List[Any]], labels: List[List[Any]]):
    lemmas_sets = [set(lemmas) for lemmas in labels]
    correct_preds = torch.zeros((len(preds), len(preds[0])), dtype=torch.bool)
    for i, lemmas in enumerate(lemmas_sets):
        for j, pred in enumerate(preds[i]):
            correct_preds[i, j] = pred in lemmas
    return correct_preds
