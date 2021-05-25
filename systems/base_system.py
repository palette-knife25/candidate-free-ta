import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from metrics import PrecisionK, MeanAveragePrecision
from metrics.mrr import build_match_matrix, MeanReciprocalRank
from options import ExperimentConfig


class BaseSystem(pl.LightningModule):
    def __init__(self, opt: ExperimentConfig, tokenizer):
        super().__init__()
        self.opt = opt
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.mask_token_id
        self.val_metrics = {}
        self.set_val_metrics()

    def set_val_metrics(self):
        self.val_metrics = {"ValidationMRR": MeanReciprocalRank(),
                            f"ValidationPrecision@{self.opt.top_k}": PrecisionK(),
                            "ValidationMAP": MeanAveragePrecision()}

    def forward(self, token_ids, type_ids, synset_ids, highway):
        return self.net(token_ids, type_ids, synset_ids, highway)

    @torch.no_grad()
    def infere_top_k(self, batch, k, insert_position=0):
        raise NotImplementedError

    def insert_mask_tokens(self, token_ids, type_ids, synset_ids, highway, n_tokens, position):
        batch_size = token_ids.shape[0]
        token_ids = torch.cat([token_ids[:, :position],
                               token_ids.new_ones(batch_size, n_tokens) * self.mask_token_id,
                               token_ids[:, position:, ]], dim=1)
        type_ids = torch.cat([type_ids[:, :position],
                              type_ids.new_zeros(batch_size, n_tokens),
                              type_ids[:, position:, ]], dim=1)
        synset_ids = torch.cat([synset_ids[:, :position],
                                synset_ids.new_zeros(batch_size, n_tokens),
                                synset_ids[:, position:, ]], dim=1)
        highway = torch.cat([highway[:, :position],
                             highway.new_ones(batch_size, n_tokens),
                             highway[:, position:, ]], dim=1)
        return token_ids, type_ids, synset_ids, highway

    def criterion(self, log_probs, labels, mask):
        raise NotImplementedError

    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, val_batch, batch_idx):
        raise NotImplementedError

    def test_step(self, val_batch, batch_idx):
        return self.validation_step(self, val_batch, batch_idx)

    def validation_step_end(self, outputs):
        correct = build_match_matrix(outputs['topk'], outputs['gt'])
        for name, metric in self.val_metrics.items():
            metric(correct)
            self.log(name, metric, on_step=False, on_epoch=True)
        return outputs

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = instantiate(self.opt.optimizer, self.parameters())
        if self.opt.scheduler is not None:
            lr_scheduler = [instantiate(self.opt.scheduler, optimizer)]
        else:
            lr_scheduler = []
        return [optimizer], lr_scheduler
