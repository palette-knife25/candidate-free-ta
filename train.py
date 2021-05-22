import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import callbacks
from options import Config, ExperimentConfig
import os

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from system import CandidateFreeTE
import pytorch_lightning as pl


def train(opt: ExperimentConfig):
    dm: pl.LightningDataModule = instantiate(opt.dataset)
    model = CandidateFreeTE(opt, dm.tokenizer)
    logger = TensorBoardLogger(".", name="", version="", default_hp_metric=False)
    
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints', filename='best_{epoch}-{ValidationMRR:.2f}',
                                        monitor='ValidationMRR', mode='min', every_n_val_epochs=1, save_last=True)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, default_root_dir="./", **opt.trainer_args)

    trainer.fit(model, datamodule=dm)

@hydra.main(config_path="configs", config_name="config")
def main(cfg : Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg.experiment)

if __name__ == "__main__":
    main()