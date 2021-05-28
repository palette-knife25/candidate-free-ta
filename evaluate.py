import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

from options import Config, ExperimentConfig
import systems


def evaluate(opt: ExperimentConfig):
	dm: pl.LightningDataModule = instantiate(opt.dataset)
	dm.setup()
	model = getattr(systems, opt.system).load_from_checkpoint(opt.trainer_args.resume_from_checkpoint,
															  opt=opt, tokenizer=dm.tokenizer)
	trainer = pl.Trainer(logger=None, default_root_dir="./", **opt.trainer_args)
	trainer.validate(model, val_dataloaders=getattr(dm, opt.evaluate_dl)())



@hydra.main(config_path="configs", config_name="config")
def main(cfg: Config) -> None:
	print(OmegaConf.to_yaml(cfg))
	evaluate(cfg.experiment)


if __name__ == "__main__":
	main()
