from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

def defaulter(mut):
    return field(default_factory=lambda: mut)

@dataclass
class ExperimentConfig:
    name: str = "test"
    data_root: str = "../../../datasets"
    
    trainer_args: Any = defaulter({'max_epochs': 30, 'profiler': None})
    dataset: Any = defaulter({})
    optimizer: Any = defaulter({
        '_target_': 'torch.optim.Adam',
        'lr': 0.001
    })
    scheduler: Any = None

    system: str = "CandidateFreeTE"
    max_tokens_lemma: int = 5
    top_k: int = 10

    net: Any = defaulter({
        '_target_': "models.KBertEnricher",
        'type_embedding_max': 7
    })

    evaluate_dl: str = "test_dataloader"

@dataclass
class Config:
    defaults: List[Any] = defaulter([{'experiment': 'basic'}])
    experiment: ExperimentConfig = ExperimentConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="experiment", name="basic", node=ExperimentConfig)