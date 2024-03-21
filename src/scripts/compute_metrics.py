import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from pytorch_lightning import Trainer

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfg
    from src.evaluation.evaluation_cfg import EvaluationCfg
    from src.evaluation.metric_computer import MetricComputer
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    evaluation: EvaluationCfg
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    seed: int
    output_metrics_path: Path


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="compute_metrics",
)
def evaluate(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    trainer = Trainer(max_epochs=-1, accelerator="gpu")
    computer = MetricComputer(cfg.evaluation)
    data_module = DataModule(cfg.dataset, cfg.data_loader)
    metrics = trainer.test(computer, datamodule=data_module)
    cfg.output_metrics_path.parent.mkdir(exist_ok=True, parents=True)
    with cfg.output_metrics_path.open("w") as f:
        json.dump(metrics[0], f)


if __name__ == "__main__":
    evaluate()
