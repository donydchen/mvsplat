from typing import Optional

from omegaconf import DictConfig

cfg: Optional[DictConfig] = None


def get_cfg() -> DictConfig:
    global cfg
    return cfg


def set_cfg(new_cfg: DictConfig) -> None:
    global cfg
    cfg = new_cfg


def get_seed() -> int:
    return cfg.seed
