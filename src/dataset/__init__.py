from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_3dovs import Dataset3dovsCfg, threedovsDataset
from .dataset_replica import DatasetReplicaCfg, ReplicaDataset
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "3dovs": threedovsDataset,
    "replica": ReplicaDataset,

}


DatasetCfg = DatasetReplicaCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
