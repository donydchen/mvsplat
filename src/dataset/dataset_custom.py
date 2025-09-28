import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

# Add the DataLoader directory to the path so we can import the custom dataset
# Since mvsplat is now in Preprocessing/Reconstruction/mvsplat, we need to go up 6 levels to reach DataLoader
# Path structure: dataset_custom.py -> dataset -> src -> mvsplat -> Reconstruction -> Preprocessing -> DataLoader
dataloader_path = Path(__file__).parent.parent.parent.parent.parent.parent
print(f"DEBUG: Adding to sys.path: {dataloader_path}")
print(f"DEBUG: custom_dataset.py exists: {(dataloader_path / 'custom_dataset.py').exists()}")
sys.path.insert(0, str(dataloader_path))  # Use insert(0, ...) to prioritize this path

from custom_dataset import CustomDataset, collate_fn
from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetCustomCfg(DatasetCfgCommon):
    name: Literal["custom"]
    dataset_path: str
    data_df_path: str
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    random_seed: int = 42
    group_by_scene: bool = True
    n_step_state: int = 1
    n_step_predict: int = 1
    n_views_context: int = 3
    n_views_target: int = 1
    original_image_size: list[int] = None
    temporal_sampling: str = "sequential"
    max_temporal_gap: int = 10
    verbose: bool = True
    view_sampler_strategy: str = "group"
    near: float = 0.7
    far: float = 21.0
    make_baseline_1: bool = True
    baseline_epsilon: float = 1e-3
    baseline_scale_bounds: bool = True
    
    def __post_init__(self):
        if self.original_image_size is None:
            self.original_image_size = [2448, 2048]


class DatasetCustom(Dataset):
    """MVsplat wrapper for CustomDataset - CLEAN PASSTHROUGH VERSION"""
    
    def __init__(
        self,
        cfg: DatasetCustomCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        # Note: view_sampler is ignored since custom dataset handles view sampling internally
        
        # Create the custom dataset
        self.dataset = CustomDataset(
            dataset_path=cfg.dataset_path,
            data_df_path=cfg.data_df_path,
            stage=stage,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            random_seed=cfg.random_seed,
            group_by_scene=cfg.group_by_scene,
            n_step_state=cfg.n_step_state,
            n_step_predict=cfg.n_step_predict,
            n_views_context=cfg.n_views_context,
            n_views_target=cfg.n_views_target,
            image_size=tuple(cfg.image_shape),
            original_image_size=cfg.original_image_size,
            temporal_sampling=cfg.temporal_sampling,
            max_temporal_gap=cfg.max_temporal_gap,
            verbose=cfg.verbose,
            view_sampler_strategy=cfg.view_sampler_strategy
        )
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, index: int) -> dict:
        """Get item and convert to MVsplat format - MINIMAL PROCESSING"""
        item = self.dataset[index]
        
        # Simply reshape from temporal format to MVSplat format
        # Your custom dataset already handles image scaling and cropping
        
        # With n_step_state=1, temporal dimension is 1, so we can squeeze it out
        # Context: remove temporal dimension [1, n_views, ...] -> [n_views, ...]
        context_extrinsics = item["context"]["extrinsics"].squeeze(0)  # [n_views, 4, 4]
        context_intrinsics = item["context"]["intrinsics"].squeeze(0)  # [n_views, 3, 3]
        context_image = item["context"]["image"].squeeze(0)  # [n_views, C, H, W]
        context_near = item["context"]["near"].squeeze(0)  # [n_views]
        context_far = item["context"]["far"].squeeze(0)  # [n_views]
        context_index = item["context"]["index"].squeeze(0)  # [n_views]
        
        # Target: remove temporal dimension [n_steps, n_views, ...] -> [n_views, ...]
        # Since we have n_step_predict=1, take the last timestep
        target_extrinsics = item["target"]["extrinsics"][-1]  # [n_views, 4, 4]
        target_intrinsics = item["target"]["intrinsics"][-1]  # [n_views, 3, 3]
        target_image = item["target"]["image"][-1]  # [n_views, C, H, W]
        target_near = item["target"]["near"][-1]  # [n_views]
        target_far = item["target"]["far"][-1]  # [n_views]
        target_index = item["target"]["index"][-1]  # [n_views]
        
        # No scaling needed - custom dataset already handles normalization
        example = {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "image": context_image,
                "near": context_near,
                "far": context_far,
                "index": context_index,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_image,
                "near": target_near,
                "far": target_far,
                "index": target_index,
            },
            "scene": item["scene"],
        }
                            
        return example