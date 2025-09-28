from pathlib import Path
from random import randrange

import hydra
import torch
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import move_data_to_device
from omegaconf import DictConfig

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.geometry.epipolar_lines import project_rays
    from src.geometry.projection import get_world_rays
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.step_tracker import StepTracker
    from src.visualization.annotation import add_label
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points
    from src.visualization.layout import add_border, hcat


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def visualize_epipolar_lines_custom(cfg_dict: DictConfig):
    device = torch.device("cuda:0")
    num_lines = 6

    # Override dataset config to use custom dataset
    cfg_dict.dataset.name = "custom"
    
    # Load configuration and initialize data module
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    dataset = iter(data_module.train_dataloader())

    for e_idx, example in enumerate(dataset):
        if e_idx > 40:
            break
            
        example = move_data_to_device(example, device)

        # Define colors for different lines
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
        
        # Get images
        _, v, _, h, w = example["context"]["image"].shape
        assert v >= 2
        source_image = example["context"]["image"][0, 0].clone()
        target_image = example["context"]["image"][0, 1].clone()
        
        scene_name = example['scene'][0] if 'scene' in example else f'sample_{e_idx}'
        
        # Draw multiple epipolar lines
        for i in range(num_lines):
            # Pick a random pixel
            x = randrange(0, w)
            y = randrange(0, h)
            xy = torch.tensor((x / w, y / h), dtype=torch.float32, device=device)

            # Generate ray and project to epipolar line
            source_extrinsics = example["context"]["extrinsics"][0, 0]
            source_intrinsics = example["context"]["intrinsics"][0, 0]
            origin, direction = get_world_rays(xy, source_extrinsics, source_intrinsics)
            
            target_extrinsics = example["context"]["extrinsics"][0, 1]
            target_intrinsics = example["context"]["intrinsics"][0, 1]
            near_val = example["context"]["near"][0, 0]
            far_val = example["context"]["far"][0, 0]
            
            projection = project_rays(
                origin,
                direction,
                target_extrinsics,
                target_intrinsics,
                near=near_val,
                far=far_val,
            )

            color = colors[i % len(colors)]
            
            # Draw point on source image
            source_image = draw_points(
                source_image, xy, color, 4, x_range=(0, 1), y_range=(0, 1)
            )

            # Draw epipolar line on target image
            target_image = draw_lines(
                target_image,
                projection["xy_min"],
                projection["xy_max"],
                color,
                4,
                x_range=(0, 1),
                y_range=(0, 1),
            )

        # Combine images and save
        source_image = add_label(source_image, "Source")
        target_image = add_label(target_image, "Target")
        together = add_border(hcat(source_image, target_image))
        
        output_dir = Path("epipolar_lines/custom")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_image(together, output_dir / f"{scene_name}_combined.png")


if __name__ == "__main__":
    visualize_epipolar_lines_custom()

