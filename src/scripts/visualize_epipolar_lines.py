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
def visualize_epipolar_lines(cfg_dict: DictConfig):
    device = torch.device("cuda:0")
    num_lines = 5

    # Boilerplate configuration stuff like in the main file...
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    # dataset = iter(data_module.train_dataloader())
    dataset = iter(data_module.test_dataloader())

    cur_radius = (
        str(cfg.dataset.view_sampler.index_path.stem).split(".")[0].split("_")[-1]
    )

    for e_idx, example in enumerate(dataset):
        # example = next(dataset)
        if e_idx > 10:
            break
        print(f"Drawing scene {example['scene'][0]}")

        example = move_data_to_device(example, device)

        # Plot a few different examples to try to get an interesting line.
        for i in range(num_lines):
            # Get a single example from the dataset.
            # example = next(dataset)
            # example = move_data_to_device(example, device)

            # Pick a random pixel to visualize the epipolar line for.
            _, v, _, h, w = example["context"]["image"].shape
            assert v >= 2
            x = randrange(0, w)
            y = randrange(0, h)
            xy = torch.tensor((x / w, y / h), dtype=torch.float32, device=device)

            # Generate the ray that corresponds to the point.
            source_extrinsics = example["context"]["extrinsics"][0, 0]
            source_intrinsics = example["context"]["intrinsics"][0, 0]
            origin, direction = get_world_rays(xy, source_extrinsics, source_intrinsics)
            target_extrinsics = example["context"]["extrinsics"][0, 1]
            target_intrinsics = example["context"]["intrinsics"][0, 1]
            projection = project_rays(
                origin,
                direction,
                target_extrinsics,
                target_intrinsics,
                near=example["context"]["near"][0, 0],
                far=example["context"]["far"][0, 0],
            )

            # Draw the point (ray) onto the source view.
            source_image = example["context"]["image"][0, 0]
            source_image = draw_points(
                source_image, xy, (1, 0, 0), 4, x_range=(0, 1), y_range=(0, 1)
            )

            # Draw the epipolar line onto the target view.
            target_image = example["context"]["image"][0, 1]
            target_image = draw_lines(
                target_image,
                projection["xy_min"],
                projection["xy_max"],
                (1, 0, 0),
                4,
                x_range=(0, 1),
                y_range=(0, 1),
            )

            # Put the images side by side.
            source_image = add_label(source_image, "Source")
            target_image = add_label(target_image, "Target")
            together = add_border(hcat(source_image, target_image))
            save_image(
                together,
                Path(f"epipolar_lines/{cur_radius}/{example['scene'][0]}_{i:0>2}.png"),
            )


if __name__ == "__main__":
    visualize_epipolar_lines()
