import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from tqdm import tqdm

from ..geometry.epipolar_lines import project_rays
from ..geometry.projection import get_world_rays, sample_image_grid
from ..misc.image_io import save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat


@dataclass
class EvaluationIndexGeneratorCfg:
    num_target_views: int
    min_distance: int
    max_distance: int
    min_overlap: float
    max_overlap: float
    output_path: Path
    save_previews: bool
    seed: int


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


class EvaluationIndexGenerator(LightningModule):
    generator: torch.Generator
    cfg: EvaluationIndexGeneratorCfg
    index: dict[str, IndexEntry | None]

    def __init__(self, cfg: EvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}

    def test_step(self, batch, batch_idx):
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        extrinsics = batch["target"]["extrinsics"][0]
        intrinsics = batch["target"]["intrinsics"][0]
        scene = batch["scene"][0]

        context_indices = torch.randperm(v, generator=self.generator)
        for context_index in tqdm(context_indices, "Finding context pair"):
            xy, _ = sample_image_grid((h, w), self.device)
            context_origins, context_directions = get_world_rays(
                rearrange(xy, "h w xy -> (h w) xy"),
                extrinsics[context_index],
                intrinsics[context_index],
            )

            # Step away from context view until the minimum overlap threshold is met.
            valid_indices = []
            for step in (1, -1):
                min_distance = self.cfg.min_distance
                max_distance = self.cfg.max_distance
                current_index = context_index + step * min_distance

                while 0 <= current_index.item() < v:
                    # Compute overlap.
                    current_origins, current_directions = get_world_rays(
                        rearrange(xy, "h w xy -> (h w) xy"),
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_current = project_rays(
                        context_origins,
                        context_directions,
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_context = project_rays(
                        current_origins,
                        current_directions,
                        extrinsics[context_index],
                        intrinsics[context_index],
                    )
                    overlap_a = projection_onto_context["overlaps_image"].float().mean()
                    overlap_b = projection_onto_current["overlaps_image"].float().mean()

                    overlap = min(overlap_a, overlap_b)
                    delta = (current_index - context_index).abs()

                    min_overlap = self.cfg.min_overlap
                    max_overlap = self.cfg.max_overlap
                    if min_overlap <= overlap <= max_overlap:
                        valid_indices.append(
                            (current_index.item(), overlap_a, overlap_b)
                        )

                    # Stop once the camera has panned away too much.
                    if overlap < min_overlap or delta > max_distance:
                        break

                    current_index += step

            if valid_indices:
                # Pick a random valid view. Index the resulting views.
                num_options = len(valid_indices)
                chosen = torch.randint(
                    0, num_options, size=tuple(), generator=self.generator
                )
                chosen, overlap_a, overlap_b = valid_indices[chosen]

                context_left = min(chosen, context_index.item())
                context_right = max(chosen, context_index.item())
                delta = context_right - context_left

                # Pick non-repeated random target views.
                while True:
                    target_views = torch.randint(
                        context_left,
                        context_right + 1,
                        (self.cfg.num_target_views,),
                        generator=self.generator,
                    )
                    if (target_views.unique(return_counts=True)[1] == 1).all():
                        break

                target = tuple(sorted(target_views.tolist()))
                self.index[scene] = IndexEntry(
                    context=(context_left, context_right),
                    target=target,
                )

                # Optionally, save a preview.
                if self.cfg.save_previews:
                    preview_path = self.cfg.output_path / "previews"
                    preview_path.mkdir(exist_ok=True, parents=True)
                    a = batch["target"]["image"][0, chosen]
                    a = add_label(a, f"Overlap: {overlap_a * 100:.1f}%")
                    b = batch["target"]["image"][0, context_index]
                    b = add_label(b, f"Overlap: {overlap_b * 100:.1f}%")
                    vis = add_border(add_border(hcat(a, b)), 1, 0)
                    vis = add_label(vis, f"Distance: {delta} frames")
                    save_image(add_border(vis), preview_path / f"{scene}.png")
                break
        else:
            # This happens if no starting frame produces a valid evaluation example.
            self.index[scene] = None

    def save_index(self) -> None:
        self.cfg.output_path.mkdir(exist_ok=True, parents=True)
        with (self.cfg.output_path / "evaluation_index.json").open("w") as f:
            json.dump(
                {k: None if v is None else asdict(v) for k, v in self.index.items()}, f
            )
