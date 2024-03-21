import os
from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from tabulate import tabulate

from ..misc.image_io import load_image, save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim


class MetricComputer(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg) -> None:
        super().__init__()
        self.cfg = cfg

    def test_step(self, batch, batch_idx):
        scene = batch["scene"][0]
        b, cv, _, _, _ = batch["context"]["image"].shape
        assert b == 1 and cv == 2
        _, v, _, _, _ = batch["target"]["image"].shape

        # Skip scenes.
        for method in self.cfg.methods:
            if not (method.path / scene).exists():
                print(f'Skipping "{scene}".')
                return

        # Load the images.
        all_images = {}
        try:
            for method in self.cfg.methods:
                images = [
                    load_image(method.path / scene / f"color/{index.item():0>6}.png")
                    for index in batch["target"]["index"][0]
                ]
                all_images[method.key] = torch.stack(images).to(self.device)
        except FileNotFoundError:
            print(f'Skipping "{scene}".')
            return

        # Compute metrics.
        all_metrics = {}
        rgb_gt = batch["target"]["image"][0]
        for key, images in all_images.items():
            all_metrics = {
                **all_metrics,
                f"lpips_{key}": compute_lpips(rgb_gt, images).mean(),
                f"ssim_{key}": compute_ssim(rgb_gt, images).mean(),
                f"psnr_{key}": compute_psnr(rgb_gt, images).mean(),
            }
        self.log_dict(all_metrics)
        self.print_preview_metrics(all_metrics)

        # Skip the rest if no side-by-side is needed.
        if self.cfg.side_by_side_path is None:
            return

        # Create side-by-side.
        scene_key = f"{batch_idx:0>6}_{scene}"
        for i in range(v):
            true_index = batch["target"]["index"][0, i]
            row = [add_label(batch["target"]["image"][0, i], "Ground Truth")]
            for method in self.cfg.methods:
                image = all_images[method.key][i]
                image = add_label(image, method.name)
                row.append(image)
            start_frame = batch["target"]["index"][0, 0]
            end_frame = batch["target"]["index"][0, -1]
            label = f"Scene {batch['scene'][0]} (frames {start_frame} to {end_frame})"
            row = add_border(add_label(hcat(*row), label, font_size=16))
            save_image(
                row,
                self.cfg.side_by_side_path / scene_key / f"{true_index:0>6}.png",
            )

        # Create an animation.
        if self.cfg.animate_side_by_side:
            (self.cfg.side_by_side_path / "videos").mkdir(exist_ok=True, parents=True)
            command = (
                'ffmpeg -y -framerate 30 -pattern_type glob -i "*.png"  -c:v libx264 '
                '-pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"'
            )
            os.system(
                f"cd {self.cfg.side_by_side_path / scene_key} && {command} "
                f"{Path.cwd()}/{self.cfg.side_by_side_path}/videos/{scene_key}.mp4"
            )

    def print_preview_metrics(self, metrics: dict[str, float]) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        table = []
        for method in self.cfg.methods:
            row = [
                f"{self.running_metrics[f'{metric}_{method.key}']:.3f}"
                for metric in ("psnr", "lpips", "ssim")
            ]
            table.append((method.key, *row))

        table = tabulate(table, ["Method", "PSNR (dB)", "LPIPS", "SSIM"])
        print(table)
