from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        projections = hcat(*render_projections(
                                gaussians_softmax,
                                256,
                                extra_label="(Softmax)",
                            )[0])
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
