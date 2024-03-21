from pathlib import Path
from random import randrange
from typing import Optional

import numpy as np
import torch
import wandb
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float
from torch import Tensor

from ....dataset.types import BatchedViews
from ....misc.heterogeneous_pairings import generate_heterogeneous_index
from ....visualization.annotation import add_label
from ....visualization.color_map import apply_color_map, apply_color_map_to_image
from ....visualization.colors import get_distinct_color
from ....visualization.drawing.lines import draw_lines
from ....visualization.drawing.points import draw_points
from ....visualization.layout import add_border, hcat, vcat
# from ...ply_export import export_ply
from ..encoder_costvolume import EncoderCostVolume
# from ..epipolar.epipolar_sampler import EpipolarSampling
from .encoder_visualizer import EncoderVisualizer
from .encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg


def box(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 new_height new_width"]:
    return add_border(add_border(image), 1, 0)


class EncoderVisualizerCostVolume(
    EncoderVisualizer[EncoderVisualizerCostVolumeCfg, EncoderCostVolume]
):
    def visualize(
        self,
        context: BatchedViews,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        # Short-circuit execution when using mvsplat.
        return {}

        visualization_dump = {}

        softmax_weights = []

        def hook(module, input, output):
            softmax_weights.append(output)

        # Register hooks to grab attention.
        handles = [
            layer[0].fn.attend.register_forward_hook(hook)
            for layer in self.encoder.epipolar_transformer.transformer.layers
        ]

        result = self.encoder.forward(
            context,
            global_step,
            visualization_dump=visualization_dump,
            deterministic=True,
        )

        # De-register hooks.
        for handle in handles:
            handle.remove()

        softmax_weights = torch.stack(softmax_weights)

        # Generate high-resolution context images that can be drawn on.
        context_images = context["image"]
        _, _, _, h, w = context_images.shape
        length = min(h, w)
        min_resolution = self.cfg.min_resolution
        scale_multiplier = (min_resolution + length - 1) // length
        if scale_multiplier > 1:
            context_images = repeat(
                context_images,
                "b v c h w -> b v c (h rh) (w rw)",
                rh=scale_multiplier,
                rw=scale_multiplier,
            )

        # This is kind of hacky for now, since we're using it for short experiments.
        if self.cfg.export_ply and wandb.run is not None:
            name = wandb.run._name.split(" ")[0]
            ply_path = Path(f"outputs/gaussians/{name}/{global_step:0>6}.ply")
            export_ply(
                context["extrinsics"][0, 0],
                result.means[0],
                visualization_dump["scales"][0],
                visualization_dump["rotations"][0],
                result.harmonics[0],
                result.opacities[0],
                ply_path,
            )

        return {
            "attention": self.visualize_attention(
                context_images,
                visualization_dump["sampling"],
                softmax_weights,
            ),
            "epipolar_samples": self.visualize_epipolar_samples(
                context_images,
                visualization_dump["sampling"],
            ),
            "epipolar_color_samples": self.visualize_epipolar_color_samples(
                context_images,
                context,
            ),
            "gaussians": self.visualize_gaussians(
                context["image"],
                result.opacities,
                result.covariances,
                result.harmonics[..., 0],  # Just visualize DC component.
            ),
            "overlaps": self.visualize_overlaps(
                context["image"],
                visualization_dump["sampling"],
                visualization_dump.get("is_monocular", None),
            ),
            "depth": self.visualize_depth(
                context,
                visualization_dump["depth"],
            ),
        }

    def visualize_attention(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        attention: Float[Tensor, "layer bvr head 1 sample"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize attention in the sample view.
        attention = rearrange(
            attention, "l (b v r) hd () s -> l b v r hd s", b=b, v=v, r=r
        )
        attention = attention[:, rb, rv, rr, :, :]
        num_layers, _, hd, _ = attention.shape

        vis = []
        for il in range(num_layers):
            vis_layer = []
            for ihd in range(hd):
                # Create colors according to attention.
                color = [get_distinct_color(i) for i, _ in enumerate(rr)]
                color = torch.tensor(color, device=attention.device)
                color = rearrange(color, "r c -> r () c")
                attn = rearrange(attention[il, :, ihd], "r s -> r s ()")
                color = rearrange(attn * color, "r s c -> (r s ) c")

                # Draw the alternating bucket lines.
                vis_layer_head = draw_lines(
                    context_images[rb, self.encoder.sampler.index_v[rv, rov]],
                    rearrange(
                        sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"
                    ),
                    rearrange(
                        sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"
                    ),
                    color,
                    3,
                    cap="butt",
                    x_range=(0, 1),
                    y_range=(0, 1),
                )
                vis_layer.append(vis_layer_head)
            vis.append(add_label(vcat(*vis_layer), f"Layer {il}"))
        vis = add_label(add_border(add_border(hcat(*vis)), 1, 0), "Keys & Values")
        vis = add_border(hcat(add_label(ray_view, "ray_view"), vis, align="top"))
        return vis

    def visualize_depth(
        self,
        context: BatchedViews,
        multi_depth: Float[Tensor, "batch view height width surface spp"],
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        multi_vis = []
        *_, srf, _ = multi_depth.shape
        for i in range(srf):
            depth = multi_depth[..., i, :]
            depth = depth.mean(dim=-1)

            # Compute relative depth and disparity.
            near = rearrange(context["near"], "b v -> b v () ()")
            far = rearrange(context["far"], "b v -> b v () ()")
            relative_depth = (depth - near) / (far - near)
            relative_disparity = 1 - (1 / depth - 1 / far) / (1 / near - 1 / far)

            relative_depth = apply_color_map_to_image(relative_depth, "turbo")
            relative_depth = vcat(*[hcat(*x) for x in relative_depth])
            relative_depth = add_label(relative_depth, "Depth")
            relative_disparity = apply_color_map_to_image(relative_disparity, "turbo")
            relative_disparity = vcat(*[hcat(*x) for x in relative_disparity])
            relative_disparity = add_label(relative_disparity, "Disparity")
            multi_vis.append(add_border(hcat(relative_depth, relative_disparity)))

        return add_border(vcat(*multi_vis))

    def visualize_overlaps(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        is_monocular: Optional[Bool[Tensor, "batch view height width"]] = None,
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        device = context_images.device
        b, v, _, h, w = context_images.shape
        green = torch.tensor([0.235, 0.706, 0.294], device=device)[..., None, None]
        rb = randrange(b)
        valid = sampling.valid[rb].float()
        ds = self.encoder.cfg.epipolar_transformer.downscale
        valid = repeat(
            valid,
            "v ov (h w) -> v ov c (h rh) (w rw)",
            c=3,
            h=h // ds,
            w=w // ds,
            rh=ds,
            rw=ds,
        )

        if is_monocular is not None:
            is_monocular = is_monocular[rb].float()
            is_monocular = repeat(is_monocular, "v h w -> v c h w", c=3, h=h, w=w)

        # Select context images in grid.
        context_images = context_images[rb]
        index, _ = generate_heterogeneous_index(v)
        valid = valid * (green + context_images[index]) / 2

        vis = vcat(*(hcat(im, hcat(*v)) for im, v in zip(context_images, valid)))
        vis = add_label(vis, "Context Overlaps")

        if is_monocular is not None:
            vis = hcat(vis, add_label(vcat(*is_monocular), "Monocular?"))

        return add_border(vis)

    def visualize_gaussians(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        opacities: Float[Tensor, "batch vrspp"],
        covariances: Float[Tensor, "batch vrspp 3 3"],
        colors: Float[Tensor, "batch vrspp 3"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        b, v, _, h, w = context_images.shape
        rb = randrange(b)
        context_images = context_images[rb]
        opacities = repeat(
            opacities[rb], "(v h w spp) -> spp v c h w", v=v, c=3, h=h, w=w
        )
        colors = rearrange(colors[rb], "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        # Color-map Gaussian covariawnces.
        det = covariances[rb].det()
        det = apply_color_map(det / det.max(), "inferno")
        det = rearrange(det, "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        return add_border(
            hcat(
                add_label(box(hcat(*context_images)), "Context"),
                add_label(box(vcat(*[hcat(*x) for x in opacities])), "Opacities"),
                add_label(
                    box(vcat(*[hcat(*x) for x in (colors * opacities)])), "Colors"
                ),
                add_label(box(vcat(*[hcat(*x) for x in colors])), "Colors (Raw)"),
                add_label(box(vcat(*[hcat(*x) for x in det])), "Determinant"),
            )
        )

    def visualize_probabilities(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        pdf: Float[Tensor, "batch view ray sample"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, _, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)
        colors = [get_distinct_color(i) for i, _ in enumerate(rr)]
        colors = torch.tensor(colors, dtype=torch.float32, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            colors,
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize probabilities in the sample view.
        pdf = pdf[rb, rv, rr]
        pdf = rearrange(pdf, "r s -> r s ()")
        colors = rearrange(colors, "r c -> r () c")
        sample_view = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(pdf * colors, "r s c -> (r s) c"),
            6,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize rescaled probabilities in the sample view.
        pdf_magnified = pdf / reduce(pdf, "r s () -> r () ()", "max")
        sample_view_magnified = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(pdf_magnified * colors, "r s c -> (r s) c"),
            6,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(
                add_label(ray_view, "Rays"),
                add_label(sample_view, "Samples"),
                add_label(sample_view_magnified, "Samples (Magnified PDF)"),
            )
        )

    def visualize_epipolar_samples(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize the samples and epipolar lines in the sample view.
        # First, draw the epipolar line in black.
        sample_view = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            sampling.xy_sample_near[rb, rv, rov, rr, 0],
            sampling.xy_sample_far[rb, rv, rov, rr, -1],
            0,
            5,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Create an alternating line color for the buckets.
        color = repeat(
            torch.tensor([0, 1], device=device),
            "ab -> r (s ab) c",
            r=len(rr),
            s=(s + 1) // 2,
            c=3,
        )
        color = rearrange(color[:, :s], "r s c -> (r s) c")

        # Draw the alternating bucket lines.
        sample_view = draw_lines(
            sample_view,
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            color,
            3,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Draw the sample points.
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            [get_distinct_color(i // s) for i in range(s * len(rr))],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(add_label(ray_view, "Ray View"), add_label(sample_view, "Sample View"))
        )

    def visualize_epipolar_color_samples(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        context: BatchedViews,
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        sampling = self.encoder.sampler(
            context["image"],
            context["extrinsics"],
            context["intrinsics"],
            context["near"],
            context["far"],
        )

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize the samples and in the sample view.
        sample_view = draw_points(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            [get_distinct_color(i // s) for i in range(s * len(rr))],
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.features[rb, rv, rov, rr], "r s c -> (r s) c"),
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(add_label(ray_view, "Ray View"), add_label(sample_view, "Sample View"))
        )
