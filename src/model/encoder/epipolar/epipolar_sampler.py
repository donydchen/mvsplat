from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Shaped
from torch import Tensor, nn

from ....geometry.epipolar_lines import project_rays
from ....geometry.projection import get_world_rays, sample_image_grid
from ....misc.heterogeneous_pairings import (
    Index,
    generate_heterogeneous_index,
    generate_heterogeneous_index_transpose,
)


@dataclass
class EpipolarSampling:
    features: Float[Tensor, "batch view other_view ray sample channel"]
    valid: Bool[Tensor, "batch view other_view ray"]
    xy_ray: Float[Tensor, "batch view ray 2"]
    xy_sample: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_near: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_far: Float[Tensor, "batch view other_view ray sample 2"]
    origins: Float[Tensor, "batch view ray 3"]
    directions: Float[Tensor, "batch view ray 3"]


class EpipolarSampler(nn.Module):
    num_samples: int
    index_v: Index
    transpose_v: Index
    transpose_ov: Index

    def __init__(
        self,
        num_views: int,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

        # Generate indices needed to sample only other views.
        _, index_v = generate_heterogeneous_index(num_views)
        t_v, t_ov = generate_heterogeneous_index_transpose(num_views)
        self.register_buffer("index_v", index_v, persistent=False)
        self.register_buffer("transpose_v", t_v, persistent=False)
        self.register_buffer("transpose_ov", t_ov, persistent=False)

    def forward(
        self,
        images: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
    ) -> EpipolarSampling:
        device = images.device
        b, v, _, _, _ = images.shape

        # Generate the rays that are projected onto other views.
        xy_ray, origins, directions = self.generate_image_rays(
            images, extrinsics, intrinsics
        )

        # Select the camera extrinsics and intrinsics to project onto. For each context
        # view, this means all other context views in the batch.
        projection = project_rays(
            rearrange(origins, "b v r xyz -> b v () r xyz"),
            rearrange(directions, "b v r xyz -> b v () r xyz"),
            rearrange(self.collect(extrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(self.collect(intrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(near, "b v -> b v () ()"),
            rearrange(far, "b v -> b v () ()"),
        )


        # Generate sample points.
        s = self.num_samples
        sample_depth = (torch.arange(s, device=device) + 0.5) / s
        sample_depth = rearrange(sample_depth, "s -> s ()")
        xy_min = projection["xy_min"].nan_to_num(posinf=0, neginf=0) 
        xy_min = xy_min * projection["overlaps_image"][..., None]
        xy_min = rearrange(xy_min, "b v ov r xy -> b v ov r () xy")
        xy_max = projection["xy_max"].nan_to_num(posinf=0, neginf=0) 
        xy_max = xy_max * projection["overlaps_image"][..., None]
        xy_max = rearrange(xy_max, "b v ov r xy -> b v ov r () xy")
        xy_sample = xy_min + sample_depth * (xy_max - xy_min)

        # The samples' shape is (batch, view, other_view, ...). However, before the
        # transpose, the view dimension refers to the view from which the ray is cast,
        # not the view from which samples are drawn. Thus, we need to transpose the
        # samples so that the view dimension refers to the view from which samples are
        # drawn. If the diagonal weren't removed for efficiency, this would be a literal
        # transpose. In our case, it's as if the diagonal were re-added, the transpose
        # were taken, and the diagonal were then removed again.
        samples = self.transpose(xy_sample)
        samples = F.grid_sample(
            rearrange(images, "b v c h w -> (b v) c h w"),
            rearrange(2 * samples - 1, "b v ov r s xy -> (b v) (ov r s) () xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        samples = rearrange(
            samples, "(b v) c (ov r s) () -> b v ov r s c", b=b, v=v, ov=v - 1, s=s
        )
        samples = self.transpose(samples)

        # Zero out invalid samples.
        samples = samples * projection["overlaps_image"][..., None, None]

        half_span = 0.5 / s
        return EpipolarSampling(
            features=samples,
            valid=projection["overlaps_image"],
            xy_ray=xy_ray,
            xy_sample=xy_sample,
            xy_sample_near=xy_min + (sample_depth - half_span) * (xy_max - xy_min),
            xy_sample_far=xy_min + (sample_depth + half_span) * (xy_max - xy_min),
            origins=origins,
            directions=directions,
        )

    def generate_image_rays(
        self,
        images: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
    ) -> tuple[
        Float[Tensor, "batch view ray 2"],  # xy
        Float[Tensor, "batch view ray 3"],  # origins
        Float[Tensor, "batch view ray 3"],  # directions
    ]:
        """Generate the rays along which Gaussians are defined. For now, these rays are
        simply arranged in a grid.
        """
        b, v, _, h, w = images.shape
        xy, _ = sample_image_grid((h, w), device=images.device)
        origins, directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            rearrange(extrinsics, "b v i j -> b v () i j"),
            rearrange(intrinsics, "b v i j -> b v () i j"),
        )
        return repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), origins, directions

    def transpose(
        self,
        x: Shaped[Tensor, "batch view other_view *rest"],
    ) -> Shaped[Tensor, "batch view other_view *rest"]:
        b, v, ov, *_ = x.shape
        t_b = torch.arange(b, device=x.device)
        t_b = repeat(t_b, "b -> b v ov", v=v, ov=ov)
        t_v = repeat(self.transpose_v, "v ov -> b v ov", b=b)
        t_ov = repeat(self.transpose_ov, "v ov -> b v ov", b=b)
        return x[t_b, t_v, t_ov]

    def collect(
        self,
        target: Shaped[Tensor, "batch view ..."],
    ) -> Shaped[Tensor, "batch view view-1 ..."]:
        b, v, *_ = target.shape
        index_b = torch.arange(b, device=target.device)
        index_b = repeat(index_b, "b -> b v ov", v=v, ov=v - 1)
        index_v = repeat(self.index_v, "v ov -> b v ov", b=b)
        return target[index_b, index_v]
