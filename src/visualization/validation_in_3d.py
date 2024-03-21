import torch
from jaxtyping import Float, Shaped
from torch import Tensor

from ..model.decoder.cuda_splatting import render_cuda_orthographic
from ..model.types import Gaussians
from ..visualization.annotation import add_label
from ..visualization.drawing.cameras import draw_cameras
from .drawing.cameras import compute_equal_aabb_with_margin


def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results


def render_projections(
    gaussians: Gaussians,
    resolution: int,
    margin: float = 0.1,
    draw_label: bool = True,
    extra_label: str = "",
) -> Float[Tensor, "batch 3 3 height width"]:
    device = gaussians.means.device
    b, _, _ = gaussians.means.shape

    # Compute the minima and maxima of the scene.
    minima = gaussians.means.min(dim=1).values
    maxima = gaussians.means.max(dim=1).values
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(
        minima, maxima, margin=margin
    )

    projections = []
    for look_axis in range(3):
        right_axis = (look_axis + 1) % 3
        down_axis = (look_axis + 2) % 3

        # Define the extrinsics for rendering.
        extrinsics = torch.zeros((b, 4, 4), dtype=torch.float32, device=device)
        extrinsics[:, right_axis, 0] = 1
        extrinsics[:, down_axis, 1] = 1
        extrinsics[:, look_axis, 2] = 1
        extrinsics[:, right_axis, 3] = 0.5 * (
            scene_minima[:, right_axis] + scene_maxima[:, right_axis]
        )
        extrinsics[:, down_axis, 3] = 0.5 * (
            scene_minima[:, down_axis] + scene_maxima[:, down_axis]
        )
        extrinsics[:, look_axis, 3] = scene_minima[:, look_axis]
        extrinsics[:, 3, 3] = 1

        # Define the intrinsics for rendering.
        extents = scene_maxima - scene_minima
        far = extents[:, look_axis]
        near = torch.zeros_like(far)
        width = extents[:, right_axis]
        height = extents[:, down_axis]

        projection = render_cuda_orthographic(
            extrinsics,
            width,
            height,
            near,
            far,
            (resolution, resolution),
            torch.zeros((b, 3), dtype=torch.float32, device=device),
            gaussians.means,
            gaussians.covariances,
            gaussians.harmonics,
            gaussians.opacities,
            fov_degrees=10.0,
        )
        if draw_label:
            right_axis_name = "XYZ"[right_axis]
            down_axis_name = "XYZ"[down_axis]
            label = f"{right_axis_name}{down_axis_name} Projection {extra_label}"
            projection = torch.stack([add_label(x, label) for x in projection])

        projections.append(projection)

    return torch.stack(pad(projections), dim=1)


def render_cameras(batch: dict, resolution: int) -> Float[Tensor, "3 3 height width"]:
    # Define colors for context and target views.
    num_context_views = batch["context"]["extrinsics"].shape[1]
    num_target_views = batch["target"]["extrinsics"].shape[1]
    color = torch.ones(
        (num_target_views + num_context_views, 3),
        dtype=torch.float32,
        device=batch["target"]["extrinsics"].device,
    )
    color[num_context_views:, 1:] = 0

    return draw_cameras(
        resolution,
        torch.cat(
            (batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0])
        ),
        torch.cat(
            (batch["context"]["intrinsics"][0], batch["target"]["intrinsics"][0])
        ),
        color,
        torch.cat((batch["context"]["near"][0], batch["target"]["near"][0])),
        torch.cat((batch["context"]["far"][0], batch["target"]["far"][0])),
    )
