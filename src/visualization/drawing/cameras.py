from typing import Optional

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import unproject
from ..annotation import add_label
from .lines import draw_lines
from .types import Scalar, sanitize_scalar


def draw_cameras(
    resolution: int,
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    color: Float[Tensor, "batch 3"],
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
    margin: float = 0.1,  # relative to AABB
    frustum_scale: float = 0.05,  # relative to image resolution
) -> Float[Tensor, "3 3 height width"]:
    device = extrinsics.device

    # Compute scene bounds.
    minima, maxima = compute_aabb(extrinsics, intrinsics, near, far)
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(
        minima, maxima, margin=margin
    )
    span = (scene_maxima - scene_minima).max()

    # Compute frustum locations.
    corner_depth = (span * frustum_scale)[None]
    frustum_corners = unproject_frustum_corners(extrinsics, intrinsics, corner_depth)
    if near is not None:
        near_corners = unproject_frustum_corners(extrinsics, intrinsics, near)
    if far is not None:
        far_corners = unproject_frustum_corners(extrinsics, intrinsics, far)

    # Project the cameras onto each axis-aligned plane.
    projections = []
    for projected_axis in range(3):
        image = torch.zeros(
            (3, resolution, resolution),
            dtype=torch.float32,
            device=device,
        )
        image_x_axis = (projected_axis + 1) % 3
        image_y_axis = (projected_axis + 2) % 3

        def project(points: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 2"]:
            x = points[..., image_x_axis]
            y = points[..., image_y_axis]
            return torch.stack([x, y], dim=-1)

        x_range, y_range = torch.stack(
            (project(scene_minima), project(scene_maxima)), dim=-1
        )

        # Draw near and far planes.
        if near is not None:
            projected_near_corners = project(near_corners)
            image = draw_lines(
                image,
                rearrange(projected_near_corners, "b p xy -> (b p) xy"),
                rearrange(projected_near_corners.roll(1, 1), "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )
        if far is not None:
            projected_far_corners = project(far_corners)
            image = draw_lines(
                image,
                rearrange(projected_far_corners, "b p xy -> (b p) xy"),
                rearrange(projected_far_corners.roll(1, 1), "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )
        if near is not None and far is not None:
            image = draw_lines(
                image,
                rearrange(projected_near_corners, "b p xy -> (b p) xy"),
                rearrange(projected_far_corners, "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )

        # Draw the camera frustums themselves.
        projected_origins = project(extrinsics[:, :3, 3])
        projected_frustum_corners = project(frustum_corners)
        start = [
            repeat(projected_origins, "b xy -> (b p) xy", p=4),
            rearrange(projected_frustum_corners.roll(1, 1), "b p xy -> (b p) xy"),
        ]
        start = rearrange(torch.cat(start, dim=0), "(r b p) xy -> (b r p) xy", r=2, p=4)
        image = draw_lines(
            image,
            start,
            repeat(projected_frustum_corners, "b p xy -> (b r p) xy", r=2),
            color=repeat(color, "b c -> (b r p) c", r=2, p=4),
            width=2,
            x_range=x_range,
            y_range=y_range,
        )

        x_name = "XYZ"[image_x_axis]
        y_name = "XYZ"[image_y_axis]
        image = add_label(image, f"{x_name}{y_name} Projection")

        # TODO: Draw axis indicators.
        projections.append(image)

    return torch.stack(projections)


def compute_aabb(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
) -> tuple[
    Float[Tensor, "3"],  # minima of the scene
    Float[Tensor, "3"],  # maxima of the scene
]:
    """Compute an axis-aligned bounding box for the camera frustums."""

    device = extrinsics.device

    # These points are included in the AABB.
    points = [extrinsics[:, :3, 3]]

    if near is not None:
        near = sanitize_scalar(near, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, near)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))

    if far is not None:
        far = sanitize_scalar(far, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, far)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))

    points = torch.cat(points, dim=0)
    return points.min(dim=0).values, points.max(dim=0).values


def compute_equal_aabb_with_margin(
    minima: Float[Tensor, "*#batch 3"],
    maxima: Float[Tensor, "*#batch 3"],
    margin: float = 0.1,
) -> tuple[
    Float[Tensor, "*batch 3"],  # minima of the scene
    Float[Tensor, "*batch 3"],  # maxima of the scene
]:
    midpoint = (maxima + minima) * 0.5
    span = (maxima - minima).max() * (1 + margin)
    scene_minima = midpoint - 0.5 * span
    scene_maxima = midpoint + 0.5 * span
    return scene_minima, scene_maxima


def unproject_frustum_corners(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    depth: Float[Tensor, "#batch"],
) -> Float[Tensor, "batch 4 3"]:
    device = extrinsics.device

    # Get coordinates for the corners. Following them in a circle makes a rectangle.
    xy = torch.linspace(0, 1, 2, device=device)
    xy = torch.stack(torch.meshgrid(xy, xy, indexing="xy"), dim=-1)
    xy = rearrange(xy, "i j xy -> (i j) xy")
    xy = xy[torch.tensor([0, 1, 3, 2], device=device)]

    # Get ray directions in camera space.
    directions = unproject(
        xy,
        torch.ones(1, dtype=torch.float32, device=device),
        rearrange(intrinsics, "b i j -> b () i j"),
    )

    # Divide by the z coordinate so that multiplying by depth will produce orthographic
    # depth (z depth) as opposed to Euclidean depth (distance from the camera).
    directions = directions / directions[..., -1:]
    directions = einsum(extrinsics[..., :3, :3], directions, "b i j, b r j -> b r i")

    origins = rearrange(extrinsics[:, :3, 3], "b xyz -> b () xyz")
    depth = rearrange(depth, "b -> b () ()")
    return origins + depth * directions
