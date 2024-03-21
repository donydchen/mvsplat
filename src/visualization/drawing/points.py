from typing import Optional

import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor

from .coordinate_conversion import generate_conversions
from .rendering import render_over_image
from .types import Pair, Scalar, Vector, sanitize_scalar, sanitize_vector


def draw_points(
    image: Float[Tensor, "3 height width"],
    points: Vector,
    color: Vector = [1, 1, 1],
    radius: Scalar = 1,
    inner_radius: Scalar = 0,
    num_msaa_passes: int = 1,
    x_range: Optional[Pair] = None,
    y_range: Optional[Pair] = None,
) -> Float[Tensor, "3 height width"]:
    device = image.device
    points = sanitize_vector(points, 2, device)
    color = sanitize_vector(color, 3, device)
    radius = sanitize_scalar(radius, device)
    inner_radius = sanitize_scalar(inner_radius, device)
    (num_points,) = torch.broadcast_shapes(
        points.shape[0],
        color.shape[0],
        radius.shape,
        inner_radius.shape,
    )

    # Convert world-space points to pixel space.
    _, h, w = image.shape
    world_to_pixel, _ = generate_conversions((h, w), device, x_range, y_range)
    points = world_to_pixel(points)

    def color_function(
        xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:
        # Define a vector between the start and end points.
        delta = xy[:, None] - points[None]
        delta_norm = delta.norm(dim=-1)
        mask = (delta_norm >= inner_radius[None]) & (delta_norm <= radius[None])

        # Determine the sample's color.
        selectable_color = color.broadcast_to((num_points, 3))
        arrangement = mask * torch.arange(num_points, device=device)
        top_color = selectable_color.gather(
            dim=0,
            index=repeat(arrangement.argmax(dim=1), "s -> s c", c=3),
        )
        rgba = torch.cat((top_color, mask.any(dim=1).float()[:, None]), dim=-1)

        return rgba

    return render_over_image(image, color_function, device, num_passes=num_msaa_passes)
