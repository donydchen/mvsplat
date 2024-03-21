from typing import Literal, Optional

import torch
from einops import einsum, repeat
from jaxtyping import Float
from torch import Tensor

from .coordinate_conversion import generate_conversions
from .rendering import render_over_image
from .types import Pair, Scalar, Vector, sanitize_scalar, sanitize_vector


def draw_lines(
    image: Float[Tensor, "3 height width"],
    start: Vector,
    end: Vector,
    color: Vector,
    width: Scalar,
    cap: Literal["butt", "round", "square"] = "round",
    num_msaa_passes: int = 1,
    x_range: Optional[Pair] = None,
    y_range: Optional[Pair] = None,
) -> Float[Tensor, "3 height width"]:
    device = image.device
    start = sanitize_vector(start, 2, device)
    end = sanitize_vector(end, 2, device)
    color = sanitize_vector(color, 3, device)
    width = sanitize_scalar(width, device)
    (num_lines,) = torch.broadcast_shapes(
        start.shape[0],
        end.shape[0],
        color.shape[0],
        width.shape,
    )

    # Convert world-space points to pixel space.
    _, h, w = image.shape
    world_to_pixel, _ = generate_conversions((h, w), device, x_range, y_range)
    start = world_to_pixel(start)
    end = world_to_pixel(end)

    def color_function(
        xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:
        # Define a vector between the start and end points.
        delta = end - start
        delta_norm = delta.norm(dim=-1, keepdim=True)
        u_delta = delta / delta_norm

        # Define a vector between each sample and the start point.
        indicator = xy - start[:, None]

        # Determine whether each sample is inside the line in the parallel direction.
        extra = 0.5 * width[:, None] if cap == "square" else 0
        parallel = einsum(u_delta, indicator, "l xy, l s xy -> l s")
        parallel_inside_line = (parallel <= delta_norm + extra) & (parallel > -extra)

        # Determine whether each sample is inside the line perpendicularly.
        perpendicular = indicator - parallel[..., None] * u_delta[:, None]
        perpendicular_inside_line = perpendicular.norm(dim=-1) < 0.5 * width[:, None]

        inside_line = parallel_inside_line & perpendicular_inside_line

        # Compute round caps.
        if cap == "round":
            near_start = indicator.norm(dim=-1) < 0.5 * width[:, None]
            inside_line |= near_start
            end_indicator = indicator = xy - end[:, None]
            near_end = end_indicator.norm(dim=-1) < 0.5 * width[:, None]
            inside_line |= near_end

        # Determine the sample's color.
        selectable_color = color.broadcast_to((num_lines, 3))
        arrangement = inside_line * torch.arange(num_lines, device=device)[:, None]
        top_color = selectable_color.gather(
            dim=0,
            index=repeat(arrangement.argmax(dim=0), "s -> s c", c=3),
        )
        rgba = torch.cat((top_color, inside_line.any(dim=0).float()[:, None]), dim=-1)

        return rgba

    return render_over_image(image, color_function, device, num_passes=num_msaa_passes)
