from typing import Optional, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from .types import Pair, sanitize_pair


@runtime_checkable
class ConversionFunction(Protocol):
    def __call__(
        self,
        xy: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, "*batch 2"]:
        pass


def generate_conversions(
    shape: tuple[int, int],
    device: torch.device,
    x_range: Optional[Pair] = None,
    y_range: Optional[Pair] = None,
) -> tuple[
    ConversionFunction,  # conversion from world coordinates to pixel coordinates
    ConversionFunction,  # conversion from pixel coordinates to world coordinates
]:
    h, w = shape
    x_range = sanitize_pair((0, w) if x_range is None else x_range, device)
    y_range = sanitize_pair((0, h) if y_range is None else y_range, device)
    minima, maxima = torch.stack((x_range, y_range), dim=-1)
    wh = torch.tensor((w, h), dtype=torch.float32, device=device)

    def convert_world_to_pixel(
        xy: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, "*batch 2"]:
        return (xy - minima) / (maxima - minima) * wh

    def convert_pixel_to_world(
        xy: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, "*batch 2"]:
        return xy / wh * (maxima - minima) + minima

    return convert_world_to_pixel, convert_pixel_to_world
