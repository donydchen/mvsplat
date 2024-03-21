import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
def generate_wobble_transformation(
    radius: Float[Tensor, "*#batch"],
    t: Float[Tensor, " time_step"],
    num_rotations: int = 1,
    scale_radius_with_t: bool = True,
) -> Float[Tensor, "*batch time_step 4 4"]:
    # Generate a translation in the image plane.
    tf = torch.eye(4, dtype=torch.float32, device=t.device)
    tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
    radius = radius[..., None]
    if scale_radius_with_t:
        radius = radius * t
    tf[..., 0, 3] = torch.sin(2 * torch.pi * num_rotations * t) * radius
    tf[..., 1, 3] = -torch.cos(2 * torch.pi * num_rotations * t) * radius
    return tf


@torch.no_grad()
def generate_wobble(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    radius: Float[Tensor, "*#batch"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 4 4"]:
    tf = generate_wobble_transformation(radius, t)
    return rearrange(extrinsics, "... i j -> ... () i j") @ tf
