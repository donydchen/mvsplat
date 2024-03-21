from jaxtyping import Float
from torch import Tensor


def relative_disparity_to_depth(
    relative_disparity: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert relative disparity, where 0 is near and 1 is far, to depth."""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    return 1 / ((1 - relative_disparity) * (disp_near - disp_far) + disp_far + eps)


def depth_to_relative_disparity(
    depth: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert depth to relative disparity, where 0 is near and 1 is far"""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    disp = 1 / (depth + eps)
    return 1 - (disp - disp_far) / (disp_near - disp_far + eps)
