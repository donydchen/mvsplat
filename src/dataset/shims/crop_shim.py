import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    original_shape = image.shape[1:]  # (h_in, w_in)

    #print(f"  Single image: {original_shape[0]}x{original_shape[1]} -> {h}x{w}")

    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # DEBUG: Print MVSplat center crop info
    #print(f"  Input size: {h_in}x{w_in} -> Output: {h_out}x{w_out}")
    #print(f"  Crop offset: row={row}, col={col}")

    # Debug intrinsics before crop
    if intrinsics.dim() >= 3:
        sample_K = intrinsics[0, 0] if intrinsics.dim() == 4 else intrinsics[0]
    else:
        sample_K = intrinsics

    pre_fx = sample_K[0, 0].item()
    pre_fy = sample_K[1, 1].item()
    pre_cx = sample_K[0, 2].item()
    pre_cy = sample_K[1, 2].item()
    #print(f"  Pre-crop intrinsics: fx={pre_fx:.4f}, fy={pre_fy:.4f}, cx={pre_cx:.4f}, cy={pre_cy:.4f}")

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    #print(f"  fx scale factor: {w_in / w_out:.4f}, fy scale factor: {h_in / h_out:.4f}")
    intrinsics[..., 0, 0] *= w_in / w_out  # fx - THIS IS WRONG FOR CENTER CROP!
    intrinsics[..., 1, 1] *= h_in / h_out  # fy - THIS IS WRONG FOR CENTER CROP!

    # Debug intrinsics after crop
    if intrinsics.dim() >= 3:
        sample_K_post = intrinsics[0, 0] if intrinsics.dim() == 4 else intrinsics[0]
    else:
        sample_K_post = intrinsics

    post_fx = sample_K_post[0, 0].item()
    post_fy = sample_K_post[1, 1].item()
    post_cx = sample_K_post[0, 2].item()
    post_cy = sample_K_post[1, 2].item()
    #print(f"  Post-crop intrinsics: fx={post_fx:.4f}, fy={post_fy:.4f}, cx={post_cx:.4f}, cy={post_cy:.4f}")

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # DEBUG: Print MVSplat rescale and crop info
    #print(f"  Input size: {h_in}x{w_in} -> Target: {h_out}x{w_out}")
    #print(f"  Scale factor: {scale_factor:.4f}")
    #print(f"  Scaled size: {h_scaled}x{w_scaled}")

    # Debug intrinsics before processing
    if intrinsics.dim() >= 3:
        sample_K = intrinsics[0, 0] if intrinsics.dim() == 4 else intrinsics[0]
    else:
        sample_K = intrinsics

    pre_fx = sample_K[0, 0].item()
    pre_fy = sample_K[1, 1].item()
    pre_cx = sample_K[0, 2].item()
    pre_cy = sample_K[1, 2].item()
    #print(f"  Input intrinsics: fx={pre_fx:.4f}, fy={pre_fy:.4f}, cx={pre_cx:.4f}, cy={pre_cy:.4f}")

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    #print(f"  Expected scale factors for intrinsics: scale_x={w_scaled/w_in:.4f}, scale_y={h_scaled/h_in:.4f}")

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    #print(f"  Input image shape: {views['image'].shape}")
    #print(f"  Input intrinsics shape: {views['intrinsics'].shape}")
    #print(f"  Target shape: {shape}")

    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)

    #print(f"  Output image shape: {images.shape}")
    #print(f"  Output intrinsics shape: {intrinsics.shape}")

    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    #print(f"  Target shape: {shape}")
    #print("=" * 60)

    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
