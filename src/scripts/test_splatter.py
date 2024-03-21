from pathlib import Path
from time import time

import torch
from einops import einsum, repeat
from jaxtyping import install_import_hook
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.misc.image_io import save_image
    from src.misc.sh_rotation import rotate_sh
    from src.model.decoder.cuda_splatting import render_cuda
    from src.visualization.camera_trajectory.spin import generate_spin


if __name__ == "__main__":
    NUM_FRAMES = 60
    NUM_GAUSSIANS = 1
    DEGREE = 4
    IMAGE_SHAPE = (512, 512)
    RESULT_PATH = Path("outputs/test_splatter")

    device = torch.device("cuda:0")

    # Generate camera parameters.
    extrinsics = generate_spin(60, device, 0.0, 10.0)
    intrinsics = torch.eye(3, dtype=torch.float32, device=device)
    intrinsics[:2, 2] = 0.5
    intrinsics[:2, :2] *= 0.5
    intrinsics = repeat(intrinsics, "i j -> b i j", b=NUM_FRAMES)

    # Generate Gaussians.
    means = torch.randn((NUM_GAUSSIANS, 3), dtype=torch.float32, device=device) * 0
    scales = torch.rand((NUM_GAUSSIANS, 3), dtype=torch.float32, device=device) * 0 + 1
    rotations = R.random(NUM_GAUSSIANS).as_matrix()
    rotations = torch.tensor(rotations, dtype=torch.float32, device=device)
    covariances = rotations @ scales.diag_embed()
    covariances = einsum(covariances, covariances, "b i j, b k j -> b i k")
    sh_coefficients = torch.randn(
        (NUM_GAUSSIANS, 3, (DEGREE + 1) ** 2), dtype=torch.float32, device=device
    )

    # https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Spherical_Harmonics.png
    # we are rolling forward, rotation-wise
    # red is blue, blue is yellow
    # we are rotating about the y axis

    sh_coefficients[:] = 0

    # sh_coefficients[:, 0, 1] = 10  # rotation does not change this // -1
    # sh_coefficients[:, 0, 2] = 10  # in/out // 0
    # sh_coefficients[:, 0, 3] = 10  # sides // 1

    sh_coefficients[:, 0, 4] = 10  # rotation does not change this // -2
    sh_coefficients[:, 0, 5] = 10  # rotation does not change this // -1
    sh_coefficients[:, 0, 6] = 10  # BRBR // 0
    sh_coefficients[:, 0, 7] = 10  # BRBR // 1
    sh_coefficients[:, 0, 8] = 10  # 2x red // 2

    opacities = torch.rand(NUM_GAUSSIANS, dtype=torch.float32, device=device) * 0 + 1

    # rotate_sh(sh_coefficients, extrinsics[0, :3, :3].inverse())

    # Render images using the CUDA splatter.
    start_time = time()
    rendered_cuda = [
        render_cuda(
            c2w[None],
            k[None],
            torch.tensor([0.1], dtype=torch.float32, device=device),
            torch.tensor([20.0], dtype=torch.float32, device=device),
            IMAGE_SHAPE,
            torch.zeros((1, 3), dtype=torch.float32, device=device),
            means[None],
            covariances[None],
            rotate_sh(sh_coefficients, c2w[:3, :3])[None],
            # sh_coefficients[None],
            opacities[None],
        )[0]
        for c2w, k in zip(tqdm(extrinsics, desc="Rendering"), intrinsics)
    ]
    print(f"CUDA rendering took {time() - start_time:.2f} seconds.")

    RESULT_PATH.mkdir(exist_ok=True, parents=True)
    for index, frame in enumerate(tqdm(rendered_cuda, "Saving images")):
        save_image(frame, RESULT_PATH / f"frame_{index:0>3}.png")

    import os

    cmd = (
        'ffmpeg -y -framerate 30 -pattern_type glob -i "*.png"  -c:v libx264 -pix_fmt '
        'yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" animation.mp4'
    )
    os.system(f"cd {RESULT_PATH} && {cmd}")

    a = 1
