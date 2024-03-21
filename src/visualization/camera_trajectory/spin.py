import numpy as np
import torch
from einops import repeat
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def generate_spin(
    num_frames: int,
    device: torch.device,
    elevation: float,
    radius: float,
) -> Float[Tensor, "frame 4 4"]:
    # Translate back along the camera's look vector.
    tf_translation = torch.eye(4, dtype=torch.float32, device=device)
    tf_translation[:2] *= -1
    tf_translation[2, 3] = -radius

    # Generate the transformation for the azimuth.
    phi = 2 * np.pi * (np.arange(num_frames) / num_frames)
    rotation_vectors = np.stack([np.zeros_like(phi), phi, np.zeros_like(phi)], axis=-1)

    azimuth = R.from_rotvec(rotation_vectors).as_matrix()
    azimuth = torch.tensor(azimuth, dtype=torch.float32, device=device)
    tf_azimuth = torch.eye(4, dtype=torch.float32, device=device)
    tf_azimuth = repeat(tf_azimuth, "i j -> b i j", b=num_frames).clone()
    tf_azimuth[:, :3, :3] = azimuth

    # Generate the transformation for the elevation.
    deg_elevation = np.deg2rad(elevation)
    elevation = R.from_rotvec(np.array([deg_elevation, 0, 0], dtype=np.float32))
    elevation = torch.tensor(elevation.as_matrix())
    tf_elevation = torch.eye(4, dtype=torch.float32, device=device)
    tf_elevation[:3, :3] = elevation

    return tf_azimuth @ tf_elevation @ tf_translation
