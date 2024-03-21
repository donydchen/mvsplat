from math import prod

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int64
from torch import Tensor


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def sample_training_rays(
    image: Float[Tensor, "batch view channel ..."],
    intrinsics: Float[Tensor, "batch view dim dim"],
    extrinsics: Float[Tensor, "batch view dim+1 dim+1"],
    num_rays: int,
) -> tuple[
    Float[Tensor, "batch ray dim"],  # origins
    Float[Tensor, "batch ray dim"],  # directions
    Float[Tensor, "batch ray 3"],  # sampled color
]:
    device = extrinsics.device
    b, v, _, *grid_shape = image.shape

    # Generate all possible target rays.
    xy, _ = sample_image_grid(tuple(grid_shape), device)
    origins, directions = get_world_rays(
        rearrange(xy, "... d -> ... () () d"),
        extrinsics,
        intrinsics,
    )
    origins = rearrange(origins, "... b v xy -> b (v ...) xy", b=b, v=v)
    directions = rearrange(directions, "... b v xy -> b (v ...) xy", b=b, v=v)
    pixels = rearrange(image, "b v c ... -> b (v ...) c")

    # Sample random rays.
    num_possible_rays = v * prod(grid_shape)
    ray_indices = torch.randint(num_possible_rays, (b, num_rays), device=device)
    batch_indices = repeat(torch.arange(b, device=device), "b -> b n", n=num_rays)

    return (
        origins[batch_indices, ray_indices],
        directions[batch_indices, ray_indices],
        pixels[batch_indices, ray_indices],
    )


def intersect_rays(
    origins_x: Float[Tensor, "*#batch 3"],
    directions_x: Float[Tensor, "*#batch 3"],
    origins_y: Float[Tensor, "*#batch 3"],
    directions_y: Float[Tensor, "*#batch 3"],
    eps: float = 1e-5,
    inf: float = 1e10,
) -> Float[Tensor, "*batch 3"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast the rays so their shapes match.
    shape = torch.broadcast_shapes(
        origins_x.shape,
        directions_x.shape,
        origins_y.shape,
        directions_y.shape,
    )
    origins_x = origins_x.broadcast_to(shape)
    directions_x = directions_x.broadcast_to(shape)
    origins_y = origins_y.broadcast_to(shape)
    directions_y = directions_y.broadcast_to(shape)

    # Detect and remove batch elements where the directions are parallel.
    parallel = einsum(directions_x, directions_y, "... xyz, ... xyz -> ...") > 1 - eps
    origins_x = origins_x[~parallel]
    directions_x = directions_x[~parallel]
    origins_y = origins_y[~parallel]
    directions_y = directions_y[~parallel]

    # Stack the rays into (2, *shape).
    origins = torch.stack([origins_x, origins_y], dim=0)
    directions = torch.stack([directions_x, directions_y], dim=0)
    dtype = origins.dtype
    device = origins.device

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "r b i, r b j -> r b i j")
    n = n - torch.eye(3, dtype=dtype, device=device).broadcast_to((2, 1, 3, 3))

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "r b i j -> b i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "r b i j, r b j -> r b i")
    rhs = reduce(rhs, "r b i -> b i", "sum")

    # Left-matrix-multiply both sides by the pseudo-inverse of lhs to find p.
    result = torch.linalg.lstsq(lhs, rhs).solution

    # Handle the case of parallel lines by setting depth to infinity.
    result_all = torch.ones(shape, dtype=dtype, device=device) * inf
    result_all[~parallel] = result
    return result_all


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)
