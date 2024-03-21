import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def interpolate_intrinsics(
    initial: Float[Tensor, "*#batch 3 3"],
    final: Float[Tensor, "*#batch 3 3"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 3 3"]:
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: Float[Tensor, "*#batch dim"],
    a_directions: Float[Tensor, "*#batch dim"],
    b_origins: Float[Tensor, "*#batch dim"],
    b_directions: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*batch dim"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast and stack the tensors.
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "... n i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")

    # Left-matrix-multiply both sides by the inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: Float[Tensor, "*#batch dim"]) -> Float[Tensor, "*#batch dim"]:
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: Float[Tensor, "*#batch 3"],
    z: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame given perpendicular, unit-length Y and Z vectors."""
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([y.cross(z), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: Float[Tensor, "*#batch 3"],
    b: Float[Tensor, "*#batch 3"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame where the Y direction is normal to the plane defined
    by unit vectors a and b. The other axes are arbitrary."""
    device = a.device

    # Replace every entry in b that's parallel to the corresponding entry in a with an
    # arbitrary vector.
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)

    # Generate the coordinate frame. The initial cross product defines the plane.
    return generate_coordinate_frame(normalize(a.cross(b)), a)


def matrix_to_euler(
    rotations: Float[Tensor, "*batch 3 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3"]:
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    rotations = torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: Float[Tensor, "*batch 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3 3"]:
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    rotations = torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 5"]:
    """Convert the extrinsics to a representation with 5 degrees of freedom:
    1. Distance from pivot point in the "X" (look cross pivot axis) direction.
    2. Distance from pivot point in the "Y" (pivot axis) direction.
    3. Distance from pivot point in the Z (look) direction
    4. Angle in plane
    5. Twist (rotation not in plane)
    """

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    # Compute the translation elements of the pivot parametrization.
    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")

    # Add the rotation elements of the pivot parametrization.
    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: Float[Tensor, "*#batch 5"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 4 4"]:
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., 3, 3] = 1
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin
    return extrinsics


def interpolate_circular(
    a: Float[Tensor, "*#batch"],
    b: Float[Tensor, "*#batch"],
    t: Float[Tensor, "*#batch"],
) -> Float[Tensor, " *batch"]:
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau

    # Consider piecewise edge cases.
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: Float[Tensor, "*#batch 5"],
    final: Float[Tensor, "*#batch 5"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 5"]:
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def interpolate_extrinsics(
    initial: Float[Tensor, "*#batch 4 4"],
    final: Float[Tensor, "*#batch 4 4"],
    t: Float[Tensor, " time_step"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch time_step 4 4"]:
    """Interpolate extrinsics by rotating around their "focus point," which is the
    least-squares intersection between the look vectors of the initial and final
    extrinsics.
    """

    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)

    # Based on the dot product between the look vectors, pick from one of two cases:
    # 1. Look vectors are parallel: interpolate about their origins' midpoint.
    # 3. Look vectors aren't parallel: interpolate about their focus point.
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps

    # Pick focus points.
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )

    # Convert to pivot parameters.
    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)

    # Interpolate the pivot parameters.
    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)

    # Convert back.
    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )
