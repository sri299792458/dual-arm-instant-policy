"""
Clean SE(3) operations for bimanual manipulation
Minimal, essential functions only
"""

import torch


def task_pose_from_grippers(T_left: torch.Tensor, T_right: torch.Tensor) -> torch.Tensor:
    """
    Compute task frame from two gripper poses.

    Task frame:
    - Origin: midpoint between grippers
    - X-axis: left → right vector
    - Z-axis: world up, orthogonalized

    Args:
        T_left: (..., 4, 4) left gripper poses in world
        T_right: (..., 4, 4) right gripper poses in world

    Returns:
        T_task: (..., 4, 4) task frame poses in world
    """
    # Extract positions
    p_left = T_left[..., :3, 3]   # (..., 3)
    p_right = T_right[..., :3, 3]  # (..., 3)

    # Midpoint
    origin = 0.5 * (p_left + p_right)

    # X-axis: left to right
    x_axis = p_right - p_left
    x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)

    # Z-axis: world up, orthogonalized to x
    z_hint = torch.tensor([0., 0., 1.], device=T_left.device, dtype=T_left.dtype)
    z_hint = z_hint.expand_as(x_axis)

    # Y = Z × X
    y_axis = torch.cross(z_hint, x_axis, dim=-1)
    y_axis = y_axis / (torch.norm(y_axis, dim=-1, keepdim=True) + 1e-8)

    # Z = X × Y (re-orthogonalize)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    z_axis = z_axis / (torch.norm(z_axis, dim=-1, keepdim=True) + 1e-8)

    # Build rotation matrix
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (..., 3, 3)

    # Build 4x4 transform
    T_task = torch.eye(4, device=T_left.device, dtype=T_left.dtype).expand(*T_left.shape[:-2], 4, 4).clone()
    T_task[..., :3, :3] = R
    T_task[..., :3, 3] = origin

    return T_task


def apply_se3(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(3) transform to points.

    Args:
        T: (..., 4, 4) transforms
        pts: (..., N, 3) points

    Returns:
        transformed: (..., N, 3) transformed points
    """
    R = T[..., :3, :3]  # (..., 3, 3)
    t = T[..., :3, 3]   # (..., 3)

    # Transform: R @ pts^T + t
    # pts: (..., N, 3) -> (..., 3, N)
    transformed = torch.matmul(R, pts.transpose(-1, -2)).transpose(-1, -2)
    transformed = transformed + t[..., None, :]

    return transformed


def kabsch_se3(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Find best rigid transform T: src -> dst using Kabsch algorithm.

    Args:
        src: (..., N, 3) source points
        dst: (..., N, 3) destination points

    Returns:
        T: (..., 4, 4) rigid transform
    """
    # Center points
    src_mean = src.mean(dim=-2, keepdim=True)
    dst_mean = dst.mean(dim=-2, keepdim=True)

    X = src - src_mean
    Y = dst - dst_mean

    # Covariance matrix
    H = torch.matmul(X.transpose(-1, -2), Y)  # (..., 3, 3)

    # SVD
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-1, -2)

    # Rotation (handle reflection)
    R = torch.matmul(V, U.transpose(-1, -2))

    # Fix reflection if det(R) < 0
    det = torch.det(R)
    V_fixed = V.clone()
    V_fixed[..., :, -1] = V[..., :, -1] * det[..., None]
    R = torch.matmul(V_fixed, U.transpose(-1, -2))

    # Translation
    t = dst_mean.squeeze(-2) - torch.matmul(R, src_mean.squeeze(-2)[..., None]).squeeze(-1)

    # Build 4x4 transform
    T = torch.eye(4, device=src.device, dtype=src.dtype).expand(*src.shape[:-2], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t

    return T
