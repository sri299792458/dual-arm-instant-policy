from __future__ import annotations

import torch

# ----------------------------
# SE(3) utilities (batched)
# ----------------------------

def make_homogeneous(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build 4x4 transforms from rotation R (...,3,3) and translation t (...,3)."""
    *batch, _, _ = R.shape
    T = torch.eye(4, device=R.device, dtype=R.dtype).expand(*batch, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T

def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert 4x4 SE(3) transform (batched)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    tinv = -(Rt @ t[..., None]).squeeze(-1)
    return make_homogeneous(Rt, tinv)

def apply_se3(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) to points.
    T: (...,4,4)
    pts: (...,N,3) or (...,3)
    returns same shape as pts
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    if pts.dim() >= 2 and pts.shape[-1] == 3:
        # (..., N, 3)
        return (R @ pts.transpose(-1, -2)).transpose(-1, -2) + t[..., None, :]
    raise ValueError(f"pts must end with 3, got {pts.shape}")

def kabsch_se3(src: torch.Tensor, dst: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Best-fit rigid transform T mapping src -> dst via Kabsch.
    src: (...,N,3), dst: (...,N,3)
    returns T: (...,4,4)
    """
    if src.shape != dst.shape or src.shape[-1] != 3:
        raise ValueError(f"src/dst shape mismatch: {src.shape} vs {dst.shape}")

    src_mean = src.mean(dim=-2, keepdim=True)
    dst_mean = dst.mean(dim=-2, keepdim=True)
    X = src - src_mean
    Y = dst - dst_mean

    # covariance: (...,3,3)
    H = X.transpose(-1, -2) @ Y
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-1, -2)

    # Proper rotation: handle reflection
    det = torch.det(V @ U.transpose(-1, -2))
    # build correction matrix
    eye = torch.eye(3, device=src.device, dtype=src.dtype).expand_as(H).clone()
    eye[..., 2, 2] = det
    R = V @ eye @ U.transpose(-1, -2)

    t = (dst_mean.squeeze(-2) - (R @ src_mean.squeeze(-2)[..., None]).squeeze(-1))
    return make_homogeneous(R, t)

def rotmat_from_two_vectors(x: torch.Tensor, z_hint: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Construct a rotation matrix whose x-axis aligns with vector x.
    z_hint is used to pick a consistent z-axis (e.g., world-up).
    x: (...,3)
    z_hint: (...,3) or None
    returns R: (...,3,3)
    """
    x = x / (x.norm(dim=-1, keepdim=True) + eps)
    if z_hint is None:
        z_hint = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype).expand_as(x)
    z = z_hint / (z_hint.norm(dim=-1, keepdim=True) + eps)

    # y = z × x, then re-orthogonalize z = x × y
    y = torch.cross(z, x, dim=-1)
    y = y / (y.norm(dim=-1, keepdim=True) + eps)
    z2 = torch.cross(x, y, dim=-1)
    z2 = z2 / (z2.norm(dim=-1, keepdim=True) + eps)

    R = torch.stack([x, y, z2], dim=-1)  # (...,3,3) columns are basis
    return R

def task_pose_from_grippers(T_left: torch.Tensor, T_right: torch.Tensor, z_hint: torch.Tensor | None = None) -> torch.Tensor:
    """Heuristic task frame from two gripper poses in world.
    - origin: midpoint between gripper origins
    - x-axis: left->right vector
    - z-axis: z_hint (default world up) projected orthogonal to x
    returns T_task_w (world) as (...,4,4)
    """
    pL = T_left[..., :3, 3]
    pR = T_right[..., :3, 3]
    origin = 0.5 * (pL + pR)
    x = (pR - pL)
    R = rotmat_from_two_vectors(x, z_hint=z_hint)
    return make_homogeneous(R, origin)
