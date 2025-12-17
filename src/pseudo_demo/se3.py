
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R, Slerp

EPS = 1e-9

def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric hat operator for R^3."""
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=np.float64)

def make_T(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = Rm.T
    Ti[:3,3] = -Rm.T @ t
    return Ti

def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B

def rot_x(theta): 
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def rot_y(theta): 
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def rot_z(theta): 
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def slerp_R(R0: np.ndarray, R1: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Slerp between two rotation matrices. Returns (len(ts),3,3)."""
    key_rots = R.from_matrix([R0, R1])
    slerp = Slerp([0.0, 1.0], key_rots)
    return slerp(ts).as_matrix()

def interp_SE3(T0: np.ndarray, T1: np.ndarray, num: int) -> np.ndarray:
    """Interpolate SE(3) with Slerp for rotation and linear for translation."""
    ts = np.linspace(0.0, 1.0, num, dtype=np.float64)
    Rts = slerp_R(T0[:3,:3], T1[:3,:3], ts)
    p0, p1 = T0[:3,3], T1[:3,3]
    pts = (1.0 - ts)[:,None]*p0[None,:] + ts[:,None]*p1[None,:]
    out = np.repeat(np.eye(4, dtype=np.float64)[None,:,:], num, axis=0)
    out[:,:3,:3] = Rts
    out[:,:3,3] = pts
    return out

def bezier_positions(p0, p1, num: int, bow: float = 0.25, up=(0,0,1.0), rng=None):
    """Cubic Bezier in R^3 with a lateral bow (for nicer approach arcs)."""
    up = np.asarray(up, dtype=np.float64)
    d = p1 - p0
    ts = np.linspace(0.0, 1.0, num, dtype=np.float64)
    if np.linalg.norm(d) < 1e-6:
        return np.repeat(p0[None,:], num, axis=0)

    n = np.cross(d, up)
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(d, np.array([0,1,0], dtype=np.float64))
        if np.linalg.norm(n) < 1e-6:
            n = np.array([1,0,0], dtype=np.float64)
    n = n / (np.linalg.norm(n) + EPS)

    offset = bow * np.linalg.norm(d) * n
    C0 = p0
    C1 = p0 + (1.0/3.0)*d + offset
    C2 = p0 + (2.0/3.0)*d + offset
    C3 = p1

    t = ts[:,None]
    omt = (1.0 - t)
    pts = (omt**3)*C0 + 3*(omt**2)*t*C1 + 3*omt*(t**2)*C2 + (t**3)*C3
    return pts

def bezier_SE3(T0: np.ndarray, T1: np.ndarray, num: int, bow: float = 0.25, up=(0,0,1.0)) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, num, dtype=np.float64)
    Rts = slerp_R(T0[:3,:3], T1[:3,:3], ts)
    pts = bezier_positions(T0[:3,3], T1[:3,3], num=num, bow=bow, up=up)
    out = np.repeat(np.eye(4, dtype=np.float64)[None,:,:], num, axis=0)
    out[:,:3,:3] = Rts
    out[:,:3,3] = pts
    return out

def jitter_T(T: np.ndarray, pos_sigma=0.002, rot_sigma_deg=1.0, rng=None) -> np.ndarray:
    """Small random perturbation (useful augmentation)."""
    rng = np.random.default_rng() if rng is None else rng
    dpos = rng.normal(0.0, pos_sigma, size=(3,))
    drot = R.from_rotvec(rng.normal(0.0, np.deg2rad(rot_sigma_deg), size=(3,)))
    Tj = T.copy()
    Tj[:3,3] = T[:3,3] + dpos
    Tj[:3,:3] = drot.as_matrix() @ T[:3,:3]
    return Tj
