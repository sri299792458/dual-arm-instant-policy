
import numpy as np
import trimesh

def sample_box_surface(half_extents, n=8000, rng=None):
    """Sample points on a box surface centered at origin.
    half_extents: (hx, hy, hz)
    """
    rng = np.random.default_rng() if rng is None else rng
    hx, hy, hz = half_extents
    mesh = trimesh.creation.box(extents=(2*hx, 2*hy, 2*hz))
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)

def sample_cylinder_surface(radius, height, n=8000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)

def sample_ellipsoid_surface(radii, n=8000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    # start from sphere and scale vertices
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.vertices = mesh.vertices * np.asarray(radii)[None,:]
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)

def sample_table_plane(x_range=(0.0, 0.6), y_range=(-0.35, 0.35), z=0.82, n=4000, rng=None):
    """Uniform points on a table plane."""
    rng = np.random.default_rng() if rng is None else rng
    xs = rng.uniform(x_range[0], x_range[1], size=(n,))
    ys = rng.uniform(y_range[0], y_range[1], size=(n,))
    zs = np.full((n,), z, dtype=np.float64)
    return np.stack([xs, ys, zs], axis=-1)

def transform_points(T, pts):
    """Apply SE3 to points. pts: (N,3)"""
    pts_h = np.concatenate([pts, np.ones((pts.shape[0],1), dtype=np.float64)], axis=-1)
    out = (T @ pts_h.T).T[:, :3]
    return out

def downsample(pts, n=2048, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if pts.shape[0] <= n:
        return pts
    idx = rng.choice(pts.shape[0], size=(n,), replace=False)
    return pts[idx]
