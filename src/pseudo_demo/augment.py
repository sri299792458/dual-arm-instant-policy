
import numpy as np
from .se3 import jitter_T

def apply_common_noise(demo, rng=None,
                       pcd_sigma=0.0015,
                       pose_pos_sigma=0.0015,
                       pose_rot_sigma_deg=1.0,
                       drop_prob=0.05):
    """Lightweight augmentation to reduce overfitting to perfect geometry.
    - Adds gaussian noise to point clouds
    - Adds small pose jitter
    - Randomly drops a fraction of points (simulated occlusion)
    """
    rng = np.random.default_rng() if rng is None else rng
    out = {k: v for k, v in demo.items()}

    pcds = out['pcds'].copy()
    pcds += rng.normal(0.0, pcd_sigma, size=pcds.shape).astype(pcds.dtype)

    # point dropout
    if drop_prob > 0:
        T, N, _ = pcds.shape
        mask = rng.random((T, N)) < drop_prob
        # replace dropped points with copies of random survivors (keeps N fixed)
        for t in range(T):
            if mask[t].any():
                survivors = np.where(~mask[t])[0]
                if len(survivors) == 0:
                    continue
                repl = rng.choice(survivors, size=int(mask[t].sum()), replace=True)
                pcds[t, mask[t]] = pcds[t, repl]

    out['pcds'] = pcds

    Tl = out['T_w_es_left'].copy()
    Tr = out['T_w_es_right'].copy()
    for t in range(Tl.shape[0]):
        Tl[t] = jitter_T(Tl[t], pos_sigma=pose_pos_sigma, rot_sigma_deg=pose_rot_sigma_deg, rng=rng).astype(Tl.dtype)
        Tr[t] = jitter_T(Tr[t], pos_sigma=pose_pos_sigma, rot_sigma_deg=pose_rot_sigma_deg, rng=rng).astype(Tr.dtype)
    out['T_w_es_left'] = Tl
    out['T_w_es_right'] = Tr
    return out

def maybe_swap_left_right(demo, rng=None, p=0.5):
    """Symmetry augmentation for tasks where left/right roles are interchangeable."""
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() > p:
        return demo
    out = {k: v for k, v in demo.items()}
    out['T_w_es_left'], out['T_w_es_right'] = out['T_w_es_right'], out['T_w_es_left']
    out['grips_left'], out['grips_right'] = out['grips_right'], out['grips_left']
    meta = out.get('meta', {})
    if isinstance(meta, dict):
        meta = dict(meta)
        meta['swapped_left_right'] = True
    out['meta'] = meta
    return out
