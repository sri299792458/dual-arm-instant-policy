
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from .tasks import Workspace, default_arm_starts, cooperative_rigid, independent_two_objects, handover
from .augment import apply_common_noise, maybe_swap_left_right

TASKS = {
    'cooperative_rigid': cooperative_rigid,
    'independent': independent_two_objects,
    'handover': handover,
}

def generate_demo(task_type: Optional[str] = None,
                  seed: Optional[int] = None,
                  augment: bool = True,
                  task_probs: Optional[Dict[str, float]] = None):
    rng = np.random.default_rng(seed)
    ws = Workspace()
    starts = default_arm_starts()

    if task_type is None:
        if task_probs is None:
            task_probs = {'cooperative_rigid': 0.55, 'handover': 0.20, 'independent': 0.25}
        keys = list(task_probs.keys())
        probs = np.array([task_probs[k] for k in keys], dtype=np.float64)
        probs /= probs.sum()
        task_type = rng.choice(keys, p=probs)

    demo = TASKS[task_type](rng=rng, ws=ws, starts=starts)

    if augment:
        demo = apply_common_noise(demo, rng=rng)
        # Only swap on symmetric tasks
        if demo['meta'].get('task_type') in ('cooperative_rigid', 'independent'):
            demo = maybe_swap_left_right(demo, rng=rng, p=0.5)

    # pack meta as JSON string for npz
    meta = demo.get('meta', {})
    demo_out = {
        'pcds': demo['pcds'].astype(np.float32),
        'T_w_es_left': demo['T_w_es_left'].astype(np.float32),
        'T_w_es_right': demo['T_w_es_right'].astype(np.float32),
        'grips_left': demo['grips_left'].astype(np.int8),
        'grips_right': demo['grips_right'].astype(np.int8),
        'meta': np.array(json.dumps(meta), dtype=np.unicode_)
    }
    return demo_out

def generate_dataset(out_dir: str,
                     num: int = 1000,
                     seed: int = 0,
                     augment: bool = True,
                     task_probs: Optional[Dict[str, float]] = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(num):
        demo_seed = int(rng.integers(0, 2**31-1))
        demo = generate_demo(seed=demo_seed, augment=augment, task_probs=task_probs)
        np.savez_compressed(out / f"traj_{i:06d}.npz", **demo)

    # write a small manifest
    manifest = {
        'num': num,
        'seed': seed,
        'augment': augment,
        'task_probs': task_probs,
        'format': {
            'pcds': '(T,N,3) world-frame point cloud',
            'T_w_es_left': '(T,4,4) left gripper pose in world',
            'T_w_es_right': '(T,4,4) right gripper pose in world',
            'grips_left': '(T,) 1=open 0=closed',
            'grips_right': '(T,) 1=open 0=closed',
            'meta': 'JSON string'
        }
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
