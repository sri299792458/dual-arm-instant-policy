
import os
import torch
import numpy as np
import shutil
from tqdm import tqdm
import sys

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import from local modules
from pseudo_data_generator import generate_traj
from utils.data_proc import sample_to_cond_demo, sample_to_live, save_sample

def generate_profiling_data(num_samples=50, save_dir='src/data/profiling_data'):
    if os.path.exists(save_dir):
        print(f"Directory {save_dir} exists. Cleaning up...")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    print(f"Generating {num_samples} pseudo-trajectories...")
    
    # We need enough pseudo-trajectories to form context
    # Each sample needs num_demos (2) context + 1 live
    # So we generate num_samples * 2 temporary trajectories to draw from
    
    num_demos = 2
    num_waypoints_demo = 10
    pred_horizon = 8
    live_spacing_trans = 0.01
    live_spacing_rot = 3
    
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }
    
    for k in tqdm(range(num_samples)):
        # Generate on-the-fly to avoid saving intermediate .npz
        demos = []
        # Generate 3 trajectories: 2 for context, 1 for live
        for _ in range(num_demos + 1):
            traj, _ = generate_traj()
            demos.append(traj)
            
        for i, sample in enumerate(demos):
            if i < num_demos:
                full_sample['demos'][i] = sample_to_cond_demo(sample, num_waypoints_demo)
            else:
                full_sample['live'] = sample_to_live(sample, pred_horizon, 2048,
                                                    live_spacing_trans, live_spacing_rot, subsample=True)
        
        # Save as .pt
        # Using None for scene_encoder means embeddings won't be pre-computed
        save_sample(full_sample, save_dir=save_dir, offset=k, scene_encoder=None)

    print(f"Generated {num_samples} samples in {save_dir}")

if __name__ == "__main__":
    generate_profiling_data()
