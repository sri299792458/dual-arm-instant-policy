#!/usr/bin/env python3
"""Test that the reference implementation can be imported and used"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config.train_config import config

print("="*60)
print("Testing Bimanual Reference Implementation Integration")
print("="*60)

# Test 1: Import SE3 utilities
print("\n[1/5] Testing SE3 utilities...")
try:
    from models.se3_utils import task_pose_from_grippers, kabsch_se3
    T_left = torch.eye(4)[None, ...]
    T_left[0, :3, 3] = torch.tensor([0.5, 0.1, 1.0])
    T_right = torch.eye(4)[None, ...]
    T_right[0, :3, 3] = torch.tensor([0.5, -0.1, 1.0])
    T_task = task_pose_from_grippers(T_left, T_right)
    print(f"✓ task_pose_from_grippers works!")
    print(f"  Task frame origin: {T_task[0, :3, 3].tolist()}")
    print(f"  Expected: [0.5, 0.0, 1.0]")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Import positional encoding
print("\n[2/5] Testing positional encoding...")
try:
    from models.positional import PositionalEncoder
    pos_enc = PositionalEncoder(3, 10)
    test_pos = torch.randn(10, 3)
    encoded = pos_enc(test_pos)
    print(f"✓ PositionalEncoder works!")
    print(f"  Input shape: {test_pos.shape}, Output shape: {encoded.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Import graph builder
print("\n[3/5] Testing graph builder...")
try:
    from models.graph_ref_adapted import BimanualGraphBuilder
    graph_builder = BimanualGraphBuilder(
        batch_size=config['batch_size'],
        num_demos=config['num_demos'],
        traj_horizon=config['traj_horizon'],
        pred_horizon=config['pred_horizon'],
        num_scene_nodes=config['num_scenes_nodes'],
        num_gripper_kp=config['num_gripper_kp'],
        num_task_kp=config['num_task_kp'],
        device=config['device'],
    )
    print(f"✓ BimanualGraphBuilder instantiated!")
    print(f"  Node types: {len(graph_builder._nodeinfo_local)} (should be 4: scene, left, right, task)")
    print(f"  Edge types: {len(graph_builder._edges_local)} (local edges)")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Import model
print("\n[4/5] Testing model...")
try:
    from models.model_ref_adapted import BimanualInstantPolicy, BimanualInstantPolicyConfig
    model_config = BimanualInstantPolicyConfig(
        batch_size=config['batch_size'],
        num_demos=config['num_demos'],
        traj_horizon=config['traj_horizon'],
        pred_horizon=config['pred_horizon'],
        num_scene_nodes=config['num_scenes_nodes'],
        num_gripper_kp=config['num_gripper_kp'],
        num_task_kp=config['num_task_kp'],
        d_model=config['hidden_dim'],
        heads=config['hidden_dim'] // 64,
        num_layers=config['num_layers'],
        diffusion_steps=config['num_diffusion_iters_train'],
        parameterization=config['parameterization'],
        device=config['device'],
    )
    model = BimanualInstantPolicy(model_config)
    print(f"✓ BimanualInstantPolicy instantiated!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test data generation
print("\n[5/5] Testing data generation...")
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from pseudo_demo.tasks import cooperative_rigid, Workspace, default_arm_starts
    import numpy as np

    rng = np.random.RandomState(42)
    ws = Workspace()
    starts = default_arm_starts()
    demo = cooperative_rigid(rng, ws, starts)

    print(f"✓ cooperative_rigid works!")
    print(f"  Keys: {list(demo.keys())}")
    print(f"  Left trajectory shape: {demo['T_w_es_left'].shape}")
    print(f"  Right trajectory shape: {demo['T_w_es_right'].shape}")
    print(f"  Point clouds shape: {demo['pcds'].shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Integration Test Complete!")
print("="*60)
